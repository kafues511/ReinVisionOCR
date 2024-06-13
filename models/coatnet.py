import torch
from torch import nn, Tensor
from einops import rearrange
from einops.layers.torch import Rearrange


__all__ = [
    "CoAtNet",
]


class PreNorm(nn.Module):
    def __init__(self, dim:int, fn:nn.Module, norm:nn.Module) -> None:
        super().__init__()

        self.norm = norm(dim)
        self.fn = fn

    def forward(self, x:Tensor, **kwargs) -> Tensor:
        return self.fn(self.norm(x), **kwargs)


class SE(nn.Module):
    def __init__(self, inp:int, oup:int, expansion:float = 0.25) -> None:
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(oup, int(inp * expansion), bias=False),
            nn.GELU(),
            nn.Linear(int(inp * expansion), oup, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x:Tensor) -> Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class FeedForward(nn.Module):
    def __init__(self, dim:int, hidden_dim:int, dropout:float = 0.0) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x:Tensor) -> Tensor:
        return self.net(x)


class MBConv(nn.Module):
    def __init__(
        self,
        inp:int,
        oup:int,
        image_size:tuple[int, int],
        downsample:bool = False,
        expansion:int = 4,
    ) -> None:
        super().__init__()

        self.downsample = downsample
        stride = 1 if not self.downsample else 2
        hidden_dim = int(inp * expansion)

        if self.downsample:
            self.pool = nn.MaxPool2d(3, 2, 1)
            self.proj = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                # down-sample in the first conv
                nn.Conv2d(inp, hidden_dim, 1, stride, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                SE(inp, hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

        self.conv = PreNorm(inp, self.conv, nn.BatchNorm2d)

    def forward(self, x:Tensor) -> Tensor:
        if self.downsample:
            return self.proj(self.pool(x)) + self.conv(x)
        else:
            return x + self.conv(x)


class Attention(nn.Module):
    def __init__(
        self,
        inp:int,
        oup:int,
        image_size:tuple[int, int],
        heads:int = 8,
        dim_head:int =  32,
        dropout:float = 0.0,
    ) -> None:
        super().__init__()

        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == inp)

        self.ih, self.iw = image_size

        self.heads = heads
        self.scale = dim_head ** -0.5

        # parameter table of relative position bias
        self.relative_bias_table = nn.Parameter(torch.zeros((2 * self.ih - 1) * (2 * self.iw - 1), heads))

        coords = torch.meshgrid((torch.arange(self.ih), torch.arange(self.iw)), indexing="ij")
        coords = torch.flatten(torch.stack(coords), 1)
        relative_coords = coords[:, :, None] - coords[:, None, :]

        relative_coords[0] += self.ih - 1
        relative_coords[1] += self.iw - 1
        relative_coords[0] *= 2 * self.iw - 1
        relative_coords = rearrange(relative_coords, "c h w -> h w c")
        relative_index = relative_coords.sum(-1).flatten().unsqueeze(1)
        self.register_buffer("relative_index", relative_index)

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(inp, inner_dim * 3, bias=False)

        if project_out:
            self.to_out = nn.Sequential(
                nn.Linear(inner_dim, oup),
                nn.Dropout(dropout),
            )
        else:
            nn.Identity()

    def forward(self, x:Tensor) -> Tensor:
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # Use "gather" for more efficiency on GPUs
        relative_bias = self.relative_bias_table.gather(0, self.relative_index.repeat(1, self.heads))
        relative_bias = rearrange(relative_bias, "(h w) c -> 1 c h w", h=self.ih*self.iw, w=self.ih*self.iw)
        dots = dots + relative_bias

        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(
        self,
        inp:int,
        oup:int,
        image_size:tuple[int, int],
        heads:int = 8,
        dim_head:int = 32,
        downsample:bool = False,
        dropout:float = 0.0,
    ) -> None:
        super().__init__()

        hidden_dim = int(inp * 4)

        self.ih, self.iw = image_size
        self.downsample = downsample

        if self.downsample:
            self.pool1 = nn.MaxPool2d(3, 2, 1)
            self.pool2 = nn.MaxPool2d(3, 2, 1)
            self.proj = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)

        self.attn = Attention(inp, oup, image_size, heads, dim_head, dropout)
        self.ff = FeedForward(oup, hidden_dim, dropout)

        self.attn = nn.Sequential(
            Rearrange("b c ih iw -> b (ih iw) c"),
            PreNorm(inp, self.attn, nn.LayerNorm),
            Rearrange("b (ih iw) c -> b c ih iw", ih=self.ih, iw=self.iw),
        )

        self.ff = nn.Sequential(
            Rearrange("b c ih iw -> b (ih iw) c"),
            PreNorm(oup, self.ff, nn.LayerNorm),
            Rearrange("b (ih iw) c -> b c ih iw", ih=self.ih, iw=self.iw),
        )

    def forward(self, x:Tensor) -> Tensor:
        if self.downsample:
            x = self.proj(self.pool1(x)) + self.attn(self.pool2(x))
        else:
            x = x + self.attn(x)
        x = x + self.ff(x)
        return x


class CoAtNet(nn.Module):
    def __init__(
        self,
        image_size:tuple[int, int],
        in_channels:int,
        num_blocks:tuple[int, int, int, int, int],
        channels:tuple[int, int, int, int, int],
        num_classes:int = 1000,
        block_types:tuple[str, str, str, str] = ("C", "C", "T", "T"),
    ) -> None:
        """コンストラクタ

        Args:
            image_size (tuple[int, int]): 入力サイズ (横幅 x 縦幅)
            in_channels (int): 入力チャンネル数
            num_blocks (tuple[int, int, int, int, int]): _description_
            channels (tuple[int, int, int, int, int]): _description_
            num_classes (int, optional): _description_. 分類数 to 1000.
            block_types (tuple[str, str, str, str], optional): "C" is MBConv, "T" is Transformer. Defaults to ("C", "C", "T", "T").
        """
        super().__init__()

        ih, iw = image_size

        block = { "C": MBConv, "T": Transformer }

        self.s0 = self.make_layer(self.conv_3x3_bn,      in_channels, channels[0], num_blocks[0], (ih// 2, iw// 2))
        self.s1 = self.make_layer(block[block_types[0]], channels[0], channels[1], num_blocks[1], (ih// 4, iw// 4))
        self.s2 = self.make_layer(block[block_types[1]], channels[1], channels[2], num_blocks[2], (ih// 8, iw// 8))
        self.s3 = self.make_layer(block[block_types[2]], channels[2], channels[3], num_blocks[3], (ih//16, iw//16))

        self.pool = nn.AvgPool2d(ih // 16, 1)
        self.fc = nn.Linear(channels[-1], num_classes, bias=False)

    def forward(self, x:Tensor) -> Tensor:
        x = self.s0(x)
        x = self.s1(x)
        x = self.s2(x)
        x = self.s3(x)

        x = self.pool(x).view(-1, x.shape[1])
        x = self.fc(x)
        return x

    def conv_3x3_bn(self, inp:int, oup:int, image_size:tuple[int, int], downsample:bool = False) -> nn.Sequential:
        stride = 1 if not downsample else 2
        return nn.Sequential(
            nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
            nn.BatchNorm2d(oup),
            nn.GELU(),
        )

    def make_layer(self, block:nn.Module, inp:int, oup:int, depth:int, image_size:tuple[int, int]) -> nn.Sequential:
        layers = nn.ModuleList([])
        for i in range(depth):
            if i == 0:
                layers.append(block(inp, oup, image_size, downsample=True))
            else:
                layers.append(block(oup, oup, image_size))
        return nn.Sequential(*layers)


if __name__ == "__main__":
    model = CoAtNet(
        (32, 32),
        1,
        (2, 2, 3, 5),
        (64, 96, 192, 384),
        1000,
        ("C", "C", "T"),
    )
    inputs = torch.randn(1, 1, 32, 32)
    outputs = model(inputs)
    print(outputs.shape)
