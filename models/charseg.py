import torch
from torch import nn, Tensor

try:
    from models.vgg16_bn import *
except ModuleNotFoundError as _:
    from vgg16_bn import *


__all__ = [
    "CharSeg",
]


class DoubleConv(nn.Module):
    def __init__(
        self,
        in_channels:int,
        mid_channels:int,
        out_channels:int,
    ) -> None:
        """コンストラクタ

        Args:
            in_channels (int): 入力チャンネル数
            mid_channels (int): 中間チャンネル数
            out_channels (int): 出力チャンネル数
        """
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + mid_channels, mid_channels, kernel_size=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x:Tensor) -> Tensor:
        x = self.conv(x)
        return x


class AttentionBlock(nn.Module):
    def __init__(self, f_g:int, f_l:int, f_int:int):
        super().__init__()

        self.w_g = nn.Sequential(
            nn.Conv2d(f_g, f_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(f_int)
        )

        self.w_x = nn.Sequential(
            nn.Conv2d(f_l, f_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(f_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(f_int, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g:Tensor, x:Tensor) -> Tensor:
        g1 = self.w_g(g)
        x1 = self.w_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)
        return psi * x


class CharSeg(nn.Module):
    def __init__(self) -> None:
        """コンストラクタ
        """
        super().__init__()

        # encoder
        self.basenet = VGG16BN(VGG16_BN_Weights.IMAGENET1K_V1)

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )

        self.attn2 = AttentionBlock(256, 128, 128)
        self.attn3 = AttentionBlock(128, 64, 64)

        # decoder
        self.conv1 = DoubleConv(512, 256, 256)
        self.conv2 = DoubleConv(256, 128, 128)
        self.conv3 = DoubleConv(128, 64, 64)

        # head
        self.conv_cls = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
        )

        init_weights(self.up2.modules())
        init_weights(self.attn2.modules())
        init_weights(self.up3.modules())
        init_weights(self.attn3.modules())

        init_weights(self.conv1.modules())
        init_weights(self.conv2.modules())
        init_weights(self.conv3.modules())

        init_weights(self.conv_cls.modules())

    def forward(self, x:Tensor) -> Tensor:
        # encode
        sources = self.basenet(x)

        # decode
        y = torch.cat((sources[0], sources[1]), dim=1)
        y = self.conv1(y)

        y = self.up2(y)
        y = torch.cat((y, self.attn2(y, sources[2])), dim=1)
        y = self.conv2(y)

        y = self.up3(y)
        y = torch.cat((y, self.attn3(y, sources[3])), dim=1)
        y = self.conv3(y)

        # head
        y = self.conv_cls(y)
        return y


if __name__ == "__main__":
    model = CharSeg()
    inputs = torch.randn(1, 3, 32, 32)
    outputs = model(inputs)
    print(outputs.shape)
