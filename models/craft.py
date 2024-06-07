import torch
import torch.nn.functional as F
from torch import Tensor, nn

try:
    from models.vgg16_bn import *
except ModuleNotFoundError as _:
    from vgg16_bn import *


__all__ = [
    "CRAFT",
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


class CRAFT(nn.Module):
    def __init__(self) -> None:
        """コンストラクタ
        """
        super().__init__()

        # Base network
        self.basenet = VGG16BN(VGG16_BN_Weights.IMAGENET1K_V1)

        # U network
        self.upconv1 = DoubleConv(1024, 512, 256)
        self.upconv2 = DoubleConv(512, 256, 128)
        self.upconv3 = DoubleConv(256, 128, 64)
        self.upconv4 = DoubleConv(128, 64, 32)

        num_class = 1 # region only
        self.conv_cls = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, num_class, kernel_size=1),
        )

        init_weights(self.upconv1.modules())
        init_weights(self.upconv2.modules())
        init_weights(self.upconv3.modules())
        init_weights(self.upconv4.modules())
        init_weights(self.conv_cls.modules())

    def forward(self, x:Tensor) -> Tensor:
        # Base network
        sources = self.basenet(x)

        # U network
        y = torch.cat((sources[0], sources[1]), dim=1)
        y = self.upconv1(y)

        y = F.interpolate(y, scale_factor=2, mode="bilinear", align_corners=False)
        y = torch.cat((y, sources[2]), dim=1)
        y = self.upconv2(y)

        y = F.interpolate(y, scale_factor=2, mode="bilinear", align_corners=False)
        y = torch.cat((y, sources[3]), dim=1)
        y = self.upconv3(y)

        y = F.interpolate(y, scale_factor=2, mode="bilinear", align_corners=False)
        y = torch.cat((y, sources[4]), dim=1)
        y = self.upconv4(y)

        y = self.conv_cls(y)

        return y


if __name__ == "__main__":
    model = CRAFT()
    inputs = torch.randn(1, 3, 864, 864)
    outputs = model(inputs)
