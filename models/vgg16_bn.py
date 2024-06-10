import torch
from torch import Tensor, nn
from torchvision.models import vgg
from torchvision.models.vgg import VGG16_BN_Weights


__all__ = [
    "VGG16BN",
    "VGG16_BN_Weights",
    "init_weights",
]


def init_weights(modules) -> None:
    for m in modules:
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.zeros_(m.bias)


class VGG16BN(nn.Module):
    def __init__(
        self,
        weights:VGG16_BN_Weights = VGG16_BN_Weights.IMAGENET1K_V1,
    ) -> None:
        super().__init__()

        vgg_pretrained_features = vgg.vgg16_bn(weights=weights).features

        self.conv1 = nn.Sequential()
        self.conv2 = nn.Sequential()
        self.conv3 = nn.Sequential()
        self.conv4 = nn.Sequential()

        # conv1_2
        for x in range(6):
            self.conv1.add_module(str(x), vgg_pretrained_features[x])
        # conv2_2
        for x in range(6, 13):
            self.conv2.add_module(str(x), vgg_pretrained_features[x])
        # conv3_3
        for x in range(13, 23):
            self.conv3.add_module(str(x), vgg_pretrained_features[x])
        # conv4_3
        for x in range(24, 32):  # ignore 23 maxpool
            self.conv4.add_module(str(x), vgg_pretrained_features[x])

    def forward(self, x:Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        h_relu1_2 = self.conv1(x)
        h_relu2_2 = self.conv2(h_relu1_2)
        h_relu3_3 = self.conv3(h_relu2_2)
        h_relu4_3 = self.conv4(h_relu3_3)
        return h_relu4_3, h_relu3_3, h_relu2_2, h_relu1_2
