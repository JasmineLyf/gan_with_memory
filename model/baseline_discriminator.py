import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from config import Config

class BaselineDiscriminator(nn.Module):
    def __init__(self):
        super(BaselineDiscriminator, self).__init__()

        def block(in_channels, out_channels, normalize=True):
            layers = [spectral_norm(nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1))]
            if normalize:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(Config.channels, 64, normalize=False),
            *block(64, 128),
            *block(128, 256),
            *block(256, 512),
        )

        ds_size = Config.img_size // 2 ** 4
        self.adv_layer = nn.Sequential(
            nn.Flatten(),
            spectral_norm(nn.Linear(512 * ds_size * ds_size, 1))
        )

    def forward(self, img):
        out = self.model(img)
        validity = self.adv_layer(out)
        return validity, None
