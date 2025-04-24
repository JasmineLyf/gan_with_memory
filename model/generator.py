import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.init_size = Config.img_size // 4
        self.latent_dim = Config.latent_dim

        self.gru = nn.GRU(self.latent_dim, self.latent_dim, batch_first=True)
        self.fusion = nn.Sequential(
            nn.Linear(self.latent_dim * 2, self.latent_dim),
            nn.Sigmoid()
        )

        self.l1 = nn.Linear(self.latent_dim, 128 * self.init_size ** 2)

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, Config.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z, h_mem=None):
        if h_mem is None or h_mem.size(1) != z.size(0):
            h_mem = torch.zeros(1, z.size(0), self.latent_dim).to(z.device)

        _, h_out = self.gru(z.unsqueeze(1), h_mem)
        fused_z = self.fusion(torch.cat([z, h_out.squeeze(0)], dim=1))

        out = self.l1(fused_z)
        out = out.view(out.size(0), 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img, h_out.detach()
