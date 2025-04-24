import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from config import Config
from model.memory_bank import MemoryBank
from model.gnn_encoder import GNNEncoder


class WGANGPDiscriminator(nn.Module):
    def __init__(self, use_gru=False, use_memory=True, use_gnn=True):
        super(WGANGPDiscriminator, self).__init__()
        self.use_gru = use_gru
        self.use_memory = use_memory
        self.use_gnn = use_gnn

        def block(in_f, out_f):
            return nn.Sequential(
                spectral_norm(nn.Conv2d(in_f, out_f, 3, 2, 1)),
                nn.LeakyReLU(0.2),
                nn.BatchNorm2d(out_f),
                spectral_norm(nn.Conv2d(out_f, out_f, 3, 1, 1)),
                nn.LeakyReLU(0.2),
                nn.Dropout2d(0.4)
            )

        self.model = nn.Sequential(
            block(Config.channels, 32),
            block(32, 64),
            block(64, 128),
        )

        ds_size = Config.img_size // 2 ** 3
        self.feature_fc = nn.Linear(128 * ds_size ** 2, Config.mem_key_dim)

        if use_gru:
            self.gru = nn.GRU(Config.mem_key_dim, Config.mem_key_dim, batch_first=True)

        if use_memory:
            self.memory_bank = MemoryBank(Config.mem_key_dim, Config.mem_val_dim, Config.mem_capacity)

        if use_gnn:
            self.gnn = GNNEncoder(Config.mem_key_dim, Config.gnn_hidden_dim)

        fuse_dim = Config.mem_key_dim
        if use_memory:
            fuse_dim += Config.mem_val_dim
        if use_gnn:
            fuse_dim += Config.gnn_hidden_dim

        self.fuse = nn.Sequential(
            nn.Linear(fuse_dim, 128),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(128, 1))
        )

    def forward(self, img, h_mem=None):
        out = self.model(img)
        out = out.view(out.size(0), -1)
        h_cur = self.feature_fc(out)

        if self.use_gru:
            if h_mem is None or h_mem.size(1) != img.size(0):
                h_mem = torch.zeros(1, img.size(0), Config.mem_key_dim).to(img.device)
            _, h_out = self.gru(h_cur.unsqueeze(1), h_mem)
        else:
            h_out = torch.zeros_like(h_cur).unsqueeze(0)

        mem_val = torch.zeros_like(h_cur) if not self.use_memory else self.memory_bank.fuse(h_cur)
        gnn_feat = torch.zeros_like(h_cur) if not self.use_gnn else self.gnn(h_cur)

        fused = torch.cat([h_cur, mem_val, gnn_feat], dim=1)
        score = self.fuse(fused)
        return score, h_out.detach()
