import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config
from model.memory_bank import MemoryBank
from model.gnn_encoder import GNNEncoder


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def block(in_f, out_f, bn=True):
            layers = [nn.Conv2d(in_f, out_f, 3, 2, 1), nn.LeakyReLU(0.2), nn.Dropout2d(0.25)]
            if bn:
                layers.append(nn.BatchNorm2d(out_f, 0.8))
            return layers

        self.model = nn.Sequential(
            *block(Config.channels, 16, bn=False),
            *block(16, 32),
            *block(32, 64),
            *block(64, 128),
        )

        ds_size = Config.img_size // 2 ** 4
        self.feature_fc = nn.Linear(128 * ds_size ** 2, Config.mem_key_dim)
        self.gru = nn.GRU(Config.mem_key_dim, Config.mem_key_dim, batch_first=True)

        self.memory_bank = MemoryBank(
            key_dim=Config.mem_key_dim,
            value_dim=Config.mem_val_dim,
            capacity=Config.mem_capacity
        )

        self.gnn = GNNEncoder(Config.mem_key_dim, Config.gnn_hidden_dim)

        self.fuse = nn.Sequential(
            nn.Linear(Config.mem_key_dim + Config.mem_val_dim + Config.gnn_hidden_dim, 128),
            nn.Sigmoid(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, img, h_mem=None):
        out = self.model(img)
        out = out.view(out.size(0), -1)
        h_cur = self.feature_fc(out)

        if h_mem is None or h_mem.size(1) != img.size(0):
            h_mem = torch.zeros(1, img.size(0), Config.mem_key_dim).to(img.device)

        _, h_out = self.gru(h_cur.unsqueeze(1), h_mem)

        # 顶层记忆融合
        self.memory_bank.add(h_cur.detach(), h_out.squeeze(0).detach())
        mem_val = self.memory_bank.fuse(h_cur)

        # 图神经网络结构编码（图谱关联）
        gnn_feat = self.gnn(h_cur)

        # 多模态融合判别
        fused = torch.cat([h_cur, mem_val, gnn_feat], dim=1)
        score = self.fuse(fused)
        return score, h_out.detach()
