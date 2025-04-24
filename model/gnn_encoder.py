import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GNNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GNNEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        # x: [B, D]
        B = x.size(0)

        # 构建邻接图：KNN 模拟（欧式距离最近的 K 个）
        sim = torch.cdist(x, x)  # [B, B]
        knn_val, knn_idx = torch.topk(-sim, k=min(5, B), dim=1)  # 越小越相似，取负号变最大值

        # 消息聚合
        agg = torch.zeros_like(x)
        for i in range(B):
            neighbors = x[knn_idx[i]]
            h = F.relu(self.fc1(neighbors))  # [K, H]
            h = h.mean(dim=0)                # mean over neighbors
            agg[i] = h

        return self.fc2(agg)  # [B, H]