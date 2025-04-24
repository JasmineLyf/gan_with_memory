import torch
import torch.nn as nn
import torch.nn.functional as F

class MemoryBank(nn.Module):
    def __init__(self, key_dim, value_dim, capacity=512):
        super(MemoryBank, self).__init__()
        self.capacity = capacity
        self.register_buffer("keys", torch.zeros(capacity, key_dim))
        self.register_buffer("values", torch.zeros(capacity, value_dim))
        self.ptr = 0
        self.full = False

    def add(self, k, v):
        bs = k.size(0)
        if self.ptr + bs > self.capacity:
            remain = self.capacity - self.ptr
            self.keys[self.ptr:] = k[:remain].detach()
            self.values[self.ptr:] = v[:remain].detach()
            self.ptr = 0
            self.full = True
            self.add(k[remain:], v[remain:])
        else:
            self.keys[self.ptr:self.ptr+bs] = k.detach()
            self.values[self.ptr:self.ptr+bs] = v.detach()
            self.ptr += bs

    def query(self, q, topk=5):
        scores = F.cosine_similarity(q.unsqueeze(1), self.keys.unsqueeze(0), dim=-1)
        topk_scores, topk_idx = torch.topk(scores, topk, dim=1)
        return self.values[topk_idx]  # shape: [B, topk, value_dim]

    def fuse(self, q):
        top_values = self.query(q)  # [B, topk, value_dim]
        fused = top_values.mean(dim=1)  # mean over topk
        return fused
