import torch
from config import Config
from model.generator import Generator
from model.wgan_gp_discriminator import WGANGPDiscriminator

# 初始化模型
G = Generator().to(Config.device)
D = WGANGPDiscriminator().to(Config.device)
G.load_state_dict(torch.load("checkpoints/best_generator.pth", map_location=Config.device))
D.load_state_dict(torch.load("checkpoints/best_discriminator.pth", map_location=Config.device))
G.eval()
D.eval()

# 使用扰动 latent 生成图像
z = torch.randn(25, Config.latent_dim).to(Config.device)
z_perturbed = z + torch.randn_like(z) * 0.1

with torch.no_grad():
    gen_imgs, _ = G(z_perturbed)
    print("🔍 Discriminator Forward Flow Check:")
    out = D.model(gen_imgs)
    out = out.view(out.size(0), -1)
    print("  [model conv out std]:", out.std().item())

    h_cur = D.feature_fc(out)
    print("  [feature_fc std]:", h_cur.std().item())

    h_mem = torch.zeros(1, h_cur.size(0), Config.mem_key_dim).to(Config.device)
    _, h_out = D.gru(h_cur.unsqueeze(1), h_mem)
    print("  [GRU h_out std]:", h_out.std().item())

    mem_val = D.memory_bank.fuse(h_cur)
    print("  [MemoryBank fuse std]:", mem_val.std().item())

    gnn_feat = D.gnn(h_cur)
    print("  [GNN feat std]:", gnn_feat.std().item())

    fused = torch.cat([h_cur, mem_val, gnn_feat], dim=1)
    print("  [concat fused std]:", fused.std().item())

    score = D.fuse(fused)
    print("  [final score std]:", score.std().item())
    print("  [final score mean]:", score.mean().item())
