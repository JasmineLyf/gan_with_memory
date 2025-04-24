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

# 生成不同扰动的 latent 向量
print("\n🔍 Debugging Discriminator Response to Generated Images")
for i in range(5):
    z = torch.randn(25, Config.latent_dim).to(Config.device)
    z_perturbed = z + torch.randn_like(z) * 0.1 * i  # 添加扰动

    with torch.no_grad():
        gen_imgs, _ = G(z_perturbed)
        fake_score, _ = D(gen_imgs)

    print(f"Iteration {i}: Avg Score = {fake_score.mean().item():.4f} | Std = {fake_score.std().item():.4f}")
    print("Sample scores:", fake_score.view(-1)[:5].cpu().numpy())

