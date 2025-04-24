import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

from config import Config
from model.generator import Generator
from model.wgan_gp_discriminator import WGANGPDiscriminator  # or use ablation version if needed

# 设置判别器结构开关
USE_GRU = True
USE_MEMORY = True
USE_GNN = True

# 初始化模型
G = Generator().to(Config.device)
D = WGANGPDiscriminator(use_gru=USE_GRU, use_memory=USE_MEMORY, use_gnn=USE_GNN).to(Config.device)
G.load_state_dict(torch.load("checkpoints/best_generator.pth", map_location=Config.device))
D.load_state_dict(torch.load("checkpoints/best_discriminator.pth", map_location=Config.device))
G.eval()
D.eval()

# 加载真实图像
transform = transforms.Compose([
    transforms.Resize(Config.img_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataloader = DataLoader(
    datasets.MNIST("./data/mnist", train=True, download=True, transform=transform),
    batch_size=25, shuffle=True
)

real_imgs, _ = next(iter(dataloader))
real_imgs = real_imgs.to(Config.device)

# 生成图像
z = torch.randn(25, Config.latent_dim).to(Config.device)
with torch.no_grad():
    gen_imgs, _ = G(z)
    real_scores = D(real_imgs)[0]
    fake_scores = D(gen_imgs)[0]

# 可视化函数
def show_images(images, scores, title, filename):
    grid = torchvision.utils.make_grid(images, nrow=5, normalize=True)
    plt.figure(figsize=(6, 6))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.axis("off")
    plt.title(f"{title} | Avg Score: {scores.mean().item():.4f}")
    plt.savefig(filename)
    print(f"✅ Saved: {filename}")
    plt.show()

# 显示并保存图像
show_images(real_imgs, real_scores, "Real Images", "real_wgan_ablation.png")
show_images(gen_imgs, fake_scores, "Generated Images", "generated_wgan_ablation.png")

# 保存单独图片
save_image(real_imgs.data, "real_final_wgan_ablation.png", nrow=5, normalize=True)
save_image(gen_imgs.data, "generated_final_wgan_ablation.png", nrow=5, normalize=True)
print("✅ Images saved: real_final_wgan_ablation.png & generated_final_wgan_ablation.png")
