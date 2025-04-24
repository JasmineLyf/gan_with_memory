import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

from config import Config
from model.generator import Generator
from model.discriminator import Discriminator

# 切换目录（适配不同运行环境）
project_root = os.path.dirname(os.path.dirname(__file__))
checkpoint_dir = os.path.join(project_root, "checkpoints")

# 加载模型
G = Generator().to(Config.device)
D = Discriminator().to(Config.device)
G.load_state_dict(torch.load("checkpoints/best_generator.pth", map_location=Config.device))
D.load_state_dict(torch.load("checkpoints/best_discriminator.pth", map_location=Config.device))
G.eval()
D.eval()

# 数据加载
transform = transforms.Compose([
    transforms.Resize(Config.img_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataloader = DataLoader(
    datasets.MNIST(os.path.join(project_root, "data/mnist"), train=True, download=True, transform=transform),
    batch_size=25, shuffle=True
)

real_imgs, _ = next(iter(dataloader))
real_imgs = real_imgs.to(Config.device)

# 生成图像
z = torch.randn(25, Config.latent_dim).to(Config.device)
with torch.no_grad():
    gen_imgs, _ = G(z)
    real_scores, _ = D(real_imgs)
    fake_scores, _ = D(gen_imgs)

# 可视化函数
def show_images(images, scores, title):
    grid = torchvision.utils.make_grid(images, nrow=5, normalize=True)
    plt.figure(figsize=(6, 6))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.axis("off")
    plt.title(f"{title} | Avg Score: {scores.mean().item():.4f}")
    plt.show()

# 显示三张图：真实图、生成图、判别图（得分对比）
print("\n🔍 Displaying Real Images")
show_images(real_imgs, real_scores, "Real Images")

print("\n🎨 Displaying Generated Images")
show_images(gen_imgs, fake_scores, "Generated Images")

# 叠加图显示对比：上半为真实图，下半为生成图
combined = torch.cat([real_imgs, gen_imgs], dim=0)
combined_scores = torch.cat([real_scores, fake_scores], dim=0)
print("\n📊 Displaying Real vs Generated Images")
show_images(combined, combined_scores, "Real (Top) vs Generated (Bottom)")

# 保存图像
save_image(real_imgs.data, "real_final.png", nrow=5, normalize=True)
save_image(gen_imgs.data, "generated_final.png", nrow=5, normalize=True)
print("✅ Saved final images to real_final.png and generated_final.png")
