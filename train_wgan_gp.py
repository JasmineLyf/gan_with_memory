import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets
import argparse

from config import Config
from model.generator import Generator
from model.wgan_gp_discriminator import WGANGPDiscriminator

# argparse for ablation control
parser = argparse.ArgumentParser()
parser.add_argument("--use_gru", action="store_true", help="Enable GRU memory module")
parser.add_argument("--use_memory", action="store_true", help="Enable MemoryBank module")
parser.add_argument("--use_gnn", action="store_true", help="Enable GNN module")
args = parser.parse_args()

os.makedirs("images_wgan", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

transform = transforms.Compose([
    transforms.Resize(Config.img_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

dataloader = torch.utils.data.DataLoader(
    datasets.MNIST("./data/mnist", train=True, download=True, transform=transform),
    batch_size=Config.batch_size,
    shuffle=True,
    num_workers=0
)

generator = Generator().to(Config.device)
discriminator = WGANGPDiscriminator(
    use_gru=args.use_gru,
    use_memory=args.use_memory,
    use_gnn=args.use_gnn
).to(Config.device)

optimizer_G = optim.Adam(generator.parameters(), lr=Config.lr, betas=(Config.b1, Config.b2))
optimizer_D = optim.Adam(discriminator.parameters(), lr=Config.lr, betas=(Config.b1, Config.b2))

Tensor = torch.cuda.FloatTensor if Config.device == "cuda" else torch.FloatTensor

# Gradient penalty
lambda_gp = 20

def compute_gradient_penalty(D, real_samples, fake_samples):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(Config.device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates, _ = D(interpolates)
    fake = torch.ones_like(d_interpolates).to(Config.device)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

n_critic = 1
patience = 10
min_delta = 0.05
best_gap = float('-inf')
no_improve_epochs = 0
g_mem = d_mem = None

for epoch in range(Config.n_epochs):
    epoch_gap_sum = 0.0
    num_steps = 0

    for i, (imgs, _) in enumerate(dataloader):
        real_imgs = imgs.to(Config.device)

        # Train Discriminator
        optimizer_D.zero_grad()
        z = torch.randn(imgs.size(0), Config.latent_dim).to(Config.device)
        gen_imgs, _ = generator(z, g_mem)

        real_score, d_mem = discriminator(real_imgs, d_mem)
        fake_score, d_mem = discriminator(gen_imgs.detach(), d_mem)

        gp = compute_gradient_penalty(discriminator, real_imgs.data, gen_imgs.data)
        d_loss = -torch.mean(real_score) + torch.mean(fake_score) + lambda_gp * gp

        d_loss.backward()
        optimizer_D.step()

        # Train Generator every n_critic steps
        if i % n_critic == 0:
            optimizer_G.zero_grad()
            z = torch.randn(imgs.size(0), Config.latent_dim).to(Config.device)
            gen_imgs, g_mem = generator(z, g_mem)
            g_loss = -torch.mean(discriminator(gen_imgs)[0])
            g_loss.backward()
            optimizer_G.step()

            gap = real_score.mean().item() - fake_score.mean().item()
            epoch_gap_sum += gap
            num_steps += 1

            print(f"[Epoch {epoch}/{Config.n_epochs}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}] [W Gap: {gap:.4f}]")

        if i % Config.sample_interval == 0:
            save_image(gen_imgs.data[:25], f"images_wgan/{epoch}_{i}.png", nrow=5, normalize=True)

    avg_gap = epoch_gap_sum / max(1, num_steps)
    print(f"[Epoch {epoch}] Avg Wasserstein Gap: {avg_gap:.4f}")

    if avg_gap - best_gap > min_delta:
        best_gap = avg_gap
        no_improve_epochs = 0

        # ✅ 保存最优模型
        torch.save(generator.state_dict(), "checkpoints/best_generator.pth")
        torch.save(discriminator.state_dict(), "checkpoints/best_discriminator.pth")
        print("✅ 模型已保存到 checkpoints/")

    else:
        no_improve_epochs += 1
        if no_improve_epochs >= patience:
            print(f"🔥 Early stopping triggered after {patience} epochs without W-Gap improvement.")
            break