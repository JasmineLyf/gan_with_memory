# train.py (fixed GRU input bug during discriminator training)

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets
from torch.autograd import Variable
import json

from config import Config
from model.generator import Generator
from model.discriminator import Discriminator

os.makedirs("images", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

transform = transforms.Compose([
    transforms.Resize(Config.img_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

train_data = datasets.MNIST("./data/mnist", train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=Config.batch_size, shuffle=True)

adversarial_loss = nn.BCELoss()

generator = Generator().to(Config.device)
discriminator = Discriminator().to(Config.device)

optimizer_G = optim.Adam(generator.parameters(), lr=Config.lr, betas=(Config.b1, Config.b2))
optimizer_D = optim.Adam(discriminator.parameters(), lr=Config.lr * 0.5, betas=(Config.b1, Config.b2))

Tensor = torch.cuda.FloatTensor if Config.device == "cuda" else torch.FloatTensor

class EarlyStopper:
    def __init__(self, patience=5):
        self.patience = patience
        self.best_g = float("inf")
        self.best_d = float("inf")
        self.counter = 0

    def check(self, g_loss, d_loss):
        improved = False
        if g_loss < self.best_g:
            self.best_g = g_loss
            improved = True
        if d_loss < self.best_d:
            self.best_d = d_loss
            improved = True

        if not improved:
            self.counter += 1
        else:
            self.counter = 0

        return self.counter >= self.patience

stopper = EarlyStopper(patience=5)

G_losses, D_losses = [], []
g_mem = d_mem = None

for epoch in range(Config.n_epochs):
    for i, (imgs, _) in enumerate(train_loader):
        valid = Variable(Tensor(imgs.size(0), 1).fill_(0.9), requires_grad=False)
        fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

        real_imgs = Variable(imgs.to(Config.device))

        # 判别器训练
        optimizer_D.zero_grad()
        real_score, d_mem = discriminator(real_imgs, d_mem)

        z_detach = Variable(Tensor(torch.randn(imgs.size(0), Config.latent_dim)))
        gen_imgs_detach, _ = generator(z_detach, g_mem)  # GRU输入修复
        fake_score, d_mem = discriminator(gen_imgs_detach.detach(), d_mem)

        d_loss = (adversarial_loss(real_score, valid) + adversarial_loss(fake_score, fake)) / 2
        d_loss.backward()
        optimizer_D.step()

        # 生成器训练每3步一次
        if i % 3 == 0:
            optimizer_G.zero_grad()
            z = Variable(Tensor(torch.randn(imgs.size(0), Config.latent_dim)))
            gen_imgs, g_mem = generator(z, g_mem)
            g_loss = adversarial_loss(discriminator(gen_imgs, d_mem)[0], valid)
            g_loss.backward()
            optimizer_G.step()
            G_losses.append(g_loss.item())

        D_losses.append(d_loss.item())

        print(f"[Epoch {epoch}/{Config.n_epochs}] [Batch {i}/{len(train_loader)}] [D loss: {d_loss.item():.4f}] [G loss: {G_losses[-1] if G_losses else 0:.4f}]")

        if i % Config.sample_interval == 0:
            save_image(gen_imgs.data[:25], f"images/{epoch}_{i}.png", nrow=5, normalize=True)

    if stopper.check(G_losses[-1] if G_losses else float("inf"), d_loss.item()):
        print("Early stopping triggered.")
        break

    torch.save(generator.state_dict(), "checkpoints/best_generator.pth")
    torch.save(discriminator.state_dict(), "checkpoints/best_discriminator.pth")

# 保存 loss 日志
with open("loss_log.json", "w") as f:
    json.dump({"G": G_losses, "D": D_losses}, f)
print("Saved loss logs to loss_log.json")