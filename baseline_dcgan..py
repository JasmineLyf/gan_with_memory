import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets
from torch.autograd import Variable

# 配置
img_size = 32
channels = 1
latent_dim = 100
batch_size = 64
n_epochs = 30
lr = 0.0002
sample_interval = 400

os.makedirs("images_baseline", exist_ok=True)
os.makedirs("checkpoints_baseline", exist_ok=True)

transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

dataloader = torch.utils.data.DataLoader(
    datasets.MNIST("./data/mnist", train=True, download=True, transform=transform),
    batch_size=batch_size,
    shuffle=True,
)

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

dev = "cuda" if torch.cuda.is_available() else "cpu"

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.init_size = img_size // 4
        self.l1 = nn.Linear(latent_dim, 128 * self.init_size ** 2)
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.size(0), 128, self.init_size, self.init_size)
        return self.conv_blocks(out)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        def block(in_f, out_f, bn=True):
            layers = [nn.Conv2d(in_f, out_f, 3, 2, 1), nn.LeakyReLU(0.2), nn.Dropout2d(0.25)]
            if bn:
                layers.append(nn.BatchNorm2d(out_f, 0.8))
            return layers

        self.model = nn.Sequential(
            *block(channels, 16, bn=False),
            *block(16, 32),
            *block(32, 64),
            *block(64, 128),
        )
        ds_size = img_size // 2 ** 4
        self.fc = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.size(0), -1)
        return self.fc(out)

# 初始化
G = Generator().to(dev)
D = Discriminator().to(dev)
optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
loss_func = nn.BCELoss()

G_losses = []
D_losses = []

for epoch in range(n_epochs):
    for i, (imgs, _) in enumerate(dataloader):
        valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

        real_imgs = imgs.to(dev)
        z = Variable(Tensor(torch.randn(imgs.size(0), latent_dim)))

        # Train G
        optimizer_G.zero_grad()
        gen_imgs = G(z)
        g_loss = loss_func(D(gen_imgs), valid)
        g_loss.backward()
        optimizer_G.step()

        # Train D
        optimizer_D.zero_grad()
        real_loss = loss_func(D(real_imgs), valid)
        fake_loss = loss_func(D(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        G_losses.append(g_loss.item())
        D_losses.append(d_loss.item())

        if i % sample_interval == 0:
            save_image(gen_imgs.data[:25], f"images_baseline/{epoch}_{i}.png", nrow=5, normalize=True)
        print(f"[Epoch {epoch}/{n_epochs}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")

# 保存模型参数
torch.save(G.state_dict(), "checkpoints_baseline/generator.pth")
torch.save(D.state_dict(), "checkpoints_baseline/discriminator.pth")

# 保存 Loss 曲线日志
with open("loss_log_baseline.json", "w") as f:
    json.dump({"G": G_losses, "D": D_losses}, f)

print("Saved baseline model and loss log.")