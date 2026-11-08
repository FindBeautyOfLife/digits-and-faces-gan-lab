import os
import glob
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


# ====================== 超参数 ======================
DATA_ROOT = "./data/faces"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device =", DEVICE)

EPOCHS = 20
BATCH_SIZE = 128
LR = 2e-4
BETAS = (0.5, 0.999)
NUM_WORKERS = 0  # Windows 下建议先用 0，稳定后再改 2

GF = 64
DF = 64
NZ = 100
NC = 3

PRINT_EVERY = 40
SAVE_EVERY = 100
LABEL_SMOOTHING = False


# ====================== 平铺目录 Dataset ======================
class FlatImageDataset(Dataset):
    def __init__(
        self, root, transform=None, exts=(".jpg", ".jpeg", ".png", ".bmp", ".webp")
    ):
        self.root = root
        self.transform = transform
        self.paths = []
        for e in exts:
            self.paths += glob.glob(os.path.join(root, f"*{e}"))
        if len(self.paths) == 0:
            raise FileNotFoundError(f"No images found in {root}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img  # 只返回图片张量


# ====================== 工具函数 ======================
def set_requires_grad(model, flag: bool):
    for p in model.parameters():
        p.requires_grad_(flag)


def randn_noise(batch_size: int, device):
    return torch.randn(batch_size, NZ, 1, 1, device=device)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


def make_targets(batch_size: int, is_real: bool, device):
    if LABEL_SMOOTHING:
        if is_real:
            return torch.empty(batch_size, 1, device=device).uniform_(0.8, 1.0)
        return torch.empty(batch_size, 1, device=device).uniform_(0.0, 0.2)
    return torch.full((batch_size, 1), 1.0 if is_real else 0.0, device=device)


# ====================== 生成器 ======================
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(
                NZ, GF * 8, kernel_size=6, stride=1, padding=0, bias=False
            ),
            nn.BatchNorm2d(GF * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                GF * 8, GF * 4, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(GF * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                GF * 4, GF * 2, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(GF * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                GF * 2, GF, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(GF),
            nn.ReLU(True),
            nn.ConvTranspose2d(GF, NC, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

        self.optimiser = torch.optim.Adam(self.parameters(), lr=LR, betas=BETAS)
        self.progress = []
        self.counter = 0

    def forward(self, z):
        return self.main(z)

    def train_step(self, D, z, targets):
        fake_x = self.forward(z)
        out = D(fake_x)
        loss = D.loss_function(out, targets)

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        self.counter += 1
        if self.counter % 10 == 0:
            self.progress.append(loss.item())

        return loss.item(), out.mean().item()


# ====================== 判别器 ======================
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(NC, DF, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(DF, DF * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(DF * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(DF * 2, DF * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(DF * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(DF * 4, DF * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(DF * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(DF * 8, 1, kernel_size=6, stride=1, padding=0, bias=False),
            nn.Sigmoid(),
        )

        self.loss_function = nn.BCELoss()
        self.optimiser = torch.optim.Adam(self.parameters(), lr=LR, betas=BETAS)
        self.progress = []
        self.counter = 0

    def forward(self, x):
        return self.main(x).view(-1, 1)  # (B,1)

    def train_step(self, x, targets):
        out = self.forward(x)
        loss = self.loss_function(out, targets)

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        self.counter += 1
        if self.counter % 10 == 0:
            self.progress.append(loss.item())

        return loss.item(), out.mean().item()


def main():
    # ====================== 数据 ======================
    tfm = transforms.Compose(
        [
            transforms.Resize(96),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_dataset = FlatImageDataset(root=DATA_ROOT, transform=tfm)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=True,
        pin_memory=(DEVICE.type == "cuda"),
    )

    # ====================== 模型 ======================
    D = Discriminator().to(DEVICE)
    G = Generator().to(DEVICE)
    D.apply(weights_init)
    G.apply(weights_init)
    D.train()
    G.train()

    fixed_noise = randn_noise(64, DEVICE)

    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Start training...")

    # ====================== 训练 ======================
    for epoch in range(EPOCHS):
        for i, real_imgs in enumerate(train_loader):
            real_imgs = real_imgs.to(DEVICE)
            bsz = real_imgs.size(0)

            # ---- 训练 D：真 ----
            real_targets = make_targets(bsz, True, DEVICE)
            lossD_real, Dx = D.train_step(real_imgs, real_targets)

            # ---- 训练 D：假 ----
            z = randn_noise(bsz, DEVICE)
            fake_imgs = G(z).detach()
            fake_targets = make_targets(bsz, False, DEVICE)
            lossD_fake, Dgz1 = D.train_step(fake_imgs, fake_targets)

            lossD = lossD_real + lossD_fake

            # ---- 训练 G：让 D(fake) -> 真 ----
            set_requires_grad(D, False)
            z2 = randn_noise(bsz, DEVICE)
            target_real = make_targets(bsz, True, DEVICE)
            lossG, Dgz2 = G.train_step(D, z2, target_real)
            set_requires_grad(D, True)

            G_losses.append(lossG)
            D_losses.append(lossD)

            if i % PRINT_EVERY == 0:
                progress = f"{i}/{len(train_loader)}"
                print(
                    f"[Epoch {epoch+1:2d}/{EPOCHS}] [{progress:>10s}] "
                    f"Loss_D: {lossD:8.4f} | Loss_G: {lossG:8.4f} | "
                    f"D(real): {Dx:6.3f} | D(fake): {Dgz1:6.3f}/{Dgz2:6.3f}"
                )

            if (iters % SAVE_EVERY == 0) or (
                (epoch == EPOCHS - 1) and (i == len(train_loader) - 1)
            ):
                with torch.no_grad():
                    fake_vis = G(fixed_noise).detach().cpu()
                img_list.append(utils.make_grid(fake_vis, padding=2, normalize=True))

            iters += 1

    # ====================== 曲线 ======================
    plt.figure(figsize=(10, 5))
    plt.title("G and D Loss during Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # ====================== 真/假图对比 ======================
    real_batch = next(iter(train_loader))

    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(
        np.transpose(
            utils.make_grid(
                real_batch.to(DEVICE)[:64], padding=5, normalize=True
            ).cpu(),
            (1, 2, 0),
        )
    )

    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
    plt.show()


if __name__ == "__main__":
    main()
