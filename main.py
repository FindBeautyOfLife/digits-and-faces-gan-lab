from dataset import MnistDataset
import torch
import torch.nn as nn
import pandas
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# ====================== 超参数 ======================
mnist_dataset = MnistDataset("./data/mnist_train.csv")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device =", device)

EPOCHS = 12
BATCH_SIZE = 128
LR = 2e-4
BETAS = (0.5, 0.999)

NUM_CLASSES = 10
Z_DIM = 100  # 噪声维度


# ====================== 工具函数 ======================
def set_requires_grad(model, flag: bool):
    for p in model.parameters():
        p.requires_grad_(flag)


def one_hot(label, num_classes=NUM_CLASSES, device="cpu"):
    """单个 label -> onehot (num_classes,)"""
    v = torch.zeros(num_classes, device=device)
    v[label] = 1.0
    return v


def randn_z(batch_size, device):
    """采样高斯噪声向量 z"""
    return torch.randn(batch_size, Z_DIM, device=device)


# ====================== 判别器 ======================
# 输入：
#   x: (B, 1, 28, 28)，像素范围 [-1,1]
#   y_onehot: (B, 10)
# 做法：
#   y_onehot 展开为 (B,10,28,28)，通道维与 x 拼接 -> (B, 11, 28, 28)
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            # 28x28 -> 14x14
            nn.Conv2d(1 + NUM_CLASSES, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 14x14 -> 7x7
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 7x7 -> 7x7（增加一层卷积增强表达能力）
            nn.Conv2d(128, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc = nn.Linear(256 * 7 * 7, 1)

        self.loss_function = nn.BCEWithLogitsLoss()
        self.optimiser = torch.optim.Adam(self.parameters(), lr=LR, betas=BETAS)
        self.counter = 0
        self.progress = []

    def forward(self, x, y_onehot):
        """
        x: (B,1,28,28)
        y_onehot: (B,10)
        返回：logits (B,1)
        """
        B = x.size(0)
        # (B,10) -> (B,10,1,1) -> (B,10,28,28)
        y_map = y_onehot.view(B, NUM_CLASSES, 1, 1).expand(-1, -1, 28, 28)
        # 通道维拼接
        x_cond = torch.cat([x, y_map], dim=1)  # (B, 1+10, 28, 28)

        feat = self.conv(x_cond)  # (B,256,7,7)
        feat = feat.view(B, -1)  # (B,256*7*7)
        out = self.fc(feat)  # (B,1)，logits
        return out

    def train_step(self, x, y_onehot, targets):
        """
        判别器单步训练
        x: (B,1,28,28)
        y_onehot: (B,10)
        targets: (B,1)  真=1 假=0 (平滑处理)
        """
        out = self.forward(x, y_onehot)
        loss = self.loss_function(out, targets)

        self.counter += 1
        if self.counter % 10 == 0:
            self.progress.append(loss.item())

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

    def plot_progress(self):
        df = pandas.DataFrame(self.progress, columns=["loss"])
        df.plot(
            figsize=(16, 8),
            alpha=0.3,
            marker=".",
            grid=True,
        )


# ====================== 生成器 ======================
# 输入：
#   z: (B,100)
#   y_onehot: (B,10)
# 拼接 -> (B,110) -> FC -> (B,256*7*7)
# reshape -> (B,256,7,7) -> 反卷积上采样 -> (B,1,28,28)
# 输出像素范围：[-1,1] （Tanh）
class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc = nn.Linear(Z_DIM + NUM_CLASSES, 256 * 7 * 7)

        self.deconv = nn.Sequential(
            # 7x7 -> 14x14
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 14x14 -> 28x28
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 保持 28x28，压到 1 通道
            nn.Conv2d(64, 1, 3, 1, 1, bias=False),
            nn.Tanh(),  # 输出 [-1,1]
        )

        self.optimiser = torch.optim.Adam(self.parameters(), lr=LR, betas=BETAS)
        self.counter = 0
        self.progress = []

    def forward(self, z, y_onehot):
        """
        z: (B,100)
        y_onehot: (B,10)
        返回：fake_x (B,1,28,28)，范围 [-1,1]
        """
        z_cond = torch.cat([z, y_onehot], dim=1)  # (B,110)
        h = self.fc(z_cond)  # (B,256*7*7)
        h = h.view(-1, 256, 7, 7)  # (B,256,7,7)
        x = self.deconv(h)  # (B,1,28,28)
        return x

    def train_step(self, D, z, y_onehot, targets):
        """
        生成器单步训练
        目标：让 D(fake_x, y) 尽量接近 1
        targets: 一般是全 1
        """
        fake_x = self.forward(z, y_onehot)  # (B,1,28,28)
        d_out = D(fake_x, y_onehot)  # (B,1)，logits
        loss = D.loss_function(d_out, targets)

        self.counter += 1
        if self.counter % 10 == 0:
            self.progress.append(loss.item())

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

    def plot_progress(self):
        df = pandas.DataFrame(self.progress, columns=["loss"])
        df.plot(
            figsize=(16, 8),
            alpha=0.3,
            marker=".",
            grid=True,
        )


# ====================== 初始化模型 ======================
D = Discriminator().to(device)
G = Generator().to(device)

train_loader = DataLoader(
    mnist_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
)

D.train()
G.train()

# ====================== 训练循环 ======================
for epoch in range(EPOCHS):
    print("epoch =", epoch + 1)
    for labels, image_vecs, onehots in train_loader:
        # 条件 one-hot
        y = onehots.to(device).float()  # (B,10)

        # 把 784 向量 reshape 成 (B,1,28,28)，并映射到 [-1,1]
        x = image_vecs.to(device).float().view(-1, 1, 28, 28)
        # 原始数据是 [0,1]，做线性变换到 [-1,1]
        x = x * 2.0 - 1.0

        batch_size = x.size(0)

        # -------- 判别器：真样本 ----------
        # label smoothing：真标签在 [0.8,1.0] 之间波动
        real_targets = torch.empty(batch_size, 1, device=device).uniform_(0.8, 1.0)
        D.train_step(x, y, real_targets)

        # -------- 判别器：假样本 ----------
        z = randn_z(batch_size, device)  # (B,100)
        fake_x = G(z, y).detach()  # (B,1,28,28)，detach 防止梯度回传到 G
        # label smoothing：假标签在 [0,0.2] 之间波动
        fake_targets = torch.empty(batch_size, 1, device=device).uniform_(0.0, 0.2)
        D.train_step(fake_x, y, fake_targets)

        # -------- 生成器：希望 D 把假样本判成真 ----------
        set_requires_grad(D, False)
        z2 = randn_z(batch_size, device)
        # 对 G 来说，目标就简单设为全 1
        target_real = torch.ones((batch_size, 1), device=device)
        G.train_step(D, z2, y, target_real)
        set_requires_grad(D, True)

# ====================== 损失曲线 ======================
D.plot_progress()
G.plot_progress()

# ====================== 展示：按 0..9 生成对应类别 ======================
G.eval()
D.eval()
with torch.no_grad():
    plt.figure(figsize=(10, 2))
    for d in range(10):
        y = one_hot(d, device=device).unsqueeze(0)  # (1,10)
        z = randn_z(1, device)  # (1,100)
        img = G(z, y)[0, 0].cpu().numpy()  # [-1,1]
        img = (img + 1.0) / 2.0  # 映射回 [0,1] 方便显示
        plt.subplot(1, 10, d + 1)
        plt.imshow(img, interpolation="none", cmap="gray")
        plt.title(str(d))
        plt.axis("off")
    plt.tight_layout()
plt.show()
