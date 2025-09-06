# bibliotecas de visualizaçÃo de dados
import os
import numpy as np
import matplotlib.pyplot as plt
import csv
#bibliotecas de aprendizado de máquina
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader,random_split
from sklearn.metrics import roc_auc_score, roc_curve
# biblioteca para lidar com dados geoespaciais
import rasterio
from rasterio import plot
import spyndex

spyndex.indices.NBRSWIR

rgb_path_1 = "/home/nicolas/git/projects/hackaton/dataset_kaggle/dataset/t1"
rgb_path_2 = "/home/nicolas/git/projects/hackaton/dataset_kaggle/dataset/t2"
mask_path = "/home/nicolas/git/projects/hackaton/dataset_kaggle/dataset/mask"

class SiameseDataset(Dataset):
    def __init__(self, T10_dir, T20_dir, mask_dir, transform=None):
        self.T10_dir = T10_dir
        self.T20_dir = T20_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.ids = sorted([
            f.split('_')[-1].replace('.tif', '')
            for f in os.listdir(T10_dir)
            if f.startswith('recorte_') and
               os.path.isfile(os.path.join(T20_dir, f)) and
               os.path.isfile(os.path.join(mask_dir, f))
        ])

    def __len__(self):
        return len(self.ids)

    def read_image(self, path):
        with rasterio.open(path) as src:
            img = src.read().astype(np.float32)
            img = np.nan_to_num(img, nan=0.0)
            img_min = img.min()
            img_max = img.max()
            if img_max - img_min > 0:
                img = (img - img_min) / (img_max - img_min)
            else:
                img = np.zeros_like(img)
        return torch.tensor(img, dtype=torch.float32)

    def read_mask(self, path):
        with rasterio.open(path) as src:
            mask = src.read(1).astype(np.float32)
            mask = np.nan_to_num(mask, nan=0.0)
            mask = np.where(mask > 0, 1.0, 0.0)
        return torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

    def __getitem__(self, idx):
        id_ = self.ids[idx]
        fname = f"recorte_{id_}.tif"
        T10_path = os.path.join(self.T10_dir, fname)
        T20_path = os.path.join(self.T20_dir, fname)
        mask_path = os.path.join(self.mask_dir, fname)

        t1 = self.read_image(T10_path)
        t2 = self.read_image(T20_path)
        mask = self.read_mask(mask_path)

        if self.transform:
            t1 = self.transform(t1)
            t2 = self.transform(t2)
            mask = self.transform(mask)

        return t1, t2, mask
    

# Modelo
class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNetSiamese(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            DoubleConv(4, 64),           # agora espera 4 canais de entrada
            nn.MaxPool2d(2),
            DoubleConv(64, 128),
            nn.MaxPool2d(2)
        )
        self.decoder = nn.Sequential(
            DoubleConv(256, 128),
            nn.Upsample(scale_factor=2),
            DoubleConv(128, 64),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        f1 = self.encoder(x1)
        f2 = self.encoder(x2)
        f = torch.cat([f1, f2], dim=1)
        out = self.decoder(f)
        if torch.isnan(out).any():
            print("⚠️ Nan detectado na saída do modelo!")
        return out

def train(model, dataloader, device, epochs=5):
    model.to(device)  # Manda o modelo pra GPU
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for t1, t2, mask in dataloader:
            # Move dados para dispositivo (GPU ou CPU)
            t1, t2, mask = t1.to(device), t2.to(device), mask.to(device)

            # Debug inputs
            # print(f"T1 min {t1.min().item()}, max {t1.max().item()}, NaN: {torch.isnan(t1).any().item()}")
            # print(f"T2 min {t2.min().item()}, max {t2.max().item()}, NaN: {torch.isnan(t2).any().item()}")
            # print(f"Mask min {mask.min().item()}, max {mask.max().item()}, NaN: {torch.isnan(mask).any().item()}")

            pred = model(t1, t2)

            # Verifica NaN na predição
            if torch.isnan(pred).any():
                print("⚠️ Nan na predição, pulando batch")
                continue

            # Confirma se pred está no intervalo [0,1]
            if pred.min() < 0 or pred.max() > 1:
                print(f"⚠️ Pred min {pred.min().item()}, max {pred.max().item()} fora do intervalo [0,1]")
                continue

            loss = criterion(pred, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "siamese_unet_model.pth")
    print("✅ Modelo salvo como 'siamese_unet_model.pth'")

def plot_sample(t1, t2, mask, pred):
    import matplotlib.pyplot as plt

    # Selecionar 3 bandas (ex: primeiras 3 para simular RGB)
    t1_rgb = t1[0, :3].permute(1,2,0).cpu().numpy()
    t2_rgb = t2[0, :3].permute(1,2,0).cpu().numpy()
    mask_img = mask[0,0].cpu().numpy()
    pred_img = pred[0,0].detach().cpu().numpy()  # <-- corrigido

    fig, axs = plt.subplots(1,4, figsize=(12,4))
    axs[0].imshow(t1_rgb)
    axs[0].set_title("T1 (RGB falso)")
    axs[1].imshow(t2_rgb)
    axs[1].set_title("T2 (RGB falso)")
    axs[2].imshow(mask_img, cmap="gray")
    axs[2].set_title("Mask")
    axs[3].imshow(pred_img, cmap="gray")
    axs[3].set_title("Pred")
    plt.show()


if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Usando GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("Usando CPU")

print("Versão do PyTorch:", torch.__version__)
print("CUDA disponível no PyTorch?:", torch.cuda.is_available())
print("Versão CUDA usada pelo PyTorch:", torch.version.cuda)
print("Versão cuDNN:", torch.backends.cudnn.version())

from torch.utils.data import DataLoader

dataset = SiameseDataset(
    "/home/nicolas/git/projects/hackaton/dataset_kaggle/dataset/t1", 
    "/home/nicolas/git/projects/hackaton/dataset_kaggle/dataset/t2", 
    "/home/nicolas/git/projects/hackaton/dataset_kaggle/dataset/mask"
)
loader = DataLoader(dataset, batch_size=1, shuffle=True)

dataset = loader.dataset
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=loader.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=loader.batch_size, shuffle=False)

# --- Inicializar modelo ---
model = UNetSiamese().to(device)
# 
# --- Treinar ---
train(model, train_loader, device, epochs=2)

# --- Avaliação no conjunto de teste ---
model.eval()
all_labels = []
all_preds = []

with torch.no_grad():
    for t1, t2, mask in test_loader:
        t1, t2, mask = t1.to(device), t2.to(device), mask.to(device)
        pred = model(t1, t2)
        
        # Flatten para usar no ROC
        all_labels.extend(mask.cpu().numpy().ravel())
        all_preds.extend(pred.cpu().numpy().ravel())

# --- Calcular AUC-ROC ---
auc = roc_auc_score(all_labels, all_preds)
fpr, tpr, thresholds = roc_curve(all_labels, all_preds)

print(f"AUC-ROC: {auc:.4f}")

# --- Plotar curva ROC ---
plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
plt.plot([0,1], [0,1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# --- Visualizar exemplo ---
t1, t2, mask = next(iter(test_loader))
pred = model(t1.to(device), t2.to(device))
plot_sample(t1, t2, mask, pred.cpu())