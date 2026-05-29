import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
from my_dataset import SELECTED_LANDMARKS, split_dataset

# ---- CONFIG ----
DATASET_DIR     = "C:/Users/sanda/OneDrive/Desktop/Archive"
PORION_IDX      = SELECTED_LANDMARKS.index('Po')
CROP_SIZE       = 256        # larger crop for Porion — more context needed
STAGE1_MODEL    = "trained_model.pth"
PORION_MODEL    = "porion_model.pth"
HISTORY_PATH    = "porion_history.json"
BATCH_SIZE      = 16
EPOCHS          = 100
LEARNING_RATE   = 1e-4
PATIENCE        = 15
STAGE1_IMG_SIZE = (512, 512)
SIGMA           = 8
# ----------------

# ---- Stage 1 UNet (frozen) ----
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=21):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, 32)
        self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128)
        self.enc4 = DoubleConv(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(256, 512)
        self.up4  = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = DoubleConv(512, 256)
        self.up3  = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = DoubleConv(256, 128)
        self.up2  = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = DoubleConv(128, 64)
        self.up1  = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = DoubleConv(64, 32)
        self.out_conv = nn.Conv2d(32, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out_conv(d1)

# ---- Dedicated Porion UNet ----
class PorionUNet(nn.Module):
    """
    Small UNet dedicated exclusively to Porion detection.
    Input: 256x256 crop centered on Stage1 Porion prediction.
    Output: single 256x256 heatmap.
    """
    def __init__(self):
        super().__init__()
        self.enc1 = DoubleConv(3, 32)
        self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128)
        self.enc4 = DoubleConv(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(256, 512)
        self.up4  = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = DoubleConv(512, 256)
        self.up3  = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = DoubleConv(256, 128)
        self.up2  = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = DoubleConv(128, 64)
        self.up1  = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = DoubleConv(64, 32)
        self.out_conv = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out_conv(d1)

# ---- Helpers ----
def heatmap_to_coord(heatmap, orig_w, orig_h):
    H, W = heatmap.shape
    idx  = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    y, x = idx
    return x * (orig_w / W), y * (orig_h / H)

def generate_heatmap(height, width, x, y, sigma=8):
    xv = np.arange(width)
    yv = np.arange(height)
    xx, yy = np.meshgrid(xv, yv)
    hm = np.exp(-((xx-x)**2 + (yy-y)**2) / (2*sigma**2))
    return hm.astype(np.float32)

def crop_patch(image, cx, cy, crop_size=256):
    orig_w, orig_h = image.size
    half = crop_size // 2
    x0   = max(0, min(int(cx) - half, orig_w - crop_size))
    y0   = max(0, min(int(cy) - half, orig_h - crop_size))
    crop = image.crop((x0, y0, x0 + crop_size, y0 + crop_size))
    return crop, x0, y0

def get_stage1_porion(model, img_path, device):
    """Get Stage1 Porion prediction in original pixel space."""
    image          = Image.open(img_path).convert("RGB")
    orig_w, orig_h = image.size
    resized        = image.resize(STAGE1_IMG_SIZE)
    tensor         = torch.from_numpy(
        np.array(resized).transpose(2,0,1)/255.0
    ).float().unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
    hm = output[0, PORION_IDX].cpu().numpy()
    x, y = heatmap_to_coord(hm, orig_w, orig_h)
    return x, y, image

# ---- Porion Dataset ----
class PorionDataset(Dataset):
    """
    Crops centered on Stage1 Porion prediction + jitter during training.
    GT heatmap centered on true Porion location within the crop.
    """
    def __init__(self, samples, stage1_cache,
                 crop_size=256, sigma=8, training=True):
        self.samples      = samples
        self.stage1_cache = stage1_cache  # dict: img_path -> (x, y)
        self.crop_size    = crop_size
        self.sigma        = sigma
        self.training     = training

        # Pre-load GT Porion coordinates
        self.gt_coords = {}
        for img_path, csv_path in samples:
            df = pd.read_csv(csv_path)
            df = df[df['Name'].isin(SELECTED_LANDMARKS)]
            df = df.set_index('Name').loc[SELECTED_LANDMARKS].reset_index()
            self.gt_coords[img_path] = (
                float(df['X'].iloc[PORION_IDX]),
                float(df['Y'].iloc[PORION_IDX])
            )

        mode = "TRAIN" if training else "VAL"
        print(f"[PorionDataset/{mode}] {len(samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, _ = self.samples[idx]
        gt_x, gt_y  = self.gt_coords[img_path]
        s1x, s1y    = self.stage1_cache[img_path]

        # Crop centered on Stage1 prediction + jitter during training
        if self.training:
            jx = np.random.normal(0, 20)  # larger jitter for Porion
            jy = np.random.normal(0, 20)
            cx, cy = s1x + jx, s1y + jy
        else:
            cx, cy = s1x, s1y

        original     = Image.open(img_path).convert("RGB")
        crop, x0, y0 = crop_patch(original, cx, cy, self.crop_size)

        # Augmentation for training
        if self.training:
            import torchvision.transforms as T
            aug   = T.ColorJitter(brightness=0.3, contrast=0.3)
            crop  = aug(crop)

        crop_tensor = torch.from_numpy(
            np.array(crop).transpose(2,0,1)/255.0
        ).float()

        # GT in crop local coordinates
        local_x = float(np.clip(gt_x - x0, 0, self.crop_size-1))
        local_y = float(np.clip(gt_y - y0, 0, self.crop_size-1))

        heatmap = generate_heatmap(
            self.crop_size, self.crop_size,
            local_x, local_y, sigma=self.sigma
        )

        return crop_tensor, torch.from_numpy(heatmap).unsqueeze(0), img_path

# ---- Training ----
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Stage 1
    stage1 = UNet(in_channels=3, out_channels=21).to(device)
    stage1.load_state_dict(torch.load(
        STAGE1_MODEL, map_location=device, weights_only=True
    ))
    stage1.eval()
    print("✅ Stage 1 loaded")

    # Get splits
    train_samples, val_samples, _ = split_dataset(DATASET_DIR)

    # Build Stage1 Porion prediction cache
    def build_cache(samples, cache_file):
        if os.path.exists(cache_file):
            print(f"  Loading {cache_file}...")
            return torch.load(cache_file, weights_only=False)
        print(f"  Computing Stage1 Porion predictions for {len(samples)} images...")
        cache = {}
        for i, (img_path, _) in enumerate(samples):
            x, y, _ = get_stage1_porion(stage1, img_path, device)
            cache[img_path] = (x, y)
            if (i+1) % 100 == 0:
                print(f"    {i+1}/{len(samples)}...")
        torch.save(cache, cache_file)
        print(f"  ✅ Saved to {cache_file}")
        return cache

    print("Building Porion prediction caches...")
    train_cache = build_cache(train_samples, "cache_porion_train.pt")
    val_cache   = build_cache(val_samples,   "cache_porion_val.pt")

    train_ds = PorionDataset(
        train_samples, train_cache, CROP_SIZE, SIGMA, training=True
    )
    val_ds   = PorionDataset(
        val_samples,   val_cache,   CROP_SIZE, SIGMA, training=False
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0
    )
    val_loader   = DataLoader(
        val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    model     = PorionUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=20, gamma=0.5
    )
    criterion = nn.MSELoss()

    best_val  = float('inf')
    patience  = 0
    history   = {'train_loss': [], 'val_loss': []}

    print(f"\nTraining Porion dedicated model...")
    print(f"Train: {len(train_loader)} batches | Val: {len(val_loader)} batches")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for crops, heatmaps, _ in train_loader:
            crops, heatmaps = crops.to(device), heatmaps.to(device)
            optimizer.zero_grad()
            loss = criterion(model(crops), heatmaps)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for crops, heatmaps, _ in val_loader:
                crops, heatmaps = crops.to(device), heatmaps.to(device)
                val_loss += criterion(model(crops), heatmaps).item()

        train_loss /= len(train_loader)
        val_loss   /= len(val_loader)
        scheduler.step()
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        print(f"Epoch [{epoch+1}/{EPOCHS}] "
              f"Train: {train_loss:.6f} | Val: {val_loss:.6f}")

        if val_loss < best_val:
            best_val  = val_loss
            patience  = 0
            torch.save(model.state_dict(), PORION_MODEL)
            print(f"  ✅ Saved (val_loss: {val_loss:.6f})")
        else:
            patience += 1
            print(f"  ⏳ No improvement ({patience}/{PATIENCE})")
            if patience >= PATIENCE:
                print(f"\n🛑 Early stopping at epoch {epoch+1}")
                break

    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f)

    plt.figure(figsize=(10,5))
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'],   label='Val')
    plt.xlabel('Epoch'); plt.ylabel('MSE Loss')
    plt.title('Porion Dedicated Model Loss')
    plt.legend(); plt.tight_layout()
    plt.savefig('porion_loss_plot.png', dpi=150)
    plt.show()
    print(f"\n✅ Done. Model saved to {PORION_MODEL}")

if __name__ == "__main__":
    train()