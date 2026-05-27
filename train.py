import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from my_dataset import MyLandmarkDataset, split_dataset
import json
import os
import matplotlib.pyplot as plt

# ---- CONFIG ----
DATASET_DIR     = "C:/Users/sanda/OneDrive/Desktop/Archive"
NUM_LANDMARKS   = 21
IMAGE_SIZE      = (512, 512)
BATCH_SIZE      = 4
EPOCHS          = 150
LEARNING_RATE   = 1e-4
PATIENCE        = 15
MODEL_SAVE_PATH = "trained_model.pth"
HISTORY_PATH    = "training_history.json"
# ----------------

# ---- U-Net ----
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
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

# ---- Setup ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

train_samples, val_samples, test_samples = split_dataset(DATASET_DIR)

# Save test sample paths so evaluate.py uses the exact same test set
with open("test_samples.json", "w") as f:
    json.dump(test_samples, f)
print(f"Test samples saved to test_samples.json")

train_ds = MyLandmarkDataset(train_samples, image_size=IMAGE_SIZE, augment=True)
val_ds   = MyLandmarkDataset(val_samples,   image_size=IMAGE_SIZE, augment=False)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

model     = UNet(in_channels=3, out_channels=NUM_LANDMARKS).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
criterion = nn.MSELoss()

# ---- Training ----
best_val_loss    = float('inf')
patience_counter = 0
history          = {'train_loss': [], 'val_loss': []}

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for heatmaps, images, _, names in train_loader:
        images, heatmaps = images.to(device), heatmaps.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), heatmaps)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for heatmaps, images, _, names in val_loader:
            images, heatmaps = images.to(device), heatmaps.to(device)
            val_loss += criterion(model(images), heatmaps).item()

    train_loss /= len(train_loader)
    val_loss   /= len(val_loader)
    scheduler.step()

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)

    print(f"Epoch [{epoch+1}/{EPOCHS}] Train: {train_loss:.6f} | Val: {val_loss:.6f}")

    if val_loss < best_val_loss:
        best_val_loss    = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"  ✅ Saved (val_loss: {val_loss:.6f})")
    else:
        patience_counter += 1
        print(f"  ⏳ No improvement ({patience_counter}/{PATIENCE})")
        if patience_counter >= PATIENCE:
            print(f"\n🛑 Early stopping at epoch {epoch+1}")
            break

# Save loss history
with open(HISTORY_PATH, "w") as f:
    json.dump(history, f)
print(f"\n✅ Training complete. History saved to {HISTORY_PATH}")

# Plot train/val loss
plt.figure(figsize=(10, 5))
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'],   label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.tight_layout()
plt.savefig('loss_plot.png', dpi=150)
plt.show()
print("Loss plot saved to loss_plot.png")