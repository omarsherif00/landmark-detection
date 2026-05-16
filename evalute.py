import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from PIL import Image
import pandas as pd
import numpy as np
import os
from my_dataset import MyLandmarkDataset, SELECTED_LANDMARKS

# ---- CONFIG ----
DATASET_DIR     = "C:/Users/sanda/OneDrive/Desktop/Archive"
NUM_LANDMARKS   = 21
IMAGE_SIZE      = (512, 512)
MODEL_SAVE_PATH = "trained_model.pth"
RANDOM_SEED     = 42
# ----------------

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

def heatmaps_to_coords(heatmaps, orig_w, orig_h):
    N, H, W = heatmaps.shape
    points = []
    for i in range(N):
        idx    = np.unravel_index(np.argmax(heatmaps[i]), heatmaps[i].shape)
        y_px, x_px = idx
        points.append((x_px * (orig_w / W), y_px * (orig_h / H)))
    return np.array(points)

def estimate_px_to_mm(gt_points):
    s_idx  = SELECTED_LANDMARKS.index('S')
    na_idx = SELECTED_LANDMARKS.index('Na')
    dist   = np.sqrt((gt_points[na_idx,0]-gt_points[s_idx,0])**2 +
                     (gt_points[na_idx,1]-gt_points[s_idx,1])**2)
    return 71.0 / dist if dist > 1 else 0.1

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    torch.manual_seed(RANDOM_SEED)
    full_ds    = MyLandmarkDataset(DATASET_DIR, image_size=IMAGE_SIZE, augment=False)
    train_size = int(0.8 * len(full_ds))
    val_size   = len(full_ds) - train_size
    _, val_ds  = random_split(full_ds, [train_size, val_size],
                              generator=torch.Generator().manual_seed(RANDOM_SEED))

    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    print(f"Test set: {len(val_ds)} images")

    model = UNet(in_channels=3, out_channels=NUM_LANDMARKS).to(device)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device, weights_only=True))
    model.eval()

    all_errors       = []
    per_lm_errors    = [[] for _ in range(NUM_LANDMARKS)]
    per_image_results = []

    with torch.no_grad():
        for images, heatmaps_gt, img_paths, _ in val_loader:
            images = images.to(device)
            output = model(images)

            img_path = img_paths[0]
            orig_w, orig_h = Image.open(img_path).size

            pred_pts = heatmaps_to_coords(output[0].cpu().numpy(), orig_w, orig_h)
            gt_pts   = heatmaps_to_coords(heatmaps_gt[0].numpy(),  orig_w, orig_h)

            px_to_mm  = estimate_px_to_mm(gt_pts)
            errors_px = np.sqrt(((pred_pts - gt_pts)**2).sum(axis=1))
            errors_mm = errors_px * px_to_mm

            all_errors.extend(errors_mm.tolist())
            for i, e in enumerate(errors_mm):
                per_lm_errors[i].append(e)

            per_image_results.append({
                'Image': os.path.basename(img_path),
                'MRE_mm': round(float(np.mean(errors_mm)), 3),
                'SD_mm':  round(float(np.std(errors_mm)),  3),
                'px_to_mm': round(px_to_mm, 4)
            })

    all_errors = np.array(all_errors)
    mre   = np.mean(all_errors)
    sd    = np.std(all_errors)
    sdr2  = np.mean(all_errors <= 2.0) * 100
    sdr25 = np.mean(all_errors <= 2.5) * 100
    sdr3  = np.mean(all_errors <= 3.0) * 100
    sdr4  = np.mean(all_errors <= 4.0) * 100

    print("\n" + "="*65)
    print("  OVERALL TEST SET RESULTS")
    print("="*65)
    print(f"  Images evaluated : {len(val_ds)}")
    print(f"  MRE ± SD         : {mre:.2f} ± {sd:.2f} mm")
    print(f"  SDR @ 2.0mm      : {sdr2:.2f}%")
    print(f"  SDR @ 2.5mm      : {sdr25:.2f}%")
    print(f"  SDR @ 3.0mm      : {sdr3:.2f}%")
    print(f"  SDR @ 4.0mm      : {sdr4:.2f}%")
    print("="*65)

    print(f"\n  PER-LANDMARK RESULTS:")
    print(f"  {'#':<4} {'Landmark':<20} {'MRE':>8} {'SD':>8} {'SDR2mm':>8} {'SDR4mm':>8}")
    print(f"  {'-'*58}")
    for i, name in enumerate(SELECTED_LANDMARKS):
        e     = np.array(per_lm_errors[i])
        flag  = " ✗" if np.mean(e) > 4 else (" ~" if np.mean(e) > 2 else " ✓")
        print(f"  {i+1:<4} {name:<20} {np.mean(e):>8.2f} {np.std(e):>8.2f} "
              f"{np.mean(e<=2)*100:>7.1f}% {np.mean(e<=4)*100:>7.1f}%{flag}")

    print(f"\n  COMPARISON WITH LITERATURE:")
    print(f"  {'Method':<25} {'MRE±SD':>12} {'2mm':>7} {'2.5mm':>7} {'3mm':>7} {'4mm':>7}")
    print(f"  {'-'*68}")
    print(f"  {'Your Model':<25} {f'{mre:.2f}±{sd:.2f}':>12} "
          f"{sdr2:>6.1f}% {sdr25:>6.1f}% {sdr3:>6.1f}% {sdr4:>6.1f}%")
    print(f"  {'Khan et al. (2024)':<25} {'1.92±7.85':>12} "
          f"{'78.54':>7} {'85.72':>7} {'89.64':>7} {'94.49':>7}")
    print(f"  {'Khalid et al. (2024)':<25} {'1.87±4.01':>12} "
          f"{'75.17':>7} {'82.43':>7} {'88.78':>7} {'93.01':>7}")
    print(f"  {'Khan et al. (2025)':<25} {'1.69±3.36':>12} "
          f"{'81.18':>7} {'87.28':>7} {'90.82':>7} {'94.82':>7}")

    # Save per-image results
    pd.DataFrame(per_image_results).to_excel("evaluation_results.xlsx", index=False)
    print(f"\n  Per-image results saved to: evaluation_results.xlsx")

if __name__ == "__main__":
    evaluate()