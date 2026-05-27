import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import json
from my_dataset import MyLandmarkDataset, SELECTED_LANDMARKS, split_dataset

# ---- CONFIG ----
DATASET_DIR     = "C:/Users/sanda/OneDrive/Desktop/Archive"
NUM_LANDMARKS   = 21
IMAGE_SIZE      = (512, 512)
MODEL_SAVE_PATH = "trained_model.pth"
OUTPUT_DIR      = "evaluation_output"
# ----------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

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

def heatmap_to_coord(heatmap, orig_w, orig_h):
    H, W = heatmap.shape
    idx  = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    y, x = idx
    return x * (orig_w / W), y * (orig_h / H)

def estimate_px_to_mm(gt_points):
    s_idx  = SELECTED_LANDMARKS.index('S')
    na_idx = SELECTED_LANDMARKS.index('Na')
    dist   = np.sqrt(
        (gt_points[na_idx, 0] - gt_points[s_idx, 0])**2 +
        (gt_points[na_idx, 1] - gt_points[s_idx, 1])**2
    )
    return 71.0 / dist if dist > 1 else 0.1

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = UNet(in_channels=3, out_channels=NUM_LANDMARKS).to(device)
    model.load_state_dict(torch.load(
        MODEL_SAVE_PATH, map_location=device, weights_only=True
    ))
    model.eval()

    if os.path.exists("test_samples.json"):
        with open("test_samples.json") as f:
            test_samples = [tuple(s) for s in json.load(f)]
        print(f"Loaded {len(test_samples)} test samples")
    else:
        _, _, test_samples = split_dataset(DATASET_DIR)

    all_errors_mm     = []
    per_lm_errors     = [[] for _ in range(NUM_LANDMARKS)]
    per_image_results = []
    sample_visuals    = []

    print(f"Evaluating on {len(test_samples)} test images...")

    for idx, (img_path, csv_path) in enumerate(test_samples):
        original       = Image.open(img_path).convert("RGB")
        orig_w, orig_h = original.size

        df     = pd.read_csv(csv_path)
        df     = df[df['Name'].isin(SELECTED_LANDMARKS)]
        df     = df.set_index('Name').loc[SELECTED_LANDMARKS].reset_index()
        gt_x   = df['X'].values.astype(np.float32)
        gt_y   = df['Y'].values.astype(np.float32)
        gt_pts = np.stack([gt_x, gt_y], axis=1)

        px_to_mm   = estimate_px_to_mm(gt_pts)
        resized    = original.resize(IMAGE_SIZE)
        img_tensor = torch.from_numpy(
            np.array(resized).transpose(2, 0, 1) / 255.0
        ).float().unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)

        hm       = output[0].cpu().numpy()
        pred_pts = np.array([
            heatmap_to_coord(hm[i], orig_w, orig_h)
            for i in range(NUM_LANDMARKS)
        ])

        errors_px = np.sqrt(((pred_pts - gt_pts)**2).sum(axis=1))
        errors_mm = errors_px * px_to_mm

        all_errors_mm.extend(errors_mm.tolist())
        for i, e in enumerate(errors_mm):
            per_lm_errors[i].append(e)

        per_image_results.append({
            'Image':    os.path.basename(img_path),
            'MRE_mm':  round(float(np.mean(errors_mm)), 3),
            'SD_mm':   round(float(np.std(errors_mm)),  3),
            'px_to_mm': round(px_to_mm, 4)
        })

        if idx < 5:
            sample_visuals.append((original, pred_pts, gt_pts, img_path, hm))

        if (idx + 1) % 20 == 0:
            print(f"  Processed {idx+1}/{len(test_samples)} images...")

    all_errors_mm = np.array(all_errors_mm)
    mre   = np.mean(all_errors_mm)
    sd    = np.std(all_errors_mm)
    sdr2  = np.mean(all_errors_mm <= 2.0) * 100
    sdr25 = np.mean(all_errors_mm <= 2.5) * 100
    sdr3  = np.mean(all_errors_mm <= 3.0) * 100
    sdr4  = np.mean(all_errors_mm <= 4.0) * 100

    print("\n" + "="*65)
    print("  OVERALL TEST SET RESULTS")
    print("="*65)
    print(f"  Images evaluated : {len(test_samples)}")
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
        e    = np.array(per_lm_errors[i])
        flag = " ✗" if np.mean(e) > 4 else (" ~" if np.mean(e) > 2 else " ✓")
        print(f"  {i+1:<4} {name:<20} {np.mean(e):>8.2f} {np.std(e):>8.2f} "
              f"{np.mean(e<=2)*100:>7.1f}% {np.mean(e<=4)*100:>7.1f}%{flag}")

    print(f"\n  COMPARISON WITH LITERATURE:")
    print(f"  {'Method':<25} {'MRE±SD':>12} {'2mm':>7} {'2.5mm':>7} {'3mm':>7} {'4mm':>7}")
    print(f"  {'-'*68}")
    print(f"  {'Your Model':<25} {f'{mre:.2f}±{sd:.2f}':>12} "
          f"{sdr2:>6.1f}% {sdr25:>6.1f}% {sdr3:>6.1f}% {sdr4:>6.1f}%")
    print(f"  {'Khan et al. (2025)':<25} {'1.69±3.36':>12} "
          f"{'81.18':>7} {'87.28':>7} {'90.82':>7} {'94.82':>7}")
    print(f"  {'Khan et al. (2024)':<25} {'1.92±7.85':>12} "
          f"{'78.54':>7} {'85.72':>7} {'89.64':>7} {'94.49':>7}")
    print(f"  {'Khalid et al. (2024)':<25} {'1.87±4.01':>12} "
          f"{'75.17':>7} {'82.43':>7} {'88.78':>7} {'93.01':>7}")

    # ---- Plots ----
    if os.path.exists("training_history.json"):
        with open("training_history.json") as f:
            history = json.load(f)
        plt.figure(figsize=(10, 5))
        plt.plot(history['train_loss'], label='Train Loss', color='blue')
        plt.plot(history['val_loss'],   label='Val Loss',   color='orange')
        plt.xlabel('Epoch'); plt.ylabel('MSE Loss')
        plt.title('Train vs Validation Loss')
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'loss_plot.png'), dpi=150)
        plt.show()
        print("✅ Loss plot saved")

    lm_mres = [np.mean(per_lm_errors[i]) for i in range(NUM_LANDMARKS)]
    colors  = ['red' if m > 4 else ('orange' if m > 2 else 'green')
               for m in lm_mres]
    plt.figure(figsize=(14, 6))
    plt.bar(SELECTED_LANDMARKS, lm_mres, color=colors)
    plt.axhline(y=2.0, color='orange', linestyle='--', label='2mm threshold')
    plt.axhline(y=4.0, color='red',    linestyle='--', label='4mm threshold')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.ylabel('MRE (mm)')
    plt.title('Per-Landmark Mean Radial Error')
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'per_landmark_mre.png'), dpi=150)
    plt.show()
    print("✅ Per-landmark MRE plot saved")

    thresholds = [2.0, 2.5, 3.0, 4.0]
    sdr_values = [sdr2, sdr25, sdr3, sdr4]
    plt.figure(figsize=(8, 5))
    plt.bar([f'{t}mm' for t in thresholds], sdr_values, color='steelblue')
    plt.ylabel('SDR (%)'); plt.ylim(0, 100)
    plt.title('Success Detection Rate')
    for i, v in enumerate(sdr_values):
        plt.text(i, v+1, f'{v:.1f}%', ha='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'sdr_plot.png'), dpi=150)
    plt.show()
    print("✅ SDR plot saved")

    plt.figure(figsize=(10, 5))
    plt.hist(all_errors_mm, bins=50, color='steelblue', edgecolor='black')
    plt.axvline(x=mre, color='red',    linestyle='--', label=f'MRE={mre:.2f}mm')
    plt.axvline(x=2.0, color='orange', linestyle='--', label='2mm threshold')
    plt.axvline(x=4.0, color='green',  linestyle='--', label='4mm threshold')
    plt.xlabel('Radial Error (mm)'); plt.ylabel('Frequency')
    plt.title('Distribution of Radial Errors')
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'error_distribution.png'), dpi=150)
    plt.show()
    print("✅ Error distribution saved")

    if test_samples:
        sample_img = Image.open(test_samples[0][0]).convert("RGB")
        import torchvision
        jitter    = torchvision.transforms.ColorJitter(
            brightness=0.3, contrast=0.3
        )
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        axes[0].imshow(sample_img)
        axes[0].set_title('Original'); axes[0].axis('off')
        for i in range(1, 5):
            axes[i].imshow(jitter(sample_img))
            axes[i].set_title(f'Augmented {i}'); axes[i].axis('off')
        plt.suptitle('ColorJitter Augmentation Examples')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'augmentation.png'), dpi=150)
        plt.show()
        print("✅ Augmentation visualization saved")

    for orig_img, pred_pts, gt_pts, img_path, heatmaps in sample_visuals:
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        draw_img  = orig_img.copy()
        draw      = ImageDraw.Draw(draw_img)
        for i in range(NUM_LANDMARKS):
            r = 6
            draw.ellipse([pred_pts[i,0]-r, pred_pts[i,1]-r,
                          pred_pts[i,0]+r, pred_pts[i,1]+r],
                         fill='red', outline='yellow')
            draw.ellipse([gt_pts[i,0]-r,   gt_pts[i,1]-r,
                          gt_pts[i,0]+r,   gt_pts[i,1]+r],
                         fill='green', outline='white')
            draw.line([pred_pts[i,0], pred_pts[i,1],
                       gt_pts[i,0],  gt_pts[i,1]],
                      fill='yellow', width=1)
        axes[0].imshow(draw_img)
        axes[0].set_title(f'Overlay — {os.path.basename(img_path)}')
        axes[0].axis('off')
        rp = mpatches.Patch(color='red',   label='Predicted')
        gp = mpatches.Patch(color='green', label='Ground Truth')
        axes[0].legend(handles=[rp, gp])
        axes[1].imshow(heatmaps[0], cmap='hot')
        axes[1].set_title(f'Heatmap — {SELECTED_LANDMARKS[0]}')
        axes[1].axis('off')
        plt.tight_layout()
        save_name = os.path.join(
            OUTPUT_DIR,
            f'overlay_{os.path.splitext(os.path.basename(img_path))[0]}.png'
        )
        plt.savefig(save_name, dpi=150)
        plt.show()
    print("✅ Qualitative overlays saved")

    if sample_visuals:
        _, _, _, _, heatmaps = sample_visuals[0]
        cols  = 7
        rows  = (NUM_LANDMARKS + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(20, rows*3))
        axes  = axes.flatten()
        for i in range(NUM_LANDMARKS):
            axes[i].imshow(heatmaps[i], cmap='hot')
            axes[i].set_title(SELECTED_LANDMARKS[i], fontsize=7)
            axes[i].axis('off')
        for i in range(NUM_LANDMARKS, len(axes)):
            axes[i].axis('off')
        plt.suptitle('Predicted Heatmaps — All 21 Landmarks')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'heatmap_grid.png'), dpi=150)
        plt.show()
        print("✅ Heatmap grid saved")

    pd.DataFrame(per_image_results).to_excel(
        os.path.join(OUTPUT_DIR, 'evaluation_results.xlsx'), index=False
    )
    lm_summary = pd.DataFrame({
        'Landmark': SELECTED_LANDMARKS,
        'MRE_mm':   [round(np.mean(per_lm_errors[i]), 3)
                     for i in range(NUM_LANDMARKS)],
        'SD_mm':    [round(np.std(per_lm_errors[i]),  3)
                     for i in range(NUM_LANDMARKS)],
        'SDR_2mm':  [round(np.mean(np.array(per_lm_errors[i])<=2)*100, 1)
                     for i in range(NUM_LANDMARKS)],
        'SDR_4mm':  [round(np.mean(np.array(per_lm_errors[i])<=4)*100, 1)
                     for i in range(NUM_LANDMARKS)],
    })
    lm_summary.to_excel(
        os.path.join(OUTPUT_DIR, 'per_landmark_results.xlsx'), index=False
    )
    print(f"\n✅ All results saved to: {OUTPUT_DIR}/")

if __name__ == "__main__":
    evaluate()