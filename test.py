import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import os

# ---- CONFIG ----
NUM_LANDMARKS   = 21
IMAGE_SIZE      = (512, 512)
MODEL_SAVE_PATH = "trained_model.pth"
PORION_MODEL    = "porion_model.pth"
CROP_SIZE       = 256
# ----------------

SELECTED_LANDMARKS = [
    'S', 'Na', 'Po', 'Or', 'ANS', 'PNS', 'Ar', 'Go', 'Me',
    'Mx-ABAM', 'Mx-ALAM', 'Mx-PBAM', 'Mx-PLAM',
    'Md-ABAM', 'Md-ALAM', 'Md-PBAM', 'Md-PLAM',
    'U1IncisalTip', 'U1RootTip', 'L1IncisalTip', 'L1RootTip'
]
PORION_IDX = SELECTED_LANDMARKS.index('Po')

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

class PorionUNet(nn.Module):
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

def heatmaps_to_coords(heatmaps, orig_w, orig_h):
    N, H, W = heatmaps.shape
    points  = []
    for i in range(N):
        idx    = np.unravel_index(np.argmax(heatmaps[i]), heatmaps[i].shape)
        y_px, x_px = idx
        points.append((x_px*(orig_w/W), y_px*(orig_h/H)))
    return points

def soft_argmax(heatmap, out_w, out_h):
    H, W    = heatmap.shape
    hm_flat = heatmap.reshape(-1).astype(np.float64)
    hm_flat = hm_flat - hm_flat.max()
    weights = np.exp(hm_flat * 30.0)
    weights = weights / (weights.sum() + 1e-9)
    weights = weights.reshape(H, W)
    xs = np.arange(W, dtype=np.float64)
    ys = np.arange(H, dtype=np.float64)
    x  = (weights.sum(axis=0) * xs).sum()
    y  = (weights.sum(axis=1) * ys).sum()
    return x * (out_w / W), y * (out_h / H)

def crop_patch(image, cx, cy, crop_size=256):
    orig_w, orig_h = image.size
    half = crop_size // 2
    x0   = max(0, min(int(cx)-half, orig_w-crop_size))
    y0   = max(0, min(int(cy)-half, orig_h-crop_size))
    crop = image.crop((x0, y0, x0+crop_size, y0+crop_size))
    return crop, x0, y0

def load_ground_truth(csv_path):
    df = pd.read_csv(csv_path)
    df = df[df['Name'].isin(SELECTED_LANDMARKS)]
    df = df.set_index('Name').loc[SELECTED_LANDMARKS].reset_index()
    return df['X'].values.astype(np.float32), \
           df['Y'].values.astype(np.float32)

def estimate_px_to_mm(points, gt_x=None, gt_y=None):
    s_idx  = SELECTED_LANDMARKS.index('S')
    na_idx = SELECTED_LANDMARKS.index('Na')
    if gt_x is not None:
        sx,sy   = gt_x[s_idx],  gt_y[s_idx]
        nax,nay = gt_x[na_idx], gt_y[na_idx]
    else:
        sx,sy   = points[s_idx]
        nax,nay = points[na_idx]
    dist = np.sqrt((nax-sx)**2+(nay-sy)**2)
    return 71.0/dist if dist>1 else 0.1

def predict_and_show(image_path, gt_csv=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Stage 1
    stage1 = UNet(in_channels=3, out_channels=NUM_LANDMARKS).to(device)
    stage1.load_state_dict(torch.load(
        MODEL_SAVE_PATH, map_location=device, weights_only=True
    ))
    stage1.eval()

    # Load Porion model if available
    has_porion = os.path.exists(PORION_MODEL)
    porion_net = None
    if has_porion:
        porion_net = PorionUNet().to(device)
        porion_net.load_state_dict(torch.load(
            PORION_MODEL, map_location=device, weights_only=True
        ))
        porion_net.eval()
        print("✅ Porion dedicated model loaded")
    else:
        print("⚠ No porion_model.pth — using Stage 1 for all landmarks")

    original       = Image.open(image_path).convert("RGB")
    orig_w, orig_h = original.size

    # Stage 1 — all landmarks
    resized    = original.resize(IMAGE_SIZE)
    img_tensor = transforms.ToTensor()(resized).unsqueeze(0).to(device)

    with torch.no_grad():
        output = stage1(img_tensor)

    heatmaps = output[0].cpu().numpy()
    points   = heatmaps_to_coords(heatmaps, orig_w, orig_h)

    # Porion refinement
    if has_porion:
        s1_po_x, s1_po_y = points[PORION_IDX]
        crop, x0, y0     = crop_patch(original, s1_po_x, s1_po_y, CROP_SIZE)
        crop_tensor       = torch.from_numpy(
            np.array(crop).transpose(2,0,1)/255.0
        ).float().unsqueeze(0).to(device)

        with torch.no_grad():
            po_hm = porion_net(crop_tensor)[0,0].cpu().numpy()

        lx, ly = soft_argmax(po_hm, CROP_SIZE, CROP_SIZE)
        points[PORION_IDX] = (x0 + lx, y0 + ly)
        print(f"  Porion: Stage1=({s1_po_x:.1f},{s1_po_y:.1f}) "
              f"→ Refined=({x0+lx:.1f},{y0+ly:.1f})")

    # Ground truth
    has_gt = gt_csv is not None and os.path.exists(gt_csv)
    gt_x = gt_y = None
    if has_gt:
        gt_x, gt_y = load_ground_truth(gt_csv)

    # Draw
    draw_img = original.copy()
    draw     = ImageDraw.Draw(draw_img)
    for i, (x, y) in enumerate(points):
        r     = 6
        color = 'cyan' if i == PORION_IDX else 'red'
        draw.ellipse([x-r,y-r,x+r,y+r], fill=color, outline='yellow')
        draw.text((x+8,y-8), SELECTED_LANDMARKS[i], fill='white')
        if has_gt:
            gx, gy = gt_x[i], gt_y[i]
            draw.ellipse([gx-r,gy-r,gx+r,gy+r], fill='green', outline='white')
            draw.line([x,y,gx,gy], fill='yellow', width=1)

    if has_gt:
        px_to_mm  = estimate_px_to_mm(points, gt_x, gt_y)
        pred_arr  = np.array(points)
        errors_mm = np.sqrt(
            (pred_arr[:,0]-gt_x)**2+(pred_arr[:,1]-gt_y)**2
        ) * px_to_mm
        mre  = np.mean(errors_mm)
        sd   = np.std(errors_mm)
        sdr2 = np.mean(errors_mm<=2.0)*100
        sdr25= np.mean(errors_mm<=2.5)*100
        sdr3 = np.mean(errors_mm<=3.0)*100
        sdr4 = np.mean(errors_mm<=4.0)*100

        print(f"\n{'='*55}")
        print(f"  ACCURACY — {os.path.basename(image_path)}")
        print(f"{'='*55}")
        print(f"  MRE ± SD    : {mre:.2f} ± {sd:.2f} mm")
        print(f"  SDR @ 2.0mm : {sdr2:.2f}%")
        print(f"  SDR @ 2.5mm : {sdr25:.2f}%")
        print(f"  SDR @ 3.0mm : {sdr3:.2f}%")
        print(f"  SDR @ 4.0mm : {sdr4:.2f}%")
        print(f"{'='*55}")
        for i, err in enumerate(errors_mm):
            flag = " ✗" if err>4 else (" ~" if err>2 else " ✓")
            marker = " ← dedicated model" if i == PORION_IDX else ""
            print(f"  {i+1:<3} {SELECTED_LANDMARKS[i]:<20} "
                  f"{err:.2f}mm{flag}{marker}")

        draw.rectangle([10,10,330,150], fill=(0,0,0))
        draw.text((15,15), f"MRE: {mre:.2f} ± {sd:.2f} mm", fill='white')
        draw.text((15,35), f"SDR 2mm:  {sdr2:.1f}%",         fill='lime')
        draw.text((15,55), f"SDR 2.5mm:{sdr25:.1f}%",        fill='lime')
        draw.text((15,75), f"SDR 3mm:  {sdr3:.1f}%",         fill='lime')
        draw.text((15,95), f"SDR 4mm:  {sdr4:.1f}%",         fill='lime')
        draw.text((15,120),"CYAN = Porion (dedicated model)", fill='cyan')

    base = os.path.splitext(os.path.basename(image_path))[0]
    pd.DataFrame({
        'Name': SELECTED_LANDMARKS,
        'X':    [round(p[0],2) for p in points],
        'Y':    [round(p[1],2) for p in points]
    }).to_excel(f"{base}_landmarks.xlsx", index=False)
    print(f"\n📊 Saved to {base}_landmarks.xlsx")

    plt.figure(figsize=(14,14))
    plt.imshow(draw_img)
    plt.title(f"Detected {NUM_LANDMARKS} Landmarks — "
              f"{os.path.basename(image_path)}", fontsize=13)
    plt.axis('off'); plt.tight_layout(); plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python test.py image.jpg")
        print("  python test.py image.jpg gt.csv --eval")
        sys.exit(1)
    image_path = sys.argv[1]
    gt_csv     = None
    if len(sys.argv)>=3 and "--eval" in sys.argv:
        gt_csv = sys.argv[2]
    predict_and_show(image_path, gt_csv)