import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import pandas as pd
import requests
import io
import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# ---- CONFIG ----
NUM_LANDMARKS   = 21
STAGE1_IMG_SIZE = (512, 512)
CROP_SIZE       = 256
STAGE1_MODEL    = "trained_model.pth"
PORION_MODEL    = "porion_model.pth"
PORION_IDX      = 2  # index of 'Po' in SELECTED_LANDMARKS
# ----------------

SELECTED_LANDMARKS = [
    'S', 'Na', 'Po', 'Or', 'ANS', 'PNS', 'Ar', 'Go', 'Me',
    'Mx-ABAM', 'Mx-ALAM', 'Mx-PBAM', 'Mx-PLAM',
    'Md-ABAM', 'Md-ALAM', 'Md-PBAM', 'Md-PLAM',
    'U1IncisalTip', 'U1RootTip', 'L1IncisalTip', 'L1RootTip'
]

# ---- Models ----
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
        self.bottleneck = nn.Sequential(
            DoubleConv(256, 512),
            nn.Dropout2d(p=0.3)
        )
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

# ---- Load models at startup ----
device = torch.device("cpu")

stage1 = UNet(in_channels=3, out_channels=NUM_LANDMARKS).to(device)
stage1.load_state_dict(torch.load(
    STAGE1_MODEL, map_location=device, weights_only=True
))
stage1.eval()

use_porion = os.path.exists(PORION_MODEL)
porion_net = None
if use_porion:
    porion_net = PorionUNet().to(device)
    porion_net.load_state_dict(torch.load(
        PORION_MODEL, map_location=device, weights_only=True
    ))
    porion_net.eval()
    print("✅ Both models loaded (Stage1 + Porion dedicated)")
else:
    print("⚠ Stage 1 only — no porion_model.pth found")

# ---- Helpers ----
def heatmap_to_coord(heatmap, orig_w, orig_h):
    H, W = heatmap.shape
    idx  = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    y, x = idx
    return x * (orig_w / W), y * (orig_h / H)

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
    x0   = max(0, min(int(cx) - half, orig_w - crop_size))
    y0   = max(0, min(int(cy) - half, orig_h - crop_size))
    crop = image.crop((x0, y0, x0 + crop_size, y0 + crop_size))
    return crop, x0, y0

def predict(image: Image.Image):
    original       = image.convert("RGB")
    orig_w, orig_h = original.size

    # Stage 1 — all landmarks
    resized    = original.resize(STAGE1_IMG_SIZE)
    img_tensor = torch.from_numpy(
        np.array(resized).transpose(2, 0, 1) / 255.0
    ).float().unsqueeze(0).to(device)

    with torch.no_grad():
        s1_out = stage1(img_tensor)

    s1_heatmaps = s1_out[0].cpu().numpy()
    coords = [
        heatmap_to_coord(s1_heatmaps[i], orig_w, orig_h)
        for i in range(NUM_LANDMARKS)
    ]

    # Porion refinement
    if use_porion and porion_net is not None:
        s1_po_x, s1_po_y = coords[PORION_IDX]
        crop, x0, y0     = crop_patch(original, s1_po_x, s1_po_y, CROP_SIZE)
        crop_tensor       = torch.from_numpy(
            np.array(crop).transpose(2, 0, 1) / 255.0
        ).float().unsqueeze(0).to(device)
        with torch.no_grad():
            po_hm = porion_net(crop_tensor)[0, 0].cpu().numpy()
        lx, ly = soft_argmax(po_hm, CROP_SIZE, CROP_SIZE)
        coords[PORION_IDX] = (x0 + lx, y0 + ly)

    return coords

# ---- FastAPI ----
app = FastAPI()

class PredictRequest(BaseModel):
    image_url: str

@app.get("/")
def root():
    stage = "Stage1 + Porion dedicated" if use_porion else "Stage1 only"
    return {"status": f"Landmark Detection API running ({stage})"}

@app.post("/predict")
def predict_endpoint(req: PredictRequest):
    try:
        if not (req.image_url.startswith("http://") or
                req.image_url.startswith("https://")):
            raise HTTPException(status_code=400, detail="Invalid URL")

        resp  = requests.get(req.image_url, timeout=10)
        image = Image.open(io.BytesIO(resp.content))
        coords = predict(image)

        df = pd.DataFrame({
            'Name': SELECTED_LANDMARKS,
            'X':    [round(c[0], 2) for c in coords],
            'Y':    [round(c[1], 2) for c in coords]
        })

        buffer = io.BytesIO()
        df.to_excel(buffer, index=False)
        buffer.seek(0)

        return StreamingResponse(
            buffer,
            media_type=(
                "application/vnd.openxmlformats-"
                "officedocument.spreadsheetml.sheet"
            ),
            headers={
                "Content-Disposition":
                    "attachment; filename=landmarks.xlsx"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))