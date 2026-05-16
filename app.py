import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import requests
import io
import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# ---- CONFIG ----
NUM_LANDMARKS = 21
IMAGE_SIZE    = (512, 512)
MODEL_FILE    = "trained_model.pth"
# ----------------

SELECTED_LANDMARKS = [
    'S', 'Na', 'Po', 'Or', 'ANS', 'PNS', 'Ar', 'Go', 'Me',
    'Mx-ABAM', 'Mx-ALAM', 'Mx-PBAM', 'Mx-PLAM',
    'Md-ABAM', 'Md-ALAM', 'Md-PBAM', 'Md-PLAM',
    'U1IncisalTip', 'U1RootTip', 'L1IncisalTip', 'L1RootTip'
]

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

# ---- Load model once at startup ----
device = torch.device("cpu")
model  = UNet(in_channels=3, out_channels=NUM_LANDMARKS).to(device)
model.load_state_dict(torch.load(MODEL_FILE, map_location=device, weights_only=True))
model.eval()
print("✅ Model loaded successfully")

# ---- FastAPI app ----
app = FastAPI()

class PredictRequest(BaseModel):
    image_url: str

def heatmaps_to_coords(heatmaps, orig_w, orig_h):
    N, H, W = heatmaps.shape
    points  = []
    for i in range(N):
        idx    = np.unravel_index(np.argmax(heatmaps[i]), heatmaps[i].shape)
        y_px, x_px = idx
        points.append((
            round(float(x_px * (orig_w / W)), 2),
            round(float(y_px * (orig_h / H)), 2)
        ))
    return points

@app.get("/")
def root():
    return {"status": "Landmark Detection API is running"}

@app.post("/predict")
def predict(req: PredictRequest):
    try:
        # Load image from URL
        if not (req.image_url.startswith("http://") or req.image_url.startswith("https://")):
            raise HTTPException(status_code=400, detail="Invalid URL")

        resp  = requests.get(req.image_url, timeout=10)
        image = Image.open(io.BytesIO(resp.content)).convert("RGB")
        orig_w, orig_h = image.size

        # Preprocess
        resized    = image.resize(IMAGE_SIZE)
        img_tensor = transforms.ToTensor()(resized).unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            output = model(img_tensor)

        heatmaps = output[0].cpu().numpy()
        points   = heatmaps_to_coords(heatmaps, orig_w, orig_h)

        # Build Excel in memory
        df = pd.DataFrame({
            'Name': SELECTED_LANDMARKS,
            'X':    [p[0] for p in points],
            'Y':    [p[1] for p in points]
        })

        buffer = io.BytesIO()
        df.to_excel(buffer, index=False)
        buffer.seek(0)

        # Return Excel file directly
        return StreamingResponse(
            buffer,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": "attachment; filename=landmarks.xlsx"}
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))