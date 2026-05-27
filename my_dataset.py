import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch

SELECTED_LANDMARKS = [
    'S', 'Na', 'Po', 'Or', 'ANS', 'PNS', 'Ar', 'Go', 'Me',
    'Mx-ABAM', 'Mx-ALAM', 'Mx-PBAM', 'Mx-PLAM',
    'Md-ABAM', 'Md-ALAM', 'Md-PBAM', 'Md-PLAM',
    'U1IncisalTip', 'U1RootTip', 'L1IncisalTip', 'L1RootTip'
]

def generate_heatmap(height, width, x, y, sigma=10):
    xv = np.arange(width)
    yv = np.arange(height)
    xx, yy = np.meshgrid(xv, yv)
    heatmap = np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma**2))
    return heatmap.astype(np.float32)

class MyLandmarkDataset(Dataset):
    def __init__(self, samples, image_size=(512, 512), augment=False, sigma=10):
        """
        samples: list of (img_path, csv_path) tuples
        """
        self.samples        = samples
        self.image_size     = image_size
        self.augment        = augment
        self.sigma          = sigma
        self.landmark_names = SELECTED_LANDMARKS.copy()
        print(f"[Dataset] {len(self.samples)} samples loaded.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, csv_path = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size
        image = image.resize(self.image_size)

        if self.augment:
            enhancer = transforms.ColorJitter(brightness=0.3, contrast=0.3)
            image    = enhancer(image)

        img_tensor = transforms.ToTensor()(image)

        df       = pd.read_csv(csv_path)
        df       = df[df['Name'].isin(SELECTED_LANDMARKS)]
        df       = df.set_index('Name').loc[SELECTED_LANDMARKS].reset_index()
        x_coords = df['X'].values.astype(np.float32)
        y_coords = df['Y'].values.astype(np.float32)

        x_scaled = x_coords * (self.image_size[0] / orig_w)
        y_scaled = y_coords * (self.image_size[1] / orig_h)

        H, W = self.image_size[1], self.image_size[0]
        heatmaps = np.zeros((len(SELECTED_LANDMARKS), H, W), dtype=np.float32)
        for i in range(len(SELECTED_LANDMARKS)):
            heatmaps[i] = generate_heatmap(H, W, x_scaled[i], y_scaled[i], sigma=self.sigma)

        return torch.from_numpy(heatmaps), img_tensor, img_path, self.landmark_names


def load_all_samples(root_dir):
    """
    Load all valid (img, csv) pairs sorted by case number.
    Returns sorted list so last 10% are the most recent cases.
    """
    image_dir = os.path.join(root_dir, "images")
    excel_dir = os.path.join(root_dir, "Excel")

    samples = []
    skipped = 0
    for fname in os.listdir(image_dir):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            case_id  = os.path.splitext(fname)[0]
            csv_path = os.path.join(excel_dir, f"{case_id}.csv")
            img_path = os.path.join(image_dir, fname)
            if os.path.exists(csv_path):
                df_check = pd.read_csv(csv_path)
                missing  = [l for l in SELECTED_LANDMARKS if l not in df_check['Name'].values]
                if missing:
                    print(f"  Skipping {fname}: missing {missing}")
                    skipped += 1
                else:
                    samples.append((img_path, csv_path))

    # Sort by case number so last 10% are the newest cases
    def sort_key(s):
        try:
            return int(os.path.splitext(os.path.basename(s[0]))[0])
        except:
            return os.path.basename(s[0])

    samples.sort(key=sort_key)
    print(f"[Dataset] Found {len(samples)} valid pairs. Skipped {skipped}.")
    return samples


def split_dataset(root_dir, train_ratio=0.8, val_ratio=0.1):
    """
    80% train, 10% val, 10% test
    Test set = last 10% of cases (newest, for manual expert review)
    """
    samples   = load_all_samples(root_dir)
    n         = len(samples)
    n_test    = int(n * (1 - train_ratio - val_ratio))
    n_val     = int(n * val_ratio)
    n_train   = n - n_val - n_test

    train_samples = samples[:n_train]
    val_samples   = samples[n_train:n_train + n_val]
    test_samples  = samples[n_train + n_val:]  # last 10% = newest cases

    print(f"[Split] Train: {len(train_samples)} | Val: {len(val_samples)} | Test: {len(test_samples)}")
    return train_samples, val_samples, test_samples