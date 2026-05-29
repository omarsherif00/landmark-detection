from huggingface_hub import HfApi
import os

api     = HfApi()
REPO_ID = "OmarSherif0/landmark-detection"

files = [
    "app.py",
    "requirements.txt",
    "Dockerfile",
    "trained_model.pth",
    "porion_model.pth"
]

for filename in files:
    if os.path.exists(filename):
        print(f"Uploading {filename}...")
        api.upload_file(
            path_or_fileobj = filename,
            path_in_repo    = filename,
            repo_id         = REPO_ID,
            repo_type       = "space"
        )
        print(f"  ✅ {filename} uploaded")
    else:
        print(f"  ⚠ {filename} not found, skipping")

print("\nDone!")