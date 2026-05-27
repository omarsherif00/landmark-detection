from huggingface_hub import HfApi
import os

api     = HfApi()
REPO_ID = "OmarSherif0/landmark-detection"

files = ["app.py", "requirements.txt", "Dockerfile", "trained_model.pth"]

# Add refine model if it exists
if os.path.exists("refine_model.pth"):
    files.append("refine_model.pth")
    print("✅ refine_model.pth found — will upload both models")
else:
    print("⚠ refine_model.pth not found — uploading Stage 1 only")

for filename in files:
    print(f"Uploading {filename}...")
    api.upload_file(
        path_or_fileobj = filename,
        path_in_repo    = filename,
        repo_id         = REPO_ID,
        repo_type       = "space"
    )
    print(f"  ✅ {filename} uploaded")

print("\nDone!")