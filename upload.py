from huggingface_hub import HfApi

api     = HfApi()
REPO_ID = "OmarSherif0/landmark-detection"

# Change Space SDK to Docker
print("Changing Space SDK to Docker...")
api.request_space_hardware(
    repo_id  = REPO_ID,
    hardware = "cpu-basic"
)

# Use create_repo with exist_ok to update SDK
from huggingface_hub import create_repo
create_repo(
    repo_id   = REPO_ID,
    repo_type = "space",
    space_sdk = "docker",
    exist_ok  = True
)
print("✅ SDK changed to Docker")

# Upload all files
for filename in ["app.py", "requirements.txt", "Dockerfile"]:
    print(f"Uploading {filename}...")
    api.upload_file(
        path_or_fileobj = filename,
        path_in_repo    = filename,
        repo_id         = REPO_ID,
        repo_type       = "space"
    )
    print(f"  ✅ {filename} uploaded")

# add to upload.py and run
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj = "README.md",
    path_in_repo    = "README.md",
    repo_id         = "OmarSherif0/landmark-detection",
    repo_type       = "space"
)
print("✅ README.md uploaded")

print("\nDone! Space will rebuild automatically.")