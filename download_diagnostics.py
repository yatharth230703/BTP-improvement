import modal
import os
from pathlib import Path

# --- Config ---
APP_NAME = "fintech-trimodal-research"
VOLUME_NAME = "fintech-data-vol"
REMOTE_MODEL_PATH = Path("/vol/models")
LOCAL_DOWNLOAD_DIR = Path("downloaded_diagnostics")

# --- Setup ---
stub = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOLUME_NAME)

# --- Define a temporary Image for reading the volume ---
image = modal.Image.from_registry("python:3.10-slim")

@stub.function(
    image=image,
    volumes={Path("/vol"): vol},
    # The timeout is set high enough to allow the file transfers
    timeout=600 
)
def download_files_remote():
    """Reads all diagnostic PNGs and saves them directly to the local directory."""
    print(f"📂 Searching for diagnostic plots in {REMOTE_MODEL_PATH}...")
    
    if not REMOTE_MODEL_PATH.exists():
        print("❌ Model path does not exist!")
        return 

    # 1. Create the local download directory if it doesn't exist
    # Note: This runs on the remote container, but we use the Modal client to manage the final save.
    
    # 2. Use Modal's built-in function to download the directory content
    # This is often more robust than reading file-by-file with open()
    print(f"⬇️ Downloading content from {REMOTE_MODEL_PATH} to local path {LOCAL_DOWNLOAD_DIR.resolve()}...")
    
    # The .to() method handles the transfer and creation of the local directory
    try:
        # We need to get the *parent* path of the files, which is /vol/models
        # and pull that specific directory.
        vol.get_path(REMOTE_MODEL_PATH).to(LOCAL_DOWNLOAD_DIR)
        print("✅ Transfer initiated successfully.")
    except Exception as e:
        print(f"❌ Failed during transfer: {e}")

if __name__ == "__main__":
    LOCAL_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Local target directory created: {LOCAL_DOWNLOAD_DIR.resolve()}")
    
    print("\n🚀 Starting remote download function...")
    with stub.run():
        download_files_remote.remote()
        
    print(f"\n🎉 Please check the '{LOCAL_DOWNLOAD_DIR}' folder for the images.")
    print("Next step: Upload HDFCBANK_diagnostics.png and SBIN_diagnostics.png.")