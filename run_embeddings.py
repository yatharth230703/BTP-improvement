import modal
import os
from pathlib import Path

APP_NAME = "fintech-embeddings-gen"
VOLUME_NAME = "fintech-data-vol"
REMOTE_MOUNT_PATH = Path("/vol")
LOCAL_SRC_DIR = Path("src")

# We use the NVIDIA image to ensure GPU drivers are perfect for Transformers
image = (
    modal.Image.from_registry("nvidia/cuda:12.2.0-devel-ubuntu22.04", add_python="3.10")
    .pip_install("torch", "pandas", "numpy", "transformers", "tqdm")
    .add_local_python_source(str(LOCAL_SRC_DIR))
)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

@app.function(
    image=image,
    gpu="H100", # Using H100 to speed up the dual generation
    timeout=7200, # Increased timeout for two passes
    volumes={REMOTE_MOUNT_PATH: volume}
)
def generate_remote():
    from src.features.text_processing import generate_finbert_embeddings
    
    news_file = REMOTE_MOUNT_PATH / "raw_upload" / "Enriched_IndianFinancialNews.csv"
    
    if not news_file.exists():
        print(f"❌ News file not found at {news_file}. Please run the upload step first.")
        return

    # --- Pass 1: Standard ProsusAI FinBERT ---
    print("\n🔹 STARTING PASS 1: ProsusAI/finbert")
    output_prosus = REMOTE_MOUNT_PATH / "processed" / "daily_embeddings_prosus.csv"
    generate_finbert_embeddings(
        str(news_file), 
        str(output_prosus), 
        model_name="ProsusAI/finbert"
    )
    
    # --- Pass 2: Indian Market Fine-Tuned FinBERT ---
    print("\n🔸 STARTING PASS 2: kdave/FineTuned_Finbert")
    output_kdave = REMOTE_MOUNT_PATH / "processed" / "daily_embeddings_kdave.csv"
    generate_finbert_embeddings(
        str(news_file), 
        str(output_kdave), 
        model_name="kdave/FineTuned_Finbert"
    )

@app.local_entrypoint()
def main():
    print("📤 Uploading Raw News Data...")
    vol = modal.Volume.from_name(VOLUME_NAME)
    
    # Ensure raw news is uploaded (force=True to be safe)
    with vol.batch_upload(force=True) as batch:
        batch.put_file(
            os.path.join("data", "raw", "Enriched_IndianFinancialNews.csv"), 
            remote_path="raw_upload/Enriched_IndianFinancialNews.csv"
        )
    
    print("🚀 Launching Dual Embedding Generation on H100...")
    generate_remote.remote()
    print("✅ Done! Both embedding files saved to Cloud Volume.")