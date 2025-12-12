import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import os

def generate_finbert_embeddings(news_path, output_path, model_name="ProsusAI/finbert", batch_size=32):
    """
    Generates daily semantic embeddings using a specified FinBERT model.
    """
    print(f"🚀 Initializing Model: {model_name}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load specific tokenizer and model
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
        model.eval()
    except Exception as e:
        print(f"❌ Error loading model {model_name}: {e}")
        return None
    
    # Load Data
    print(f"📂 Loading News Data from {news_path}...")
    if not os.path.exists(news_path):
        print(f"❌ News file not found at {news_path}")
        return None
        
    df = pd.read_csv(news_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Combine Title and Description
    df['Title'] = df['Title'].fillna('')
    df['Description'] = df['Description'].fillna('')
    df['Full_Text'] = df['Title'] + ". " + df['Description']
    
    # Filter short text
    df = df[df['Full_Text'].str.len() > 5]
    
    # Storage
    all_embeddings = []
    
    print(f"🧠 Generating Embeddings with {model_name}...")
    texts = df['Full_Text'].tolist()
    
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i : i + batch_size]
            
            # Tokenize
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
            
            # Forward pass
            outputs = model(**inputs)
            
            # Get [CLS] embeddings
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(embeddings)
            
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    df['embedding'] = list(all_embeddings)
    
    print("∑ Aggregating Embeddings by Date...")
    daily_embeddings = df.groupby('Date')['embedding'].apply(lambda x: np.mean(np.vstack(x), axis=0))
    
    # Convert to DataFrame
    emb_df = pd.DataFrame(daily_embeddings.tolist(), index=daily_embeddings.index)
    emb_df.columns = [f'emb_{i}' for i in range(emb_df.shape[1])]
    
    # Save
    print(f"💾 Saving {model_name} Embeddings to {output_path}...")
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    emb_df.to_csv(output_path)
    
    return emb_df