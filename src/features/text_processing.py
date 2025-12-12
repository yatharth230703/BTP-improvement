import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import os

def generate_finbert_scores(news_path, output_path, model_name="ProsusAI/finbert", batch_size=32):
    """
    Generates Daily Sentiment SCORES (Pos, Neg, Neu) instead of Embeddings.
    Matches the methodology of Karadas et al. (2025).
    """
    print(f"🚀 Initializing FinBERT for Scoring: {model_name}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # FIX: Use SequenceClassification to get probabilities, not just embeddings
        model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        model.eval()
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None
    
    print(f"📂 Loading News Data from {news_path}...")
    if not os.path.exists(news_path): return None
    df = pd.read_csv(news_path)
    
    # Combine text
    df['Full_Text'] = df['Title'].fillna('') + ". " + df['Description'].fillna('')
    df = df[df['Full_Text'].str.len() > 5]
    
    # Storage
    sentiment_scores = []
    
    print("🧠 Calculating Sentiment Probabilities...")
    texts = df['Full_Text'].tolist()
    
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i : i + batch_size]
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
            outputs = model(**inputs)
            
            # Apply Softmax to get probabilities (Pos, Neg, Neu)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            sentiment_scores.append(probs.cpu().numpy())
            
    # Flatten and assign columns
    scores = np.concatenate(sentiment_scores, axis=0)
    # ProsusAI/finbert labels: [Positive, Negative, Neutral] (Verify specific model config if needed)
    df['Prob_Pos'] = scores[:, 0]
    df['Prob_Neg'] = scores[:, 1]
    df['Prob_Neu'] = scores[:, 2]
    
    # --- PAPER REPLICATION LOGIC ---
    # Calculate a "Net Sentiment Score" for each article: (Pos - Neg)
    df['Sentiment_Score'] = df['Prob_Pos'] - df['Prob_Neg']
    
    print("∑ Aggregating Daily Sentiment...")
    # We aggregate by taking the MEAN score and the COUNT of articles (Volume)
    daily_stats = df.groupby('Date').agg({
        'Sentiment_Score': 'mean',  # The average mood
        'Prob_Pos': 'mean',
        'Prob_Neg': 'mean',
        'Title': 'count'            # Volume of news (Proxy for engagement)
    })
    
    daily_stats.rename(columns={'Title': 'News_Volume'}, inplace=True)
    
    # Save
    print(f"💾 Saving Sentiment Scores to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    daily_stats.to_csv(output_path)
    return daily_stats