# File: streamlit_app/utils/scorer.py

import os
import joblib
import pickle
import nltk
import pandas as pd
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import ast

# Import our other utility modules
from .parser import scrape_page, parse_html
from .features import engineer_features

# --- NLTK Data Path (Bulletproof) ---
NLTK_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'nltk_data')
nltk.data.path.append(NLTK_DATA_DIR)

# --- Model & Data Loading (Cached) ---
@st.cache_resource
def load_models():
    """Loads all models into memory."""
    print("Loading models...")
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, '..', 'models', 'quality_model.pkl')
    
    model_pipeline = joblib.load(model_path)
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    punkt_path = os.path.join(NLTK_DATA_DIR, 'tokenizers', 'punkt', 'english.pickle')
    with open(punkt_path, 'rb') as f:
        punkt_tokenizer = pickle.load(f)
        
    print("Models loaded successfully.")
    return model_pipeline, sbert_model, punkt_tokenizer

@st.cache_data
def load_corpus_embeddings():
    """Loads the corpus embeddings for similarity checks."""
    print("Loading corpus embeddings...")
    base_dir = os.path.dirname(__file__)
    features_path = os.path.join(base_dir, '..', 'data', 'features.csv')
    
    df_corpus = pd.read_csv(features_path)
    corpus_embeddings = np.array(df_corpus['embedding'].apply(ast.literal_eval).tolist())
    corpus_urls = df_corpus['url']
    print("Corpus loaded.")
    return corpus_embeddings, corpus_urls

# --- Main Analysis Function ---

def analyze_url(url, models, corpus):
    """
    Main function to analyze a single URL for SEO quality
    AND check for similarity against our corpus.
    """
    model_pipeline, sbert_model, punkt_tokenizer = models
    corpus_embeddings, corpus_urls = corpus
    
    # 1. Scrape (from parser.py)
    html = scrape_page(url)
    if not html: return {"error": f"Failed to scrape URL: {url}"}
    
    # 2. Parse (from parser.py)
    title, body_text = parse_html(html)
    if not body_text: return {"error": "Could not parse main content from page."}

    # 3. Engineer Features (from features.py)
    features = engineer_features(body_text, punkt_tokenizer)
    features_df = pd.DataFrame([features])
    
    # 4. Predict Quality
    try:
        prediction = model_pipeline.predict(features_df)
        quality_label = prediction[0]
    except Exception as e:
        print(f"Model prediction error: {e}")
        quality_label = "Error"
        
    # 5. Check for Similarity
    SIMILARITY_THRESHOLD = 0.80
    similar_to = []
    try:
        new_embedding = sbert_model.encode([body_text], show_progress_bar=False)
        sim_scores = cosine_similarity(new_embedding, corpus_embeddings)[0]
        matches = np.where(sim_scores > SIMILARITY_THRESHOLD)[0]
        
        for i in matches:
            # Correct logic: do not match the page to itself
            if corpus_urls[i] != url:
                similar_to.append({
                    "url": corpus_urls[i],
                    "similarity": round(float(sim_scores[i]), 2)
                })
    except Exception as e:
        print(f"Similarity check error: {e}")

    # 6. Build the JSON output
    result = {
        "url": url,
        "title": title,
        "word_count": features['word_count'],
        "readability": round(features['flesch_reading_ease'], 2),
        "quality_label": quality_label,
        "is_thin": bool(features['word_count'] < 500),
        "similar_to": similar_to
    }
    
    return result