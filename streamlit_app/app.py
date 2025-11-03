# File: streamlit_app/app.py

import streamlit as st
import pandas as pd
from utils.scorer import load_models, load_corpus_embeddings, analyze_url

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="SEO Content Analyzer",
    page_icon="🔎",
    layout="wide"
)

# --- 2. Load Models & Data ---
with st.spinner("Loading models... This may take a moment."):
    # This now calls our scorer.py functions
    models = load_models()
    corpus = load_corpus_embeddings()

st.success("Models and data loaded successfully!")

# --- 3. App Title & Description ---
st.title("🔎 SEO Content Quality & Duplicate Detector")
st.markdown("""
This app analyzes any live URL to assess its content quality for SEO. 
It uses a machine learning model to score the content and semantic search to find potential duplicates against a known dataset.
""")

# --- 4. User Input ---
url_to_analyze = st.text_input(
    "Enter a URL to analyze:", 
    placeholder="https://www.example.com/my-article"
)

if st.button("Analyze Content", type="primary"):
    if not url_to_analyze or not url_to_analyze.startswith('http'):
        st.error("Please enter a valid URL (e.g., 'https://...')")
    else:
        # --- 5. Run Analysis ---
        with st.spinner(f"Analyzing {url_to_analyze}... This involves scraping, feature engineering, and 2 model predictions..."):
            # This now calls our main function from scorer.py
            result = analyze_url(url_to_analyze, models, corpus)
        
        # --- 6. Display Results ---
        if "error" in result:
            st.error(result["error"])
        else:
            st.success("Analysis Complete!")
            
            st.subheader(f"Results for: {result['title']}")
            
            # --- Metrics Row ---
            col1, col2, col3 = st.columns(3)
            
            quality_label = result['quality_label']
            if quality_label == "High":
                col1.metric("Quality Score", quality_label, "🚀 Excellent")
            elif quality_label == "Medium":
                col1.metric("Quality Score", quality_label, "👍 Average")
            else:
                # --- THIS IS THE CORRECTED LINE ---
                col1.metric("Quality Score", quality_label, "⚠️ Poor")
            
            word_count = result['word_count']
            col2.metric("Word Count", f"{word_count} words", "Thin Content" if result['is_thin'] else "Sufficient Content")
            
            readability = result['readability']
            col3.metric("Readability (Flesch)", f"{readability}", "Hard to Read" if readability < 50 else "Easy to Read")

            # --- Duplicate Detection ---
            st.subheader("Duplicate Content Check")
            if not result['similar_to']:
                st.info("✅ No duplicates found in our dataset.")
            else:
                st.warning(f"Found {len(result['similar_to'])} similar articles!")
                df_dupes = pd.DataFrame(result['similar_to'])
                st.dataframe(df_dupes, use_container_width=True)
            
            # --- Raw JSON ---
            with st.expander("Show Raw JSON Output"):
                st.json(result)