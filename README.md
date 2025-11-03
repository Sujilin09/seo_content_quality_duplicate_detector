# SEO Content Quality & Duplicate Detector

This project is a complete machine learning pipeline that analyzes web content for SEO quality and detects duplicate articles. It ingests raw HTML, engineers advanced NLP features, trains a robust classifier, and serves the results in a real-time Streamlit application.



## 🚀 Key Features
* **HTML Parsing:** Efficiently parses raw HTML to extract clean, readable body text.
* **Advanced Feature Engineering:** Engineers 7 distinct features, including readability scores (Flesch, Gunning Fog) and interaction terms (`readability * word_count`).
* **Semantic Duplicate Detection:** Uses `Sentence-Transformers` (S-BERT) to find *semantically similar* content, which is far more powerful than basic keyword matching.
* **Robust Quality Scoring:** Employs a hypertuned `Support Vector Classifier (SVC)` model to accurately predict content quality (Low, Medium, High).
* **Real-Time Analysis:** A single function (and Streamlit app) can scrape, parse, and score any live URL in seconds.

---

## 🔧 Setup & Installation

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/yourusername/seo-content-detector](https://github.com/yourusername/seo-content-detector)
    cd seo-content-detector
    ```

2.  **Create & Activate Virtual Environment:**
    ```bash
    # Create
    python -m venv venv
    
    # Activate (Windows)
    .\venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    # Use the venv's python to install
    .\venv\Scripts\python.exe -m pip install -r requirements.txt
    ```
4.  **Download NLTK Data:**
    The pipeline automatically downloads the 'punkt' tokenizer to a local `nltk_data` folder on first run.

---

## 🏃 Quick Start

### 1. Jupyter Notebook
The main analysis and model training pipeline is in the notebook.

1.  **Launch Jupyter:**
    ```bash
    jupyter notebook
    ```
2.  **Open and Run:**
    Navigate to `notebooks/` and open `seo_pipeline.ipynb`. Click `Kernel > Restart & Run All`.

### 2. Streamlit Web App
To run the interactive web demo locally:

1.  **Ensure you are in your `venv`**.
2.  **Run Streamlit from the root folder:**
    ```bash
    .\venv\Scripts\python.exe -m streamlit run streamlit_app/app.py
    ```

---

## 🌟 Deployed Streamlit App (Bonus)

**(Deployed Streamlit URL: [LINK-TO-YOUR-DEPLOYED-APP-HERE])**

---

## 🧠 Key Decisions & Project Rationale

* **Embeddings: `S-BERT` over `TF-IDF`**. I chose `Sentence-Transformers` because they understand *semantic meaning*, not just keywords. This allows the tool to find articles that "mean" the same thing, which is a far superior method for duplicate detection.
* **Preprocessing: Minimalist by Design**. I intentionally avoided aggressive text cleaning (like stopword removal) because our key features (Readability scores and S-BERT embeddings) *require* natural, grammatically correct language to function accurately.
* **Model Selection: `SVC` over `RandomForest/LogisticRegression`**. This was the most critical decision.
    * `Logistic Regression` **underfit** and failed to identify the 'Medium' class (0.44 F1-Score).
    * `Random Forest` **overfit** and "memorized" the simple, rule-based labels, giving an untrustworthy 100% accuracy.
    * `SVC (Support Vector Classifier)` provided the perfect "Goldilocks" balance. It's powerful enough to find the complex non-linear patterns of the 'Medium' class but was controlled via tuning to prevent overfitting, resulting in a robust, realistic model.
* **Feature Engineering: The Key to Performance**. The model's success in finding the 'Medium' class (0.71 F1-score) was a direct result of engineering new features like `gunning_fog` and `readability_length_interaction`, which provided signals beyond simple word count.
* **Validation: `GridSearchCV` + `5-Fold CV`**. A single 70/30 split on a small dataset is unreliable. I used 5-Fold Cross-Validation with a GridSearch to find the best, most trustworthy model parameters.

---

## 📊 Results Summary

The final tuned `SVC` model demonstrated strong performance, significantly beating the baseline and proving its ability to handle the nuanced, imbalanced classes.

### Duplicate Detection
* **Duplicate Pairs Found:** 10
* **Example:**
    | url1 | url2 | similarity |
    | :--- | :--- | :--- |
    | `https://ma...` | `https://en...` | 0.85915 |
    | `https://en...` | `https://sin...` | 0.846509 |
    | `https://en...` | `https://sin...` | 0.841784 |


### Model Performance (Tuned SVC on 70/30 Split):

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Low Quality** | 0.91 | 0.83 | 0.87 | 12 |
| **Medium Quality**| 0.71 | 0.71 | 0.71 | 7 |
| **High Quality** | 0.67 | 1.00 | 0.80 | 2 |

---
**Overall Accuracy:** 0.81
<br>
**Baseline Accuracy:** 0.71


### Top 3 Most Important Features:
1.  `readability_length_interaction` (importance: 0.264)
2.  `word_count` (importance: 0.228)
3.  `flesch_reading_ease` (importance: 0.194)

---

## ⚠️ Limitations

* **Tiny Dataset:** With only ~69 rows, the model is "data starved." Its performance is promising but would be greatly improved with more training data.
* **Synthetic Labels:** The quality labels were rule-based. A production model would need to be trained on labels from human SEO experts.
* **Real-Time Similarity:** The `similar_to` function **only** compares a new URL against the original ~60 articles in the `features.csv` dataset, not the entire internet.
