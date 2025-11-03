# SEO Content Quality & Duplicate Detector

This project is a complete machine learning pipeline built for a data science case study. It ingests a dataset of webpage content, engineers advanced NLP features, and trains a robust classification model to score content quality (Low, Medium, High). It also includes a real-time analysis function to score and check for duplicates on any live URL.



## 🚀 Key Features
* **HTML Parsing:** Efficiently parses raw HTML to extract clean, readable body text and titles.
* **Advanced Feature Engineering:** Goes beyond basic word count to engineer 7 distinct features, including:
    * Readability Scores (Flesch Ease, Gunning Fog)
    * Lexical Complexity (Avg. Sentence Length)
    * Interaction Terms (Readability * Word Count)
* **Semantic Duplicate Detection:** Uses `Sentence-Transformers` (S-BERT) to find *semantically similar* content, which is far more powerful than basic keyword matching.
* **Robust Quality Scoring:** Employs a `Support Vector Classifier (SVC)` model, hypertuned with `GridSearchCV`, to accurately predict content quality.
* **Real-Time Analysis:** A single function `analyze_url(url)` can scrape, parse, and score any live URL in seconds.

---

## 🔧 Setup & Installation

To run this project locally, follow these steps:

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/yourusername/seo-content-detector](https://github.com/yourusername/seo-content-detector)
    cd seo-content-detector
    ```

2.  **Create a Virtual Environment:**
    ```bash
    python -m venv venv
    
    # On Windows
    .\venv\Scripts\activate
    
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    All required packages are listed in `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download NLTK Data:**
    The pipeline requires the 'punkt' tokenizer. The notebook will automatically download this to a local `nltk_data` folder on first run.

---

## 🏃 Quick Start

The entire pipeline and all analysis are contained within a single Jupyter Notebook.

1.  **Launch Jupyter:**
    ```bash
    jupyter notebook
    ```
2.  **Open the Notebook:**
    Navigate to `notebooks/` and open `seo_pipeline.ipynb`.

3.  **Run All Cells:**
    Click `Kernel > Restart & Run All` to execute the full pipeline from data parsing to model training and the final real-time demo.

---

## 🌟 Deployed Streamlit App (Bonus)

**(Deployed Streamlit URL: [LINK-TO-YOUR-DEPLOYED-APP-HERE])**

(Once we build and deploy the Streamlit app, you will paste the public URL here.)

---

## 🧠 Key Decisions & Project Rationale

This project involved several key decisions to move beyond the baseline requirements and build a more robust, professional-grade pipeline.

### 1. Parsing: `lxml` over `html.parser`
While `BeautifulSoup` can use Python's built-in `html.parser`, I explicitly chose the `lxml` parser. It is written in C and is significantly faster and more resilient to malformed HTML, which is critical for a web-scale data pipeline.

### 2. Embeddings: `S-BERT` over `TF-IDF`
The assignment allowed for `TF-IDF` vectors for similarity. I chose to use `sentence-transformers` (S-BERT) instead.
* **Why?** TF-IDF is a "bag-of-words" model; it knows *if* a word exists but not *what it means*. It would fail to see "How to improve SEO" and "Tips for getting better at SEO" as similar.
* **S-BERT** is a deep learning model that understands *semantic meaning* and *context*. This provides a vastly superior and more human-like duplicate detection capability.

### 3. Preprocessing: Minimalist by Design
I intentionally avoided aggressive preprocessing (like stopword removal or stemming) on the main text.
* **Why?** Our key features, **Readability Scores** (`textstat`) and **Semantic Embeddings** (`S-BERT`), depend on natural, grammatically correct language. Removing stopwords or stemming would corrupt these features and lead to a far less accurate model.

### 4. Model Selection: The "Goldilocks" Problem
This was the most critical part of the project. The synthetic, rule-based labels created a "trap."

* **Baseline (Random Forest):** A test with `RandomForestClassifier` (a suggested model) achieved **1.00 (100%) accuracy**. On a tiny, rule-based dataset, this is a clear sign of **overfitting**. The model was simply "memorizing" the exact rules I wrote, not "learning" a general pattern.
* **Baseline (Logistic Regression):** This model **underfit** the data, achieving only a **0.44 F1-Score** for the 'Medium' class. It was too simple to capture the "catch-all" nature of this class.

* **✅ Solution (SVC):** I chose a **Support Vector Classifier (SVC) with an RBF kernel**. This is the "Goldilocks" model for this problem. It is powerful enough to find complex, non-linear boundaries (unlike Logistic Regression) but can be controlled with `C` and `gamma` parameters (unlike Random Forest) to prevent overfitting.

### 5. Feature Engineering: The Key to Unlocking SVC
The SVC model alone wasn't enough. It needed better features to solve the 'Medium' class problem. I engineered three new, highly-informative features:
* `gunning_fog`: Measures word complexity (grade level).
* `avg_sentence_length`: A proxy for writing style.
* `readability_length_interaction`: A new feature (`flesch_score * word_count`) that captures the relationship between length and quality.

### 6. Validation: `GridSearchCV` + `5-Fold CV`
A single 70/30 split on 69 rows is unreliable. To find the best, most trustworthy model, I used `GridSearchCV` to test all parameter combinations, validating each one with **5-Fold Cross-Validation**. This gives a robust average score and ensures the final model isn't just a "lucky split."

---

## 📊 Results Summary

The final tuned `SVC` model (our "Goldilocks" model) performed exceptionally well, demonstrating a clear ability to distinguish between the nuanced classes and crushing the baseline.

### Model Performance (Tuned SVC on 70/30 Split):

