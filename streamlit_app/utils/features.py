# File: streamlit_app/utils/features.py

import numpy as np
import textstat

def engineer_features(body_text, tokenizer):
    """Calculates all features our SVC model needs."""
    word_count = len(body_text.split())
    
    # Use the passed tokenizer
    sentence_count = len(tokenizer.tokenize(body_text))
    
    flesch_reading_ease, gunning_fog = 0, 0
    if word_count >= 10:
        try:
            flesch_reading_ease = textstat.flesch_reading_ease(body_text)
            gunning_fog = textstat.gunning_fog(body_text)
        except Exception: pass
        
    avg_sentence_length = np.where(sentence_count > 0, word_count / sentence_count, 0)
    keyword_density = 0 # Known limitation
    readability_length_interaction = flesch_reading_ease * word_count
    
    feature_dict = {
        'word_count': word_count,
        'sentence_count': sentence_count,
        'flesch_reading_ease': flesch_reading_ease,
        'avg_sentence_length': float(avg_sentence_length),
        'keyword_density': keyword_density,
        'gunning_fog': gunning_fog,
        'readability_length_interaction': readability_length_interaction
    }
    return feature_dict