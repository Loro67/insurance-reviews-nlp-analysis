import os
from pathlib import Path

# Root directories
BASE_DIR   = Path(Path(__file__).parent.parent)
DATA_DIR   = BASE_DIR / "data"
MODEL_DIR  = BASE_DIR / "model"
VISU_DIR   = BASE_DIR / "visualizations"

# Data files 
CLEANED_DATA        = DATA_DIR / "reviews_step5.parquet"   # final enriched dataset
STEP4_DATA          = DATA_DIR / "reviews_step4.parquet"
STEP2_DATA          = DATA_DIR / "reviews_step2.parquet"
INSURER_SUMMARIES   = DATA_DIR / "insurer_summaries.csv"   # per-insurer summaries (Step 2)

TFIDF_EN_PKL        = DATA_DIR / "tfidf_en.pkl"
TFIDF_FR_PKL        = DATA_DIR / "tfidf_fr.pkl"

# http://localhost:8502
# Saved models 
LR_CLASSIFIER_EN    = MODEL_DIR / "lr_rating_en.pkl"   # rating (5-class)
LR_CLASSIFIER_FR    = MODEL_DIR / "lr_rating_fr.pkl"
LR_THEME_CLASSIFIER_EN = MODEL_DIR / "lr_theme_en.pkl"

# TF-IDF fitted for classifier (separate from Step-3 vectorizers)
LR_TFIDF_EN         = MODEL_DIR / "lr_tfidf_en.pkl"
LR_TFIDF_FR         = MODEL_DIR / "lr_tfidf_fr.pkl"
LR_THEME_TFIDF_EN   = MODEL_DIR / "tfidf_theme_en.pkl"

# Word2Vec models (gensim .model files)
W2V_EN_MODEL        = MODEL_DIR / "word2vec_en.model"
W2V_FR_MODEL        = MODEL_DIR / "word2vec_fr.model"

# GloVe cache (pickled gensim KeyedVectors)
GLOVE_CACHE         = MODEL_DIR / "glove-wiki-gigaword-100.vecs"

# HuggingFace model IDs
SUMMARIZER_MODEL    = "sshleifer/distilbart-cnn-12-6"
QA_MODEL            = "deepset/roberta-base-squad2"
BERT_SENTIMENT      = "nlptown/bert-base-multilingual-uncased-sentiment"
TRANSLATION_MODEL   = "Helsinki-NLP/opus-mt-fr-en"


# Column names
COL_ID              = "id"
COL_TEXT_EN         = "avis_cor_en"    # cleaned English review
COL_TEXT_FR         = "avis_cor"       # cleaned French review
COL_TEXT_RAW        = "avis"           # original French
COL_SUMMARY         = "avis_summary"   # extractive summary (Step 2)
COL_RATING          = "note"           # ground-truth star rating (1-5)
COL_INSURER         = "assureur"
COL_PRODUCT         = "produit"
COL_THEME           = "theme_enriched" # enriched theme (Step 4)
COL_TOPIC_LDA       = "topic_lda_en"
COL_PRED_RATING     = "pred_lr_rating"
COL_PRED_SENTIMENT  = "pred_lr_sentiment"
COL_PRED_THEME      = "pred_theme_lr"
COL_DATE            = "date_publication"

# UI / app settings
APP_TITLE           = "Insurance Reviews — NLP Dashboard"
APP_ICON            = "D"
TOP_K_RETRIEVAL     = 10          # default number of search results
MIN_REVIEW_CHARS    = 20          # minimum input length
MAX_SUMMARY_WORDS   = 130         # abstractive summary cap

# Optional RAG configuration
OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_BASE     = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
RAG_LLM_MODEL       = os.getenv("RAG_LLM_MODEL", "gpt-4o-mini")
RAG_MAX_TOKENS      = int(os.getenv("RAG_MAX_TOKENS", "350"))
RAG_TOP_K           = int(os.getenv("RAG_TOP_K", "5"))

# Star labels (0-indexed → display)
STAR_LABELS = {0: "⭐ 1 star", 1: "⭐⭐ 2 stars", 2: "⭐⭐⭐ 3 stars",
               3: "⭐⭐⭐⭐ 4 stars", 4: "⭐⭐⭐⭐⭐ 5 stars"}
SENTIMENT_LABELS = {0: "Negative", 1: "Neutral", 2: "Positive"}

THEME_COLORS = {
    "Pricing & Value":     "#E53935",
    "Claims Handling":     "#FB8C00",
    "Customer Service":    "#8E24AA",
    "Contract Management": "#1E88E5",
    "Health Coverage":     "#43A047",
    "Positive Sentiment":  "#00ACC1",
    "Other":               "#757575",
}
