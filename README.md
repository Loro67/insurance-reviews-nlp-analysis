# NLP Project 2: Insurance Reviews Analysis

**ESILV A4 DIA6 — 2026**  
**Authors:** Leo WINTER & Alvaro SERERO

## Overview
This project analyzes French insurance reviews with a complete NLP pipeline:

- Step 1: data cleaning, spelling correction, EDA, and n-grams
- Step 2: summarization, translation, QA, and generation
- Step 3: topic modeling and business themes
- Step 4: embeddings, TensorBoard projector, semantic search, and theme enrichment
- Step 5: supervised learning for rating, sentiment, and theme prediction
- Step 6: Streamlit applications for prediction, insurer analysis, explanation, retrieval, RAG, and QA

The final Streamlit app combines local classical models, local extractive QA, embedding-based retrieval, and an optional RAG page.

## Repository Structure
```text
NLP_2/
├── app/
│   ├── app.py
│   ├── config.py
│   ├── pages/
│   └── utils/
├── data/
│   ├── reviews_clean.parquet
│   ├── reviews_step2.parquet
│   ├── reviews_step3.parquet
│   ├── reviews_step4.parquet
│   ├── reviews_step5.parquet
│   └── insurer_summaries.csv
├── model/
│   ├── word2vec_en.model
│   ├── word2vec_fr.model
│   ├── lr_tfidf_en.pkl
│   ├── lr_rating_en.pkl
│   ├── tfidf_theme_en.pkl
│   └── lr_theme_en.pkl
├── notebooks/
│   ├── 1_data_cleaning_eda.ipynb
│   ├── 2_summary_translation_qa_generation.ipynb
│   ├── 3_topic_modeling.ipynb
│   ├── 4_embeddings.ipynb
│   └── 5_supervised_learning.ipynb
└── visualizations/
```

## Main Outputs
- `data/reviews_step5.parquet`: final enriched dataset used by the app
- `model/lr_tfidf_en.pkl` + `model/lr_rating_en.pkl`: local rating classifier
- `model/tfidf_theme_en.pkl` + `model/lr_theme_en.pkl`: local theme classifier
- `data/tfidf_en.pkl` + `data/tfidf_fr.pkl`: retrieval vectorizers from Step 3
- `logs/step5/`: TensorBoard projector exports for embedding models

## Installation
Install notebook dependencies:

```bash
pip install -r requirements.txt
```

Install Streamlit app dependencies:

```bash
pip install -r app/requirements.txt
```

## Run Order
Run the notebooks in this order:

1. `notebooks/1_data_cleaning_eda.ipynb`
2. `notebooks/2_summary_translation_qa_generation.ipynb`
3. `notebooks/3_topic_modeling.ipynb`
4. `notebooks/4_embeddings.ipynb`
5. `notebooks/5_supervised_learning.ipynb`

This sequence produces the parquet files, model artifacts, TensorBoard logs, and comparison plots used in the app.

## Launch the App
From the repository root:

```bash
streamlit run app/app.py
```

The app includes these pages:

- `Prediction`: local star rating and theme prediction, with optional FR -> EN translation for French input
- `Insurer Analysis`: insurer summaries, metrics, review search, and theme analysis
- `Explanation`: SHAP / LR-based word-level explanation for rating predictions
- `Information Retrieval`: TF-IDF, BM25, and Word2Vec search
- `RAG`: optional external LLM page over retrieved reviews
- `Question Answering`: local extractive QA with `deepset/roberta-base-squad2`

## Notes for Submission
- The `RAG` page is the only page that may require an external API.
- You can configure the `RAG` page with `OPENAI_API_KEY`, `OPENAI_API_BASE`, and `RAG_LLM_MODEL`.
- The `Question Answering` page is fully local and does not require API keys.
