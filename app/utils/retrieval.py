from __future__ import annotations

import re
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def _get_search_texts(df: pd.DataFrame, text_col: str) -> pd.Series:
    """Prefer pre-tokenized text so retrieval matches the fitted vectorizer vocabulary."""
    def normalize_tokens(value) -> str:
        if isinstance(value, (list, tuple, np.ndarray)):
            return " ".join(str(token) for token in value if str(token).strip())
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return ""
        return str(value)

    if text_col == "avis_cor_en" and "tokens_en" in df.columns:
        return df["tokens_en"].apply(normalize_tokens)
    if text_col == "avis_cor" and "tokens_fr" in df.columns:
        return df["tokens_fr"].apply(normalize_tokens)
    return df[text_col].fillna("").astype(str)


def tfidf_search(
    query: str,
    df: pd.DataFrame,
    tfidf_vectorizer,
    text_col: str,
    top_k: int = 10,
) -> pd.DataFrame:
    """
    Retrieve top-k documents by TF-IDF cosine similarity.

    Args:
        query: raw user query string
        df: the full review dataset
        tfidf_vectorizer: fitted sklearn TfidfVectorizer
        text_col: column name containing the text to search over
        top_k: number of results

    Returns:
        Subset of df with added 'score' column, sorted by score descending.
    """
    from utils.preprocessing import prepare_for_tfidf

    if tfidf_vectorizer is None:
        return keyword_search(query, df, text_col, top_k)

    clean_query = prepare_for_tfidf(query)
    if not clean_query.strip():
        return pd.DataFrame()

    texts = _get_search_texts(df, text_col)

    try:
        corpus_matrix = tfidf_vectorizer.transform(texts)
        query_vec     = tfidf_vectorizer.transform([clean_query])
        scores        = cosine_similarity(query_vec, corpus_matrix).flatten()
        top_idx       = scores.argsort()[-top_k:][::-1]
        result        = df.iloc[top_idx].copy()
        result["score"] = scores[top_idx]
        return result[result["score"] > 0].reset_index(drop=True)
    except Exception:
        return keyword_search(query, df, text_col, top_k)


def keyword_search(
    query: str,
    df: pd.DataFrame,
    text_col: str,
    top_k: int = 10,
) -> pd.DataFrame:
    """Simple case-insensitive keyword search as a fallback."""
    terms   = query.lower().split()
    if not terms:
        return pd.DataFrame()

    texts  = df[text_col].fillna("").str.lower()
    scores = texts.apply(lambda t: sum(t.count(term) for term in terms))
    mask   = scores > 0
    result = df[mask].copy()
    result["score"] = scores[mask]
    return result.nlargest(top_k, "score").reset_index(drop=True)

def bm25_search(
    query: str,
    df: pd.DataFrame,
    text_col: str,
    top_k: int = 10,
    k1: float = 1.5,
    b: float = 0.75,
) -> pd.DataFrame:
    """
    BM25 retrieval — pure-Python implementation, no external library needed.

    BM25 is an improvement over TF-IDF that accounts for document length.
    """
    from utils.preprocessing import simple_tokenize

    query_tokens = simple_tokenize(query)
    if not query_tokens:
        return pd.DataFrame()

    corpus = _get_search_texts(df, text_col).tolist()
    tokenized = [simple_tokenize(doc) for doc in corpus]

    # Document frequencies
    N = len(tokenized)
    avgdl = sum(len(d) for d in tokenized) / max(N, 1)

    df_counts: dict = {}
    for tokens in tokenized:
        for t in set(tokens):
            df_counts[t] = df_counts.get(t, 0) + 1

    scores = np.zeros(N)
    for term in query_tokens:
        df_t = df_counts.get(term, 0)
        if df_t == 0:
            continue
        idf = np.log((N - df_t + 0.5) / (df_t + 0.5) + 1)
        for i, tokens in enumerate(tokenized):
            tf = tokens.count(term)
            dl = len(tokens)
            scores[i] += idf * (tf * (k1 + 1)) / (
                tf + k1 * (1 - b + b * dl / max(avgdl, 1))
            )

    top_idx = scores.argsort()[-top_k:][::-1]
    result  = df.iloc[top_idx].copy()
    result["score"] = scores[top_idx]
    return result[result["score"] > 0].reset_index(drop=True)


def embedding_search(
    query: str,
    df: pd.DataFrame,
    corpus_vectors: np.ndarray,
    w2v_model,
    idf_map: dict,
    top_k: int = 10,
) -> pd.DataFrame:
    """
    Retrieve top-k documents by cosine similarity of Word2Vec sentence vectors.

    Args:
        query: user query (English)
        df: review dataset
        corpus_vectors: (n_docs, dim) array of pre-computed sentence vectors
        w2v_model: loaded gensim Word2Vec model
        idf_map: {word: idf_score} mapping from the fitted TF-IDF vectorizer
        top_k: number of results to return

    Returns:
        Subset of df with added 'score' column.
    """
    from utils.preprocessing import simple_tokenize

    if corpus_vectors is None or w2v_model is None:
        return keyword_search(query, df, "avis_cor_en", top_k=top_k)

    tokens = simple_tokenize(query)
    wv     = w2v_model.wv
    dim    = wv.vector_size

    vecs, weights = [], []
    for w in tokens:
        if w in wv:
            vecs.append(wv[w])
            weights.append(idf_map.get(w, 1.0))

    if not vecs:
        return keyword_search(query, df, "avis_cor_en", top_k)

    q_vec  = np.average(vecs, axis=0, weights=weights).reshape(1, -1).astype(np.float32)
    scores = cosine_similarity(q_vec, corpus_vectors).flatten()
    top_idx = scores.argsort()[-top_k:][::-1]
    result  = df.iloc[top_idx].copy()
    result["score"] = scores[top_idx]
    return result[result["score"] > 0].reset_index(drop=True)


def build_rag_context(docs: pd.DataFrame, text_col: str, max_words: int = 800) -> str:
    """
    Concatenate retrieved documents into a single context string for the LLM.
    Caps the total at max_words to stay within the LLM's context window.
    """
    chunks = []
    total  = 0
    for _, row in docs.iterrows():
        text  = str(row.get(text_col, "")).strip()
        words = text.split()
        if total + len(words) > max_words:
            words = words[: max_words - total]
            chunks.append(" ".join(words))
            break
        chunks.append(text)
        total += len(words)

    return "\n\n".join(f"[Doc {i+1}]: {c}" for i, c in enumerate(chunks))
