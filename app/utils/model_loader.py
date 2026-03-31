from __future__ import annotations
import pickle
import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# Local imports
import sys
sys.path.append(str(Path(__file__).parent.parent))
import config as cfg


class _LocalSeq2SeqGenerator:
    """Lightweight seq2seq wrapper for summarization / translation across transformers versions."""

    def __init__(self, model_name: str, output_key: str):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self._torch = torch
        self.output_key = output_key
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.eval()

    def __call__(self, text, max_length: int = 128, min_length: int = 0,
                 truncation: bool = True, do_sample: bool = False, **kwargs):
        if isinstance(text, list):
            text = text[0] if text else ""
        text = str(text or "").strip()
        if not text:
            return [{self.output_key: ""}]

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=truncation,
            max_length=512,
        )
        with self._torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_length=max_length,
                min_length=max(min_length, 0),
                do_sample=do_sample,
            )
        decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        return [{self.output_key: decoded[0].strip() if decoded else ""}]


class _LocalQuestionAnswerer:
    """Local extractive QA wrapper that does not depend on the deprecated pipeline task registry."""

    def __init__(self, model_name: str):
        import torch
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.model.eval()

    def __call__(self, question: str, context: str):
        question = str(question or "").strip()
        context = str(context or "").strip()
        if not question or not context:
            return {"answer": "", "score": 0.0, "start": 0, "end": 0}

        encoded = self.tokenizer(
            question,
            context,
            return_tensors="pt",
            truncation="only_second",
            max_length=512,
            return_offsets_mapping=True,
        )
        offset_mapping = encoded.pop("offset_mapping")[0].tolist()
        sequence_ids = encoded.sequence_ids(0)

        with self._torch.no_grad():
            outputs = self.model(**encoded)

        start_probs = self._torch.softmax(outputs.start_logits[0], dim=-1)
        end_probs = self._torch.softmax(outputs.end_logits[0], dim=-1)

        candidate_starts = [
            idx for idx in self._torch.argsort(start_probs, descending=True).tolist()
            if idx < len(sequence_ids) and sequence_ids[idx] == 1
        ][:20]
        candidate_ends = [
            idx for idx in self._torch.argsort(end_probs, descending=True).tolist()
            if idx < len(sequence_ids) and sequence_ids[idx] == 1
        ][:20]

        best = None
        best_score = 0.0
        for start_idx in candidate_starts:
            for end_idx in candidate_ends:
                if end_idx < start_idx or end_idx - start_idx > 30:
                    continue
                start_char, _ = offset_mapping[start_idx]
                _, end_char = offset_mapping[end_idx]
                if end_char <= start_char:
                    continue
                score = float(start_probs[start_idx] * end_probs[end_idx])
                if score > best_score:
                    best_score = score
                    best = (start_char, end_char)

        if best is None:
            return {"answer": "", "score": 0.0, "start": 0, "end": 0}

        start_char, end_char = best
        answer = context[start_char:end_char].strip()
        return {
            "answer": answer,
            "score": best_score,
            "start": start_char,
            "end": end_char,
        }


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    """Load the review dataset with strict format handling."""
    path = cfg.CLEANED_DATA
    if not path.exists():
        for fallback in [cfg.STEP4_DATA, cfg.STEP2_DATA]:
            p = fallback
            if p.exists():
                path = p
                break
        else:
            st.warning(f"No dataset found at {path}")
            return pd.DataFrame()

    try:
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)

        for col in [cfg.COL_TEXT_EN, cfg.COL_TEXT_FR, cfg.COL_INSURER]:
            if col in df.columns:
                df[col] = df[col].fillna("").astype(str)

        if "avis_summary" in df.columns:
            df[cfg.COL_SUMMARY] = df["avis_summary"]

        return df

    except Exception as e:
        st.error(f"Critical error loading {path.name}: {e}")
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_insurer_summaries() -> pd.DataFrame:
    """Load per-insurer abstractive summaries with robust delimiter/encoding detection."""
    path = cfg.INSURER_SUMMARIES
    if not path.exists():
        return pd.DataFrame(columns=["assureur", "summary_overall",
                                      "summary_complaints", "avg_rating", "n_reviews"])

    encodings = ['utf-8', 'latin-1']
    separators = [',', ';', '\t']

    for enc in encodings:
        for sep in separators:
            try:
                df = pd.read_csv(path, encoding=enc, sep=sep)

                if "assureur" in df.columns or "summary_overall" in df.columns:
                    return df
            except Exception:
                continue

    try:
        return pd.read_csv(path, sep=None, engine='python', encoding='latin-1')
    except Exception as e:
        st.error(f"Failed to parse {path.name}: {e}")
        return pd.DataFrame()

@st.cache_resource(show_spinner=False)
def load_tfidf_vectorizers() -> Tuple[Optional[object], Optional[object]]:
    """Load fitted TF-IDF vectorizers (EN and FR) from Step 3."""
    tfidf_en = tfidf_fr = None
    for attr, path in [("tfidf_en", cfg.TFIDF_EN_PKL),
                        ("tfidf_fr", cfg.TFIDF_FR_PKL)]:
        if path.exists():
            try:
                with open(path, "rb") as f:
                    obj = pickle.load(f)
                if attr == "tfidf_en":
                    tfidf_en = obj
                else:
                    tfidf_fr = obj
            except Exception:
                pass
    return tfidf_en, tfidf_fr

@st.cache_resource(show_spinner=False)
def load_classifier(lang: str = "en") -> Tuple[Optional[object], Optional[object]]:
    """
    Load the TF-IDF + LR classifier saved from Step 5.

    Returns (tfidf, classifier) or (None, None) if not found.
    The app will fall back to a live HuggingFace BERT pipeline if missing.
    """
    lang = (lang or "en").lower()
    candidates = []
    if lang == "fr":
        candidates = [
            (cfg.LR_TFIDF_FR, cfg.LR_CLASSIFIER_FR),
            (cfg.MODEL_DIR / "tfidf_fr.pkl", cfg.MODEL_DIR / "lr_rating_fr.pkl"),
        ]
    else:
        candidates = [
            (cfg.LR_TFIDF_EN, cfg.LR_CLASSIFIER_EN),
            (cfg.MODEL_DIR / "tfidf_en.pkl", cfg.MODEL_DIR / "lr_rating_en.pkl"),
        ]

    for tfidf_path, clf_path in candidates:
        if tfidf_path.exists() and clf_path.exists():
            try:
                with open(tfidf_path, "rb") as f:
                    tfidf = pickle.load(f)
                with open(clf_path, "rb") as f:
                    clf = pickle.load(f)
                return tfidf, clf
            except Exception:
                pass

    return None, None


@st.cache_resource(show_spinner=False)
def load_theme_classifier(lang: str = "en") -> Tuple[Optional[object], Optional[object]]:
    """Load the TF-IDF + LR theme classifier exported from Step 5."""
    if (lang or "en").lower() != "en":
        return None, None
    if cfg.LR_THEME_TFIDF_EN.exists() and cfg.LR_THEME_CLASSIFIER_EN.exists():
        try:
            with open(cfg.LR_THEME_TFIDF_EN, "rb") as f:
                tfidf = pickle.load(f)
            with open(cfg.LR_THEME_CLASSIFIER_EN, "rb") as f:
                clf = pickle.load(f)
            return tfidf, clf
        except Exception:
            pass
    return None, None


@st.cache_resource(show_spinner=False)
def load_bert_sentiment_pipeline():
    """Load the BERT multilingual sentiment pipeline (fallback classifier)."""
    try:
        from transformers import pipeline
        pipe = pipeline(
            "text-classification",
            model=cfg.BERT_SENTIMENT,
            framework="pt",
            device=-1,   # CPU
        )
        return pipe
    except Exception as e:
        st.warning(f"Could not load BERT sentiment model: {e}")
        return None


@st.cache_resource(show_spinner=False)
def load_summarizer():
    """Load the DistilBART summarization pipeline."""
    try:
        from transformers import pipeline
        try:
            return pipeline(
                "summarization",
                model=cfg.SUMMARIZER_MODEL,
                framework="pt",
                device=-1,
            )
        except KeyError:
            return _LocalSeq2SeqGenerator(cfg.SUMMARIZER_MODEL, "summary_text")
    except Exception as e:
        st.warning(f"Could not load summarization model: {e}")
        return None

@st.cache_resource(show_spinner=False)
def load_translation_pipeline():
    """Load the French-to-English translation pipeline used by the app."""
    try:
        from transformers import pipeline
        try:
            return pipeline(
                "translation",
                model=cfg.TRANSLATION_MODEL,
                framework="pt",
                device=-1,
            )
        except KeyError:
            return _LocalSeq2SeqGenerator(cfg.TRANSLATION_MODEL, "translation_text")
    except Exception as e:
        st.warning(f"Could not load translation model: {e}")
        return None

@st.cache_resource(show_spinner=False)
def load_qa_pipeline():
    """Load the extractive QA pipeline (RoBERTa-SQuAD2)."""
    try:
        from transformers import pipeline
        try:
            return pipeline(
                "question-answering",
                model=cfg.QA_MODEL,
                framework="pt",
                device=-1,
            )
        except KeyError:
            return _LocalQuestionAnswerer(cfg.QA_MODEL)
    except Exception as e:
        st.warning(f"Could not load QA model: {e}")
        return None


@st.cache_resource(show_spinner=False)
def load_word2vec_models():
    """Load EN and FR Word2Vec models from Step 4."""
    w2v_en = w2v_fr = None
    try:
        from gensim.models import Word2Vec
        if cfg.W2V_EN_MODEL.exists():
            w2v_en = Word2Vec.load(str(cfg.W2V_EN_MODEL))
        if cfg.W2V_FR_MODEL.exists():
            w2v_fr = Word2Vec.load(str(cfg.W2V_FR_MODEL))
    except Exception:
        pass
    return w2v_en, w2v_fr

@st.cache_data(show_spinner=False)
def build_sentence_vectors(df: pd.DataFrame) -> Optional[np.ndarray]:
    """
    Build TF-IDF weighted sentence vectors from Word2Vec embeddings.
    Returns (n_docs, embed_dim) float32 array or None if models unavailable.
    """
    w2v_en, _ = load_word2vec_models()
    tfidf_en, _ = load_tfidf_vectorizers()
    if w2v_en is None or tfidf_en is None:
        return None

    wv      = w2v_en.wv
    idf_map = dict(zip(tfidf_en.get_feature_names_out(), tfidf_en.idf_))
    dim     = wv.vector_size

    def sentence_vec(tokens):
        if not isinstance(tokens, list):
            return np.zeros(dim, dtype=np.float32)
        vecs, weights = [], []
        for w in tokens:
            if w in wv:
                vecs.append(wv[w])
                weights.append(idf_map.get(w, 1.0))
        if not vecs:
            return np.zeros(dim, dtype=np.float32)
        return np.average(vecs, axis=0, weights=weights).astype(np.float32)

    col = cfg.COL_TEXT_EN
    if "tokens_en" in df.columns:
        token_col = df["tokens_en"]
    else:
        # Fallback: tokenize from text
        from utils.preprocessing import simple_tokenize
        token_col = df[col].apply(simple_tokenize)

    vecs = np.vstack([sentence_vec(t) for t in token_col])
    return vecs
