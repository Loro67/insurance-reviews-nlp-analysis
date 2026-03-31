import streamlit as st
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import config as cfg
from utils.ui_helpers import (
    page_header,
    metric_card,
    probability_chart,
    info_box,
    star_rating_badge,
    PRIMARY,
    ACCENT_GREEN,
    ACCENT_RED,
)
from utils.model_loader import (
    load_classifier,
    load_theme_classifier,
    load_bert_sentiment_pipeline,
    load_translation_pipeline,
)
from utils.preprocessing import prepare_for_tfidf


def predict_with_lr(text: str, tfidf, clf):
    """Run TF-IDF + LR inference on raw text. Returns (label_0idx, probs array)."""
    cleaned = prepare_for_tfidf(text)
    vec = tfidf.transform([cleaned])
    probs = clf.predict_proba(vec)[0]
    pred = int(np.argmax(probs))
    return pred, probs


def predict_with_bert(text: str, pipe):
    """Run BERT multilingual sentiment inference. Returns (label_0idx, probs array)."""
    truncated = " ".join(text.split()[:400])
    result = pipe(truncated, truncation=True, max_length=512)[0]
    label_idx = int(result["label"].split()[0]) - 1
    score = result["score"]

    probs = np.full(5, (1 - score) / 4)
    probs[label_idx] = score
    return label_idx, probs


def predict_theme(text: str, tfidf, clf):
    cleaned = prepare_for_tfidf(text)
    vec = tfidf.transform([cleaned])
    probs = clf.predict_proba(vec)[0]
    pred_idx = int(np.argmax(probs))
    labels = list(clf.classes_)
    return labels[pred_idx], labels, probs


def translate_french_input(text: str, translator) -> str:
    translated = translator(text[:1000], max_length=512, truncation=True)
    return translated[0]["translation_text"].strip()


def sentiment_from_star(star: int):
    if star <= 2:
        return "Negative", ACCENT_RED
    if star == 3:
        return "Neutral", "#FB8C00"
    return "Positive", ACCENT_GREEN


def render():
    page_header("", "Prediction")

    with st.spinner("Loading local models ..."):
        tfidf_rating, clf_rating = load_classifier()
        tfidf_theme, clf_theme = load_theme_classifier()

    model_available = tfidf_rating is not None and clf_rating is not None
    theme_available = tfidf_theme is not None and clf_theme is not None
    bert_pipe = None

    if not model_available:
        info_box(
            "Local rating classifier not found. Falling back to multilingual BERT for "
            "rating only until notebook 5 exports `lr_tfidf_en.pkl` and `lr_rating_en.pkl`.",
            kind="warning",
        )
        with st.spinner("Loading BERT fallback model ..."):
            bert_pipe = load_bert_sentiment_pipeline()

    if not theme_available:
        info_box(
            "Local theme classifier not found. Run notebook 5 to export "
            "`tfidf_theme_en.pkl` and `lr_theme_en.pkl` for subject detection.",
            kind="warning",
        )

    if not model_available and bert_pipe is None:
        info_box(
            "No rating classifier is available. Please export the Step 5 models first.",
            kind="error",
        )
        return

    st.markdown("#### Enter a review")
    example_texts = {
        "— choose an example —": "",
        "Claims issue": (
            "My claim was refused after weeks of waiting. Customer service kept asking for the "
            "same documents and nobody could explain the decision."
        ),
        "Pricing complaint": (
            "The premium increased again this year and the coverage is worse than before. "
            "It is becoming too expensive for the value provided."
        ),
        "Positive service": (
            "Very satisfied with this insurer. Fast reimbursement, clear communication and a "
            "simple subscription process. I would recommend them."
        ),
    }

    col_ex, col_lang = st.columns([3, 1])
    with col_ex:
        selected = st.selectbox("Load an example", list(example_texts.keys()))
    with col_lang:
        language_hint = st.radio("Input language", ["French", "English"], index=1, horizontal=True)

    user_text = st.text_area(
        "Review text",
        value=example_texts[selected],
        height=160,
        placeholder="Paste or type your review here ...",
        label_visibility="collapsed",
    )

    st.markdown(
        f"<small style='color:#888;'>{len(user_text.split())} words</small>",
        unsafe_allow_html=True,
    )

    if st.button("Predict Rating and Subject", type="primary", use_container_width=True):
        if len(user_text.strip()) < cfg.MIN_REVIEW_CHARS:
            info_box(
                f"Please enter at least {cfg.MIN_REVIEW_CHARS} characters.",
                kind="warning",
            )
            return

        with st.spinner("Running local inference ..."):
            try:
                inference_text = user_text
                translated_text = None
                if language_hint == "French":
                    translator = load_translation_pipeline()
                    if translator is None:
                        info_box(
                            "The translation model could not be loaded, so French input cannot be "
                            "processed right now.",
                            kind="error",
                        )
                        return
                    translated_text = translate_french_input(user_text, translator)
                    inference_text = translated_text

                if model_available:
                    pred_idx, probs = predict_with_lr(inference_text, tfidf_rating, clf_rating)
                    model_name = "TF-IDF + Logistic Regression"
                else:
                    pred_idx, probs = predict_with_bert(inference_text, bert_pipe)
                    model_name = "BERT multilingual fallback"

                star = pred_idx + 1
                confidence = float(probs[pred_idx])
                sentiment, sent_color = sentiment_from_star(star)

                pred_theme = "Unavailable"
                theme_labels = []
                theme_probs = np.array([])
                if theme_available:
                    pred_theme, theme_labels, theme_probs = predict_theme(
                        inference_text, tfidf_theme, clf_theme
                    )
            except Exception as e:
                info_box(f"Prediction error: {e}", kind="error")
                return

        st.divider()
        st.markdown("### Results")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            metric_card("Predicted Rating", star_rating_badge(star), color=PRIMARY)
        with col2:
            metric_card("Confidence", f"{confidence*100:.1f}%", color=PRIMARY)
        with col3:
            metric_card("Sentiment", sentiment, color=sent_color)
        with col4:
            metric_card("Predicted Subject", pred_theme, color=PRIMARY)

        star_labels = [f"{i} star{'s' if i > 1 else ''}" for i in range(1, 6)]
        probability_chart(star_labels, probs.tolist(), "Rating Probability Distribution")

        if theme_available and len(theme_labels) == len(theme_probs):
            with st.expander("Subject probabilities", expanded=True):
                order = np.argsort(theme_probs)[::-1]
                ordered_labels = [str(theme_labels[i]) for i in order]
                ordered_probs = [float(theme_probs[i]) for i in order]
                probability_chart(
                    ordered_labels,
                    ordered_probs,
                    "Theme Probability Distribution",
                )

        if language_hint == "French" and translated_text:
            with st.expander("English text used for the model"):
                st.markdown(translated_text)

        with st.expander("Model details"):
            st.markdown(f"**Rating model:** {model_name}")
            if theme_available:
                st.markdown("**Subject model:** TF-IDF + Logistic Regression on `theme_enriched`")
            else:
                st.markdown("**Subject model:** unavailable")
            if language_hint == "French":
                st.markdown(
                    "**Input handling:** French reviews are translated to English first, then "
                    "passed to the local rating and subject classifiers."
                )
            st.markdown(
                "**Pipeline:** clean text -> TF-IDF transform -> local classifier(s) -> "
                "real-time rating, sentiment, and subject prediction"
            )
