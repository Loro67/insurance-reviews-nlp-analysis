import streamlit as st
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import config as cfg
from utils.ui_helpers import page_header, metric_card, empty_state, info_box, PRIMARY
from utils.model_loader import (
    load_dataset,
    load_insurer_summaries,
    load_summarizer,
    load_translation_pipeline,
)


def run_abstractive_summary(text: str, summarizer) -> str:
    """Generate a short abstractive summary from free-form text."""
    word_count = len(text.split())
    if word_count < 20:
        return text

    max_l = min(150, max(30, word_count // 2))
    result = summarizer(
        text.encode("utf-8", errors="ignore").decode("utf-8"),
        min_length=15,
        max_length=max_l,
        truncation=True,
        do_sample=False,
    )
    return result[0]["summary_text"].strip()


def run_translation(text: str, translator) -> str:
    """Translate French text to English using the Step 2 model."""
    result = translator(text[:1000], max_length=512, truncation=True)
    return result[0]["translation_text"]


def _filtered_reviews(df: pd.DataFrame, insurer: str, min_rating: int, theme: str, keyword: str):
    reviews = df.copy()
    if insurer != "All insurers":
        reviews = reviews[reviews[cfg.COL_INSURER] == insurer]
    if min_rating > 1:
        reviews = reviews[reviews[cfg.COL_RATING] >= min_rating]
    if theme != "All themes" and cfg.COL_THEME in reviews.columns:
        reviews = reviews[reviews[cfg.COL_THEME] == theme]
    if keyword.strip():
        mask = reviews[cfg.COL_TEXT_EN].str.contains(keyword, case=False, na=False)
        mask = mask | reviews[cfg.COL_TEXT_FR].str.contains(keyword, case=False, na=False)
        reviews = reviews[mask]
    return reviews


def _theme_metrics(reviews: pd.DataFrame) -> pd.DataFrame:
    if reviews.empty or cfg.COL_THEME not in reviews.columns:
        return pd.DataFrame(columns=["theme", "avg_rating", "n_reviews"])
    return (
        reviews.groupby(cfg.COL_THEME)
        .agg(avg_rating=(cfg.COL_RATING, "mean"), n_reviews=(cfg.COL_RATING, "size"))
        .reset_index()
        .rename(columns={cfg.COL_THEME: "theme"})
        .sort_values(["avg_rating", "n_reviews"], ascending=[False, False])
        .round({"avg_rating": 2})
    )


def render():
    page_header("", "Insurer Analysis")

    try:
        df = load_dataset()
        insurer_summaries = load_insurer_summaries()
    except Exception as e:
        st.error(f"Error loading files: {e}")
        return

    if df.empty:
        empty_state("No review data available.")
        return

    insurers = ["All insurers"] + sorted(df[cfg.COL_INSURER].dropna().unique().tolist())
    themes = ["All themes"]
    if cfg.COL_THEME in df.columns:
        themes += sorted(df[cfg.COL_THEME].dropna().unique().tolist())

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        insurer = st.selectbox("Insurer", insurers)
    with col2:
        min_rating = st.slider("Minimum star rating", 1, 5, 1)
    with col3:
        theme_filter = st.selectbox("Theme filter", themes)

    keyword = st.text_input(
        "Review search",
        placeholder="Search reviews by keyword, claim, pricing, cancellation, reimbursement ...",
    )

    filtered = _filtered_reviews(df, insurer, min_rating, theme_filter, keyword)
    if filtered.empty:
        empty_state("No reviews match the current filters.")
        return

    avg_rating = filtered[cfg.COL_RATING].mean()
    neg_share = (filtered[cfg.COL_RATING] <= 2).mean() * 100
    top_theme = (
        filtered[cfg.COL_THEME].mode().iloc[0]
        if cfg.COL_THEME in filtered.columns and not filtered[cfg.COL_THEME].mode().empty
        else "Unavailable"
    )

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        metric_card("Reviews", f"{len(filtered):,}", color=PRIMARY)
    with m2:
        metric_card("Average Rating", f"{avg_rating:.2f} / 5", color=PRIMARY)
    with m3:
        metric_card("Negative Share", f"{neg_share:.1f}%", color=PRIMARY)
    with m4:
        metric_card("Dominant Theme", str(top_theme), color=PRIMARY)

    st.markdown("### Insurer Summaries")
    if insurer != "All insurers" and not insurer_summaries.empty and "assureur" in insurer_summaries.columns:
        row = insurer_summaries[insurer_summaries["assureur"] == insurer]
        if not row.empty:
            st.info(f"**Overall summary:** {row.iloc[0]['summary_overall']}")
            st.warning(f"**Complaint summary:** {row.iloc[0]['summary_complaints']}")
        else:
            info_box(
                "No pre-generated insurer summary is available for this insurer. "
                "The review metrics below still reflect the filtered corpus.",
                kind="info",
            )
    else:
        overview = (
            df.groupby(cfg.COL_INSURER)
            .agg(avg_rating=(cfg.COL_RATING, "mean"), n_reviews=(cfg.COL_RATING, "size"))
            .sort_values(["avg_rating", "n_reviews"], ascending=[False, False])
            .head(10)
            .round({"avg_rating": 2})
        )
        st.dataframe(overview, use_container_width=True)

    st.markdown("### Ratings Overview")
    rating_counts = (
        filtered[cfg.COL_RATING]
        .value_counts()
        .sort_index()
        .rename_axis("rating")
        .reset_index(name="count")
    )
    st.bar_chart(rating_counts.set_index("rating"))

    theme_metrics = _theme_metrics(filtered)
    if not theme_metrics.empty:
        tcol1, tcol2 = st.columns([1, 1])
        with tcol1:
            st.markdown("### Average Rating by Theme")
            st.bar_chart(theme_metrics.set_index("theme")["avg_rating"])
        with tcol2:
            st.markdown("### Theme Breakdown")
            st.dataframe(theme_metrics, use_container_width=True, hide_index=True)

    st.markdown("### Matching Reviews")
    preview_cols = [cfg.COL_INSURER, cfg.COL_PRODUCT, cfg.COL_RATING, cfg.COL_TEXT_EN]
    if cfg.COL_THEME in filtered.columns:
        preview_cols.append(cfg.COL_THEME)
    if cfg.COL_SUMMARY in filtered.columns:
        preview_cols.append(cfg.COL_SUMMARY)

    preview = filtered[preview_cols].copy()
    preview = preview.rename(
        columns={
            cfg.COL_INSURER: "Insurer",
            cfg.COL_PRODUCT: "Product",
            cfg.COL_RATING: "Rating",
            cfg.COL_THEME: "Theme",
            cfg.COL_TEXT_EN: "Review",
            cfg.COL_SUMMARY: "Summary",
        }
    )
    st.dataframe(preview.head(25), use_container_width=True, hide_index=True)

    for i, row in filtered.head(10).reset_index(drop=True).iterrows():
        stars = "⭐" * int(row[cfg.COL_RATING])
        theme = row.get(cfg.COL_THEME, "Unavailable")
        title = f"Review {i+1} | {row[cfg.COL_INSURER]} | {stars} | {theme}"
        with st.expander(title):
            st.markdown(row.get(cfg.COL_TEXT_EN, ""))
            if str(row.get(cfg.COL_TEXT_FR, "")).strip():
                st.markdown("---")
                st.markdown("**Original French**")
                st.markdown(row[cfg.COL_TEXT_FR])
            summary = row.get(cfg.COL_SUMMARY, "")
            if str(summary).strip():
                st.markdown("---")
                st.markdown("**Extractive summary**")
                st.info(summary)

    with st.expander("Optional custom summarize / translate tools"):
        text_input = st.text_area(
            "French or English text",
            height=180,
            placeholder="Paste a review or insurer description here ...",
        )
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Summarize text", use_container_width=True):
                with st.spinner("Generating summary ..."):
                    try:
                        model = load_summarizer()
                        if model is None:
                            info_box("Summarizer model could not be loaded.", kind="error")
                        else:
                            st.success(run_abstractive_summary(text_input, model))
                    except Exception as e:
                        st.error(f"Summarization error: {e}")
        with c2:
            if st.button("Translate FR -> EN", use_container_width=True):
                with st.spinner("Translating ..."):
                    try:
                        translator = load_translation_pipeline()
                        if translator is None:
                            info_box("Translation model could not be loaded.", kind="error")
                        else:
                            st.info(run_translation(text_input, translator))
                    except Exception as e:
                        st.error(f"Translation error: {e}")
