import streamlit as st
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import config as cfg
from utils.ui_helpers import (
    page_header, info_box, metric_card, highlight_text, PRIMARY,
    ACCENT_GREEN, ACCENT_RED, empty_state,
)
from utils.model_loader import load_classifier, load_dataset
from utils.preprocessing import prepare_for_tfidf, clean_text


def coeff_word_importance(text: str, tfidf, clf, pred_class: int, top_n: int = 15):
    """
    Returns list of (word, score) where score > 0 pushes toward pred_class,
    score < 0 pushes away. Uses LR coefficients × TF-IDF weights.
    """
    cleaned   = prepare_for_tfidf(text)
    vec       = tfidf.transform([cleaned])
    feature_names = tfidf.get_feature_names_out()

    # LR coefficient for the predicted class
    if hasattr(clf, "coef_"):
        coef  = clf.coef_[pred_class]   # (n_features,)
        dense = vec.toarray()[0]        # (n_features,)
        contrib = coef * dense          # element-wise

        # Keep only non-zero (words actually in the input)
        non_zero = [(feature_names[i], contrib[i])
                    for i in range(len(contrib)) if dense[i] > 0]
        non_zero.sort(key=lambda x: abs(x[1]), reverse=True)
        return non_zero[:top_n]
    return []


def _extract_shap_for_class(shap_values, class_idx: int) -> np.ndarray:
    """
    Safely extract the (n_samples, n_features) SHAP array for one class,
    regardless of whether shap_values is a list or a 3-D ndarray.
    """
    if isinstance(shap_values, list):
        # list of (n_samples, n_features) arrays — one per class
        return shap_values[class_idx]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        # shape: (n_samples, n_features, n_classes)
        return shap_values[:, :, class_idx]
    else:
        # Binary / single-output: shap_values is already (n_samples, n_features)
        return shap_values


def shap_word_importance(text: str, tfidf, clf, pred_class: int,
                          background_texts: list, top_n: int = 15):
    """
    Compute SHAP values for a single input using a background sample.
    Returns list of (word, shap_value) for the predicted class.

    Handles both the list-of-arrays and 3-D ndarray formats returned by
    different SHAP / sklearn versions, and aligns feature dimensions safely.
    """
    import shap

    feature_names = tfidf.get_feature_names_out()
    bg_matrix     = tfidf.transform(background_texts).toarray()
    test_vec      = tfidf.transform([prepare_for_tfidf(text)]).toarray()

    explainer   = shap.LinearExplainer(
        clf, bg_matrix, feature_perturbation="interventional"
    )
    shap_values = explainer.shap_values(test_vec)

    sv_all    = _extract_shap_for_class(shap_values, pred_class)  # (1, n_features) or (n_features,)
    sv_row    = sv_all[0] if sv_all.ndim == 2 else sv_all

    # Align dimensions (guard against version mismatches)
    n_features  = min(len(sv_row), test_vec.shape[1], len(feature_names))
    sv_row      = sv_row[:n_features]
    input_vec   = test_vec[0, :n_features]
    feat_names  = feature_names[:n_features]

    non_zero = [
        (feat_names[i], float(sv_row[i]))
        for i in range(n_features)
        if input_vec[i] > 0   # only words actually present in the text
    ]
    non_zero.sort(key=lambda x: abs(x[1]), reverse=True)
    return non_zero[:top_n]


def plot_word_importance(word_scores: list, pred_star: int, method: str = "SHAP"):
    """Render a Plotly horizontal bar chart for word importance."""
    if not word_scores:
        empty_state("No word importance data to display.")
        return

    words  = [w for w, _ in word_scores]
    scores = [s for _, s in word_scores]
    colors = [ACCENT_GREEN if s > 0 else ACCENT_RED for s in scores]

    try:
        import plotly.graph_objects as go

        fig = go.Figure(go.Bar(
            x=scores[::-1],
            y=words[::-1],
            orientation="h",
            marker_color=colors[::-1],
            text=[f"{s:+.4f}" for s in scores[::-1]],
            textposition="outside",
        ))
        fig.update_layout(
            title=f"{method} — Word contributions for {pred_star} star prediction",
            xaxis_title="Contribution score",
            height=max(300, len(words) * 25 + 100),
            margin=dict(l=10, r=60, t=40, b=10),
            plot_bgcolor="white",
        )
        fig.add_vline(x=0, line_color="gray", line_width=1)
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        # Text fallback
        for word, score in word_scores:
            color = ACCENT_GREEN if score > 0 else ACCENT_RED
            icon  = "▲" if score > 0 else "▼"
            st.markdown(
                f"<span style='color:{color}'>{icon} **{word}** ({score:+.4f})</span>",
                unsafe_allow_html=True,
            )


def render():
    page_header(
        "", "Explainability"
    )

    with st.spinner("Loading classifier …"):
        tfidf, clf = load_classifier()

    if tfidf is None or clf is None:
        info_box(
            "The TF-IDF + LR rating artifacts (<code>lr_tfidf_en.pkl</code> and "
            "<code>lr_rating_en.pkl</code>) are required.",
            kind="error",
        )
        return

    # Load a background sample for SHAP
    df = load_dataset()
    use_shap = False

    try:
        import shap  # noqa: F401
        use_shap = True
    except ImportError:
        info_box(
            "SHAP is not installed. Using <b>LR coefficient × TF-IDF</b> as a fallback. "
            "Install with <code>pip install shap</code> for full SHAP values.",
            kind="info",
        )

    feature_names = tfidf.get_feature_names_out()

    st.markdown("#### Enter a review to explain")

    examples = {
        "— choose an example —": "",
        "Negative": (
            "The insurance refused my claim without any justification. "
            "Customer service is completely unresponsive. Very expensive for nothing."
        ),
        "Positive": (
            "Excellent service, fast reimbursement, very satisfied with the coverage. "
            "I highly recommend this insurer."
        ),
        "Mixed": (
            "The price increased again this year. The claim process was okay but slow. "
            "Overall acceptable but not great."
        ),
    }

    selected = st.selectbox("Load an example", list(examples.keys()))
    user_text = st.text_area(
        "Review text",
        value=examples[selected],
        height=140,
        label_visibility="collapsed",
        placeholder="Enter text to explain …",
    )

    top_n = st.slider("Number of words to show", min_value=5, max_value=25, value=15, step=5)

    if st.button("Explain Prediction", type="primary", use_container_width=True):
        if len(user_text.strip()) < cfg.MIN_REVIEW_CHARS:
            info_box("Please enter at least 20 characters.", kind="warning")
            return

        # Run prediction first
        cleaned   = prepare_for_tfidf(user_text)
        vec       = tfidf.transform([cleaned])
        probs     = clf.predict_proba(vec)[0]
        pred_idx  = int(np.argmax(probs))
        pred_star = pred_idx + 1
        confidence= float(probs[pred_idx])

        st.divider()

        # Prediction summary
        col1, col2, col3 = st.columns(3)
        with col1:
            metric_card("Predicted", "⭐" * pred_star, color=PRIMARY)
        with col2:
            metric_card("Confidence", f"{confidence*100:.1f}%", color=PRIMARY)
        with col3:
            method_used = "SHAP LinearExplainer" if use_shap else "LR Coefficients × TF-IDF"
            metric_card("Method", method_used, color=PRIMARY)

        st.markdown("### 🔍 Word Importance")

        with st.spinner("Computing word contributions …"):
            if use_shap and not df.empty:
                bg_size  = min(500, len(df))
                bg_texts = df[cfg.COL_TEXT_EN].fillna("").sample(
                    bg_size, random_state=42
                ).apply(prepare_for_tfidf).tolist()

                try:
                    word_scores = shap_word_importance(
                        user_text, tfidf, clf, pred_idx, bg_texts, top_n=top_n
                    )
                    method = "SHAP"
                except Exception as e:
                    st.warning(f"SHAP failed ({e}), falling back to coefficient-based explanation.")
                    word_scores = coeff_word_importance(user_text, tfidf, clf, pred_idx, top_n)
                    method = "LR Coefficients"
            else:
                word_scores = coeff_word_importance(user_text, tfidf, clf, pred_idx, top_n)
                method = "LR Coefficients"

        plot_word_importance(word_scores, pred_star, method=method)

        # Highlighted text
        if word_scores:
            pos_words = [w for w, s in word_scores if s > 0]
            neg_words = [w for w, s in word_scores if s < 0]

            st.markdown("### Highlighted Text")
            highlighted = highlight_text(user_text, pos_words, color="#C8E6C9")
            highlighted = highlight_text(highlighted, neg_words, color="#FFCDD2")

            st.markdown(
                f"""
                <div style="
                    background:#FAFAFA;
                    padding:1rem;
                    border-radius:8px;
                    border:1px solid #DDD;
                    line-height:1.8;
                    font-size:0.95rem;
                ">
                    {highlighted}
                </div>
                <p style="font-size:0.78rem; color:#888; margin-top:0.4rem;">
                    Green = pushes toward {pred_star} &nbsp;|&nbsp; Red = pushes away
                </p>
                """,
                unsafe_allow_html=True,
            )

        # Per-class SHAP bar chart (global overview if SHAP is available)
        if use_shap and word_scores:
            with st.expander("Class-by-class contribution breakdown"):
                try:
                    import shap
                    bg_mat    = tfidf.transform(bg_texts).toarray()
                    t_vec     = tfidf.transform([cleaned]).toarray()
                    explainer = shap.LinearExplainer(
                        clf, bg_mat, feature_perturbation="interventional"
                    )
                    shap_vals = explainer.shap_values(t_vec)

                    for cls_idx in range(5):
                        cls_star = cls_idx + 1
                        cls_name = f"{cls_star} star{'s' if cls_star > 1 else ''}"

                        # Robust extraction (list vs 3-D ndarray)
                        sv_all = _extract_shap_for_class(shap_vals, cls_idx)
                        sv_row = sv_all[0] if sv_all.ndim == 2 else sv_all

                        # Align dimensions safely
                        n_f       = min(len(sv_row), t_vec.shape[1], len(feature_names))
                        sv_row    = sv_row[:n_f]
                        t_dense   = t_vec[0, :n_f]
                        feat_trim = feature_names[:n_f]

                        cls_scores = [
                            (feat_trim[i], float(sv_row[i]))
                            for i in range(n_f)
                            if t_dense[i] > 0
                        ]
                        cls_scores.sort(key=lambda x: abs(x[1]), reverse=True)
                        st.markdown(f"**{cls_name}**")
                        plot_word_importance(cls_scores[:8], cls_star, method="SHAP")
                except Exception as e:
                    st.warning(f"Per-class breakdown failed: {e}")
