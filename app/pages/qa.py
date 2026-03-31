import streamlit as st
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import config as cfg
from utils.ui_helpers import (
    page_header,
    info_box,
    metric_card,
    empty_state,
    PRIMARY,
    ACCENT_GREEN,
    highlight_text,
)
from utils.model_loader import (
    load_dataset,
    load_tfidf_vectorizers,
    load_word2vec_models,
    build_sentence_vectors,
    load_qa_pipeline,
)
from utils.retrieval import tfidf_search, bm25_search, embedding_search


def _select_context_docs(df, text_col: str, max_words: int = 1500):
    docs = []
    total = 0
    for _, row in df.iterrows():
        text = str(row.get(text_col, "")).strip()
        if not text:
            continue
        words = text.split()
        if total >= max_words:
            break
        if total + len(words) > max_words:
            remaining = max_words - total
            text = " ".join(words[:remaining])
            words = text.split()
        docs.append((row, text))
        total += len(words)
    return docs


def _run_extract_qa(question: str, docs, qa_pipe):
    answers = []
    for row, context in docs:
        result = qa_pipe(question=question, context=context)
        answer = str(result.get("answer", "")).strip()
        if answer:
            answers.append(
                {
                    "answer": answer,
                    "score": float(result.get("score", 0.0)),
                    "start": int(result.get("start", 0)),
                    "end": int(result.get("end", 0)),
                    "context": context,
                    "row": row,
                }
            )
    answers.sort(key=lambda x: x["score"], reverse=True)
    return answers


def render():
    page_header("", "Question Answering")

    df = load_dataset()
    tfidf_en, _ = load_tfidf_vectorizers()
    w2v_en, _ = load_word2vec_models()
    qa_pipe = load_qa_pipeline()

    if df.empty:
        info_box("Dataset not loaded.", kind="error")
        return
    if qa_pipe is None:
        info_box(
            "The local QA model could not be loaded. Please install the app dependencies "
            "and rerun the app.",
            kind="error",
        )
        return

    st.markdown("#### Retrieval Configuration")
    c1, c2 = st.columns(2)
    with c1:
        retrieval_method = st.selectbox(
            "Search method",
            ["BM25 (Recommended)", "TF-IDF Cosine", "Embedding (Word2Vec)"],
        )
    with c2:
        top_k = st.slider("Reviews to inspect", min_value=3, max_value=25, value=8, step=1)

    with st.expander("Filter the knowledge base", expanded=False):
        fcol1, fcol2 = st.columns(2)
        with fcol1:
            insurers = ["All"] + sorted(df[cfg.COL_INSURER].dropna().unique().tolist())
            f_insurer = st.selectbox("Limit to insurer", insurers)
        with fcol2:
            products = ["All"] + sorted(df[cfg.COL_PRODUCT].dropna().unique().tolist())
            f_product = st.selectbox("Limit to product", products)

    corpus = df.copy()
    if f_insurer != "All":
        corpus = corpus[corpus[cfg.COL_INSURER] == f_insurer]
    if f_product != "All":
        corpus = corpus[corpus[cfg.COL_PRODUCT] == f_product]

    st.divider()
    st.markdown("#### Ask a grounded question about the reviews")

    example_questions = {
        "— choose an example —": "",
        "Claims": "Why are customers dissatisfied with the claims process?",
        "Pricing": "What do customers say about price increases and value for money?",
        "Customer service": "How is customer service described by reviewers?",
        "Cancellation": "Why do customers complain about cancellation or contract renewal?",
    }
    selected_ex = st.selectbox("Load an example question", list(example_questions.keys()))
    question = st.text_input(
        "Your question",
        value=example_questions[selected_ex],
        placeholder="Ask a question grounded in the insurance review corpus ...",
        label_visibility="collapsed",
    )

    if st.button("Answer with Local QA", type="primary", use_container_width=True):
        if len(question.strip()) < 5:
            st.warning("Please enter a longer question.")
            return
        if corpus.empty:
            empty_state("No reviews match the selected filters.")
            return

        with st.spinner(f"Retrieving supporting reviews with {retrieval_method} ..."):
            try:
                if "BM25" in retrieval_method:
                    retrieved = bm25_search(question, corpus, cfg.COL_TEXT_EN, top_k=top_k)
                elif "TF-IDF" in retrieval_method:
                    retrieved = tfidf_search(question, corpus, tfidf_en, cfg.COL_TEXT_EN, top_k=top_k)
                else:
                    corpus_vecs = build_sentence_vectors(corpus)
                    idf_map = dict(zip(tfidf_en.get_feature_names_out(), tfidf_en.idf_)) if tfidf_en else {}
                    retrieved = embedding_search(question, corpus, corpus_vecs, w2v_en, idf_map, top_k=top_k)
            except Exception as e:
                st.error(f"Retrieval failed: {e}")
                return

        if retrieved.empty:
            empty_state("No relevant reviews found for this question.")
            return

        docs = _select_context_docs(retrieved, cfg.COL_TEXT_EN, max_words=1500)
        with st.spinner("Running extractive QA over the retrieved review context ..."):
            answers = _run_extract_qa(question, docs, qa_pipe)

        if not answers:
            empty_state("The QA model could not extract a confident answer from the retrieved reviews.")
            return

        best = answers[0]
        st.divider()
        st.markdown("### Answer")
        st.markdown(
            f"""
            <div style="
                background:#E8F5E9;
                border-left:5px solid {ACCENT_GREEN};
                padding:1.2rem;
                border-radius:10px;
                font-size:1.05rem;
                line-height:1.6;
                color:#1B5E20;
            ">
                {best['answer']}
            </div>
            """,
            unsafe_allow_html=True,
        )

        m1, m2, m3 = st.columns(3)
        with m1:
            metric_card("Answer Confidence", f"{best['score']*100:.1f}%", color=PRIMARY)
        with m2:
            metric_card("Reviews Retrieved", str(len(retrieved)), color=PRIMARY)
        with m3:
            metric_card("QA Model", "RoBERTa-SQuAD2", color=PRIMARY)

        st.markdown("### Best Supporting Review")
        best_row = best["row"]
        highlighted = highlight_text(best["context"], [best["answer"]], color="#C8E6C9")
        st.markdown(
            f"""
            <div style="
                background:#FAFAFA;
                padding:1rem;
                border-radius:8px;
                border:1px solid #DDD;
                line-height:1.8;
                font-size:0.96rem;
            ">
                {highlighted}
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(
            f"{best_row.get(cfg.COL_INSURER, 'Unknown')} | "
            f"{best_row.get(cfg.COL_PRODUCT, 'Unknown product')} | "
            f"{'⭐' * int(best_row.get(cfg.COL_RATING, 0))}"
        )

        with st.expander("Alternative answer candidates", expanded=True):
            for idx, candidate in enumerate(answers[:5], start=1):
                row = candidate["row"]
                st.markdown(
                    f"**#{idx}** {candidate['answer']}  "
                    f"(confidence: {candidate['score']*100:.1f}%)"
                )
                st.caption(
                    f"{row.get(cfg.COL_INSURER, 'Unknown')} | "
                    f"{row.get(cfg.COL_PRODUCT, 'Unknown product')} | "
                    f"{'⭐' * int(row.get(cfg.COL_RATING, 0))}"
                )
                st.markdown(
                    highlight_text(candidate["context"], [candidate["answer"]], color="#FFF59D"),
                    unsafe_allow_html=True,
                )
                st.divider()
