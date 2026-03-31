import streamlit as st
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import config as cfg
from utils.ui_helpers import page_header, info_box, metric_card, empty_state, PRIMARY
from utils.model_loader import (
    load_dataset,
    load_tfidf_vectorizers,
    load_word2vec_models,
    build_sentence_vectors,
)
from utils.retrieval import tfidf_search, bm25_search, embedding_search, build_rag_context


def call_llm(prompt: str, api_key: str, base_url: str, model: str, max_tokens: int) -> str:
    """Call an OpenAI-compatible chat endpoint."""
    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=api_key if api_key else "no-key",
            base_url=base_url,
        )
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an analyst of customer reviews for French insurance companies. "
                        "Answer using only the retrieved review context. If the context is "
                        "insufficient, say so clearly and do not invent facts."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
    except ImportError:
        return (
            "The `openai` package is not installed. Run `pip install -r app/requirements.txt` "
            "and restart the app."
        )
    except Exception as e:
        return f"LLM call failed: {e}"


def build_rag_prompt(question: str, context: str) -> str:
    return f"""You are analysing customer reviews of French insurance companies.

RETRIEVED CONTEXT:
{context}

QUESTION:
{question}

Instructions:
- Use only the context above.
- Be concise and specific.
- If the context does not answer the question, say so directly.
- When possible, mention the main insurers, themes, or review patterns behind the answer.
"""


def render():
    page_header(
        "",
        "RAG Analysis",
        "Use retrieved reviews as grounded context for an external LLM answer.",
    )

    df = load_dataset()
    tfidf_en, _ = load_tfidf_vectorizers()
    w2v_en, _ = load_word2vec_models()

    if df.empty:
        info_box("Dataset not loaded. Please run the notebooks first.", kind="error")
        return

    with st.expander("LLM configuration", expanded=not cfg.OPENAI_API_KEY):
        st.markdown(
            "This page is the only one that can rely on an external LLM. "
            "It works with OpenAI-compatible endpoints, including local Ollama."
        )
        c1, c2 = st.columns(2)
        with c1:
            api_key = st.text_input(
                "API key",
                value=cfg.OPENAI_API_KEY,
                type="password",
                help="Leave empty only if your endpoint does not require a key.",
            )
        with c2:
            api_base = st.text_input(
                "API base URL",
                value=cfg.OPENAI_API_BASE,
                help="For example https://api.openai.com/v1 or http://localhost:11434/v1",
            )

        c3, c4 = st.columns(2)
        with c3:
            model_id = st.text_input("Model", value=cfg.RAG_LLM_MODEL)
        with c4:
            max_tokens = st.slider("Max answer tokens", 100, 1024, cfg.RAG_MAX_TOKENS, 50)

        top_k = st.slider("Retrieved reviews", 2, 15, cfg.RAG_TOP_K, 1)
        retrieval_method = st.selectbox(
            "Retrieval method",
            ["BM25 (Recommended)", "TF-IDF Cosine", "Embedding (Word2Vec)"],
        )

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

    st.markdown("#### Ask a grounded question about the reviews")
    example_questions = {
        "— choose an example —": "",
        "Claims": "What are the main complaints about the claims process?",
        "Pricing": "How do customers describe price increases and value for money?",
        "Customer service": "What patterns appear in customer service reviews?",
        "Top insurers": "Which insurers receive the strongest positive feedback and why?",
    }
    selected_ex = st.selectbox("Load an example question", list(example_questions.keys()))
    question = st.text_input(
        "Your question",
        value=example_questions[selected_ex],
        placeholder="Ask a broad analytical question grounded in the corpus ...",
        label_visibility="collapsed",
    )

    if st.button("Ask the RAG system", type="primary", use_container_width=True):
        if len(question.strip()) < 5:
            info_box("Please enter a longer question.", kind="warning")
            return
        if corpus.empty:
            empty_state("No reviews match the selected filters.")
            return
        if not api_key and "localhost" not in api_base.lower() and "ollama" not in api_base.lower():
            info_box(
                "Please provide an API key, or point the base URL to a local compatible endpoint.",
                kind="warning",
            )
            return

        with st.spinner(f"Retrieving evidence with {retrieval_method} ..."):
            try:
                if "BM25" in retrieval_method:
                    retrieved = bm25_search(question, corpus, cfg.COL_TEXT_EN, top_k=top_k)
                elif "TF-IDF" in retrieval_method:
                    retrieved = tfidf_search(question, corpus, tfidf_en, cfg.COL_TEXT_EN, top_k=top_k)
                else:
                    corpus_vecs = build_sentence_vectors(corpus)
                    if corpus_vecs is None or w2v_en is None:
                        info_box(
                            "Embedding retrieval is unavailable right now. Falling back to TF-IDF.",
                            kind="warning",
                        )
                        retrieved = tfidf_search(question, corpus, tfidf_en, cfg.COL_TEXT_EN, top_k=top_k)
                    else:
                        idf_map = (
                            dict(zip(tfidf_en.get_feature_names_out(), tfidf_en.idf_))
                            if tfidf_en is not None
                            else {}
                        )
                        retrieved = embedding_search(
                            question,
                            corpus,
                            corpus_vecs,
                            w2v_en,
                            idf_map,
                            top_k=top_k,
                        )
            except Exception as e:
                info_box(f"Retrieval failed: {e}", kind="error")
                return

        if retrieved.empty:
            empty_state("No relevant reviews were retrieved for this question.")
            return

        context = build_rag_context(retrieved, cfg.COL_TEXT_EN, max_words=900)
        prompt = build_rag_prompt(question, context)

        with st.spinner("Generating the answer with the external LLM ..."):
            answer = call_llm(
                prompt=prompt,
                api_key=api_key,
                base_url=api_base,
                model=model_id,
                max_tokens=max_tokens,
            )

        st.divider()
        st.markdown("### Generated Answer")
        st.markdown(
            f"""
            <div style="
                background:#E8F5E9;
                border-left:4px solid #43A047;
                padding:1rem 1.2rem;
                border-radius:8px;
                font-size:1rem;
                line-height:1.7;
            ">
                {answer}
            </div>
            """,
            unsafe_allow_html=True,
        )

        m1, m2, m3 = st.columns(3)
        with m1:
            metric_card("Retrieved Reviews", str(len(retrieved)), color=PRIMARY)
        with m2:
            avg_rating = retrieved[cfg.COL_RATING].mean() if cfg.COL_RATING in retrieved.columns else 0
            metric_card("Average Rating", f"{avg_rating:.2f} / 5", color=PRIMARY)
        with m3:
            metric_card("LLM Model", model_id, color=PRIMARY)

        st.markdown("### Retrieved Evidence")
        st.caption("These reviews were passed to the LLM as retrieval context.")
        for idx, (_, row) in enumerate(retrieved.iterrows(), start=1):
            stars = "⭐" * int(row.get(cfg.COL_RATING, 0))
            insurer = str(row.get(cfg.COL_INSURER, "Unknown insurer"))
            product = str(row.get(cfg.COL_PRODUCT, "Unknown product"))
            theme = str(row.get(cfg.COL_THEME, "Unavailable"))
            score = float(row.get("score", 0))
            text = str(row.get(cfg.COL_TEXT_EN, "")).strip()

            with st.expander(
                f"Doc {idx} | {insurer} | {product} | {stars} | {theme} | score {score:.3f}"
            ):
                st.markdown(text)

        with st.expander("Prompt sent to the LLM"):
            st.code(prompt, language="text")
