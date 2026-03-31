import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

import config as cfg

# Page config 
st.set_page_config(
    page_title   = cfg.APP_TITLE,
    page_icon    = cfg.APP_ICON,
    layout       = "wide",
    initial_sidebar_state = "expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
    /* Global font & background */
    html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }

    /* Hide default Streamlit page navigation */
    [data-testid="stSidebarNav"] { display: none; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0D47A1 0%, #1565C0 100%);
    }
    [data-testid="stSidebar"] * { color: white !important; }
    [data-testid="stSidebar"] .stRadio label { color: white !important; }
    [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.2); }

    /* Main content area */
    .main .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

    /* Buttons */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #1565C0, #0D47A1);
        border: none;
        color: white;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1.5rem;
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #1976D2, #1565C0);
    }

    /* Expanders */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #1565C0;
    }

    /* Dataframe */
    .stDataFrame thead th { background: #1565C0 !important; color: white !important; }

    /* Success / info boxes */
    .stSuccess { border-radius: 8px; }
    .stInfo    { border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding: 1rem 0 0.5rem;">
            <div style="font-size:2.5rem;"></div>
            <div style="font-size:1.1rem; font-weight:700; letter-spacing:0.03em;">
                Insurance NLP
            </div>
            <div style="font-size:0.78rem; opacity:0.75; margin-top:0.2rem;">
                ESILV A4 DIA6 — 2026
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    PAGES = {
        "Prediction":            "prediction",
        "Insurer Analysis":      "summary",
        "Explanation":           "explanation",
        "Information Retrieval": "retrieval",
        "RAG":                   "rag",
        "Question Answering":    "qa",
    }

    selected_label = st.radio(
        "Navigation",
        list(PAGES.keys()),
        label_visibility="collapsed",
    )
    page_key = PAGES[selected_label]

    st.divider()

    # Dataset stats 
    try:
        from utils.model_loader import load_dataset
        df_meta = load_dataset()
        if not df_meta.empty:
            st.markdown("**Dataset**")
            st.markdown(f"- **{len(df_meta):,}** reviews")
            if cfg.COL_INSURER in df_meta.columns:
                st.markdown(f"- **{df_meta[cfg.COL_INSURER].nunique()}** insurers")
            if cfg.COL_PRODUCT in df_meta.columns:
                st.markdown(f"- **{df_meta[cfg.COL_PRODUCT].nunique()}** products")
            if cfg.COL_RATING in df_meta.columns:
                avg = df_meta[cfg.COL_RATING].mean()
                st.markdown(f"- **{avg:.2f}** avg. rating ")
    except Exception:
        pass

    st.divider()
    st.markdown(
        "<div style='font-size:0.72rem; opacity:0.6; text-align:center;'>"
        "Leo WINTER · Alvaro SERERO<br>NLP Project 2 — 2026"
        "</div>",
        unsafe_allow_html=True,
    )

# Page routing
if page_key == "prediction":
    from pages.prediction import render
    render()

elif page_key == "summary":
    from pages.summary import render
    render()

elif page_key == "explanation":
    from pages.explanation import render
    render()

elif page_key == "retrieval":
    from pages.retrieval import render
    render()

elif page_key == "rag":
    from pages.rag import render
    render()

elif page_key == "qa":
    from pages.qa import render
    render()
