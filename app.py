# import os
# import sys
# import json
# import pandas as pd
# import streamlit as st
# from dotenv import load_dotenv
# load_dotenv()

# # Make sure src/ is on the path
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# from vector_store import load_vector_store, build_vector_store, CHROMA_DIR
# from retriever    import HybridRetriever
# from chatbot      import GroupTheoryChatbot

# # ------------------------------------------------------------------
# # Page config
# # ------------------------------------------------------------------
# st.set_page_config(
#     page_title = "Group Theory RAG Chatbot",
#     page_icon  = "∞",
#     layout     = "wide",
#     initial_sidebar_state = "expanded",
# )

# # ------------------------------------------------------------------
# # Custom CSS
# # ------------------------------------------------------------------
# st.markdown("""
# <style>
# @import url('https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400;600&display=swap');

# html, body, [class*="css"] { font-family: 'EB Garamond', Georgia, serif; }
# .stApp { background-color: #0f0e0b; color: #e8e0d0; }
# [data-testid="stSidebar"] { background-color: #161410; border-right: 1px solid #2a2418; }

# .user-bubble {
#     background: #1e1a14; border-left: 3px solid #c9a84c;
#     padding: 12px 16px; border-radius: 0 8px 8px 0; margin: 8px 0; font-size: 1.05rem;
# }
# .assistant-bubble {
#     background: #141210; border-left: 3px solid #6b8f71;
#     padding: 14px 18px; border-radius: 0 8px 8px 0; margin: 8px 0;
#     font-size: 1.05rem; line-height: 1.7;
# }
# .source-tag {
#     display: inline-block; background: #1a2a1c; color: #8fb894;
#     font-family: 'JetBrains Mono', monospace; font-size: 0.72rem;
#     padding: 2px 8px; border-radius: 4px; margin: 4px 2px 0 0;
#     border: 1px solid #2a4a2e; text-decoration: none;
# }
# .source-tag:hover { background: #2a4a2e; }
# .qa-tag { background: #2a1e0a; color: #c9a84c; border-color: #4a3418; }
# .confidence-bar { height: 3px; border-radius: 2px; margin-top: 6px; }
# .stat-box {
#     background: #1a1710; border: 1px solid #2a2418; border-radius: 6px;
#     padding: 12px; text-align: center; margin: 4px 0;
# }
# .stat-number { font-size: 1.8rem; font-weight: 600; color: #c9a84c; font-family: 'JetBrains Mono', monospace; }
# .stat-label  { font-size: 0.8rem; color: #8a7d65; font-family: 'JetBrains Mono', monospace; }
# h1, h2, h3  { font-family: 'EB Garamond', serif; }

# .stTextInput input, .stChatInput textarea {
#     background: #1a1710 !important; color: #e8e0d0 !important;
#     border: 1px solid #2a2418 !important; font-family: 'EB Garamond', serif !important;
#     font-size: 1.05rem !important;
# }
# .stButton button {
#     background: #1e1a14; color: #c9a84c; border: 1px solid #c9a84c;
#     font-family: 'JetBrains Mono', monospace; font-size: 0.85rem;
# }
# .stButton button:hover { background: #c9a84c; color: #0f0e0b; }
# </style>
# """, unsafe_allow_html=True)


# # ------------------------------------------------------------------
# # Load / initialise chatbot
# # ------------------------------------------------------------------
# @st.cache_resource(show_spinner="Loading knowledge base…")
# def load_chatbot():
#     if not os.path.exists(CHROMA_DIR) or not os.listdir(CHROMA_DIR):
#         st.warning("Vector store not found — running full pipeline…")
#         from scraper      import run_scraper
#         from qa_generator import run_qa_generator
#         run_scraper()
#         run_qa_generator()
#         build_vector_store()

#     doc_col, qa_col = load_vector_store()
#     retriever = HybridRetriever(doc_col, qa_col)
#     return GroupTheoryChatbot(retriever), doc_col, qa_col


# # ------------------------------------------------------------------
# # Stats
# # ------------------------------------------------------------------
# def get_stats(doc_col, qa_col) -> dict:
#     raw_dir = os.path.join(os.path.dirname(__file__), "data", "raw")
#     n_pages = len([f for f in os.listdir(raw_dir) if f.endswith(".txt")]) if os.path.exists(raw_dir) else 0
#     n_words = 0
#     if os.path.exists(raw_dir):
#         for f in os.listdir(raw_dir):
#             if f.endswith(".txt"):
#                 with open(os.path.join(raw_dir, f)) as fh:
#                     n_words += len(fh.read().split())
#     return {"pages": n_pages, "qa": qa_col.count(), "chunks": doc_col.count(), "words": n_words}


# # ------------------------------------------------------------------
# # Sidebar
# # ------------------------------------------------------------------
# def render_sidebar(doc_col, qa_col):
#     with st.sidebar:
#         st.markdown("## ∞ Group Theory\n### RAG Chatbot")
#         st.markdown("---")

#         # Stats
#         st.markdown("**Knowledge Base**")
#         stats = get_stats(doc_col, qa_col)
#         col1, col2 = st.columns(2)
#         with col1:
#             st.markdown(f'<div class="stat-box"><div class="stat-number">{stats["pages"]}</div><div class="stat-label">pages scraped</div></div>', unsafe_allow_html=True)
#             st.markdown(f'<div class="stat-box"><div class="stat-number">{stats["chunks"]}</div><div class="stat-label">doc chunks</div></div>', unsafe_allow_html=True)
#         with col2:
#             st.markdown(f'<div class="stat-box"><div class="stat-number">{stats["qa"]}</div><div class="stat-label">Q/A pairs</div></div>', unsafe_allow_html=True)
#             st.markdown(f'<div class="stat-box"><div class="stat-number">{stats["words"]//1000}k</div><div class="stat-label">words indexed</div></div>', unsafe_allow_html=True)

#         st.markdown("---")
#         st.markdown("**Source**")
#         st.markdown("📖 [Wikipedia — Group Theory](https://en.wikipedia.org/wiki/Group_theory)")

#         st.markdown("---")
#         st.markdown("**Retrieval Mode**")
#         st.markdown("🟡 **Q/A Match** — direct answer from curated pairs  \n🟢 **Vector Search** — synthesised from Wikipedia chunks")

#         st.markdown("---")
#         if st.button("🗑  Clear Chat", use_container_width=True):
#             st.session_state.messages = []
#             if "chatbot" in st.session_state:
#                 st.session_state.chatbot.clear_history()
#             st.rerun()

#         # Sample questions
#         st.markdown("---")
#         st.markdown("**Sample Questions**")
#         samples = [
#             "What is a normal subgroup?",
#             "State and explain Lagrange's theorem.",
#             "What is the kernel of a homomorphism?",
#             "How do cosets partition a group?",
#             "What are the Sylow theorems?",
#         ]
#         for q in samples:
#             if st.button(q, use_container_width=True, key=f"sample_{q}"):
#                 st.session_state.pending_query = q
#                 st.rerun()

#         # ── Q/A Dataset viewer (full, with search) ──────────────────
#         st.markdown("---")
#         with st.expander("📋 View Q/A Dataset"):
#             qa_path = os.path.join(os.path.dirname(__file__), "data", "processed", "qa_dataset.csv")
#             if os.path.exists(qa_path):
#                 df = pd.read_csv(qa_path)
#                 st.caption(f"{len(df)} pairs total")

#                 search = st.text_input("Filter by keyword", key="qa_search", placeholder="e.g. coset")
#                 if search:
#                     mask = df["question"].str.contains(search, case=False, na=False)
#                     df_show = df[mask]
#                 else:
#                     df_show = df

#                 st.dataframe(
#                     df_show[["question", "answer", "source_page"]],
#                     use_container_width=True,
#                     height=300,
#                 )
#             else:
#                 st.info("Q/A dataset not yet generated.")

#         # ── Upload additional Q/A dataset ───────────────────────────
#         st.markdown("---")
#         with st.expander("➕ Upload Additional Q/A Pairs"):
#             st.caption("Upload a CSV with columns: question, answer, source_page")
#             uploaded = st.file_uploader("Choose CSV file", type=["csv"], key="qa_upload")
#             if uploaded:
#                 try:
#                     new_df = pd.read_csv(uploaded)
#                     required = {"question", "answer", "source_page"}
#                     if not required.issubset(new_df.columns):
#                         st.error(f"CSV must have columns: {required}")
#                     else:
#                         st.success(f"Preview — {len(new_df)} rows loaded")
#                         st.dataframe(new_df.head(5), use_container_width=True)
#                         if st.button("Merge & Rebuild Vector Store", key="merge_btn"):
#                             qa_path = os.path.join(os.path.dirname(__file__), "data", "processed", "qa_dataset.csv")
#                             if os.path.exists(qa_path):
#                                 existing = pd.read_csv(qa_path)
#                                 combined = pd.concat([existing, new_df], ignore_index=True)
#                                 combined = combined.drop_duplicates(subset=["question"])
#                             else:
#                                 combined = new_df
#                             combined.to_csv(qa_path, index=False)
#                             with st.spinner("Rebuilding vector store…"):
#                                 build_vector_store()
#                             st.success(f"Done! Total pairs: {len(combined)}")
#                             st.cache_resource.clear()
#                             st.rerun()
#                 except Exception as e:
#                     st.error(f"Error reading file: {e}")

#         # ── Source pages list ────────────────────────────────────────
#         st.markdown("---")
#         with st.expander("🔗 All Source Pages"):
#             raw_dir = os.path.join(os.path.dirname(__file__), "data", "raw")
#             manifest_path = os.path.join(os.path.dirname(__file__), "data", "processed", "pages_manifest.json")
#             if os.path.exists(manifest_path):
#                 with open(manifest_path) as f:
#                     manifest = json.load(f)
#                 for page in manifest:
#                     st.markdown(
#                         f"[{page['display_name']}]({page['url']}) — {page['word_count']:,} words",
#                         unsafe_allow_html=False,
#                     )
#             elif os.path.exists(raw_dir):
#                 files = [f.replace("_", " ").replace(".txt", "").title()
#                          for f in os.listdir(raw_dir) if f.endswith(".txt")]
#                 for name in sorted(files):
#                     st.markdown(f"• {name}")
#             else:
#                 st.info("No pages scraped yet.")

#         st.markdown("---")
#         st.markdown(
#             "<small style='color:#4a4030'>AGAI-03 · Group Theory RAG<br>Built with Streamlit + ChromaDB + Claude</small>",
#             unsafe_allow_html=True,
#         )


# # ------------------------------------------------------------------
# # Message rendering
# # ------------------------------------------------------------------
# def render_message(msg: dict):
#     if msg["role"] == "user":
#         st.markdown(f'<div class="user-bubble">🎓 {msg["content"]}</div>', unsafe_allow_html=True)
#     else:
#         st.markdown(f'<div class="assistant-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
#         if msg.get("sources"):
#             mode_cls  = "qa-tag"  if msg.get("used_qa") else "source-tag"
#             mode_icon = "🟡 Q/A"  if msg.get("used_qa") else "🟢 Vector"
#             tags = f'<span class="{mode_cls} source-tag">{mode_icon}</span>'
#             for i, src in enumerate(msg["sources"]):
#                 urls = msg.get("source_urls", [])
#                 url  = urls[i] if i < len(urls) and urls[i] else None
#                 if url:
#                     tags += f'<a href="{url}" target="_blank" class="source-tag">📄 {src}</a>'
#                 else:
#                     tags += f'<span class="source-tag">📄 {src}</span>'
#             conf      = msg.get("confidence", 0)
#             bar_color = "#c9a84c" if msg.get("used_qa") else "#6b8f71"
#             tags += f'<div class="confidence-bar" style="width:{int(conf*100)}%; background:{bar_color};" title="Confidence: {conf:.0%}"></div>'
#             st.markdown(tags, unsafe_allow_html=True)


# # ------------------------------------------------------------------
# # Main
# # ------------------------------------------------------------------
# def main():
#     if "messages" not in st.session_state:
#         st.session_state.messages = []

#     chatbot, doc_col, qa_col = load_chatbot()
#     if "chatbot" not in st.session_state:
#         st.session_state.chatbot = chatbot

#     render_sidebar(doc_col, qa_col)

#     st.markdown(
#         "<h1 style='color:#c9a84c; margin-bottom:0'>∞ Group Theory Tutor</h1>"
#         "<p style='color:#8a7d65; font-style:italic; margin-top:4px'>"
#         "A RAG-powered assistant for undergraduate abstract algebra</p>",
#         unsafe_allow_html=True,
#     )
#     st.markdown("---")

#     for msg in st.session_state.messages:
#         render_message(msg)

#     pending    = st.session_state.pop("pending_query", None)
#     user_input = st.chat_input("Ask about group theory…") or pending

#     if user_input:
#         st.session_state.messages.append({"role": "user", "content": user_input})
#         render_message({"role": "user", "content": user_input})

#         with st.spinner("Searching knowledge base…"):
#             response = st.session_state.chatbot.chat(user_input)

#         msg = {
#             "role":        "assistant",
#             "content":     response.answer,
#             "sources":     response.sources,
#             "source_urls": response.source_urls,
#             "used_qa":     response.used_qa,
#             "confidence":  response.confidence,
#         }
#         st.session_state.messages.append(msg)
#         render_message(msg)
#         st.rerun()


# if __name__ == "__main__":
#     main()

import os
import sys
import json
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

# Make sure src/ is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from vector_store import load_vector_store, build_vector_store, CHROMA_DIR
from retriever    import HybridRetriever
from chatbot      import GroupTheoryChatbot

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(
    page_title = "Group Theory RAG Chatbot",
    page_icon  = "∞",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ------------------------------------------------------------------
# Custom CSS
# ------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'EB Garamond', Georgia, serif; }
.stApp { background-color: #0f0e0b; color: #e8e0d0; }
[data-testid="stSidebar"] { background-color: #161410; border-right: 1px solid #2a2418; }

.user-bubble {
    background: #1e1a14; border-left: 3px solid #c9a84c;
    padding: 12px 16px; border-radius: 0 8px 8px 0; margin: 8px 0; font-size: 1.05rem;
}
.assistant-bubble {
    background: #141210; border-left: 3px solid #6b8f71;
    padding: 14px 18px; border-radius: 0 8px 8px 0; margin: 8px 0;
    font-size: 1.05rem; line-height: 1.7;
}
.source-tag {
    display: inline-block; background: #1a2a1c; color: #8fb894;
    font-family: 'JetBrains Mono', monospace; font-size: 0.72rem;
    padding: 2px 8px; border-radius: 4px; margin: 4px 2px 0 0;
    border: 1px solid #2a4a2e; text-decoration: none;
}
.source-tag:hover { background: #2a4a2e; }
.qa-tag { background: #2a1e0a; color: #c9a84c; border-color: #4a3418; }
.confidence-bar { height: 3px; border-radius: 2px; margin-top: 6px; }
.confidence-widget {
    display: flex; align-items: center; gap: 8px;
    margin-top: 8px; padding: 6px 10px;
    background: #1a1710; border: 1px solid #2a2418;
    border-radius: 6px; width: fit-content;
}
.confidence-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem; color: #8a7d65; white-space: nowrap;
}
.confidence-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem; font-weight: 600;
}
.confidence-track {
    width: 80px; height: 6px; background: #2a2418;
    border-radius: 3px; overflow: hidden;
}
.confidence-fill { height: 100%; border-radius: 3px; }
.confidence-desc {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem; color: #6a5f50; font-style: italic;
}
.stat-box {
    background: #1a1710; border: 1px solid #2a2418; border-radius: 6px;
    padding: 12px; text-align: center; margin: 4px 0;
}
.stat-number { font-size: 1.8rem; font-weight: 600; color: #c9a84c; font-family: 'JetBrains Mono', monospace; }
.stat-label  { font-size: 0.8rem; color: #8a7d65; font-family: 'JetBrains Mono', monospace; }
h1, h2, h3  { font-family: 'EB Garamond', serif; }

.stTextInput input, .stChatInput textarea {
    background: #1a1710 !important; color: #e8e0d0 !important;
    border: 1px solid #2a2418 !important; font-family: 'EB Garamond', serif !important;
    font-size: 1.05rem !important;
}
.stButton button {
    background: #1e1a14; color: #c9a84c; border: 1px solid #c9a84c;
    font-family: 'JetBrains Mono', monospace; font-size: 0.85rem;
}
.stButton button:hover { background: #c9a84c; color: #0f0e0b; }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------
# Load / initialise chatbot
# ------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading knowledge base…")
def load_chatbot():
    if not os.path.exists(CHROMA_DIR) or not os.listdir(CHROMA_DIR):
        st.warning("Vector store not found — running full pipeline…")
        from scraper      import run_scraper
        from qa_generator import run_qa_generator
        run_scraper()
        run_qa_generator()
        build_vector_store()

    doc_col, qa_col = load_vector_store()
    retriever = HybridRetriever(doc_col, qa_col)
    return GroupTheoryChatbot(retriever), doc_col, qa_col


# ------------------------------------------------------------------
# Stats
# ------------------------------------------------------------------
def get_stats(doc_col, qa_col) -> dict:
    raw_dir = os.path.join(os.path.dirname(__file__), "data", "raw")
    n_pages = len([f for f in os.listdir(raw_dir) if f.endswith(".txt")]) if os.path.exists(raw_dir) else 0
    n_words = 0
    if os.path.exists(raw_dir):
        for f in os.listdir(raw_dir):
            if f.endswith(".txt"):
                with open(os.path.join(raw_dir, f)) as fh:
                    n_words += len(fh.read().split())
    return {"pages": n_pages, "qa": qa_col.count(), "chunks": doc_col.count(), "words": n_words}


# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
def render_sidebar(doc_col, qa_col):
    with st.sidebar:
        st.markdown("## ∞ Group Theory\n### RAG Chatbot")
        st.markdown("---")

        # Stats
        st.markdown("**Knowledge Base**")
        stats = get_stats(doc_col, qa_col)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f'<div class="stat-box"><div class="stat-number">{stats["pages"]}</div><div class="stat-label">pages scraped</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="stat-box"><div class="stat-number">{stats["chunks"]}</div><div class="stat-label">doc chunks</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="stat-box"><div class="stat-number">{stats["qa"]}</div><div class="stat-label">Q/A pairs</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="stat-box"><div class="stat-number">{stats["words"]//1000}k</div><div class="stat-label">words indexed</div></div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("**Source**")
        st.markdown("📖 [Wikipedia — Group Theory](https://en.wikipedia.org/wiki/Group_theory)")

        st.markdown("---")
        st.markdown("**Retrieval Mode**")
        st.markdown("🟡 **Q/A Match** — direct answer from curated pairs  \n🟢 **Vector Search** — synthesised from Wikipedia chunks")

        st.markdown("---")
        if st.button("🗑  Clear Chat", use_container_width=True):
            st.session_state.messages = []
            if "chatbot" in st.session_state:
                st.session_state.chatbot.clear_history()
            st.rerun()

        # Sample questions
        st.markdown("---")
        st.markdown("**Sample Questions**")
        samples = [
            "What is a normal subgroup?",
            "State and explain Lagrange's theorem.",
            "What is the kernel of a homomorphism?",
            "How do cosets partition a group?",
            "What are the Sylow theorems?",
        ]
        for q in samples:
            if st.button(q, use_container_width=True, key=f"sample_{q}"):
                st.session_state.pending_query = q
                st.rerun()

        # ── Q/A Dataset viewer (full, with search) ──────────────────
        st.markdown("---")
        with st.expander("📋 View Q/A Dataset"):
            qa_path = os.path.join(os.path.dirname(__file__), "data", "processed", "qa_dataset.csv")
            if os.path.exists(qa_path):
                df = pd.read_csv(qa_path)
                st.caption(f"{len(df)} pairs total")

                search = st.text_input("Filter by keyword", key="qa_search", placeholder="e.g. coset")
                if search:
                    mask = df["question"].str.contains(search, case=False, na=False)
                    df_show = df[mask]
                else:
                    df_show = df

                st.dataframe(
                    df_show[["question", "answer", "source_page"]],
                    use_container_width=True,
                    height=300,
                )
            else:
                st.info("Q/A dataset not yet generated.")

        # ── Upload additional Q/A dataset ───────────────────────────
        st.markdown("---")
        with st.expander("➕ Upload Additional Q/A Pairs"):
            st.caption("Upload a CSV with columns: question, answer, source_page")
            uploaded = st.file_uploader("Choose CSV file", type=["csv"], key="qa_upload")
            if uploaded:
                try:
                    new_df = pd.read_csv(uploaded)
                    required = {"question", "answer", "source_page"}
                    if not required.issubset(new_df.columns):
                        st.error(f"CSV must have columns: {required}")
                    else:
                        st.success(f"Preview — {len(new_df)} rows loaded")
                        st.dataframe(new_df.head(5), use_container_width=True)
                        if st.button("Merge & Rebuild Vector Store", key="merge_btn"):
                            qa_path = os.path.join(os.path.dirname(__file__), "data", "processed", "qa_dataset.csv")
                            if os.path.exists(qa_path):
                                existing = pd.read_csv(qa_path)
                                combined = pd.concat([existing, new_df], ignore_index=True)
                                combined = combined.drop_duplicates(subset=["question"])
                            else:
                                combined = new_df
                            combined.to_csv(qa_path, index=False)
                            with st.spinner("Rebuilding vector store…"):
                                build_vector_store()
                            st.success(f"Done! Total pairs: {len(combined)}")
                            st.cache_resource.clear()
                            st.rerun()
                except Exception as e:
                    st.error(f"Error reading file: {e}")

        # ── Source pages list ────────────────────────────────────────
        st.markdown("---")
        with st.expander("🔗 All Source Pages"):
            raw_dir = os.path.join(os.path.dirname(__file__), "data", "raw")
            manifest_path = os.path.join(os.path.dirname(__file__), "data", "processed", "pages_manifest.json")
            if os.path.exists(manifest_path):
                with open(manifest_path) as f:
                    manifest = json.load(f)
                for page in manifest:
                    st.markdown(
                        f"[{page['display_name']}]({page['url']}) — {page['word_count']:,} words",
                        unsafe_allow_html=False,
                    )
            elif os.path.exists(raw_dir):
                files = [f.replace("_", " ").replace(".txt", "").title()
                         for f in os.listdir(raw_dir) if f.endswith(".txt")]
                for name in sorted(files):
                    st.markdown(f"• {name}")
            else:
                st.info("No pages scraped yet.")

        st.markdown("---")
        st.markdown(
            "<small style='color:#4a4030'>AGAI-03 · Group Theory RAG<br>Built with Streamlit + ChromaDB + Claude</small>",
            unsafe_allow_html=True,
        )


# ------------------------------------------------------------------
# Message rendering
# ------------------------------------------------------------------
def confidence_colour(conf: float, used_qa: bool) -> str:
    """Return a hex colour based on confidence level."""
    if used_qa:
        return "#c9a84c"   # gold for Q/A hits
    if conf >= 0.75:
        return "#6b8f71"   # green — high
    if conf >= 0.50:
        return "#8fb8c4"   # blue — medium
    if conf >= 0.30:
        return "#c9a84c"   # amber — low
    return "#a05050"       # red — very low


def confidence_label(conf: float, used_qa: bool) -> str:
    """Return a human-readable confidence description."""
    if used_qa:
        return "direct match"
    if conf >= 0.75:
        return "high"
    if conf >= 0.50:
        return "medium"
    if conf >= 0.30:
        return "low"
    return "very low"


def render_message(msg: dict):
    if msg["role"] == "user":
        st.markdown(f'<div class="user-bubble">🎓 {msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="assistant-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
        if msg.get("sources"):
            mode_cls  = "qa-tag"  if msg.get("used_qa") else "source-tag"
            mode_icon = "🟡 Q/A"  if msg.get("used_qa") else "🟢 Vector"

            # Source tags
            tags = f'<span class="{mode_cls} source-tag">{mode_icon}</span>'
            for i, src in enumerate(msg["sources"]):
                urls = msg.get("source_urls", [])
                url  = urls[i] if i < len(urls) and urls[i] else None
                if url:
                    tags += f'<a href="{url}" target="_blank" class="source-tag">📄 {src}</a>'
                else:
                    tags += f'<span class="source-tag">📄 {src}</span>'
            st.markdown(tags, unsafe_allow_html=True)

            # Confidence widget
            conf       = msg.get("confidence", 0)
            used_qa    = msg.get("used_qa", False)
            colour     = confidence_colour(conf, used_qa)
            desc       = confidence_label(conf, used_qa)
            fill_width = int(conf * 100)
            retrieval  = "Q/A dataset" if used_qa else "vector search"

            st.markdown(f"""
            <div class="confidence-widget">
                <span class="confidence-label">confidence</span>
                <span class="confidence-value" style="color:{colour}">{conf:.0%}</span>
                <div class="confidence-track">
                    <div class="confidence-fill" style="width:{fill_width}%; background:{colour};"></div>
                </div>
                <span class="confidence-desc">{desc} · via {retrieval}</span>
            </div>""", unsafe_allow_html=True)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    if "messages" not in st.session_state:
        st.session_state.messages = []

    chatbot, doc_col, qa_col = load_chatbot()
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = chatbot

    render_sidebar(doc_col, qa_col)

    st.markdown(
        "<h1 style='color:#c9a84c; margin-bottom:0'>∞ Group Theory Tutor</h1>"
        "<p style='color:#8a7d65; font-style:italic; margin-top:4px'>"
        "A RAG-powered assistant for undergraduate abstract algebra</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    for msg in st.session_state.messages:
        render_message(msg)

    pending    = st.session_state.pop("pending_query", None)
    user_input = st.chat_input("Ask about group theory…") or pending

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        render_message({"role": "user", "content": user_input})

        with st.spinner("Searching knowledge base…"):
            response = st.session_state.chatbot.chat(user_input)

        msg = {
            "role":        "assistant",
            "content":     response.answer,
            "sources":     response.sources,
            "source_urls": response.source_urls,
            "used_qa":     response.used_qa,
            "confidence":  response.confidence,
        }
        st.session_state.messages.append(msg)
        render_message(msg)
        st.rerun()


if __name__ == "__main__":
    main()