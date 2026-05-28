# ∞ Group Theory RAG Chatbot
### AGAI-03 Assignment — RAG-Powered Chatbot with Hybrid Q/A Retrieval

A Retrieval-Augmented Generation chatbot that answers undergraduate-level group theory questions, built on Wikipedia's group theory article cluster with a two-stage hybrid retrieval pipeline.

---

## Project Overview

| Component | Detail |
|---|---|
| **Source** | 20 Wikipedia pages on group theory topics |
| **Level** | Undergraduate years 1–2 |
| **LLM** | Claude Sonnet (Anthropic API) |
| **Embeddings** | `all-MiniLM-L6-v2` (SentenceTransformers) |
| **Vector DB** | ChromaDB (persistent, local) |
| **Retrieval** | Hybrid: Q/A semantic match → document vector fallback |
| **UI** | Streamlit |

---

## Repository Structure

```
rag-chatbot/
├── data/
│   ├── raw/                  # Scraped Wikipedia pages (.txt)
│   ├── processed/
│   │   ├── pages_manifest.json
│   │   ├── qa_dataset.csv    # Generated Q/A pairs
│   │   ├── qa_dataset.json
│   │   └── chroma_db/        # Persistent ChromaDB
├── src/
│   ├── scraper.py            # Wikipedia scraper
│   ├── qa_generator.py       # Synthetic Q/A generation via Claude
│   ├── vector_store.py       # Chroma collection builder
│   ├── retriever.py          # Hybrid retrieval logic
│   └── chatbot.py            # Orchestration + memory
├── app.py                    # Streamlit application
├── requirements.txt
└── README.md
```

---

## Setup & Running Locally

### 1. Clone and install dependencies

```bash
git clone https://github.com/yourusername/rag-chatbot-yourname
cd rag-chatbot-yourname
pip install -r requirements.txt
```

### 2. Set your API key

Create a `.env` file in the project root:

```
ANTHROPIC_API_KEY=sk-ant-...
```

Or export it directly:
```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Run the pipeline (first time only)

```bash
# Step 1: Scrape Wikipedia pages
python src/scraper.py

# Step 2: Generate Q/A pairs
python src/qa_generator.py

# Step 3: Build vector store
python src/vector_store.py
```

> **Note:** The Streamlit app will also run this pipeline automatically on first launch if the data directory is empty.

### 4. Launch the app

```bash
streamlit run app.py
```

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────┐
│         Hybrid Retriever            │
│                                     │
│  Stage 1: Q/A Semantic Search       │
│  ┌─────────────────────────────┐    │
│  │  Embed query → cosine sim   │    │
│  │  against 100–200 Q/A pairs  │    │
│  │  (ChromaDB qa_pairs)        │    │
│  └────────────┬────────────────┘    │
│               │ confidence < 0.65?  │
│               ▼                     │
│  Stage 2: Document Vector Search    │
│  ┌─────────────────────────────┐    │
│  │  Retrieve top-4 chunks from │    │
│  │  ~500-word document chunks  │    │
│  │  (ChromaDB documents)       │    │
│  └─────────────────────────────┘    │
└──────────────────┬──────────────────┘
                   │
                   ▼
         Claude Sonnet (LLM)
         + conversation history
                   │
                   ▼
            Final Answer
         + source citations
```

---

## Scraped Pages

20 curated Wikipedia articles covering:

- Group, Subgroup, Coset, Normal Subgroup, Quotient Group
- Group Homomorphism, Isomorphism Theorems
- Cyclic Group, Symmetric Group, Dihedral Group, Abelian Group
- Lagrange's Theorem, Sylow Theorems
- Group Action, Orbit-Stabiliser Theorem
- Simple Group, Free Group, Permutation Group
- Centralizer and Normalizer, Conjugacy Class

---

## Hybrid Retrieval Logic

The retriever uses a **confidence threshold** (cosine distance < 0.35) to decide which path to take:

| Path | Trigger | Output |
|---|---|---|
| 🟡 Q/A Match | Distance to nearest Q/A question < threshold | Pre-written answer, passed to LLM for light formatting |
| 🟢 Vector Search | No confident Q/A match | Top-4 document chunks, synthesised by LLM |

This gives fast, accurate answers for common questions while gracefully handling novel queries.

---

## Author

**[Your Name]**  
AGAI-03 · May 2026
