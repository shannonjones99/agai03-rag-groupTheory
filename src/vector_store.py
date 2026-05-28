import os
import json
import re
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

RAW_DIR    = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROC_DIR   = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db")

CHUNK_SIZE    = 500   # words per document chunk
CHUNK_OVERLAP = 50    # word overlap between chunks

# Using a lightweight but effective sentence-transformer model
EMBED_MODEL = "all-MiniLM-L6-v2"


def get_embedding_function():
    """Return the SentenceTransformer embedding function for Chroma."""
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )


def word_chunks(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Split text into overlapping word-level chunks."""
    words = text.split()
    step  = size - overlap
    for start in range(0, max(1, len(words) - overlap), step):
        chunk = " ".join(words[start : start + size])
        if len(chunk.strip()) > 100:
            yield chunk


def safe_id(base: str, index: int) -> str:
    """Create a Chroma-safe document ID."""
    clean = re.sub(r"[^\w\-]", "_", base.lower())[:40]
    return f"{clean}_{index:04d}"


def build_document_collection(client: chromadb.Client, ef) -> chromadb.Collection:
    """Chunk all raw .txt files and store in the 'documents' collection."""
    collection = client.get_or_create_collection(
        name="documents",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    txt_files = sorted(f for f in os.listdir(RAW_DIR) if f.endswith(".txt"))
    print(f"\n  Building 'documents' collection from {len(txt_files)} files...")

    total_chunks = 0
    for filename in txt_files:
        filepath = os.path.join(RAW_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        first_line   = content.split("\n")[0]
        source_page  = first_line.lstrip("# ").strip() or filename
        source_url   = ""
        for line in content.split("\n")[:3]:
            if line.startswith("Source:"):
                source_url = line.replace("Source:", "").strip()
                break

        ids, docs, metas = [], [], []
        for i, chunk in enumerate(word_chunks(content)):
            ids.append(safe_id(source_page, i))
            docs.append(chunk)
            metas.append({"source_page": source_page, "source_url": source_url, "chunk_index": i})
            total_chunks += 1

        if ids:
            # Upsert in batches of 100
            batch_size = 100
            for b in range(0, len(ids), batch_size):
                collection.upsert(
                    ids       = ids[b : b + batch_size],
                    documents = docs[b : b + batch_size],
                    metadatas = metas[b : b + batch_size],
                )

        print(f"    {source_page:35s} → {len(ids)} chunks")

    print(f"  Total document chunks: {total_chunks}")
    return collection


def build_qa_collection(client: chromadb.Client, ef) -> chromadb.Collection:
    """Store Q/A questions in the 'qa_pairs' collection."""
    collection = client.get_or_create_collection(
        name="qa_pairs",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    qa_path = os.path.join(PROC_DIR, "qa_dataset.csv")
    if not os.path.exists(qa_path):
        print(f"  [WARN] No Q/A dataset found at {qa_path}. Skipping QA collection.")
        return collection

    df = pd.read_csv(qa_path).dropna(subset=["question", "answer"])
    print(f"\n  Building 'qa_pairs' collection from {len(df)} pairs...")

    ids, docs, metas = [], [], []
    for i, row in df.iterrows():
        ids.append(f"qa_{i:05d}")
        docs.append(str(row["question"]))
        metas.append({
            "answer":      str(row["answer"]),
            "source_page": str(row.get("source_page", "")),
        })

    batch_size = 100
    for b in range(0, len(ids), batch_size):
        collection.upsert(
            ids       = ids[b : b + batch_size],
            documents = docs[b : b + batch_size],
            metadatas = metas[b : b + batch_size],
        )

    print(f"  Total Q/A pairs indexed: {len(df)}")
    return collection


def build_vector_store() -> tuple[chromadb.Collection, chromadb.Collection]:
    """Build both Chroma collections. Returns (doc_collection, qa_collection)."""
    os.makedirs(CHROMA_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Group Theory RAG — Vector Store Builder")
    print(f"  Embedding model : {EMBED_MODEL}")
    print(f"  Chroma path     : {CHROMA_DIR}")
    print(f"{'='*60}")

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    ef     = get_embedding_function()

    doc_col = build_document_collection(client, ef)
    qa_col  = build_qa_collection(client, ef)

    print(f"\n{'='*60}")
    print(f"  Vector store built successfully!")
    print(f"  Documents collection : {doc_col.count()} chunks")
    print(f"  QA collection        : {qa_col.count()} pairs")
    print(f"{'='*60}\n")

    return doc_col, qa_col


def load_vector_store() -> tuple[chromadb.Collection, chromadb.Collection]:
    """Load existing Chroma collections (must call build_vector_store first)."""
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    ef     = get_embedding_function()
    doc_col = client.get_collection("documents", embedding_function=ef)
    qa_col  = client.get_collection("qa_pairs",  embedding_function=ef)
    return doc_col, qa_col


if __name__ == "__main__":
    build_vector_store()