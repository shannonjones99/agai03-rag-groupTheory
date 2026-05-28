import os
import json
import re
import time
import pandas as pd
import anthropic
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

RAW_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
OUTPUT_CSV    = os.path.join(os.path.dirname(__file__), '..', 'data', 'qa_dataset.csv')

# How many Q/A pairs to request per page chunk
QA_PER_CHUNK = 2

# Max characters per chunk sent to the LLM (keeps prompts manageable)
CHUNK_SIZE = 3000

QA_SYSTEM_PROMPT = """You are an expert mathematician specialising in abstract algebra and group theory.
Your task is to generate high-quality question-answer pairs from the provided text,
suitable for undergraduate students (first or second year).

Rules:
- Generate exactly the number of Q/A pairs requested.
- Cover a mix of: definitions, theorems, examples, and conceptual understanding.
- Questions should be clear and self-contained (no "according to the text…").
- Answers should be accurate, concise (2–5 sentences), and use correct mathematical notation.
- Use Unicode for symbols where helpful: ∀, ∃, ∈, ⊆, ≅, φ, etc.
- Do NOT invent facts not supported by the provided text.

Respond ONLY with a JSON array. No preamble, no markdown fences. Example format:
[
  {
    "question": "What is a group homomorphism?",
    "answer": "A group homomorphism is a map φ: G → H between two groups such that φ(ab) = φ(a)φ(b) for all a, b ∈ G. It preserves the group structure. The kernel of φ is the set of elements mapped to the identity in H, and is always a normal subgroup of G."
  }
]
"""

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> list[str]:
    """Split text into overlapping chunks at paragraph boundaries."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, current = [], ""
    for para in paragraphs:
        if len(current) + len(para) > chunk_size and current:
            chunks.append(current.strip())
            # 20% overlap: keep last paragraph of previous chunk
            last_para = current.strip().split("\n\n")[-1]
            current = last_para + "\n\n" + para
        else:
            current += "\n\n" + para
    if current.strip():
        chunks.append(current.strip())
    return chunks

def generate_qa_for_chunk(client: anthropic.Anthropic,chunk: str,source_page: str,n: int = QA_PER_CHUNK,) -> list[dict]:
    """Call Claude to generate n Q/A pairs from a text chunk."""

    prompt = (
        f"Generate {n} question-answer pairs from the following group theory text.\n\n"
        f"NOTE: To help you manage the content, it has been split into chunks. Each chunk may be a section of text, a definition, an example, or a theorem from the original Wikipedia page. Use the information in each chunk to generate relevant Q/A pairs, but do not assume any information beyond what is provided in that chunk. This chunk of text is from a Wikipedia page about {source_page}.\n\n"
        f"TEXT:\n{chunk}"
    )
    try:
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=2000,
            system=QA_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()

        # Strip accidental markdown fences
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        pairs = json.loads(raw)

        # Attach source page
        for pair in pairs:
            pair["source_page"] = source_page
        return pairs
    
    except (json.JSONDecodeError, anthropic.APIError) as e:
        print(f"    [ERROR] QA generation failed for chunk: {e}")
        return []

def run_qa_generator(delay: float = 1.0) -> pd.DataFrame:
    """Generate Q/A pairs for all scraped pages and save to CSV + JSON."""

    os.makedirs(PROC_DIR, exist_ok=True)
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    txt_files = sorted(
        file for file in os.listdir(RAW_DIR) if file.endswith(".txt")
    )

    if not txt_files:
        raise FileNotFoundError(
            f"No .txt files found in {RAW_DIR}. Run scraper.py first."
        )

    print(f"\n{'='*60}")
    print(f"  Group Theory RAG — Q/A Generator")
    print(f"  Files to process: {len(txt_files)}")
    print(f"{'='*60}\n")

    all_pairs = []

    for filename in txt_files:
        filepath = os.path.join(RAW_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # Extract display name from first line (e.g. "# Normal Subgroup")
        first_line = content.split("\n")[0]
        source_page = first_line.lstrip("# ").strip() or filename

        print(f"Processing: {source_page}")
        chunks = chunk_text(content)
        page_pairs = []

        for i, chunk in enumerate(chunks):
            pairs = generate_qa_for_chunk(client, chunk, source_page, n=QA_PER_CHUNK)
            page_pairs.extend(pairs)
            print(f"  Chunk {i+1}/{len(chunks)}: +{len(pairs)} pairs")
            time.sleep(delay)

        all_pairs.extend(page_pairs)
        print(f"  → {len(page_pairs)} pairs total for {source_page}\n")

    df = pd.DataFrame(all_pairs, columns=["question", "answer", "source_page"])
    # Deduplicate on question text (case-insensitive)
    df = df.drop_duplicates(subset=["question"])
    df = df.reset_index(drop=True)

    csv_path  = os.path.join(PROC_DIR, "qa_dataset.csv")
    df.to_csv(csv_path,   index=False)

    print(f"{'='*60}")
    print(f"  Q/A generation complete!")
    print(f"  Total pairs: {len(df)}")
    print(f"  Saved → {csv_path}")
    print(f"{'='*60}\n")

    return df

if __name__ == "__main__":
    run_qa_generator()