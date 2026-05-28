
import os
import json
import time
import re
import requests
from bs4 import BeautifulSoup


PAGES = [
    ("https://en.wikipedia.org/wiki/Group_theory", "Group Theory"),
    ("https://en.wikipedia.org/wiki/Abelian_group", "Abelian Group"),
    ("https://en.wikipedia.org/wiki/Permutation_group", "Permutation Group"),
    ("https://en.wikipedia.org/wiki/Normal_subgroup", "Normal Subgroup"),
    ("https://en.wikipedia.org/wiki/Cyclic_group", "Cyclic Group"),
    ("https://en.wikipedia.org/wiki/Cayley_table", "Cayley Table"),
    ("https://en.wikipedia.org/wiki/Symmetric_group", "Symmetric Group"),
    ("https://en.wikipedia.org/wiki/Dihedral_group", "Dihedral Group"),
    ("https://en.wikipedia.org/wiki/Coset", "Coset"),
    ("https://en.wikipedia.org/wiki/Lagrange%27s_theorem_(group_theory)", "Lagrange's Theorem"),
    ("https://en.wikipedia.org/wiki/Quotient_group", "Quotient Group"),
    ("https://en.wikipedia.org/wiki/Subgroup", "Subgroup"),
    ("https://en.wikipedia.org/wiki/Group_homomorphism", "Group Homomorphism"),
    ("https://en.wikipedia.org/wiki/Isomorphism_theorems", "Isomorphism Theorems")

]

RAW_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

# Tags whose content we always discard
SKIP_TAGS = {"script", "style", "sup", "table", "figure", "figcaption"}

# Section headings that are boilerplate / not useful for RAG
SKIP_SECTIONS = {
    "references", "external links", "see also", "notes",
    "further reading", "bibliography", "footnotes",
}

def clean_text(soup: BeautifulSoup) -> str:
    """Extract readable information from a Wikipedia page's soup object."""

    # Remove unwanted tags in-place
    for tag in soup.find_all(SKIP_TAGS):
        tag.decompose()

    # Remove Wikipedia edit-section links, navboxes, infoboxes, etc.
    for cls in ["mw-editsection", "navbox", "infobox", "reflist",
                "mw-references-wrap", "hatnote", "sidebar"]:
        for el in soup.find_all(class_=cls):
            el.decompose()

    content_div = soup.find("div", {"id": "mw-content-text"})
    if not content_div:
        return ""

    chunks = []
    current_section = ""

    for element in content_div.find_all(["h2", "h3", "h4", "p", "li", "dl"]):
        if element.name in ("h2", "h3", "h4"):
            heading = element.get_text(" ", strip=True).lower()
            heading_size = int(element.name[1])
            current_section = heading
            if current_section not in SKIP_SECTIONS:
                chunks.append(f"\n{heading_size * '#'} {element.get_text(' ', strip=True)}\n")
        else:

            text = element.get_text(" ", strip=True)
            # Drop very short or purely numeric lines
            if len(text) < 30 or re.fullmatch(r"[\d\s\.\,\-]+", text):
                continue
            chunks.append(text)

    raw = "\n".join(chunks)

    # Collapse multiple blank lines
    raw = re.sub(r"\n{3,}", "\n\n", raw)

    return raw.strip()

def scrape_page(url: str, display_name: str) -> dict | None:
    """Fetch and clean one Wikipedia page. Returns a metadata dict or None."""

    try:
        resp = requests.get(
            url,
            headers={"User-Agent": "GroupTheoryRAGBot/1.0 (educational project)"},
            timeout=15,
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  [ERROR] Could not fetch {url}: {e}")
        return None

    soup = BeautifulSoup(resp.text, "html.parser")
    text = clean_text(soup)

    if len(text) < 200:
        print(f"  [WARN]  Very little text extracted for {display_name}, skipping.")
        return None

    # Save individual file
    safe_name = re.sub(r"[^\w\-]", "_", display_name.lower())
    filepath = os.path.join(RAW_DIR, f"{safe_name}.txt")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"# {display_name}\n")
        f.write(f"Source: {url}\n\n")
        f.write(text)

    word_count = len(text.split())
    print(f"  [OK]    {display_name:35s} — {word_count:,} words → {filepath}")

    return {
        "display_name": display_name,
        "url": url,
        "text": text,
        "word_count": word_count,
    }


def run_scraper(delay: float = 1.5) -> list[dict]:
    """Scrape all target pages with a polite delay between requests."""

    os.makedirs(RAW_DIR,  exist_ok=True)
    os.makedirs(PROC_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Group Theory RAG — Wikipedia Scraper")
    print(f"  Target: {len(PAGES)} pages   Delay: {delay}s between requests")
    print(f"{'='*60}\n")

    manifest = []
    for url, name in PAGES:
        print(f"Fetching: {name}")
        result = scrape_page(url, name)
        if result:
            manifest.append(result)
        time.sleep(delay)

    # Save manifest
    manifest_path = os.path.join(PROC_DIR, "pages_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    total_words = sum(p["word_count"] for p in manifest)
    print(f"\n{'='*60}")
    print(f"  Scraped {len(manifest)}/{len(PAGES)} pages successfully")
    print(f"  Total words: {total_words:,}")
    print(f"  Manifest saved → {manifest_path}")
    print(f"{'='*60}\n")

    return manifest


if __name__ == "__main__":
    run_scraper()