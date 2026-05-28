from __future__ import annotations
from dataclasses import dataclass
import chromadb

# Distance threshold for Q/A match (Chroma returns cosine *distance*, 0 = identical)
# Anything below this is considered a confident Q/A hit.
QA_THRESHOLD  = 0.35

# Number of document chunks to retrieve in the fallback path
TOP_K_DOCS    = 4


@dataclass
class RetrievalResult:
    """Encapsulates what the retriever returns to the chatbot."""
    answer:       str | None     # Pre-written answer (QA hit) or None
    context:      list[str]      # Document chunks (fallback) or []
    sources:      list[str]      # Human-readable source labels
    source_urls:  list[str]      # URLs for citations
    used_qa:      bool           # True → Stage 1 hit; False → Stage 2 fallback
    qa_question:  str | None     # Matched QA question (for display)
    confidence:   float          # 1 - cosine_distance (0–1)


class HybridRetriever:
    """
    Performs hybrid retrieval over two Chroma collections.

    Parameters
    ----------
    doc_collection : chromadb.Collection
        The 'documents' collection of chunked raw text.
    qa_collection  : chromadb.Collection
        The 'qa_pairs' collection of question strings.
    qa_threshold   : float
        Maximum cosine distance to accept a Q/A hit (lower = stricter).
    top_k          : int
        Number of document chunks to return in the fallback path.
    """

    def __init__(
        self,
        doc_collection: chromadb.Collection,
        qa_collection:  chromadb.Collection,
        qa_threshold:   float = QA_THRESHOLD,
        top_k:          int   = TOP_K_DOCS,
    ):
        self.docs       = doc_collection
        self.qa         = qa_collection
        self.threshold  = qa_threshold
        self.top_k      = top_k

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, query: str) -> RetrievalResult:
        """
        Run the two-stage hybrid retrieval for a user query.

        Returns a RetrievalResult with everything the chatbot needs
        to either return a direct answer or synthesise one via LLM.
        """
        # Stage 1: try Q/A match
        qa_result = self._qa_search(query)
        if qa_result is not None:
            return qa_result

        # Stage 2: fallback to document search
        return self._doc_search(query)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _qa_search(self, query: str) -> RetrievalResult | None:
        """
        Search the Q/A collection.
        Returns a RetrievalResult if a good match is found, else None.
        """
        if self.qa.count() == 0:
            return None

        results = self.qa.query(
            query_texts = [query],
            n_results   = 1,
            include     = ["documents", "metadatas", "distances"],
        )

        distance = results["distances"][0][0]
        confidence = round(1.0 - distance, 4)

        if distance > self.threshold:
            return None   # Low confidence → fall through to Stage 2

        matched_question = results["documents"][0][0]
        meta             = results["metadatas"][0][0]
        answer           = meta.get("answer", "")
        source_page      = meta.get("source_page", "Wikipedia")

        return RetrievalResult(
            answer      = answer,
            context     = [],
            sources     = [source_page],
            source_urls = [],
            used_qa     = True,
            qa_question = matched_question,
            confidence  = confidence,
        )

    def _doc_search(self, query: str) -> RetrievalResult:
        """
        Fallback: retrieve top-k document chunks.
        Returns a RetrievalResult with context chunks for the LLM.
        """
        n = min(self.top_k, self.docs.count())
        if n == 0:
            return RetrievalResult(
                answer=None, context=[], sources=[], source_urls=[],
                used_qa=False, qa_question=None, confidence=0.0,
            )

        results = self.docs.query(
            query_texts = [query],
            n_results   = n,
            include     = ["documents", "metadatas", "distances"],
        )

        chunks      = results["documents"][0]
        metas       = results["metadatas"][0]
        distances   = results["distances"][0]

        # Best confidence = closest chunk
        confidence  = round(1.0 - distances[0], 4) if distances else 0.0

        sources     = []
        source_urls = []
        seen        = set()
        for meta in metas:
            page = meta.get("source_page", "Wikipedia")
            url  = meta.get("source_url",  "")
            if page not in seen:
                sources.append(page)
                source_urls.append(url)
                seen.add(page)

        return RetrievalResult(
            answer      = None,
            context     = chunks,
            sources     = sources,
            source_urls = source_urls,
            used_qa     = False,
            qa_question = None,
            confidence  = confidence,
        )
