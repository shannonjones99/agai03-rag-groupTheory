from __future__ import annotations
import os
from dataclasses import dataclass, field
import anthropic
from retriever import HybridRetriever, RetrievalResult

MAX_HISTORY = 6   # number of past (user, assistant) turns to keep in context

SYSTEM_PROMPT = """You are a knowledgeable and patient tutor specialising in abstract algebra,
specifically group theory at undergraduate level (years 1–2).

Your answers should be:
- Mathematically precise but accessible to a first/second-year student
- Structured clearly, with definitions before examples
- Rich with notation where helpful: use ∀, ∃, ∈, ⊆, ≅, ⟨g⟩, ker φ, im φ, G/N, etc.
- Honest about the limits of what the provided context contains

You will be given either:
  (a) A direct answer from a curated Q/A database — present this naturally, 
      expanding slightly for clarity if needed.
  (b) Relevant text chunks from Wikipedia — synthesise these into a clear, 
      coherent answer without repeating the raw text verbatim.

If the context does not contain enough information to answer confidently, 
say so clearly and suggest what the student might look up."""


@dataclass
class ChatMessage:
    role:    str   # "user" or "assistant"
    content: str


@dataclass
class ChatResponse:
    answer:      str
    sources:     list[str]
    source_urls: list[str]
    used_qa:     bool
    qa_question: str | None
    confidence:  float
    retrieval:   RetrievalResult


class GroupTheoryChatbot:
    """
    Orchestrates retrieval + LLM generation with conversation memory.

    Parameters
    ----------
    retriever : HybridRetriever
        The two-stage retriever over Chroma collections.
    """

    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever
        self.client    = anthropic.Anthropic()
        self.history:  list[ChatMessage] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(self, user_query: str) -> ChatResponse:
        """
        Process a user query end-to-end.
        Retrieves context, calls Claude, updates history.
        """
        # 1. Retrieve context via hybrid search
        result = self.retriever.retrieve(user_query)

        # 2. Build the user-turn content with injected context
        augmented_query = self._build_augmented_query(user_query, result)

        # 3. Assemble message history for Claude
        messages = self._build_messages(augmented_query)

        # 4. Call Claude
        response = self.client.messages.create(
            model     = "claude-sonnet-4-20250514",
            max_tokens= 1024,
            system    = SYSTEM_PROMPT,
            messages  = messages,
        )
        answer = response.content[0].text.strip()

        # 5. Update memory
        self.history.append(ChatMessage("user",      user_query))
        self.history.append(ChatMessage("assistant", answer))
        # Keep history bounded
        if len(self.history) > MAX_HISTORY * 2:
            self.history = self.history[-(MAX_HISTORY * 2):]

        return ChatResponse(
            answer      = answer,
            sources     = result.sources,
            source_urls = result.source_urls,
            used_qa     = result.used_qa,
            qa_question = result.qa_question,
            confidence  = result.confidence,
            retrieval   = result,
        )

    def clear_history(self):
        """Reset conversation memory."""
        self.history = []

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_augmented_query(self, query: str, result: RetrievalResult) -> str:
        """Inject retrieved context into the user query."""
        if result.used_qa and result.answer:
            context_block = (
                f"[CONTEXT — from curated Q/A database]\n"
                f"Matched question: {result.qa_question}\n"
                f"Stored answer: {result.answer}\n"
                f"Source: {', '.join(result.sources)}"
            )
        elif result.context:
            chunks_text  = "\n\n---\n\n".join(result.context)
            sources_text = ", ".join(result.sources)
            context_block = (
                f"[CONTEXT — from Wikipedia ({sources_text})]\n"
                f"{chunks_text}"
            )
        else:
            context_block = "[CONTEXT — no relevant material found in the knowledge base]"

        return f"{context_block}\n\n[STUDENT QUESTION]\n{query}"

    def _build_messages(self, augmented_query: str) -> list[dict]:
        """Build the messages list including conversation history."""
        messages = []

        # Add recent history
        for msg in self.history:
            messages.append({"role": msg.role, "content": msg.content})

        # Add current query
        messages.append({"role": "user", "content": augmented_query})
        return messages