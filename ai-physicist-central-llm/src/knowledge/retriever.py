"""Physics knowledge retrieval system with optional embedding backend."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

logger = logging.getLogger(__name__)

# Optional deps
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    _HAS_ST = True
except Exception:
    SentenceTransformer = None  # type: ignore
    _HAS_ST = False

try:
    import faiss  # type: ignore
    _HAS_FAISS = True
except Exception:
    faiss = None  # type: ignore
    _HAS_FAISS = False


class PhysicsRetriever:
    """
    Retrieval system for physics knowledge.

    Two modes:
      1) Embedding-based (SentenceTransformers + FAISS) if deps are available.
      2) Keyword fallback (works everywhere), deterministic and fast.
    """

    def __init__(self, embedding_model: str = "BAAI/bge-small-en-v1.5") -> None:
        self.embedding_model_name = embedding_model
        self.documents: List[Dict[str, Any]] = []
        self._index = None
        self._embeddings = None
        self._backend = "keyword"  # or "embeddings"

        if _HAS_ST and _HAS_FAISS:
            try:
                self._st_model = SentenceTransformer(self.embedding_model_name)
                self._backend = "embeddings"
                logger.info(f"Retriever using embeddings backend: {self.embedding_model_name}")
            except Exception as e:
                logger.warning(f"Failed to init embedding model; falling back to keyword. Error: {e}")
                self._st_model = None
                self._backend = "keyword"
        else:
            self._st_model = None
            logger.info("Retriever using keyword backend (sentence-transformers/faiss not available).")

    # ----------------- Loading -----------------

    def load_corpus(self, corpus: Union[str, Path, List[Dict[str, Any]], Dict[str, Any]]) -> None:
        """
        Load documents from:
         - path to JSON file
         - list of dicts
         - dict with {"documents": [...]}
        """
        if isinstance(corpus, (str, Path)):
            path = Path(corpus)
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = corpus

        if isinstance(data, list):
            self.documents = data
        elif isinstance(data, dict):
            if "documents" in data and isinstance(data["documents"], list):
                self.documents = data["documents"]
            else:
                # A dict per document? Normalize to list
                self.documents = [data]
        else:
            raise ValueError("Unsupported corpus format")

        # Normalize minimal fields
        for d in self.documents:
            d.setdefault("id", d.get("doc_id", ""))
            d.setdefault("title", d.get("title", ""))
            d.setdefault("content", d.get("content", ""))

        logger.info(f"Loaded {len(self.documents)} documents into retriever")

    # ----------------- Indexing -----------------

    def build_index(self) -> None:
        """Build an index for the chosen backend."""
        if not self.documents:
            logger.warning("No documents loaded; using small default corpus.")
            self._load_default_corpus()

        if self._backend == "embeddings" and self._st_model is not None:
            try:
                texts = [d.get("content", "") or (d.get("title", "")) for d in self.documents]
                self._embeddings = self._st_model.encode(texts, show_progress_bar=False)
                import numpy as np  # local import to avoid hard dep

                dim = self._embeddings.shape[1]
                self._index = faiss.IndexFlatL2(dim)  # type: ignore[attr-defined]
                self._index.add(self._embeddings.astype("float32"))
                logger.info(f"FAISS index built with {len(self.documents)} docs, dim={dim}")
            except Exception as e:
                logger.warning(f"Embedding index failed; switching to keyword backend. Error: {e}")
                self._backend = "keyword"
                self._index = None
                self._embeddings = None
        else:
            # Keyword backend doesn't need an index
            self._index = None
            self._embeddings = None
            logger.info("Keyword backend ready (no index needed).")

    # ----------------- Search -----------------

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Return top-k documents (full dicts). Fields: id/title/content/...
        """
        if not self.documents:
            self._load_default_corpus()

        if self._backend == "embeddings" and self._st_model is not None and self._index is not None:
            return self._search_embeddings(query, k)
        return self._search_keyword(query, k)

    # ----------------- Backends -----------------

    def _search_embeddings(self, query: str, k: int) -> List[Dict[str, Any]]:
        import numpy as np  # local import

        q_emb = self._st_model.encode([query])  # type: ignore[union-attr]
        distances, indices = self._index.search(q_emb.astype("float32"), k)  # type: ignore[union-attr]
        hits: List[Dict[str, Any]] = []
        for idx in indices[0]:
            if 0 <= idx < len(self.documents):
                hits.append(self.documents[idx])
        return hits

    def _search_keyword(self, query: str, k: int) -> List[Dict[str, Any]]:
        q = query.lower().strip()
        scored: List[tuple[Dict[str, Any], int]] = []

        for d in self.documents:
            text = (d.get("title", "") + " " + d.get("content", "")).lower()
            score = 0
            if q and q in text:
                score += 10
            for w in q.split():
                score += text.count(w)
            if score > 0:
                scored.append((d, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [d for d, _ in scored[:k]]

    # ----------------- Defaults -----------------

    def _load_default_corpus(self) -> None:
        self.documents = [
            {"id": "1", "title": "Newton 2nd Law", "content": "Newton's second law: F = ma"},
            {"id": "2", "title": "Pendulum", "content": "Period of pendulum: T = 2π√(L/g)"},
            {"id": "3", "title": "Kinetic Energy", "content": "Kinetic energy: KE = 1/2 m v^2"},
            {"id": "4", "title": "Coulomb's Law", "content": "Electrostatic force: F = k q1 q2 / r^2"},
            {"id": "5", "title": "Ideal Gas Law", "content": "Equation of state: PV = nRT"},
        ]
        logger.info("Loaded built-in mini corpus as fallback.")
