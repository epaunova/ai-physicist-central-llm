"""
Physics Corpus Module
=====================
Management of physics knowledge documents for RAG.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Iterable

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Single document in the physics corpus."""
    id: str
    title: str
    content: str
    category: str = "general"
    source: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    timestamp: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.id:
            self.id = self._generate_id()
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def _generate_id(self) -> str:
        content_hash = hashlib.md5((self.title + self.content).encode("utf-8")).hexdigest()[:8]
        return f"doc_{content_hash}"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        return cls(**data)

    def get_text(self, include_title: bool = True) -> str:
        return f"{self.title}\n\n{self.content}" if include_title else self.content

    def extract_keywords(self, max_keywords: int = 10) -> List[str]:
        text = (self.title + " " + self.content).lower()
        stopwords = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at",
            "to", "for", "of", "with", "by", "from", "is", "are", "was", "were"
        }
        words = re.findall(r"\b[a-z]+\b", text)
        words = [w for w in words if w not in stopwords and len(w) > 3]
        word_freq = Counter(words)
        return [w for w, _ in word_freq.most_common(max_keywords)]


class PhysicsCorpus:
    """
    Manages a collection of physics documents.

    Responsibilities:
    - Document storage and retrieval
    - Corpus statistics
    - Import/export (JSON)
    - Preprocessing
    """

    def __init__(self, name: str = "Physics Knowledge Base") -> None:
        self.name = name
        self.documents: List[Document] = []
        self.document_index: Dict[str, Document] = {}
        self.category_index: Dict[str, List[Document]] = defaultdict(list)
        self.metadata: Dict[str, Any] = {
            "created": datetime.now().isoformat(),
            "version": "1.0",
            "description": "Physics knowledge corpus for RAG",
        }
        self.stats = {
            "total_documents": 0,
            "total_words": 0,
            "categories": set(),
            "sources": set(),
        }

    # ---------- add / load / save ----------

    def add_document(
        self,
        title: str,
        content: str,
        category: str = "general",
        source: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None,
        document_id: Optional[str] = None,
    ) -> Document:
        doc = Document(
            id=document_id or "",
            title=title,
            content=content,
            category=category,
            source=source,
            metadata=metadata or {},
        )
        self._add(doc)
        return doc

    def _add(self, doc: Document) -> None:
        self.documents.append(doc)
        self.document_index[doc.id] = doc
        self.category_index[doc.category].append(doc)
        self._update_stats(doc)
        logger.info(f"Added document: {doc.id} - {doc.title}")

    def add_documents(self, documents: Iterable[Union[Dict[str, Any], Document]]) -> List[Document]:
        added: List[Document] = []
        for item in documents:
            if isinstance(item, Document):
                self._add(item)
                added.append(item)
            else:
                added.append(
                    self.add_document(
                        title=item.get("title", ""),
                        content=item.get("content", ""),
                        category=item.get("category", "general"),
                        source=item.get("source", "unknown"),
                        metadata=item.get("metadata", {}),
                        document_id=item.get("id"),
                    )
                )
        logger.info(f"Added {len(added)} documents to corpus")
        return added

    def load_from_file(self, filepath: str) -> None:
        path = Path(filepath)
        if not path.exists():
            logger.warning(f"Corpus file not found: {path}")
            # Fallback to repo default
            package_corpus = Path(__file__).resolve().parents[2] / "data/corpus/physics_abstracts.json"
            if package_corpus.exists():
                path = package_corpus
            else:
                logger.error("No corpus file found")
                return

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            self.add_documents(data)
        elif isinstance(data, dict):
            if "documents" in data:
                self.add_documents(data["documents"])
            if "metadata" in data:
                self.metadata.update(data["metadata"])

        logger.info(f"Loaded {len(self.documents)} documents from {path}")

    def save_to_file(self, filepath: str) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "metadata": self.metadata,
            "documents": [d.to_dict() for d in self.documents],
            "statistics": {
                "total_documents": self.stats["total_documents"],
                "total_words": self.stats["total_words"],
                "categories": list(self.stats["categories"]),
                "sources": list(self.stats["sources"]),
            },
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved corpus to {path}")

    # ---------- queries ----------

    def get_document(self, doc_id: str) -> Optional[Document]:
        return self.document_index.get(doc_id)

    def get_documents_by_category(self, category: str) -> List[Document]:
        return self.category_index.get(category, [])

    def search_documents(self, query: str, category: Optional[str] = None, limit: Optional[int] = None) -> List[Document]:
        """Very simple keyword search (no embeddings)."""
        q = query.lower()
        results: List[tuple[Document, int]] = []

        pool = self.category_index[category] if category else self.documents
        for doc in pool:
            text = (doc.title + " " + doc.content).lower()
            score = 0
            if q in text:
                score += 10
            for w in q.split():
                count = text.count(w)
                if count:
                    score += count
            if score > 0:
                results.append((doc, score))

        results.sort(key=lambda x: x[1], reverse=True)
        if limit:
            results = results[:limit]
        return [d for d, _ in results]

    # ---------- helpers ----------

    def _update_stats(self, doc: Document) -> None:
        self.stats["total_documents"] += 1
        self.stats["total_words"] += len(doc.content.split())
        self.stats["categories"].add(doc.category)
        self.stats["sources"].add(doc.source)

    def preprocess_documents(self, lowercase: bool = True, remove_special: bool = True, min_length: int = 10) -> None:
        processed: List[Document] = []
        for doc in self.documents:
            content = doc.content
            if lowercase:
                content = content.lower()
            if remove_special:
                content = re.sub(r"[^a-zA-Z0-9\s\.\,\!\?\-\(\)]", "", content)
            if len(content.split()) >= min_length:
                doc.content = content
                processed.append(doc)
            else:
                logger.warning(f"Removing short document: {doc.id}")
        self.documents = processed
        logger.info(f"Preprocessed {len(processed)} documents")

    def filter_documents(
        self,
        categories: Optional[List[str]] = None,
        sources: Optional[List[str]] = None,
        min_words: Optional[int] = None,
        max_words: Optional[int] = None,
    ) -> "PhysicsCorpus":
        filt = PhysicsCorpus(name=f"{self.name} (filtered)")
        for doc in self.documents:
            if categories and doc.category not in categories:
                continue
            if sources and doc.source not in sources:
                continue
            wc = len(doc.content.split())
            if min_words and wc < min_words:
                continue
            if max_words and wc > max_words:
                continue
            filt.add_document(
                title=doc.title,
                content=doc.content,
                category=doc.category,
                source=doc.source,
                metadata=doc.metadata,
                document_id=doc.id,
            )
        logger.info(f"Created filtered corpus with {len(filt.documents)} documents")
        return filt

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "total_documents": self.stats["total_documents"],
            "total_words": self.stats["total_words"],
            "avg_words_per_doc": (
                self.stats["total_words"] / self.stats["total_documents"] if self.stats["total_documents"] else 0
            ),
            "categories": list(self.stats["categories"]),
            "num_categories": len(self.stats["categories"]),
            "sources": list(self.stats["sources"]),
            "documents_by_category": {c: len(docs) for c, docs in self.category_index.items()},
        }

    # ---------- dunder ----------

    def __len__(self) -> int:
        return len(self.documents)

    def __getitem__(self, index: int) -> Document:
        return self.documents[index]

    def __iter__(self):
        return iter(self.documents)

    def __repr__(self) -> str:
        return f"PhysicsCorpus(name='{self.name}', documents={len(self.documents)}, categories={len(self.stats['categories'])})"


# Convenience functions
def load_default_corpus() -> PhysicsCorpus:
    return CorpusBuilder.create_default_corpus()


class CorpusBuilder:
    """Create a physics corpus from various sources."""

    @staticmethod
    def from_textbook(filepath: str, chapter_separator: str = "Chapter", min_section_length: int = 100) -> PhysicsCorpus:
        corpus = PhysicsCorpus(name="Textbook Corpus")
        with open(filepath, "r", encoding="utf-8") as f:
            raw = f.read()
        chapters = raw.split(chapter_separator)
        for i, ch in enumerate(chapters[1:], 1):
            lines = ch.strip().split("\n")
            if not lines:
                continue
            title = f"Chapter {i}: {lines[0][:50]}"
            body = "\n".join(lines[1:])
            if len(body) >= min_section_length:
                corpus.add_document(title=title, content=body, category="textbook", source=filepath)
        return corpus

    @staticmethod
    def from_arxiv_papers(papers: List[Dict[str, str]]) -> PhysicsCorpus:
        corpus = PhysicsCorpus(name="arXiv Papers")
        for p in papers:
            corpus.add_document(
                title=p.get("title", ""),
                content=p.get("abstract", ""),
                category=p.get("category", "physics"),
                source="arXiv",
                metadata={"authors": p.get("authors", []), "arxiv_id": p.get("id", ""), "published": p.get("published", "")},
            )
        return corpus

    @staticmethod
    def create_default_corpus() -> PhysicsCorpus:
        corpus = PhysicsCorpus(name="Default Physics Corpus")
        corpus.add_document(
            title="Newton's Laws of Motion",
            content=(
                "Newton's three laws of motion form the foundation of classical mechanics. "
                "First law (Inertia): An object at rest stays at rest and an object in motion "
                "stays in motion unless acted upon by an external force. "
                "Second law: F = ma. Third law: For every action, there is an equal and opposite reaction."
            ),
            category="classical_mechanics",
            source="textbook",
        )
        corpus.add_document(
            title="Simple Harmonic Motion",
            content=(
                "Simple harmonic motion occurs when the restoring force is proportional to displacement. "
                "The period of a simple pendulum is T = 2π√(L/g). For a mass-spring system, T = 2π√(m/k)."
            ),
            category="classical_mechanics",
            source="textbook",
        )
        corpus.add_document(
            title="Coulomb's Law and Electric Fields",
            content=(
                "Coulomb's law: F = k q1 q2 / r^2. The electric field of a point charge is E = kQ / r^2. "
                "Field lines point away from positive charges and toward negative charges."
            ),
            category="electromagnetism",
            source="textbook",
        )
        corpus.add_document(
            title="Laws of Thermodynamics",
            content=(
                "First law: ΔU = Q - W. Second law: Entropy of an isolated system always increases. "
                "Third law: Entropy approaches zero as temperature approaches absolute zero."
            ),
            category="thermodynamics",
            source="textbook",
        )
        corpus.add_document(
            title="Heisenberg Uncertainty Principle",
            content=(
                "Δx Δp ≥ ℏ/2. This expresses a fundamental limit on simultaneous knowledge of position and momentum."
            ),
            category="quantum_mechanics",
            source="textbook",
        )
        corpus.add_document(
            title="Special Relativity",
            content=(
                "Postulates: physics is the same in all inertial frames; the speed of light c is constant. "
                "Time dilation: Δt = γ Δt0 with γ = 1/√(1 − v^2/c^2). Mass–energy equivalence: E = mc^2."
            ),
            category="relativity",
            source="textbook",
        )
        return corpus
