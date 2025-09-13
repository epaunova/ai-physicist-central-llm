cat > src/knowledge/physics_corpus.py << 'EOF'
"""
Physics Corpus Module
=====================
Management of physics knowledge documents
"""

import json
import os
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
import logging
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Single document in the physics corpus"""
    id: str
    title: str
    content: str
    category: str = "general"
    source: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    timestamp: Optional[str] = None
    
    def __post_init__(self):
        """Post-initialization processing"""
        # Generate ID if not provided
        if not self.id:
            self.id = self._generate_id()
        
        # Set timestamp if not provided
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
    
    def _generate_id(self) -> str:
        """Generate unique ID based on content"""
        content_hash = hashlib.md5(
            (self.title + self.content).encode()
        ).hexdigest()[:8]
        return f"doc_{content_hash}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        """Create from dictionary"""
        return cls(**data)
    
    def get_text(self, include_title: bool = True) -> str:
        """Get full text for embedding"""
        if include_title:
            return f"{self.title}\n\n{self.content}"
        return self.content
    
    def extract_keywords(self, max_keywords: int = 10) -> List[str]:
        """Extract keywords from document"""
        # Simple keyword extraction (can be improved with NLTK/spaCy)
        text = (self.title + " " + self.content).lower()
        
        # Remove common words
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
            'to', 'for', 'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were'
        }
        
        # Extract words
        words = re.findall(r'\b[a-z]+\b', text)
        words = [w for w in words if w not in stopwords and len(w) > 3]
        
        # Count frequencies
        word_freq = Counter(words)
        
        # Return top keywords
        return [word for word, _ in word_freq.most_common(max_keywords)]


class PhysicsCorpus:
    """
    Manages a collection of physics documents
    
    This class handles:
    - Document storage and retrieval
    - Corpus statistics
    - Import/export functionality
    - Document preprocessing
    """
    
    def __init__(self, name: str = "Physics Knowledge Base"):
        """Initialize corpus"""
        self.name = name
        self.documents: List[Document] = []
        self.document_index: Dict[str, Document] = {}
        self.category_index: Dict[str, List[Document]] = defaultdict(list)
        self.metadata: Dict[str, Any] = {
            "created": datetime.now().isoformat(),
            "version": "1.0",
            "description": "Physics knowledge corpus for RAG"
        }
        
        # Statistics
        self.stats = {
            "total_documents": 0,
            "total_words": 0,
            "categories": set(),
            "sources": set()
        }
    
    def add_document(
        self,
        title: str,
        content: str,
        category: str = "general",
        source: str = "unknown",
        metadata: Optional[Dict] = None,
        document_id: Optional[str] = None
    ) -> Document:
        """
        Add a single document to corpus
        
        Args:
            title: Document title
            content: Document content
            category: Physics category
            source: Document source
            metadata: Additional metadata
            document_id: Optional specific ID
            
        Returns:
            Created Document object
        """
        doc = Document(
            id=document_id or "",
            title=title,
            content=content,
            category=category,
            source=source,
            metadata=metadata or {}
        )
        
        # Add to collections
        self.documents.append(doc)
        self.document_index[doc.id] = doc
        self.category_index[category].append(doc)
        
        # Update statistics
        self._update_stats(doc)
        
        logger.info(f"Added document: {doc.id} - {doc.title}")
        return doc
    
    def add_documents(self, documents: List[Union[Dict, Document]]) -> List[Document]:
        """
        Add multiple documents
        
        Args:
            documents: List of documents (dicts or Document objects)
            
        Returns:
            List of added Document objects
        """
        added = []
        
        for doc_data in documents:
            if isinstance(doc_data, Document):
                # Already a Document object
                self.documents.append(doc_data)
                self.document_index[doc_data.id] = doc_data
                self.category_index[doc_data.category].append(doc_data)
                self._update_stats(doc_data)
                added.append(doc_data)
            elif isinstance(doc_data, dict):
                # Convert from dictionary
                doc = self.add_document(
                    title=doc_data.get("title", ""),
                    content=doc_data.get("content", ""),
                    category=doc_data.get("category", "general"),
                    source=doc_data.get("source", "unknown"),
                    metadata=doc_data.get("metadata", {}),
                    document_id=doc_data.get("id")
                )
                added.append(doc)
        
        logger.info(f"Added {len(added)} documents to corpus")
        return added
    
    def _update_stats(self, doc: Document):
        """Update corpus statistics"""
        self.stats["total_documents"] += 1
        self.stats["total_words"] += len(doc.content.split())
        self.stats["categories"].add(doc.category)
        self.stats["sources"].add(doc.source)
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get document by ID"""
        return self.document_index.get(doc_id)
    
    def get_documents_by_category(self, category: str) -> List[Document]:
        """Get all documents in a category"""
        return self.category_index.get(category, [])
    
    def search_documents(
        self,
        query: str,
        category: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Document]:
        """
        Simple keyword search (without embeddings)
        
        Args:
            query: Search query
            category: Filter by category
            limit: Maximum results
            
        Returns:
            List of matching documents
        """
        query_lower = query.lower()
        results = []
        
        # Search in appropriate document set
        search_docs = (
            self.category_index[category] if category
            else self.documents
        )
        
        for doc in search_docs:
            # Simple scoring based on occurrences
            score = 0
            text_lower = (doc.title + " " + doc.content).lower()
            
            # Check for exact phrase
            if query_lower in text_lower:
                score += 10
            
            # Check for individual words
            for word in query_lower.split():
                if word in text_lower:
                    score += text_lower.count(word)
            
            if score > 0:
                results.append((doc, score))
        
        # Sort by score
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Apply limit
        if limit:
            results = results[:limit]
        
        return [doc for doc, _ in results]
    
    def load_from_file(self, filepath: str) -> None:
        """
        Load corpus from JSON file
        
        Args:
            filepath: Path to JSON file
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            logger.warning(f"Corpus file not found: {filepath}")
            # Try to load from package data
            package_corpus = Path(__file__).parent.parent.parent / "data/corpus/physics_abstracts.json"
            if package_corpus.exists():
                filepath = package_corpus
            else:
                logger.error("No corpus file found")
                return
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different formats
        if isinstance(data, list):
            # List of documents
            self.add_documents(data)
        elif isinstance(data, dict):
            # Full corpus format
            if "documents" in data:
                self.add_documents(data["documents"])
            if "metadata" in data:
                self.metadata.update(data["metadata"])
        
        logger.info(f"Loaded {len(self.documents)} documents from {filepath}")
    
    def save_to_file(self, filepath: str) -> None:
        """
        Save corpus to JSON file
        
        Args:
            filepath: Path to save file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "metadata": self.metadata,
            "documents": [doc.to_dict() for doc in self.documents],
            "statistics": {
                "total_documents": self.stats["total_documents"],
                "total_words": self.stats["total_words"],
                "categories": list(self.stats["categories"]),
                "sources": list(self.stats["sources"])
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved corpus to {filepath}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get corpus statistics"""
        return {
            "name": self.name,
            "total_documents": self.stats["total_documents"],
            "total_words": self.stats["total_words"],
            "avg_words_per_doc": (
                self.stats["total_words"] / self.stats["total_documents"]
                if self.stats["total_documents"] > 0 else 0
            ),
            "categories": list(self.stats["categories"]),
            "num_categories": len(self.stats["categories"]),
            "sources": list(self.stats["sources"]),
            "documents_by_category": {
                cat: len(docs) for cat, docs in self.category_index.items()
            }
        }
    
    def preprocess_documents(
        self,
        lowercase: bool = True,
        remove_special: bool = True,
        min_length: int = 10
    ) -> None:
        """
        Preprocess all documents
        
        Args:
            lowercase: Convert to lowercase
            remove_special: Remove special characters
            min_length: Minimum document length
        """
        processed = []
        
        for doc in self.documents:
            # Process content
            content = doc.content
            
            if lowercase:
                content = content.lower()
            
            if remove_special:
                # Keep alphanumeric and basic punctuation
                content = re.sub(r'[^a-zA-Z0-9\s\.\,\!\?\-\(\)]', '', content)
            
            # Check minimum length
            if len(content.split()) >= min_length:
                doc.content = content
                processed.append(doc)
            else:
                logger.warning(f"Removing short document: {doc.id}")
        
        self.documents = processed
        logger.info(f"Preprocessed {len(processed)} documents")
    
    def merge_corpus(self, other: "PhysicsCorpus") -> None:
        """Merge another corpus into this one"""
        for doc in other.documents:
            if doc.id not in self.document_index:
                self.add_document(
                    title=doc.title,
                    content=doc.content,
                    category=doc.category,
                    source=doc.source,
                    metadata=doc.metadata,
                    document_id=doc.id
                )
        
        logger.info(f"Merged corpus: added {len(other.documents)} documents")
    
    def filter_documents(
        self,
        categories: Optional[List[str]] = None,
        sources: Optional[List[str]] = None,
        min_words: Optional[int] = None,
        max_words: Optional[int] = None
    ) -> "PhysicsCorpus":
        """
        Create filtered corpus
        
        Args:
            categories: Categories to include
            sources: Sources to include
            min_words: Minimum word count
            max_words: Maximum word count
            
        Returns:
            New filtered PhysicsCorpus
        """
        filtered_corpus = PhysicsCorpus(name=f"{self.name} (filtered)")
        
        for doc in self.documents:
            # Apply filters
            if categories and doc.category not in categories:
                continue
            if sources and doc.source not in sources:
                continue
            
            word_count = len(doc.content.split())
            if min_words and word_count < min_words:
                continue
            if max_words and word_count > max_words:
                continue
            
            # Add to filtered corpus
            filtered_corpus.add_document(
                title=doc.title,
                content=doc.content,
                category=doc.category,
                source=doc.source,
                metadata=doc.metadata,
                document_id=doc.id
            )
        
        logger.info(f"Created filtered corpus with {len(filtered_corpus.documents)} documents")
        return filtered_corpus
    
    def __len__(self) -> int:
        """Number of documents in corpus"""
        return len(self.documents)
    
    def __getitem__(self, index: int) -> Document:
        """Get document by index"""
        return self.documents[index]
    
    def __iter__(self):
        """Iterate over documents"""
        return iter(self.documents)
    
    def __repr__(self) -> str:
        return (
            f"PhysicsCorpus(name='{self.name}', "
            f"documents={len(self.documents)}, "
            f"categories={len(self.stats['categories'])})"
        )


class CorpusBuilder:
    """
    Builder for creating physics corpus from various sources
    """
    
    @staticmethod
    def from_textbook(
        filepath: str,
        chapter_separator: str = "Chapter",
        min_section_length: int = 100
    ) -> PhysicsCorpus:
        """Create corpus from textbook file"""
        corpus = PhysicsCorpus(name="Textbook Corpus")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split into chapters
        chapters = content.split(chapter_separator)
        
        for i, chapter in enumerate(chapters[1:], 1):  # Skip before first chapter
            lines = chapter.strip().split('\n')
            if not lines:
                continue
            
            title = f"Chapter {i}: {lines[0][:50]}"
            content = '\n'.join(lines[1:])
            
            if len(content) >= min_section_length:
                corpus.add_document(
                    title=title,
                    content=content,
                    category="textbook",
                    source=filepath
                )
        
        return corpus
    
    @staticmethod
    def from_arxiv_papers(papers: List[Dict[str, str]]) -> PhysicsCorpus:
        """Create corpus from arXiv papers"""
        corpus = PhysicsCorpus(name="arXiv Papers")
        
        for paper in papers:
            corpus.add_document(
                title=paper.get("title", ""),
                content=paper.get("abstract", ""),
                category=paper.get("category", "physics"),
                source="arXiv",
                metadata={
                    "authors": paper.get("authors", []),
                    "arxiv_id": paper.get("id", ""),
                    "published": paper.get("published", "")
                }
            )
        
        return corpus
    
    @staticmethod
    def create_default_corpus() -> PhysicsCorpus:
        """Create default physics corpus with common topics"""
        corpus = PhysicsCorpus(name="Default Physics Corpus")
        
        # Classical Mechanics
        corpus.add_document(
            title="Newton's Laws of Motion",
            content="Newton's three laws of motion form the foundation of classical mechanics. "
                   "First law (Inertia): An object at rest stays at rest and an object in motion "
                   "stays in motion unless acted upon by an external force. "
                   "Second law: F = ma, force equals mass times acceleration. "
                   "Third law: For every action, there is an equal and opposite reaction.",
            category="classical_mechanics",
            source="textbook"
        )
        
        corpus.add_document(
            title="Simple Harmonic Motion",
            content="Simple harmonic motion occurs when the restoring force is proportional to "
                   "displacement. The period of a simple pendulum is T = 2π√(L/g), where L is "
                   "the length and g is gravitational acceleration. For a mass-spring system, "
                   "T = 2π√(m/k), where m is mass and k is spring constant.",
            category="classical_mechanics",
            source="textbook"
        )
        
        # Electromagnetism
        corpus.add_document(
            title="Coulomb's Law and Electric Fields",
            content="Coulomb's law describes the force between electric charges: F = kq₁q₂/r², "
                   "where k = 8.99×10⁹ N⋅m²/C². The electric field E = F/q = kQ/r² for a point "
                   "charge. Electric field lines point away from positive charges and toward "
                   "negative charges.",
            category="electromagnetism",
            source="textbook"
        )
        
        corpus.add_document(
            title="Faraday's Law of Induction",
            content="Faraday's law states that the induced EMF in a circuit is equal to the "
                   "negative rate of change of magnetic flux: ε = -dΦ/dt. This is the principle "
                   "behind electric generators and transformers. Lenz's law determines the "
                   "direction of induced current.",
            category="electromagnetism",
            source="textbook"
        )
        
        # Thermodynamics
        corpus.add_document(
            title="Laws of Thermodynamics",
            content="First law: Energy cannot be created or destroyed (ΔU = Q - W). "
                   "Second law: Entropy of an isolated system always increases. "
                   "Third law: Entropy approaches zero as temperature approaches absolute zero. "
                   "Zeroth law: If A is in thermal equilibrium with B, and B with C, then A is "
                   "in equilibrium with C.",
            category="thermodynamics",
            source="textbook"
        )
        
        corpus.add_document(
            title="Ideal Gas Law",
            content="The ideal gas law relates pressure, volume, and temperature: PV = nRT, "
                   "where n is the number of moles and R = 8.314 J/(mol⋅K) is the gas constant. "
                   "For isothermal processes, PV = constant. For adiabatic processes, "
                   "PVᵞ = constant, where γ is the heat capacity ratio.",
            category="thermodynamics",
            source="textbook"
        )
        
        # Quantum Mechanics
        corpus.add_document(
            title="Heisenberg Uncertainty Principle",
            content="The uncertainty principle states that certain pairs of physical properties "
                   "cannot be simultaneously known with arbitrary precision: ΔxΔp ≥ ℏ/2, "
                   "where ℏ = h/2π = 1.055×10⁻³⁴ J⋅s. This is not a measurement limitation "
                   "but a fundamental property of quantum systems.",
            category="quantum_mechanics",
            source="textbook"
        )
        
        corpus.add_document(
            title="Wave-Particle Duality",
            content="Matter exhibits both wave and particle properties. The de Broglie wavelength "
                   "λ = h/p = h/(mv) relates a particle's momentum to its wavelength. "
                   "The double-slit experiment demonstrates wave-particle duality: particles "
                   "create interference patterns when not observed, but behave as particles "
                   "when measured.",
            category="quantum_mechanics",
            source="textbook"
        )
        
        corpus.add_document(
            title="Schrödinger Equation",
            content="The time-dependent Schrödinger equation iℏ∂Ψ/∂t = ĤΨ describes the "
                   "evolution of quantum systems. The time-independent version ĤΨ = EΨ "
                   "gives stationary states. The wave function Ψ contains all information "
                   "about the quantum system, with |Ψ|² giving probability density.",
            category="quantum_mechanics",
            source="textbook"
        )
        
        # Special Relativity
        corpus.add_document(
            title="Special Relativity",
            content="Einstein's special relativity is based on two postulates: the laws of physics "
                   "are the same in all inertial frames, and the speed of light c is constant. "
                   "Time dilation: Δt = γΔt₀, where γ = 1/√(1 - v²/c²). "
                   "Length contraction: L = L₀/γ. Mass-energy equivalence: E = mc².",
            category="relativity",
            source="textbook"
        )
        
        return corpus


# Convenience functions
def load_default_corpus() -> PhysicsCorpus:
    """Load the default physics corpus"""
    return CorpusBuilder.create_default_corpus()


def create_corpus_from_file(filepath: str) -> PhysicsCorpus:
    """Create corpus from file"""
    corpus = PhysicsCorpus()
    corpus.load_from_file(filepath)
    return corpus
EOF
