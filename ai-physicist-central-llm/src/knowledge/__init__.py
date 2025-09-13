cat > src/knowledge/__init__.py << 'EOF'
"""
Knowledge Module
================
RAG and retrieval systems for physics knowledge

This module provides:
- Document retrieval with FAISS
- Physics corpus management
- Embedding generation
- Context ranking and filtering
"""

__version__ = "0.1.0"

# Core imports
from .retriever import PhysicsRetriever
from .physics_corpus import PhysicsCorpus, Document, CorpusBuilder

# Public API
__all__ = [
    "PhysicsRetriever",
    "PhysicsCorpus",
    "Document",
    "CorpusBuilder",
    "create_knowledge_base",
    "load_corpus",
    "search_physics"
]

# Configuration
DEFAULT_CONFIG = {
    "embedding_model": "BAAI/bge-small-en-v1.5",
    "index_type": "faiss",
    "top_k": 5,
    "corpus_path": "data/corpus/physics_abstracts.json",
    "cache_embeddings": True,
    "max_seq_length": 512
}

# Check for optional dependencies
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: FAISS not available. Install faiss-cpu or faiss-gpu for better performance.")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not available. Install for embedding generation.")


def create_knowledge_base(
    documents=None,
    corpus_path=None,
    embedding_model=None,
    **kwargs
):
    """
    Create a complete knowledge base
    
    Args:
        documents: List of documents or dict
        corpus_path: Path to corpus file
        embedding_model: Model for embeddings
        **kwargs: Additional configuration
        
    Returns:
        Tuple of (corpus, retriever)
        
    Example:
        >>> corpus, retriever = create_knowledge_base(
        ...     corpus_path="data/corpus/physics.json"
        ... )
        >>> results = retriever.search("pendulum period")
    """
    # Create corpus
    corpus = PhysicsCorpus()
    
    if corpus_path:
        corpus.load_from_file(corpus_path)
    elif documents:
        if isinstance(documents, list):
            corpus.add_documents(documents)
        elif isinstance(documents, dict):
            corpus.add_documents([documents])
    else:
        # Load default corpus
        corpus.load_from_file(DEFAULT_CONFIG["corpus_path"])
    
    # Create retriever
    embedding_model = embedding_model or DEFAULT_CONFIG["embedding_model"]
    retriever = PhysicsRetriever(
        corpus=corpus,
        embedding_model=embedding_model,
        **kwargs
    )
    
    # Build index
    retriever.build_index()
    
    return corpus, retriever


def load_corpus(path=None):
    """
    Load physics corpus from file
    
    Args:
        path: Path to corpus file (JSON)
        
    Returns:
        PhysicsCorpus object
        
    Example:
        >>> corpus = load_corpus("data/corpus/physics.json")
        >>> print(f"Loaded {len(corpus)} documents")
    """
    path = path or DEFAULT_CONFIG["corpus_path"]
    corpus = PhysicsCorpus()
    corpus.load_from_file(path)
    return corpus


def search_physics(
    query,
    corpus=None,
    retriever=None,
    top_k=None,
    **kwargs
):
    """
    Quick search function
    
    Args:
        query: Search query
        corpus: PhysicsCorpus object (optional)
        retriever: PhysicsRetriever object (optional)
        top_k: Number of results
        **kwargs: Additional search parameters
        
    Returns:
        List of relevant documents
        
    Example:
        >>> results = search_physics("Newton's second law")
        >>> for doc in results:
        ...     print(doc['title'], doc['score'])
    """
    # Create retriever if not provided
    if retriever is None:
        if corpus is None:
            corpus = load_corpus()
        retriever = PhysicsRetriever(corpus=corpus)
        retriever.build_index()
    
    # Search
    top_k = top_k or DEFAULT_CONFIG["top_k"]
    return retriever.search(query, k=top_k, **kwargs)


class KnowledgeConfig:
    """Configuration for knowledge module"""
    
    # Supported embedding models
    EMBEDDING_MODELS = {
        "small": "BAAI/bge-small-en-v1.5",
        "base": "BAAI/bge-base-en-v1.5",
        "large": "BAAI/bge-large-en-v1.5",
        "minilm": "sentence-transformers/all-MiniLM-L6-v2",
        "mpnet": "sentence-transformers/all-mpnet-base-v2",
        "e5": "intfloat/e5-base-v2"
    }
    
    # Index types
    INDEX_TYPES = {
        "flat": "IndexFlatL2",
        "ivf": "IndexIVFFlat",
        "hnsw": "IndexHNSWFlat",
        "lsh": "IndexLSH"
    }
    
    # Physics categories
    PHYSICS_CATEGORIES = [
        "classical_mechanics",
        "electromagnetism",
        "thermodynamics",
        "quantum_mechanics",
        "relativity",
        "optics",
        "atomic_physics",
        "nuclear_physics",
        "particle_physics",
        "condensed_matter",
        "astrophysics",
        "mathematical_physics"
    ]
    
    @classmethod
    def get_model(cls, name="small"):
        """Get embedding model by name"""
        return cls.EMBEDDING_MODELS.get(name, name)
    
    @classmethod
    def list_models(cls):
        """List available embedding models"""
        return list(cls.EMBEDDING_MODELS.keys())


# Knowledge statistics
class KnowledgeStats:
    """Track knowledge base statistics"""
    
    def __init__(self):
        self.total_documents = 0
        self.total_searches = 0
        self.avg_retrieval_time = 0.0
        self.cache_hits = 0
        self.cache_misses = 0
    
    def update_search(self, time_taken):
        """Update search statistics"""
        self.total_searches += 1
        # Update running average
        self.avg_retrieval_time = (
            (self.avg_retrieval_time * (self.total_searches - 1) + time_taken) 
            / self.total_searches
        )
    
    def get_stats(self):
        """Get statistics dictionary"""
        return {
            "total_documents": self.total_documents,
            "total_searches": self.total_searches,
            "avg_retrieval_time": self.avg_retrieval_time,
            "cache_hit_rate": self.cache_hits / (self.cache_hits + self.cache_misses)
            if (self.cache_hits + self.cache_misses) > 0 else 0.0
        }


# Global stats instance
knowledge_stats = KnowledgeStats()


# Initialization
def _init_module():
    """Initialize knowledge module"""
    import logging
    logger = logging.getLogger(__name__)
    
    if not FAISS_AVAILABLE:
        logger.warning("FAISS not available - using numpy fallback")
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        logger.warning("Sentence transformers not available - using mock embeddings")
    
    logger.info(f"Knowledge module initialized with config: {DEFAULT_CONFIG}")

_init_module()
EOF
