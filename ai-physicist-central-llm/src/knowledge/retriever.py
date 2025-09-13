"""Physics knowledge retrieval system"""

import json
import numpy as np
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
import faiss


class PhysicsRetriever:
    """Retrieval system for physics knowledge"""
    
    def __init__(self, embedding_model: str = "BAAI/bge-small-en-v1.5"):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.documents = []
        self.embeddings = None
        self.index = None
        
    def load_corpus(self, corpus_path: str):
        """Load physics corpus from file"""
        with open(corpus_path, 'r') as f:
            self.documents = json.load(f)
            
    def build_index(self):
        """Build FAISS index for similarity search"""
        if not self.documents:
            # Load default corpus
            self._load_default_corpus()
            
        # Create embeddings
        texts = [doc.get('content', '') for doc in self.documents]
        self.embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        # Build FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings.astype('float32'))
        
    def search(self, query: str, k: int = 5) -> List[str]:
        """Search for relevant documents"""
        if not self.index:
            return []
            
        # Encode query
        query_embedding = self.embedding_model.encode([query])
        
        # Search
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Return documents
        results = []
        for idx in indices[0]:
            if idx < len(self.documents):
                results.append(self.documents[idx].get('content', ''))
                
        return results
    
    def _load_default_corpus(self):
        """Load default physics corpus"""
        self.documents = [
            {"id": "1", "content": "Newton's second law: F = ma, where F is force, m is mass, a is acceleration"},
            {"id": "2", "content": "Period of pendulum: T = 2π√(L/g), where L is length, g is gravitational acceleration"},
            {"id": "3", "content": "Kinetic energy: KE = ½mv², where m is mass, v is velocity"},
            {"id": "4", "content": "Coulomb's law: F = k(q₁q₂/r²), where k is Coulomb's constant, q is charge, r is distance"},
            {"id": "5", "content": "Ideal gas law: PV = nRT, where P is pressure, V is volume, n is moles, R is gas constant, T is temperature"},
        ]
