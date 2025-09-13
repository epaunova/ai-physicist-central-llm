cat > src/brain/__init__.py << 'EOF'
"""
Brain Module - Central LLM Orchestration
========================================

This module contains the core language model components:
- BaseModel: Base LLM wrapper for any model
- PhysicsLLM: Physics-specialized model with RAG and tools integration
"""

# Version info
__version__ = "0.1.0"

# Imports
from .specialized_model import PhysicsLLM

# Try to import optional components
try:
    from .base_model import BaseModel
    _has_base_model = True
except ImportError:
    BaseModel = None
    _has_base_model = False

# Public API
__all__ = [
    "PhysicsLLM",
    "BaseModel",
    "BrainConfig",
    "get_available_models",
    "MODEL_REGISTRY"
]

# Configuration class
class BrainConfig:
    """Configuration for brain module"""
    
    DEFAULT_MODEL = "meta-llama/Llama-3.2-8B-Instruct"
    DEFAULT_MAX_LENGTH = 512
    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_TOP_P = 0.9
    
    # Performance settings
    USE_CACHE = True
    BATCH_SIZE = 4
    
    # Model paths (for future use)
    MODEL_PATHS = {
        "base": "models/base",
        "finetuned": "models/finetuned",
        "lora": "models/lora_adapter"
    }
    
    @classmethod
    def get_config(cls):
        """Get current configuration"""
        return {
            "model": cls.DEFAULT_MODEL,
            "max_length": cls.DEFAULT_MAX_LENGTH,
            "temperature": cls.DEFAULT_TEMPERATURE,
            "use_cache": cls.USE_CACHE
        }

# Model registry for dynamic loading
MODEL_REGISTRY = {
    "base": {
        "class": BaseModel if _has_base_model else None,
        "description": "Base LLM without specialization",
        "model_name": "meta-llama/Llama-3.2-3B-Instruct"
    },
    "physics": {
        "class": PhysicsLLM,
        "description": "Physics-specialized with RAG and tools",
        "model_name": "meta-llama/Llama-3.2-8B-Instruct",
        "features": ["rag", "tools", "unit_validation"]
    },
    "physics_finetuned": {
        "class": PhysicsLLM,
        "description": "Fine-tuned on physics QA",
        "model_name": "meta-llama/Llama-3.2-8B-Instruct",
        "adapter": "models/lora_adapter",
        "features": ["rag", "tools", "unit_validation", "lora"]
    }
}

def get_available_models():
    """Get list of available models"""
    return {
        name: {
            "description": info["description"],
            "available": info["class"] is not None,
            "features": info.get("features", [])
        }
        for name, info in MODEL_REGISTRY.items()
    }

def create_model(model_type: str = "physics", **kwargs):
    """
    Factory function to create models
    
    Args:
        model_type: Type of model ('base', 'physics', 'physics_finetuned')
        **kwargs: Additional configuration
        
    Returns:
        Model instance
        
    Example:
        >>> model = create_model("physics", use_rag=True, use_tools=True)
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(MODEL_REGISTRY.keys())}")
    
    model_info = MODEL_REGISTRY[model_type]
    model_class = model_info["class"]
    
    if model_class is None:
        raise ImportError(f"Model {model_type} is not available. Missing dependencies?")
    
    # Merge default config with user kwargs
    config = BrainConfig.get_config()
    config.update(kwargs)
    
    return model_class(config)

# Module initialization message (optional, for debugging)
def _init_message():
    """Print initialization info"""
    models = get_available_models()
    available = [name for name, info in models.items() if info["available"]]
    print(f"Brain module loaded. Available models: {available}")

# Lazy loading helper
class LazyLoader:
    """Lazy load models only when needed"""
    
    _instances = {}
    
    @classmethod
    def get_physics_model(cls):
        """Get singleton PhysicsLLM instance"""
        if "physics" not in cls._instances:
            cls._instances["physics"] = PhysicsLLM()
        return cls._instances["physics"]
    
    @classmethod
    def clear_cache(cls):
        """Clear all cached instances"""
        cls._instances.clear()

# Module-level convenience functions
def answer_physics_question(question: str, **kwargs):
    """
    Quick function to answer physics questions
    
    Args:
        question: Physics question to answer
        **kwargs: Additional parameters
        
    Returns:
        Answer dictionary
        
    Example:
        >>> answer_physics_question("What is F=ma?")
    """
    model = LazyLoader.get_physics_model()
    return model.answer(question, **kwargs)

# Export convenience
physics_model = LazyLoader.get_physics_model  # Function reference for lazy loading
EOF
