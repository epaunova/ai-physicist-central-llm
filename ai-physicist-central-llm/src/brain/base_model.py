cat > src/brain/base_model.py << 'EOF'
"""
Base Model Module
=================
Base wrapper for language model operations with support for multiple backends
"""

import os
import json
import time
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import model libraries
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Using mock mode.")

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available. Using mock mode.")


class ModelBackend(Enum):
    """Supported model backends"""
    TRANSFORMERS = "transformers"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    MOCK = "mock"  # For testing without actual models


@dataclass
class GenerationConfig:
    """Configuration for text generation"""
    max_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    num_beams: int = 1
    do_sample: bool = True
    repetition_penalty: float = 1.1
    length_penalty: float = 1.0
    early_stopping: bool = True
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None


@dataclass
class ModelConfig:
    """Configuration for model initialization"""
    model_name: str = "meta-llama/Llama-3.2-3B-Instruct"
    backend: ModelBackend = ModelBackend.TRANSFORMERS
    device: str = "auto"  # auto, cpu, cuda, mps
    dtype: str = "float16"  # float16, float32, bfloat16
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    use_cache: bool = True
    trust_remote_code: bool = False
    local_files_only: bool = False
    revision: str = "main"


class BaseModel:
    """
    Base wrapper for language models with multiple backend support
    
    This class provides a unified interface for different LLM backends
    including HuggingFace Transformers, OpenAI API, and mock mode for testing.
    """
    
    def __init__(
        self, 
        config: Optional[Union[ModelConfig, Dict]] = None,
        generation_config: Optional[Union[GenerationConfig, Dict]] = None
    ):
        """
        Initialize base model
        
        Args:
            config: Model configuration
            generation_config: Generation parameters
        """
        # Handle config
        if config is None:
            self.config = ModelConfig()
        elif isinstance(config, dict):
            self.config = ModelConfig(**config)
        else:
            self.config = config
            
        # Handle generation config
        if generation_config is None:
            self.generation_config = GenerationConfig()
        elif isinstance(generation_config, dict):
            self.generation_config = GenerationConfig(**generation_config)
        else:
            self.generation_config = generation_config
        
        # Model components
        self.model = None
        self.tokenizer = None
        self.device = self._setup_device()
        
        # Performance tracking
        self.stats = {
            "total_queries": 0,
            "total_tokens_generated": 0,
            "total_time": 0.0,
            "avg_tokens_per_second": 0.0
        }
        
        # Cache for responses (optional)
        self.cache = {} if self.config.use_cache else None
        
        logger.info(f"Initialized BaseModel with backend: {self.config.backend}")
    
    def _setup_device(self) -> torch.device:
        """Setup computation device"""
        if not TORCH_AVAILABLE:
            return None
            
        if self.config.device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
                logger.info("Using Apple Metal Performance Shaders")
            else:
                device = torch.device("cpu")
                logger.info("Using CPU")
        else:
            device = torch.device(self.config.device)
            
        return device
    
    def load_model(self) -> None:
        """Load model and tokenizer based on backend"""
        
        if self.config.backend == ModelBackend.MOCK:
            self._load_mock_model()
        elif self.config.backend == ModelBackend.TRANSFORMERS:
            self._load_transformers_model()
        elif self.config.backend == ModelBackend.OPENAI:
            self._load_openai_model()
        else:
            raise NotImplementedError(f"Backend {self.config.backend} not implemented")
            
        logger.info(f"Model loaded: {self.config.model_name}")
    
    def _load_transformers_model(self) -> None:
        """Load HuggingFace Transformers model"""
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers not available, using mock mode")
            self._load_mock_model()
            return
            
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=self.config.trust_remote_code,
                local_files_only=self.config.local_files_only,
                revision=self.config.revision
            )
            
            # Set padding token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Determine dtype
            dtype_map = {
                "float16": torch.float16,
                "float32": torch.float32,
                "bfloat16": torch.bfloat16
            }
            torch_dtype = dtype_map.get(self.config.dtype, torch.float32)
            
            # Load model with appropriate settings
            model_kwargs = {
                "pretrained_model_name_or_path": self.config.model_name,
                "torch_dtype": torch_dtype,
                "trust_remote_code": self.config.trust_remote_code,
                "local_files_only": self.config.local_files_only,
                "revision": self.config.revision
            }
            
            # Add quantization if requested
            if self.config.load_in_8bit:
                model_kwargs["load_in_8bit"] = True
            elif self.config.load_in_4bit:
                model_kwargs["load_in_4bit"] = True
            else:
                model_kwargs["device_map"] = "auto" if self.device else None
            
            self.model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
            
            # Move to device if not using device_map
            if self.device and not (self.config.load_in_8bit or self.config.load_in_4bit):
                self.model = self.model.to(self.device)
                
            # Set to eval mode
            self.model.eval()
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.info("Falling back to mock mode")
            self._load_mock_model()
    
    def _load_mock_model(self) -> None:
        """Load mock model for testing"""
        self.config.backend = ModelBackend.MOCK
        logger.info("Using mock model for testing")
    
    def _load_openai_model(self) -> None:
        """Load OpenAI API model"""
        raise NotImplementedError("OpenAI backend not yet implemented")
    
    def generate(
        self, 
        prompt: str, 
        generation_config: Optional[GenerationConfig] = None,
        return_full_text: bool = False,
        stream: bool = False
    ) -> Union[str, Dict[str, Any]]:
        """
        Generate text from prompt
        
        Args:
            prompt: Input prompt
            generation_config: Override default generation config
            return_full_text: Return prompt + generation (vs just generation)
            stream: Stream tokens as they're generated
            
        Returns:
            Generated text or full response dict
        """
        start_time = time.time()
        
        # Check cache
        if self.cache is not None and prompt in self.cache:
            logger.info("Cache hit!")
            return self.cache[prompt]
        
        # Use provided config or default
        gen_config = generation_config or self.generation_config
        
        # Generate based on backend
        if self.config.backend == ModelBackend.MOCK:
            response = self._generate_mock(prompt, gen_config)
        elif self.config.backend == ModelBackend.TRANSFORMERS:
            response = self._generate_transformers(prompt, gen_config, return_full_text)
        else:
            raise NotImplementedError(f"Generation not implemented for {self.config.backend}")
        
        # Update stats
        elapsed = time.time() - start_time
        self.stats["total_queries"] += 1
        self.stats["total_time"] += elapsed
        
        # Cache response
        if self.cache is not None:
            self.cache[prompt] = response
            
        return response
    
    def _generate_transformers(
        self, 
        prompt: str, 
        config: GenerationConfig,
        return_full_text: bool
    ) -> str:
        """Generate using Transformers model"""
        if not self.model or not self.tokenizer:
            logger.warning("Model not loaded, using mock generation")
            return self._generate_mock(prompt, config)
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt",
                truncation=True,
                max_length=config.max_length
            )
            
            if self.device:
                inputs = inputs.to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=config.max_length,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    top_k=config.top_k,
                    num_beams=config.num_beams,
                    do_sample=config.do_sample,
                    repetition_penalty=config.repetition_penalty,
                    length_penalty=config.length_penalty,
                    early_stopping=config.early_stopping,
                    pad_token_id=config.pad_token_id or self.tokenizer.pad_token_id,
                    eos_token_id=config.eos_token_id or self.tokenizer.eos_token_id
                )
            
            # Decode
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Return based on preference
            if return_full_text:
                return generated_text
            else:
                # Remove prompt from response
                return generated_text[len(prompt):].strip()
                
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return self._generate_mock(prompt, config)
    
    def _generate_mock(self, prompt: str, config: GenerationConfig) -> str:
        """Generate mock response for testing"""
        mock_responses = {
            "pendulum": "The period of a pendulum is T = 2π√(L/g)",
            "newton": "Newton's second law states that F = ma",
            "energy": "Energy is conserved in isolated systems",
            "quantum": "Quantum mechanics describes subatomic particles",
            "default": f"Mock response to: {prompt[:50]}..."
        }
        
        # Find matching response
        prompt_lower = prompt.lower()
        for key, response in mock_responses.items():
            if key in prompt_lower:
                return response
                
        return mock_responses["default"]
    
    def batch_generate(
        self, 
        prompts: List[str],
        generation_config: Optional[GenerationConfig] = None,
        batch_size: int = 4
    ) -> List[str]:
        """
        Generate responses for multiple prompts
        
        Args:
            prompts: List of prompts
            generation_config: Generation configuration
            batch_size: Process this many prompts at once
            
        Returns:
            List of generated texts
        """
        responses = []
        
        # Process in batches
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            
            # Generate for each prompt in batch
            for prompt in batch:
                response = self.generate(prompt, generation_config)
                responses.append(response)
                
        return responses
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if self.stats["total_queries"] > 0:
            self.stats["avg_time_per_query"] = (
                self.stats["total_time"] / self.stats["total_queries"]
            )
        return self.stats
    
    def clear_cache(self) -> None:
        """Clear response cache"""
        if self.cache is not None:
            self.cache.clear()
            logger.info("Cache cleared")
    
    def save_model(self, path: str) -> None:
        """Save model to disk"""
        if self.model and self.config.backend == ModelBackend.TRANSFORMERS:
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
            logger.info(f"Model saved to {path}")
        else:
            logger.warning("Model saving not supported for this backend")
    
    @classmethod
    def from_pretrained(cls, path: str) -> "BaseModel":
        """Load model from disk"""
        config = ModelConfig(model_name=path, local_files_only=True)
        model = cls(config)
        model.load_model()
        return model
    
    def __repr__(self) -> str:
        return (
            f"BaseModel(model='{self.config.model_name}', "
            f"backend={self.config.backend}, "
            f"device={self.device})"
        )

# Convenience function
def create_base_model(model_name: str = None, **kwargs) -> BaseModel:
    """Quick function to create a base model"""
    config = ModelConfig(model_name=model_name) if model_name else ModelConfig()
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    model = BaseModel(config)
    model.load_model()
    return model
EOF
