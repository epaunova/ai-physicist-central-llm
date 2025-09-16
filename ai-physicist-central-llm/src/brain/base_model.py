"""
Base Model Module
=================
Base wrapper for language model operations with support for multiple backends.
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any, Union

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional deps
try:
    import torch  # type: ignore
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Using mock mode.")

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available. Using mock mode.")


class ModelBackend(Enum):
    TRANSFORMERS = "transformers"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    MOCK = "mock"


@dataclass
class GenerationConfig:
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
    Base wrapper for language models with multiple backend support.
    Supports HuggingFace Transformers, (future) OpenAI/Anthropic backends, and a mock mode.
    """

    def __init__(
        self,
        config: Optional[Union[ModelConfig, Dict[str, Any]]] = None,
        generation_config: Optional[Union[GenerationConfig, Dict[str, Any]]] = None,
    ):
        if config is None:
            self.config = ModelConfig()
        elif isinstance(config, dict):
            self.config = ModelConfig(**config)
        else:
            self.config = config

        if generation_config is None:
            self.generation_config = GenerationConfig()
        elif isinstance(generation_config, dict):
            self.generation_config = GenerationConfig(**generation_config)
        else:
            self.generation_config = generation_config

        self.model: Any = None
        self.tokenizer: Any = None
        self.device: Any = self._setup_device()

        self.stats: Dict[str, Any] = {
            "total_queries": 0,
            "total_tokens_generated": 0,
            "total_time": 0.0,
            "avg_tokens_per_second": 0.0,
        }

        self.cache: Optional[Dict[str, Any]] = {} if self.config.use_cache else None
        logger.info(f"Initialized BaseModel with backend: {self.config.backend}")

    def _setup_device(self) -> Any:
        """Setup computation device if torch is available."""
        if not TORCH_AVAILABLE:
            return None

        if self.config.device == "auto":
            if torch.cuda.is_available():
                logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
                return torch.device("cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore
                logger.info("Using Apple Metal Performance Shaders (MPS)")
                return torch.device("mps")
            logger.info("Using CPU")
            return torch.device("cpu")
        else:
            return torch.device(self.config.device)

    def load_model(self) -> None:
        """Load model and tokenizer based on backend."""
        try:
            if self.config.backend == ModelBackend.TRANSFORMERS:
                self._load_transformers_model()
            elif self.config.backend == ModelBackend.OPENAI:
                self._load_openai_model()
            elif self.config.backend == ModelBackend.ANTHROPIC:
                self._load_anthropic_model()
            else:
                self._load_mock_model()
        except Exception as e:
            logger.error(f"load_model error: {e}")
            self._load_mock_model()

        logger.info(f"Model loaded: {self.config.model_name} (backend={self.config.backend})")

    def _load_transformers_model(self) -> None:
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers not available; falling back to mock.")
            self._load_mock_model()
            return

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=self.config.trust_remote_code,
                local_files_only=self.config.local_files_only,
                revision=self.config.revision,
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            torch_dtype = None
            if TORCH_AVAILABLE:
                dtype_map = {
                    "float16": torch.float16,
                    "float32": torch.float32,
                    "bfloat16": torch.bfloat16,
                }
                torch_dtype = dtype_map.get(self.config.dtype, torch.float32)

            model_kwargs: Dict[str, Any] = {
                "pretrained_model_name_or_path": self.config.model_name,
                "trust_remote_code": self.config.trust_remote_code,
                "local_files_only": self.config.local_files_only,
                "revision": self.config.revision,
            }
            if torch_dtype is not None:
                model_kwargs["torch_dtype"] = torch_dtype

            if self.config.load_in_8bit:
                model_kwargs["load_in_8bit"] = True
            elif self.config.load_in_4bit:
                model_kwargs["load_in_4bit"] = True
            else:
                if TORCH_AVAILABLE and self.device is not None:
                    model_kwargs["device_map"] = "auto"

            self.model = AutoModelForCausalLM.from_pretrained(**model_kwargs)

            if TORCH_AVAILABLE and self.device is not None and not (self.config.load_in_8bit or self.config.load_in_4bit):
                self.model = self.model.to(self.device)

            self.model.eval()

        except Exception as e:
            logger.error(f"Failed to load transformers model: {e}")
            self._load_mock_model()

    def _load_openai_model(self) -> None:
        raise NotImplementedError("OpenAI backend not yet implemented")

    def _load_anthropic_model(self) -> None:
        raise NotImplementedError("Anthropic backend not yet implemented")

    def _load_mock_model(self) -> None:
        self.config.backend = ModelBackend.MOCK
        self.model = None
        self.tokenizer = None
        logger.info("Using mock model (no dependencies required)")

    def generate(
        self,
        prompt: str,
        generation_config: Optional[GenerationConfig] = None,
        return_full_text: bool = False,
        stream: bool = False,
    ) -> Union[str, Dict[str, Any]]:
        start = time.time()

        if self.cache is not None and prompt in self.cache:
            return self.cache[prompt]

        gen_cfg = generation_config or self.generation_config

        if self.config.backend == ModelBackend.TRANSFORMERS:
            text = self._generate_transformers(prompt, gen_cfg, return_full_text)
        elif self.config.backend == ModelBackend.MOCK:
            text = self._generate_mock(prompt, gen_cfg)
        else:
            raise NotImplementedError(f"Generation not implemented for backend={self.config.backend}")

        elapsed = time.time() - start
        self.stats["total_queries"] += 1
        self.stats["total_time"] += elapsed

        if self.cache is not None:
            self.cache[prompt] = text
        return text

    def _generate_transformers(
        self,
        prompt: str,
        cfg: GenerationConfig,
        return_full_text: bool,
    ) -> str:
        if self.model is None or self.tokenizer is None or not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers unavailable or not loaded; using mock.")
            return self._generate_mock(prompt, cfg)

        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=cfg.max_length,
            )
            if TORCH_AVAILABLE and self.device is not None:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():  # type: ignore
                outputs = self.model.generate(
                    **inputs,
                    max_length=cfg.max_length,
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                    top_k=cfg.top_k,
                    num_beams=cfg.num_beams,
                    do_sample=cfg.do_sample,
                    repetition_penalty=cfg.repetition_penalty,
                    length_penalty=cfg.length_penalty,
                    early_stopping=cfg.early_stopping,
                    pad_token_id=cfg.pad_token_id or self.tokenizer.pad_token_id,
                    eos_token_id=cfg.eos_token_id or self.tokenizer.eos_token_id,
                )

            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if return_full_text or not generated.startswith(prompt):
                return generated
            return generated[len(prompt) :].strip()
        except Exception as e:
            logger.error(f"Transformers generation failed: {e}")
            return self._generate_mock(prompt, cfg)

    def _generate_mock(self, prompt: str, cfg: GenerationConfig) -> str:
        prompt_lower = prompt.lower()
        if "pendulum" in prompt_lower:
            return "The period of a pendulum is T = 2π√(L/g)."
        if "newton" in prompt_lower:
            return "Newton's second law states that F = m a."
        if "planck" in prompt_lower:
            return "Dimensions of Planck's constant: [M L^2 T^-1]."
        return f"Mock response: {prompt[:60]}..."

    def batch_generate(
        self,
        prompts: List[str],
        generation_config: Optional[GenerationConfig] = None,
        batch_size: int = 4,
    ) -> List[str]:
        out: List[str] = []
        for i in range(0, len(prompts), batch_size):
            for p in prompts[i : i + batch_size]:
                out.append(self.generate(p, generation_config))
        return out

    def get_stats(self) -> Dict[str, Any]:
        if self.stats["total_queries"] > 0:
            self.stats["avg_time_per_query"] = self.stats["total_time"] / self.stats["total_queries"]
        return self.stats

    def clear_cache(self) -> None:
        if self.cache is not None:
            self.cache.clear()
            logger.info("Cache cleared")

    def save_model(self, path: str) -> None:
        if self.model and self.config.backend == ModelBackend.TRANSFORMERS:
            self.model.save_pretrained(path)
            if self.tokenizer:
                self.tokenizer.save_pretrained(path)
            logger.info(f"Model saved to {path}")
        else:
            logger.warning("Model saving not supported for this backend")

    @classmethod
    def from_pretrained(cls, path: str) -> "BaseModel":
        cfg = ModelConfig(model_name=path, local_files_only=True)
        mdl = cls(cfg)
        mdl.load_model()
        return mdl

    def __repr__(self) -> str:
        return f"BaseModel(model='{self.config.model_name}', backend={self.config.backend}, device={self.device})"


def create_base_model(model_name: Optional[str] = None, **kwargs: Any) -> BaseModel:
    cfg = ModelConfig(model_name=model_name) if model_name else ModelConfig()
    for k, v in kwargs.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    m = BaseModel(cfg)
    m.load_model()
    return m
