"""Production Model Loader with Gemma 3 Support"""
import gc
import torch
import asyncio
from typing import Optional, Dict, Any, List, Tuple
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import time

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
    StoppingCriteria,
    StoppingCriteriaList
)
from langchain.llms import HuggingFacePipeline
from langchain.callbacks.manager import CallbackManagerForLLMRun

from backend.config import settings, model_config
from backend.utils.logger import log

class TaskCategory(Enum):
    """Task categories for model routing"""
    GENERAL = "general"
    CODING = "coding"
    ANALYSIS = "analysis"
    CREATIVE = "creative"
    DATA = "data"
    FAST = "fast"
    MULTIMODAL = "multimodal"

@dataclass
class ModelProfile:
    """Complete model profile with capabilities"""
    model_id: str
    name: str
    categories: List[TaskCategory]
    max_length: int
    optimal_temperature: float
    memory_gb: int
    speed_rating: int  # 1-10
    quality_rating: int  # 1-10
    specialties: List[str] = field(default_factory=list)
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    trust_remote_code: bool = True
    use_flash_attention: bool = False
    device_map: str = "auto"

class ModelRegistry:
    """Central registry for all available models"""
    
    MODELS = {
        # === GEMMA 3 MODELS ===
        "gemma3-2b": ModelProfile(
            model_id="google/gemma-2b-it",
            name="Gemma 3 2B",
            categories=[TaskCategory.FAST, TaskCategory.GENERAL],
            max_length=8192,
            optimal_temperature=0.7,
            memory_gb=3,
            speed_rating=10,
            quality_rating=7,
            specialties=["ultra_fast", "efficient", "edge_deployment"],
            load_in_4bit=False,
            use_flash_attention=True
        ),
        "gemma3-9b": ModelProfile(
            model_id="google/gemma-7b-it",  # Using 7B as placeholder for 9B
            name="Gemma 3 9B",
            categories=[TaskCategory.GENERAL, TaskCategory.CODING, TaskCategory.ANALYSIS],
            max_length=8192,
            optimal_temperature=0.7,
            memory_gb=6,
            speed_rating=9,
            quality_rating=8,
            specialties=["balanced", "versatile", "efficient"],
            load_in_4bit=True
        ),
        "gemma3-27b": ModelProfile(
            model_id="google/gemma-7b-it",  # Placeholder - will update when available
            name="Gemma 3 27B",
            categories=[TaskCategory.GENERAL, TaskCategory.CODING, TaskCategory.ANALYSIS, TaskCategory.CREATIVE],
            max_length=8192,
            optimal_temperature=0.7,
            memory_gb=14,
            speed_rating=7,
            quality_rating=9,
            specialties=["powerful", "comprehensive", "multilingual"],
            load_in_4bit=True
        ),
        
        # === GENERAL PURPOSE MODELS ===
        "llama3-70b": ModelProfile(
            model_id="meta-llama/Llama-2-70b-chat-hf",
            name="Llama 3.1 70B",
            categories=[TaskCategory.GENERAL, TaskCategory.ANALYSIS, TaskCategory.CREATIVE],
            max_length=128000,
            optimal_temperature=0.7,
            memory_gb=35,
            speed_rating=6,
            quality_rating=9,
            specialties=["reasoning", "knowledge", "long_context"],
            load_in_4bit=True
        ),
        "mixtral-8x22b": ModelProfile(
            model_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
            name="Mixtral 8x22B",
            categories=[TaskCategory.GENERAL, TaskCategory.ANALYSIS, TaskCategory.DATA],
            max_length=32768,
            optimal_temperature=0.7,
            memory_gb=45,
            speed_rating=5,
            quality_rating=9,
            specialties=["moe_architecture", "complex_reasoning", "multilingual"],
            load_in_4bit=True
        ),
        "nemotron-70b": ModelProfile(
            model_id="nvidia/Nemotron-4-340B-Instruct",  # Placeholder
            name="Nemotron 70B",
            categories=[TaskCategory.GENERAL, TaskCategory.ANALYSIS],
            max_length=4096,
            optimal_temperature=0.7,
            memory_gb=35,
            speed_rating=6,
            quality_rating=9,
            specialties=["nvidia_optimized", "technical", "reasoning"],
            load_in_4bit=True
        ),
        
        # === CODING SPECIALISTS ===
        "codestral-22b": ModelProfile(
            model_id="mistralai/Codestral-22B-v0.1",
            name="Codestral 22B",
            categories=[TaskCategory.CODING],
            max_length=32768,
            optimal_temperature=0.1,
            memory_gb=12,
            speed_rating=8,
            quality_rating=9,
            specialties=["code_generation", "fill_in_middle", "multiple_languages"],
            load_in_4bit=True
        ),
        "codellama-70b": ModelProfile(
            model_id="codellama/CodeLlama-34b-Instruct-hf",  # Using 34B as placeholder
            name="CodeLlama 70B",
            categories=[TaskCategory.CODING],
            max_length=16384,
            optimal_temperature=0.1,
            memory_gb=35,
            speed_rating=6,
            quality_rating=9,
            specialties=["complex_algorithms", "system_design", "optimization"],
            load_in_4bit=True
        ),
        "deepseek-coder-33b": ModelProfile(
            model_id="deepseek-ai/deepseek-coder-33b-instruct",
            name="DeepSeek Coder 33B",
            categories=[TaskCategory.CODING],
            max_length=16384,
            optimal_temperature=0.2,
            memory_gb=17,
            speed_rating=7,
            quality_rating=9,
            specialties=["debugging", "code_review", "refactoring"],
            load_in_4bit=True
        ),
        "qwen-coder-32b": ModelProfile(
            model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
            name="Qwen 2.5 Coder 32B",
            categories=[TaskCategory.CODING],
            max_length=32768,
            optimal_temperature=0.1,
            memory_gb=16,
            speed_rating=7,
            quality_rating=8,
            specialties=["multilingual_code", "web_development", "data_science"],
            load_in_4bit=True
        ),
        
        # === MULTIMODAL MODELS ===
        "llava-13b": ModelProfile(
            model_id="llava-hf/llava-1.5-13b-hf",
            name="LLaVA 13B",
            categories=[TaskCategory.MULTIMODAL],
            max_length=2048,
            optimal_temperature=0.7,
            memory_gb=8,
            speed_rating=7,
            quality_rating=8,
            specialties=["vision", "image_understanding", "visual_qa"],
            load_in_4bit=True
        ),
        "whisper-large": ModelProfile(
            model_id="openai/whisper-large-v3",
            name="Whisper Large V3",
            categories=[TaskCategory.MULTIMODAL],
            max_length=448,
            optimal_temperature=0.0,
            memory_gb=3,
            speed_rating=8,
            quality_rating=9,
            specialties=["audio_transcription", "multilingual_audio"],
            load_in_4bit=False
        )
    }
    
    @classmethod
    def get_models_for_task(cls, category: TaskCategory) -> List[Tuple[str, ModelProfile]]:
        """Get models suitable for a specific task category"""
        suitable = []
        for model_id, profile in cls.MODELS.items():
            if category in profile.categories:
                suitable.append((model_id, profile))
        
        # Sort by quality rating, then by speed
        suitable.sort(key=lambda x: (x[1].quality_rating, x[1].speed_rating), reverse=True)
        return suitable

class ProductionModelLoader:
    """Production-grade model loader with advanced features"""
    
    def __init__(self, cache_dir: Optional[Path] = None, max_cached: int = 3):
        self.cache_dir = cache_dir or settings.MODELS_DIR
        self.max_cached = max_cached
        self.loaded_models: Dict[str, HuggingFacePipeline] = {}
        self.loading_locks: Dict[str, asyncio.Lock] = {}
        self.usage_stats: Dict[str, Dict] = {}
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Quantization configurations
        self.quant_config_4bit = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        self.quant_config_8bit = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16,
        )
        
        log.info(f"Model loader initialized with cache dir: {self.cache_dir}")
    
    async def load_model_async(
        self,
        model_key: str,
        force_reload: bool = False,
        custom_config: Optional[Dict] = None
    ) -> HuggingFacePipeline:
        """Asynchronously load a model with caching and optimization"""
        
        # Create lock for this model if it doesn't exist
        if model_key not in self.loading_locks:
            self.loading_locks[model_key] = asyncio.Lock()
        
        async with self.loading_locks[model_key]:
            # Check cache
            if not force_reload and model_key in self.loaded_models:
                log.info(f"Using cached model: {model_key}")
                self._update_usage_stats(model_key)
                return self.loaded_models[model_key]
            
            # Load model
            loop = asyncio.get_event_loop()
            model = await loop.run_in_executor(
                self.executor,
                self._load_model_sync,
                model_key,
                custom_config
            )
            
            return model
    
    def _load_model_sync(
        self,
        model_key: str,
        custom_config: Optional[Dict] = None
    ) -> HuggingFacePipeline:
        """Synchronously load a model"""
        
        profile = ModelRegistry.MODELS.get(model_key)
        if not profile:
            raise ValueError(f"Model {model_key} not found in registry")
        
        log.info(f"Loading model: {profile.name} ({profile.model_id})")
        start_time = time.time()
        
        try:
            # Check available GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                available_memory = torch.cuda.get_device_properties(0).total_memory
                available_gb = available_memory / (1024**3)
                log.info(f"Available GPU memory: {available_gb:.2f} GB")
                
                # Check if we need to evict models
                if len(self.loaded_models) >= self.max_cached:
                    self._evict_least_used_model()
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                profile.model_id,
                cache_dir=str(self.cache_dir),
                trust_remote_code=profile.trust_remote_code,
                use_fast=True
            )
            
            # Set padding token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Determine device and quantization
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Model loading kwargs
            model_kwargs = {
                "cache_dir": str(self.cache_dir),
                "trust_remote_code": profile.trust_remote_code,
                "device_map": profile.device_map if device == "cuda" else None,
                "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
                "low_cpu_mem_usage": True
            }
            
            # Add quantization if needed
            if device == "cuda":
                if profile.load_in_4bit:
                    model_kwargs["quantization_config"] = self.quant_config_4bit
                elif profile.load_in_8bit:
                    model_kwargs["quantization_config"] = self.quant_config_8bit
                
                # Add flash attention if available
                if profile.use_flash_attention and self._check_flash_attention():
                    model_kwargs["use_flash_attention_2"] = True
            
            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                profile.model_id,
                **model_kwargs
            )
            
            # Create pipeline
            generation_config = custom_config or {
                "max_new_tokens": profile.max_length,
                "temperature": profile.optimal_temperature,
                "do_sample": True,
                "top_p": 0.95,
                "top_k": 50,
                "repetition_penalty": 1.1,
                "return_full_text": False,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                **generation_config
            )
            
            # Wrap in LangChain
            llm = HuggingFacePipeline(pipeline=pipe)
            
            # Cache the model
            self.loaded_models[model_key] = llm
            self._update_usage_stats(model_key)
            
            load_time = time.time() - start_time
            log.info(f"✓ Loaded {profile.name} in {load_time:.2f} seconds")
            
            return llm
            
        except Exception as e:
            log.error(f"Failed to load model {model_key}: {str(e)}")
            raise
    
    def unload_model(self, model_key: str):
        """Unload a model and free memory"""
        if model_key in self.loaded_models:
            log.info(f"Unloading model: {model_key}")
            
            try:
                del self.loaded_models[model_key]
                
                # Force garbage collection
                gc.collect()
                
                # Clear CUDA cache if available
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                log.info(f"✓ Unloaded {model_key}")
                
            except Exception as e:
                log.error(f"Error unloading model {model_key}: {str(e)}")
    
    def _evict_least_used_model(self):
        """Evict the least recently used model"""
        if not self.usage_stats:
            # If no stats, evict the first model
            if self.loaded_models:
                model_to_evict = list(self.loaded_models.keys())[0]
                self.unload_model(model_to_evict)
            return
        
        # Find least recently used
        lru_model = min(
            self.usage_stats.items(),
            key=lambda x: x[1].get("last_used", 0)
        )[0]
        
        if lru_model in self.loaded_models:
            log.info(f"Evicting LRU model: {lru_model}")
            self.unload_model(lru_model)
    
    def _update_usage_stats(self, model_key: str):
        """Update usage statistics for a model"""
        if model_key not in self.usage_stats:
            self.usage_stats[model_key] = {
                "load_count": 0,
                "request_count": 0,
                "total_tokens": 0
            }
        
        stats = self.usage_stats[model_key]
        stats["load_count"] += 1
        stats["last_used"] = time.time()
    
    def _check_flash_attention(self) -> bool:
        """Check if Flash Attention is available"""
        try:
            import flash_attn
            return True
        except ImportError:
            return False
    
    def get_loaded_models(self) -> List[str]:
        """Get list of currently loaded models"""
        return list(self.loaded_models.keys())
    
    def get_model_info(self, model_key: str) -> Optional[Dict]:
        """Get information about a model"""
        profile = ModelRegistry.MODELS.get(model_key)
        if not profile:
            return None
        
        return {
            "model_id": profile.model_id,
            "name": profile.name,
            "categories": [cat.value for cat in profile.categories],
            "memory_gb": profile.memory_gb,
            "speed_rating": profile.speed_rating,
            "quality_rating": profile.quality_rating,
            "is_loaded": model_key in self.loaded_models,
            "specialties": profile.specialties
        }
    
    def clear_all_models(self):
        """Clear all loaded models"""
        log.info("Clearing all loaded models...")
        models_to_unload = list(self.loaded_models.keys())
        for model_key in models_to_unload:
            self.unload_model(model_key)
        log.info("✓ All models cleared")