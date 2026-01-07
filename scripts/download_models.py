#!/usr/bin/env python3
"""Download and cache all required models"""
import os
import sys
from pathlib import Path
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    BlipProcessor,
    BlipForConditionalGeneration
)
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model list with priorities
LANGUAGE_MODELS = [
    # Gemma 3 Models (Priority)
    ("google/gemma-2b-it", "gemma3-2b", True),
    ("google/gemma-7b-it", "gemma3-9b", True),
    
    # General Models
    ("mistralai/Mistral-7B-Instruct-v0.2", "mistral-7b", False),
    ("mistralai/Mixtral-8x7B-Instruct-v0.1", "mixtral-8x7b", True),
    
    # Coding Models
    ("codellama/CodeLlama-13b-Instruct-hf", "codellama-13b", True),
    ("deepseek-ai/deepseek-coder-6.7b-instruct", "deepseek-coder", False),
    
    # Small Models
    ("microsoft/phi-2", "phi-2", False),
]

MULTIMODAL_MODELS = [
    ("openai/whisper-base", "whisper", False),
    ("Salesforce/blip-image-captioning-base", "blip", False),
]

EMBEDDING_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-base-en-v1.5",
]

def download_language_models(cache_dir: Path):
    """Download language models"""
    logger.info("Downloading language models...")
    
    for model_id, name, priority in LANGUAGE_MODELS:
        if not priority and os.getenv("DOWNLOAD_ALL") != "true":
            logger.info(f"Skipping {name} (non-priority)")
            continue
            
        try:
            logger.info(f"Downloading {name} ({model_id})...")
            
            # Download tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                cache_dir=str(cache_dir),
                trust_remote_code=True
            )
            
            # Download model (weights only)
            if "t5" in model_id.lower():
                model_class = AutoModelForSeq2SeqLM
            else:
                model_class = AutoModelForCausalLM
            
            # Just download, don't load to GPU
            _ = model_class.from_pretrained(
                model_id,
                cache_dir=str(cache_dir),
                trust_remote_code=True,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            
            logger.info(f"✓ Downloaded {name}")
            
            # Clear memory
            del _
            torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"Failed to download {name}: {e}")

def download_multimodal_models(cache_dir: Path):
    """Download multimodal models"""
    logger.info("Downloading multimodal models...")
    
    for model_id, name, _ in MULTIMODAL_MODELS:
        try:
            logger.info(f"Downloading {name} ({model_id})...")
            
            if "whisper" in model_id:
                WhisperProcessor.from_pretrained(model_id, cache_dir=str(cache_dir))
                WhisperForConditionalGeneration.from_pretrained(
                    model_id,
                    cache_dir=str(cache_dir),
                    torch_dtype=torch.float16
                )
            elif "blip" in model_id:
                BlipProcessor.from_pretrained(model_id, cache_dir=str(cache_dir))
                BlipForConditionalGeneration.from_pretrained(
                    model_id,
                    cache_dir=str(cache_dir),
                    torch_dtype=torch.float16
                )
            
            logger.info(f"✓ Downloaded {name}")
            
        except Exception as e:
            logger.error(f"Failed to download {name}: {e}")

def download_embedding_models(cache_dir: Path):
    """Download embedding models"""
    logger.info("Downloading embedding models...")
    
    for model_id in EMBEDDING_MODELS:
        try:
            logger.info(f"Downloading {model_id}...")
            _ = SentenceTransformer(model_id, cache_folder=str(cache_dir))
            logger.info(f"✓ Downloaded {model_id}")
            del _
            
        except Exception as e:
            logger.error(f"Failed to download {model_id}: {e}")

def main():
    """Main download function"""
    # Determine cache directory
    if len(sys.argv) > 1:
        cache_dir = Path(sys.argv[1])
    else:
        cache_dir = Path("/opt/sovereign-ai-suite/models")
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Using cache directory: {cache_dir}")
    logger.info("Starting model downloads...")
    logger.info("This process may take 1-3 hours depending on your internet speed.")
    
    # Download models
    download_language_models(cache_dir)
    download_multimodal_models(cache_dir)
    download_embedding_models(cache_dir)
    
    logger.info("✓ All downloads complete!")
    
    # Print summary
    model_files = list(cache_dir.glob("**/*.bin")) + list(cache_dir.glob("**/*.safetensors"))
    total_size = sum(f.stat().st_size for f in model_files) / (1024**3)
    
    logger.info(f"Total models downloaded: {len(model_files)}")
    logger.info(f"Total disk space used: {total_size:.2f} GB")

if __name__ == "__main__":
    main()