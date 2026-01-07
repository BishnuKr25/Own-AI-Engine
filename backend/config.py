"""Production configuration for Sovereign AI Suite"""
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from pydantic import BaseSettings, Field
import yaml

class Settings(BaseSettings):
    """Production settings with environment variable support"""
    
    # Application
    APP_NAME: str = "Sovereign AI Suite"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    MODELS_DIR: Path = BASE_DIR / "models"
    DATA_DIR: Path = BASE_DIR / "data"
    LOGS_DIR: Path = BASE_DIR / "logs"
    UPLOAD_DIR: Path = DATA_DIR / "uploads"
    
    # API Configuration
    API_HOST: str = Field(default="0.0.0.0", env="API_HOST")
    API_PORT: int = Field(default=8000, env="API_PORT")
    API_WORKERS: int = Field(default=1, env="API_WORKERS")
    API_KEY_HEADER: str = "X-API-Key"
    
    # Security
    SECRET_KEY: str = Field(default=None, env="SECRET_KEY")
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # MongoDB
    MONGODB_URL: str = Field(default="mongodb://localhost:27017", env="MONGODB_URL")
    DATABASE_NAME: str = Field(default="sovereign_ai", env="DATABASE_NAME")
    
    # Model Configuration
    MAX_CACHED_MODELS: int = Field(default=3, env="MAX_CACHED_MODELS")
    DEFAULT_TEMPERATURE: float = 0.7
    DEFAULT_MAX_TOKENS: int = 2048
    MODEL_TIMEOUT: int = 300  # seconds
    
    # Council of Experts
    MAX_EXPERTS: int = 5
    ENABLE_GEMMA3_FAST_MODE: bool = True
    PRELOAD_MODELS: List[str] = ["gemma-2b"]
    
    # RAG Configuration
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    VECTOR_DB_TYPE: str = "chromadb"  # or "faiss"
    
    # Multimodal
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB
    ALLOWED_EXTENSIONS: List[str] = [
        ".pdf", ".docx", ".txt", ".csv", ".xlsx",
        ".png", ".jpg", ".jpeg", ".gif",
        ".mp3", ".wav", ".m4a"
    ]
    
    # Performance
    ENABLE_CACHE: bool = True
    CACHE_TTL: int = 3600
    ENABLE_MONITORING: bool = True
    
    class Config:
        env_file = ".env"
        case_sensitive = True
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create directories if they don't exist
        for dir_path in [self.MODELS_DIR, self.DATA_DIR, self.LOGS_DIR, self.UPLOAD_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Generate secret key if not provided
        if not self.SECRET_KEY:
            import secrets
            self.SECRET_KEY = secrets.token_hex(32)

# Load model configurations from YAML
def load_model_config() -> Dict[str, Any]:
    """Load model configurations from YAML file"""
    config_path = Path(__file__).parent.parent / "config" / "models_config.yaml"
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}

settings = Settings()
model_config = load_model_config()