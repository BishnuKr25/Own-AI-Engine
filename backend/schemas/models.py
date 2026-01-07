"""Pydantic models for request/response validation"""
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class TaskType(str, Enum):
    """Task types for routing"""
    GENERAL = "general"
    CODING = "coding"
    ANALYSIS = "analysis"
    CREATIVE = "creative"
    DATA_GENERATION = "data_generation"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    QUICK = "quick"

class QueryRequest(BaseModel):
    """Query request model"""
    query: str = Field(..., min_length=1, max_length=10000)
    context: Optional[str] = Field(None, max_length=5000)
    task_type: Optional[TaskType] = None
    num_experts: int = Field(default=3, ge=1, le=5)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=1, le=8192)
    stream: bool = Field(default=False)
    use_rag: bool = Field(default=True)
    fast_mode: bool = Field(default=False)
    
    @validator('query')
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()

class QueryResponse(BaseModel):
    """Query response model"""
    query_id: str
    success: bool
    answer: str
    task_type: TaskType
    experts_consulted: List[str]
    confidence_score: float
    processing_time: float
    tokens_used: int
    metadata: Dict[str, Any] = {}
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class FileUploadRequest(BaseModel):
    """File upload request model"""
    filename: str
    file_type: str
    query: Optional[str] = None
    process_type: str = Field(default="analyze", regex="^(analyze|convert|extract|summarize)$")

class UserCreate(BaseModel):
    """User creation model"""
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., regex="^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\\.[a-zA-Z0-9-.]+$")
    role: str = Field(default="user", regex="^(user|developer|admin)$")
    organization: Optional[str] = None

class APIKey(BaseModel):
    """API Key model"""
    key: str
    user_id: str
    created_at: datetime
    last_used: Optional[datetime] = None
    is_active: bool = True

class ModelInfo(BaseModel):
    """Model information"""
    model_id: str
    name: str
    categories: List[str]
    memory_gb: int
    speed_rating: int
    quality_rating: int
    is_loaded: bool = False