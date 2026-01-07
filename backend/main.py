"""Production FastAPI Application"""
from fastapi import FastAPI, HTTPException, Depends, Header, File, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import uvicorn
from typing import Optional, Dict, Any
from datetime import datetime
import uuid

from motor.motor_asyncio import AsyncIOMotorClient

from backend.config import settings
from backend.schemas.models import (
    QueryRequest, QueryResponse, FileUploadRequest,
    UserCreate, TaskType
)
from backend.core.model_loader import ProductionModelLoader
from backend.core.council_of_experts import CouncilOfExperts
from backend.core.task_router import TaskRouter
from backend.services.auth_service import AuthService
from backend.services.rag_service import RAGService
from backend.services.multimodal_service import MultimodalService
from backend.utils.logger import log

# Global instances
db_client: Optional[AsyncIOMotorClient] = None
model_loader: Optional[ProductionModelLoader] = None
council: Optional[CouncilOfExperts] = None
task_router: Optional[TaskRouter] = None
auth_service: Optional[AuthService] = None
rag_service: Optional[RAGService] = None
multimodal_service: Optional[MultimodalService] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    global db_client, model_loader, council, task_router
    global auth_service, rag_service, multimodal_service
    
    log.info("Starting Sovereign AI Suite...")
    
    # Initialize database
    db_client = AsyncIOMotorClient(settings.MONGODB_URL)
    log.info("✓ MongoDB connected")
    
    # Initialize services
    auth_service = AuthService(db_client)
    rag_service = RAGService(db_client)
    multimodal_service = MultimodalService()
    await multimodal_service.initialize()
    
    # Initialize model loader
    model_loader = ProductionModelLoader(
        cache_dir=settings.MODELS_DIR,
        max_cached=settings.MAX_CACHED_MODELS
    )
    
    # Initialize council of experts
    council = CouncilOfExperts(model_loader)
    await council.initialize()
    
    # Initialize task router
    task_router = TaskRouter()
    
    log.info("✓ All systems initialized")
    
    yield
    
    # Shutdown
    log.info("Shutting down...")
    
    if model_loader:
        model_loader.clear_all_models()
    
    if db_client:
        db_client.close()
    
    log.info("✓ Shutdown complete")

# Create FastAPI app
app = FastAPI(
    title="Sovereign AI Suite API",
    description="Production-grade self-hosted AI system with Council of Experts",
    version=settings.VERSION,
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency for API key validation
async def verify_api_key(api_key: str = Header(None, alias="X-API-Key")) -> Dict:
    """Verify API key dependency"""
    if not api_key:
        raise HTTPException(status_code=401, detail="API key required")
    
    user_info = await auth_service.verify_api_key(api_key)
    if not user_info:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return user_info

# Health check
@app.get("/health")
async def health_check():
    """System health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": settings.VERSION,
        "components": {
            "database": db_client is not None,
            "model_loader": model_loader is not None,
            "council": council is not None,
            "models_loaded": len(model_loader.get_loaded_models()) if model_loader else 0
        }
    }

# Main query endpoint
@app.post("/api/v1/query", response_model=QueryResponse)
async def process_query(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    user_info: Dict = Depends(verify_api_key)
):
    """Process a query through the Council of Experts"""
    
    query_id = str(uuid.uuid4())
    log.info(f"Processing query {query_id} from user {user_info['username']}")
    
    try:
        # Get RAG context if enabled
        rag_context = None
        if request.use_rag and rag_service:
            rag_context = await rag_service.get_context(request.query)
        
        # Combine contexts
        full_context = request.context
        if rag_context:
            full_context = f"{request.context}\n\n{rag_context}" if request.context else rag_context
        
        # Process through council
        result = await council.process_query(
            query=request.query,
            task_type=request.task_type,
            context=full_context,
            num_experts=request.num_experts,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            use_cache=settings.ENABLE_CACHE,
            fast_mode=request.fast_mode
        )
        
        # Log query in background
        background_tasks.add_task(
            log_query,
            user_info["user_id"],
            query_id,
            request.dict(),
            result
        )
        
        return QueryResponse(
            query_id=query_id,
            **result
        )
        
    except Exception as e:
        log.error(f"Query processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# File upload endpoint
@app.post("/api/v1/upload")
async def upload_file(
    file: UploadFile = File(...),
    query: Optional[str] = None,
    process_type: str = "analyze",
    user_info: Dict = Depends(verify_api_key)
):
    """Upload and process a file"""
    
    log.info(f"Processing file upload: {file.filename}")
    
    # Check file size
    if file.size > settings.MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large")
    
    # Check file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=415, detail="File type not supported")
    
    try:
        # Save file
        file_path = settings.UPLOAD_DIR / f"{uuid.uuid4()}{file_ext}"
        content = await file.read()
        
        with open(file_path, "wb") as f:
            f.write(content)
        
        # Process based on file type
        if file_ext in [".jpg", ".jpeg", ".png", ".gif"]:
            result = await multimodal_service.process_image(file_path, query)
        elif file_ext in [".mp3", ".wav", ".m4a"]:
            result = await multimodal_service.process_audio(file_path)
        elif file_ext in [".pdf", ".docx", ".txt"]:
            # Add to RAG if specified
            if process_type == "add_to_knowledge":
                result = await rag_service.add_document(str(file_path))
            else:
                result = await multimodal_service.process_document(file_path)
        else:
            result = {"success": False, "error": "Unsupported file type"}
        
        return result
        
    except Exception as e:
        log.error(f"File processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Model management endpoints
@app.get("/api/v1/models")
async def list_models(user_info: Dict = Depends(verify_api_key)):
    """List available models"""
    
    from backend.core.model_loader import ModelRegistry
    
    models = []
    for key, profile in ModelRegistry.MODELS.items():
        model_info = model_loader.get_model_info(key) if model_loader else {}
        models.append({
            "key": key,
            "name": profile.name,
            "categories": [cat.value for cat in profile.categories],
            "memory_gb": profile.memory_gb,
            "speed_rating": profile.speed_rating,
            "quality_rating": profile.quality_rating,
            "is_loaded": model_info.get("is_loaded", False),
            "specialties": profile.specialties
        })
    
    return {
        "models": models,
        "loaded": model_loader.get_loaded_models() if model_loader else []
    }

# User management endpoint
@app.post("/api/v1/users")
async def create_user(
    user_data: UserCreate,
    user_info: Dict = Depends(verify_api_key)
):
    """Create a new user (admin only)"""
    
    if user_info["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        result = await auth_service.create_user(user_data)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

# Analytics endpoint
@app.get("/api/v1/analytics")
async def get_analytics(
    user_info: Dict = Depends(verify_api_key),
    days: int = 7
):
    """Get usage analytics"""
    
    if user_info["role"] not in ["admin", "developer"]:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    # Get analytics from database
    analytics = await db_client[settings.DATABASE_NAME].query_logs.aggregate([
        {
            "$match": {
                "timestamp": {
                    "$gte": datetime.utcnow() - timedelta(days=days)
                }
            }
        },
        {
            "$group": {
                "_id": "$task_type",
                "count": {"$sum": 1},
                "avg_time": {"$avg": "$processing_time"},
                "avg_confidence": {"$avg": "$confidence_score"}
            }
        }
    ]).to_list(None)
    
    return {
        "period_days": days,
        "analytics": analytics
    }

# Helper function for logging queries
async def log_query(user_id: str, query_id: str, request: Dict, result: Dict):
    """Log query for analytics"""
    try:
        log_entry = {
            "query_id": query_id,
            "user_id": user_id,
            "request": request,
            "task_type": result.get("task_type"),
            "experts_consulted": result.get("experts_consulted"),
            "processing_time": result.get("processing_time"),
            "confidence_score": result.get("confidence_score"),
            "timestamp": datetime.utcnow()
        }
        
        await db_client[settings.DATABASE_NAME].query_logs.insert_one(log_entry)
        
    except Exception as e:
        log.error(f"Failed to log query: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        workers=settings.API_WORKERS
    )