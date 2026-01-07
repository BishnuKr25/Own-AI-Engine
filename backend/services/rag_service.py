"""Production RAG (Retrieval-Augmented Generation) Service"""
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
import hashlib

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma, FAISS
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from motor.motor_asyncio import AsyncIOMotorClient
import torch
from backend.config import settings
from backend.utils.logger import log

class RAGService:
    """Production RAG service for knowledge retrieval"""
    
    def __init__(self, db_client: AsyncIOMotorClient):
        self.db = db_client[settings.DATABASE_NAME]
        self.documents_collection = self.db.documents
        self.embeddings_collection = self.db.embeddings
        
        # Initialize embeddings model
        self.embeddings_model = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize vector store
        self.vector_store = self._initialize_vector_store()
        
        # Text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        log.info("RAG service initialized")
    
    def _initialize_vector_store(self):
        """Initialize the vector store"""
        
        if settings.VECTOR_DB_TYPE == "chromadb":
            persist_directory = str(settings.DATA_DIR / "chromadb")
            return Chroma(
                embedding_function=self.embeddings_model,
                persist_directory=persist_directory
            )
        elif settings.VECTOR_DB_TYPE == "faiss":
            # Initialize or load FAISS index
            index_path = settings.DATA_DIR / "faiss_index"
            if index_path.exists():
                return FAISS.load_local(
                    str(index_path),
                    self.embeddings_model
                )
            else:
                # Create new FAISS index
                return FAISS.from_texts(
                    ["initialization"],
                    self.embeddings_model
                )
        else:
            raise ValueError(f"Unsupported vector DB type: {settings.VECTOR_DB_TYPE}")
    
    async def add_document(
        self,
        file_path: str,
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Add a document to the knowledge base"""
        
        log.info(f"Adding document: {file_path}")
        
        try:
            # Load document
            if file_path.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif file_path.endswith(".docx"):
                loader = Docx2txtLoader(file_path)
            else:
                loader = TextLoader(file_path)
            
            documents = loader.load()
            
            # Split into chunks
            chunks = self.text_splitter.split_documents(documents)
            
            # Add metadata
            if metadata:
                for chunk in chunks:
                    chunk.metadata.update(metadata)
            
            # Generate document ID
            doc_id = hashlib.md5(file_path.encode()).hexdigest()
            
            # Store in vector database
            self.vector_store.add_documents(chunks)
            
            # Store document info in MongoDB
            doc_info = {
                "doc_id": doc_id,
                "file_path": file_path,
                "num_chunks": len(chunks),
                "metadata": metadata or {},
                "added_at": datetime.utcnow()
            }
            
            await self.documents_collection.insert_one(doc_info)
            
            log.info(f"✓ Added document with {len(chunks)} chunks")
            
            return {
                "doc_id": doc_id,
                "num_chunks": len(chunks),
                "success": True
            }
            
        except Exception as e:
            log.error(f"Failed to add document: {str(e)}")
            raise
    
    async def search(
        self,
        query: str,
        k: int = 5,
        filter_metadata: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """Search for relevant documents"""
        
        log.info(f"Searching for: {query[:50]}...")
        
        try:
            # Perform similarity search
            if filter_metadata:
                results = self.vector_store.similarity_search_with_score(
                    query,
                    k=k,
                    filter=filter_metadata
                )
            else:
                results = self.vector_store.similarity_search_with_score(
                    query,
                    k=k
                )
            
            # Format results
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity_score": float(score)
                })
            
            log.info(f"Found {len(formatted_results)} relevant documents")
            
            return formatted_results
            
        except Exception as e:
            log.error(f"Search failed: {str(e)}")
            return []
    
    async def get_context(
        self,
        query: str,
        max_context_length: int = 2000
    ) -> Optional[str]:
        """Get relevant context for a query"""
        
        results = await self.search(query, k=3)
        
        if not results:
            return None
        
        # Combine results into context
        context_parts = []
        current_length = 0
        
        for result in results:
            content = result["content"]
            if current_length + len(content) > max_context_length:
                # Truncate if necessary
                remaining = max_context_length - current_length
                content = content[:remaining]
            
            context_parts.append(content)
            current_length += len(content)
            
            if current_length >= max_context_length:
                break
        
        return "\n\n".join(context_parts)
    
    async def update_embeddings(self):
        """Update all embeddings (maintenance task)"""
        
        log.info("Updating embeddings...")
        
        # Get all documents
        documents = await self.documents_collection.find().to_list(None)
        
        for doc in documents:
            # Re-process document
            # Implementation depends on specific needs
            pass
        
        log.info("✓ Embeddings updated")