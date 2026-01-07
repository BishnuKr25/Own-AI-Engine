"""Production Council of Experts Implementation"""
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import json
import hashlib

from backend.core.model_loader import ProductionModelLoader, ModelRegistry, TaskCategory
from backend.schemas.models import TaskType
from backend.config import settings
from backend.utils.logger import log

@dataclass
class ExpertResponse:
    """Response from an expert model"""
    model_key: str
    model_name: str
    response: str
    confidence: float
    processing_time: float
    tokens_generated: int
    metadata: Dict[str, Any]

class CouncilOfExperts:
    """Production Council of Experts with Gemma 3 optimization"""
    
    # Expert selection strategies with Gemma 3 27B added
    EXPERT_STRATEGIES = {
        TaskType.GENERAL: {
            "primary": ["gemma3-27b", "llama3-70b", "mixtral-8x22b"],
            "secondary": ["gemma3-9b", "nemotron-70b"],
            "fast": ["gemma3-2b"]
        },
        TaskType.CODING: {
            "primary": ["codestral-22b", "codellama-70b", "deepseek-coder-33b"],
            "secondary": ["qwen-coder-32b", "gemma3-27b"],
            "fast": ["gemma3-9b"]
        },
        TaskType.ANALYSIS: {
            "primary": ["gemma3-27b", "mixtral-8x22b", "llama3-70b"],
            "secondary": ["nemotron-70b", "gemma3-9b"],
            "fast": ["gemma3-2b"]
        },
        TaskType.CREATIVE: {
            "primary": ["gemma3-27b", "llama3-70b", "mixtral-8x22b"],
            "secondary": ["gemma3-9b"],
            "fast": ["gemma3-2b"]
        },
        TaskType.DATA_GENERATION: {
            "primary": ["mixtral-8x22b", "gemma3-27b"],
            "secondary": ["llama3-70b", "gemma3-9b"],
            "fast": ["gemma3-2b"]
        },
        TaskType.QUICK: {
            "primary": ["gemma3-2b"],
            "secondary": ["gemma3-9b"],
            "fast": ["gemma3-2b"]
        }
    }
    
    def __init__(self, model_loader: ProductionModelLoader):
        self.model_loader = model_loader
        self.preloaded_models = {}
        self.response_cache = {}
        self.synthesis_model = None
        
    async def initialize(self):
        """Initialize council with preloaded models"""
        log.info("Initializing Council of Experts...")
        
        # Preload fast models if enabled
        if settings.ENABLE_GEMMA3_FAST_MODE:
            for model_key in settings.PRELOAD_MODELS:
                try:
                    log.info(f"Preloading {model_key} for fast responses...")
                    model = await self.model_loader.load_model_async(model_key)
                    self.preloaded_models[model_key] = model
                    log.info(f"✓ Preloaded {model_key}")
                except Exception as e:
                    log.error(f"Failed to preload {model_key}: {str(e)}")
        
        log.info("✓ Council of Experts initialized")
    
    async def process_query(
        self,
        query: str,
        task_type: Optional[TaskType] = None,
        context: Optional[str] = None,
        num_experts: int = 3,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        use_cache: bool = True,
        fast_mode: bool = False
    ) -> Dict[str, Any]:
        """Process a query through the council"""
        
        start_time = datetime.utcnow()
        query_id = self._generate_query_id(query)
        
        log.info(f"Processing query {query_id[:8]}... Type: {task_type}, Fast: {fast_mode}")
        
        # Check cache if enabled
        if use_cache and query_id in self.response_cache:
            log.info(f"Cache hit for query {query_id[:8]}")
            cached = self.response_cache[query_id]
            cached["from_cache"] = True
            return cached
        
        # Determine task type if not provided
        if not task_type:
            task_type = self._classify_query(query)
            log.info(f"Auto-classified query as: {task_type}")
        
        # Select experts based on strategy
        selected_experts = self._select_experts(
            task_type=task_type,
            num_experts=num_experts,
            fast_mode=fast_mode
        )
        
        log.info(f"Selected experts: {selected_experts}")
        
        # Prepare enhanced query
        enhanced_query = self._prepare_query(query, context)
        
        # Get responses from experts
        expert_responses = await self._get_expert_responses(
            query=enhanced_query,
            experts=selected_experts,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Synthesize final response
        final_answer = await self._synthesize_responses(
            query=query,
            responses=expert_responses,
            task_type=task_type
        )
        
        # Calculate metrics
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        total_tokens = sum(r.tokens_generated for r in expert_responses)
        avg_confidence = sum(r.confidence for r in expert_responses) / len(expert_responses) if expert_responses else 0
        
        result = {
            "query_id": query_id,
            "success": True,
            "answer": final_answer,
            "task_type": task_type,
            "experts_consulted": [r.model_key for r in expert_responses],
            "confidence_score": avg_confidence,
            "processing_time": processing_time,
            "tokens_used": total_tokens,
            "metadata": {
                "num_experts": len(expert_responses),
                "fast_mode": fast_mode,
                "used_cache": False,
                "individual_responses": [
                    {
                        "model": r.model_name,
                        "confidence": r.confidence,
                        "time": r.processing_time,
                        "tokens": r.tokens_generated
                    }
                    for r in expert_responses
                ]
            },
            "timestamp": datetime.utcnow()
        }
        
        # Cache the result
        if use_cache:
            self.response_cache[query_id] = result
            # Limit cache size
            if len(self.response_cache) > 100:
                oldest = min(self.response_cache.items(), key=lambda x: x[1].get("timestamp", datetime.min))
                del self.response_cache[oldest[0]]
        
        return result
    
    def _classify_query(self, query: str) -> TaskType:
        """Classify query into task type"""
        query_lower = query.lower()
        
        # Keyword-based classification
        if any(kw in query_lower for kw in ["write code", "implement", "function", "debug", "fix bug"]):
            return TaskType.CODING
        elif any(kw in query_lower for kw in ["analyze", "evaluate", "compare", "assess"]):
            return TaskType.ANALYSIS
        elif any(kw in query_lower for kw in ["story", "poem", "creative", "imagine"]):
            return TaskType.CREATIVE
        elif any(kw in query_lower for kw in ["generate data", "csv", "json", "dataset"]):
            return TaskType.DATA_GENERATION
        elif any(kw in query_lower for kw in ["summarize", "summary", "brief"]):
            return TaskType.SUMMARIZATION
        elif any(kw in query_lower for kw in ["translate", "translation"]):
            return TaskType.TRANSLATION
        elif len(query.split()) < 10:
            return TaskType.QUICK
        else:
            return TaskType.GENERAL
    
    def _select_experts(
        self,
        task_type: TaskType,
        num_experts: int,
        fast_mode: bool
    ) -> List[str]:
        """Select appropriate experts for the task"""
        
        strategy = self.EXPERT_STRATEGIES.get(task_type, self.EXPERT_STRATEGIES[TaskType.GENERAL])
        
        if fast_mode:
            # Use fast models only
            return strategy["fast"][:1]
        
        # Select primary experts first
        selected = []
        for expert in strategy["primary"]:
            if expert in ModelRegistry.MODELS:
                selected.append(expert)
                if len(selected) >= num_experts:
                    break
        
        # Add secondary experts if needed
        if len(selected) < num_experts:
            for expert in strategy.get("secondary", []):
                if expert in ModelRegistry.MODELS and expert not in selected:
                    selected.append(expert)
                    if len(selected) >= num_experts:
                        break
        
        return selected[:num_experts]
    
    def _prepare_query(self, query: str, context: Optional[str]) -> str:
        """Prepare query with context"""
        if not context:
            return query
        
        return f"""Context:
{context}

Query:
{query}

Please provide a comprehensive and accurate response based on the context provided."""
    
    async def _get_expert_responses(
        self,
        query: str,
        experts: List[str],
        temperature: float,
        max_tokens: int
    ) -> List[ExpertResponse]:
        """Get responses from selected experts"""
        
        responses = []
        
        for expert_key in experts:
            try:
                start = datetime.utcnow()
                
                # Check if preloaded
                if expert_key in self.preloaded_models:
                    model = self.preloaded_models[expert_key]
                else:
                    model = await self.model_loader.load_model_async(expert_key)
                
                # Generate response
                response_text = model(
                    query,
                    temperature=temperature,
                    max_new_tokens=max_tokens
                )
                
                # Calculate metrics
                processing_time = (datetime.utcnow() - start).total_seconds()
                tokens_generated = len(response_text.split())
                confidence = self._calculate_confidence(response_text)
                
                # Create expert response
                profile = ModelRegistry.MODELS[expert_key]
                expert_response = ExpertResponse(
                    model_key=expert_key,
                    model_name=profile.name,
                    response=response_text,
                    confidence=confidence,
                    processing_time=processing_time,
                    tokens_generated=tokens_generated,
                    metadata={}
                )
                
                responses.append(expert_response)
                log.info(f"✓ Got response from {profile.name} ({processing_time:.2f}s)")
                
                # Unload model if not preloaded (memory management)
                if expert_key not in self.preloaded_models and expert_key != "gemma3-27b":
                    self.model_loader.unload_model(expert_key)
                
            except Exception as e:
                log.error(f"Error getting response from {expert_key}: {str(e)}")
                continue
        
        return responses
    
    async def _synthesize_responses(
        self,
        query: str,
        responses: List[ExpertResponse],
        task_type: TaskType
    ) -> str:
        """Synthesize multiple expert responses into final answer"""
        
        if not responses:
            return "I apologize, but I couldn't generate a response at this time. Please try again."
        
        if len(responses) == 1:
            return responses[0].response
        
        # Use Gemma 3 27B for synthesis if available
        synthesis_model_key = "gemma3-27b"
        
        try:
            if not self.synthesis_model:
                self.synthesis_model = await self.model_loader.load_model_async(synthesis_model_key)
            
            # Prepare synthesis prompt
            responses_text = "\n\n".join([
                f"Expert {i+1} ({r.model_name}, Confidence: {r.confidence:.2f}):\n{r.response}"
                for i, r in enumerate(responses)
            ])
            
            synthesis_prompt = f"""You are tasked with synthesizing multiple expert responses into a single, comprehensive answer.

Original Query: {query}
Task Type: {task_type}

Expert Responses:
{responses_text}

Please synthesize these responses into a single, high-quality answer that:
1. Combines the best insights from all experts
2. Resolves any contradictions
3. Provides a complete and coherent response
4. Maintains accuracy and clarity

Synthesized Answer:"""
            
            final_answer = self.synthesis_model(synthesis_prompt, temperature=0.3, max_new_tokens=2048)
            
            return final_answer.strip()
            
        except Exception as e:
            log.error(f"Synthesis failed: {str(e)}")
            # Fallback: return highest confidence response
            best_response = max(responses, key=lambda x: x.confidence)
            return best_response.response
    
    def _calculate_confidence(self, response: str) -> float:
        """Calculate confidence score for a response"""
        
        if not response:
            return 0.0
        
        # Basic heuristics for confidence
        score = 0.5  # Base score
        
        # Length factor
        word_count = len(response.split())
        if word_count > 50:
            score += 0.2
        if word_count > 200:
            score += 0.1
        
        # Structure indicators
        if any(marker in response for marker in ["1.", "2.", "First", "Second", "```"]):
            score += 0.1
        
        # Uncertainty markers (negative)
        uncertainty_markers = [
            "i'm not sure", "i don't know", "perhaps", "maybe",
            "might be", "could be", "uncertain"
        ]
        for marker in uncertainty_markers:
            if marker in response.lower():
                score -= 0.1
        
        return max(0.0, min(1.0, score))
    
    def _generate_query_id(self, query: str) -> str:
        """Generate unique ID for query"""
        return hashlib.sha256(query.encode()).hexdigest()