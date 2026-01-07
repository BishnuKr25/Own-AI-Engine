"""Task routing and orchestration"""
from typing import Dict, Any, Optional, List
from enum import Enum

from backend.schemas.models import TaskType
from backend.core.model_loader import TaskCategory, ModelRegistry
from backend.utils.logger import log

class TaskComplexity(Enum):
    """Task complexity levels"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"

class TaskRouter:
    """Intelligent task routing system"""
    
    def __init__(self):
        self.routing_rules = self._initialize_routing_rules()
        
    def _initialize_routing_rules(self) -> Dict:
        """Initialize routing rules"""
        return {
            TaskType.GENERAL: {
                "complexity_threshold": 0.5,
                "preferred_models": ["gemma3-27b", "llama3-70b"],
                "max_experts": 3
            },
            TaskType.CODING: {
                "complexity_threshold": 0.6,
                "preferred_models": ["codestral-22b", "codellama-70b"],
                "max_experts": 4
            },
            TaskType.ANALYSIS: {
                "complexity_threshold": 0.7,
                "preferred_models": ["gemma3-27b", "mixtral-8x22b"],
                "max_experts": 3
            },
            TaskType.CREATIVE: {
                "complexity_threshold": 0.4,
                "preferred_models": ["gemma3-27b", "llama3-70b"],
                "max_experts": 2
            },
            TaskType.DATA_GENERATION: {
                "complexity_threshold": 0.5,
                "preferred_models": ["mixtral-8x22b", "gemma3-27b"],
                "max_experts": 2
            },
            TaskType.QUICK: {
                "complexity_threshold": 0.2,
                "preferred_models": ["gemma3-2b"],
                "max_experts": 1
            }
        }
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze query and determine routing strategy"""
        
        # Calculate complexity
        complexity = self._calculate_complexity(query)
        
        # Determine task type
        task_type = self._classify_task(query)
        
        # Get routing strategy
        strategy = self.routing_rules.get(task_type, self.routing_rules[TaskType.GENERAL])
        
        # Determine number of experts
        if complexity.value == "simple":
            num_experts = 1
        elif complexity.value == "moderate":
            num_experts = 2
        elif complexity.value == "complex":
            num_experts = 3
        else:
            num_experts = strategy["max_experts"]
        
        # Select models
        selected_models = self._select_models(
            task_type=task_type,
            complexity=complexity,
            num_experts=num_experts
        )
        
        return {
            "task_type": task_type,
            "complexity": complexity.value,
            "num_experts": num_experts,
            "selected_models": selected_models,
            "estimated_time": self._estimate_processing_time(selected_models),
            "strategy": strategy
        }
    
    def _calculate_complexity(self, query: str) -> TaskComplexity:
        """Calculate query complexity"""
        
        score = 0.0
        
        # Length factor
        word_count = len(query.split())
        if word_count < 20:
            score += 0.1
        elif word_count < 50:
            score += 0.3
        elif word_count < 100:
            score += 0.5
        else:
            score += 0.7
        
        # Technical terms
        technical_terms = [
            "algorithm", "architecture", "optimize", "implement",
            "analyze", "complex", "integrate", "performance",
            "debug", "refactor", "scale", "distributed"
        ]
        tech_count = sum(1 for term in technical_terms if term in query.lower())
        score += min(tech_count * 0.1, 0.3)
        
        # Multi-part questions
        if query.count("?") > 1 or query.count(";") > 1:
            score += 0.2
        
        # Code indicators
        if "```" in query or "def " in query or "class " in query:
            score += 0.2
        
        # Normalize score
        score = min(score, 1.0)
        
        # Map to complexity level
        if score < 0.3:
            return TaskComplexity.SIMPLE
        elif score < 0.5:
            return TaskComplexity.MODERATE
        elif score < 0.8:
            return TaskComplexity.COMPLEX
        else:
            return TaskComplexity.VERY_COMPLEX
    
    def _classify_task(self, query: str) -> TaskType:
        """Classify task type from query"""
        
        query_lower = query.lower()
        
        # Pattern matching for task classification
        patterns = {
            TaskType.CODING: [
                "code", "function", "implement", "debug", "error",
                "class", "method", "api", "algorithm"
            ],
            TaskType.ANALYSIS: [
                "analyze", "evaluate", "compare", "assess",
                "review", "examine", "investigate"
            ],
            TaskType.CREATIVE: [
                "write", "story", "poem", "creative",
                "imagine", "fiction", "narrative"
            ],
            TaskType.DATA_GENERATION: [
                "generate", "csv", "json", "data",
                "dataset", "table", "excel"
            ],
            TaskType.SUMMARIZATION: [
                "summarize", "summary", "brief", "overview"
            ],
            TaskType.TRANSLATION: [
                "translate", "translation", "convert language"
            ]
        }
        
        # Score each task type
        scores = {}
        for task_type, keywords in patterns.items():
            score = sum(1 for kw in keywords if kw in query_lower)
            scores[task_type] = score
        
        # Get highest scoring task type
        if scores:
            best_match = max(scores, key=scores.get)
            if scores[best_match] > 0:
                return best_match
        
        # Default based on length
        if len(query.split()) < 10:
            return TaskType.QUICK
        
        return TaskType.GENERAL
    
    def _select_models(
        self,
        task_type: TaskType,
        complexity: TaskComplexity,
        num_experts: int
    ) -> List[str]:
        """Select appropriate models based on task and complexity"""
        
        strategy = self.routing_rules.get(task_type, self.routing_rules[TaskType.GENERAL])
        preferred = strategy["preferred_models"]
        
        selected = []
        
        # For simple tasks, use fast models
        if complexity == TaskComplexity.SIMPLE:
            if "gemma3-2b" in ModelRegistry.MODELS:
                selected.append("gemma3-2b")
        else:
            # Add preferred models
            for model in preferred:
                if model in ModelRegistry.MODELS:
                    selected.append(model)
                    if len(selected) >= num_experts:
                        break
        
        return selected[:num_experts]
    
    def _estimate_processing_time(self, models: List[str]) -> float:
        """Estimate processing time based on selected models"""
        
        if not models:
            return 0.0
        
        total_time = 0.0
        
        for model_key in models:
            profile = ModelRegistry.MODELS.get(model_key)
            if profile:
                # Estimate based on speed rating (1-10 scale)
                base_time = 10.0  # Base time in seconds
                model_time = base_time * (11 - profile.speed_rating) / 10
                total_time += model_time
        
        # Add overhead for synthesis
        total_time += 2.0
        
        return total_time