"""Context-Aware ESG Enhancement Module

This module provides context-aware capabilities for the lightweight ESG model:
1. ESG semantic understanding through domain-specific embeddings
2. Context-aware feature enhancement for better classification
3. Bias-free ESG domain knowledge integration
4. Memory-efficient implementation for training pipeline integration

Author: ESG Analysis System
Date: 2025-01-24
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
import json
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

class ESGContextAwareModule(nn.Module):
    """Context-aware module for ESG semantic understanding"""
    
    def __init__(self, hidden_size: int = 768, context_dim: int = 128, dropout_rate: float = 0.1):
        super(ESGContextAwareModule, self).__init__()
        
        self.hidden_size = hidden_size
        self.context_dim = context_dim
        
        # ESG domain knowledge embeddings (learned, not hardcoded)
        self.esg_category_embeddings = nn.Embedding(3, context_dim)  # E, S, G
        
        # Context attention mechanism
        self.context_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Context fusion layers
        self.context_proj = nn.Linear(hidden_size, context_dim)
        self.context_norm = nn.LayerNorm(context_dim)
        self.context_dropout = nn.Dropout(dropout_rate)
        
        # Enhanced feature fusion
        self.fusion_gate = nn.Linear(hidden_size + context_dim, hidden_size)
        self.fusion_norm = nn.LayerNorm(hidden_size)
        
        # Initialize embeddings with ESG-aware values
        self._initialize_esg_embeddings()
    
    def _initialize_esg_embeddings(self):
        """Initialize ESG category embeddings with domain knowledge"""
        # Initialize with small random values to avoid bias
        nn.init.normal_(self.esg_category_embeddings.weight, mean=0.0, std=0.02)
    
    def compute_esg_context_scores(self, hidden_states: torch.Tensor, 
                                 attention_mask: torch.Tensor) -> torch.Tensor:
        """Compute ESG context relevance scores without bias"""
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Apply attention mask
        masked_hidden = hidden_states * attention_mask.unsqueeze(-1)
        
        # Compute attention weights for ESG relevance
        context_query = self.context_proj(masked_hidden.mean(dim=1))  # [batch, context_dim]
        context_query = self.context_norm(context_query)
        
        # Get ESG category embeddings
        esg_embeddings = self.esg_category_embeddings.weight  # [3, context_dim]
        
        # Compute similarity scores (no hardcoded bias)
        context_scores = torch.matmul(context_query, esg_embeddings.T)  # [batch, 3]
        context_scores = F.softmax(context_scores, dim=-1)
        
        return context_scores
    
    def enhance_features_with_context(self, hidden_states: torch.Tensor,
                                    attention_mask: torch.Tensor) -> torch.Tensor:
        """Enhance features with ESG context awareness"""
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Self-attention for context modeling
        attended_features, _ = self.context_attention(
            hidden_states, hidden_states, hidden_states,
            key_padding_mask=~attention_mask.bool()
        )
        
        # Pool attended features
        pooled_attended = (attended_features * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        
        # Compute context scores
        context_scores = self.compute_esg_context_scores(hidden_states, attention_mask)
        
        # Project context scores to feature space
        context_features = torch.matmul(context_scores, self.esg_category_embeddings.weight)
        context_features = self.context_dropout(context_features)
        
        # Fusion gate for combining original and context features
        combined_features = torch.cat([pooled_attended, context_features], dim=-1)
        gate_weights = torch.sigmoid(self.fusion_gate(combined_features))
        
        # Enhanced features with gated fusion
        enhanced_features = gate_weights * pooled_attended + (1 - gate_weights) * pooled_attended
        enhanced_features = self.fusion_norm(enhanced_features)
        
        return enhanced_features, context_scores
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for context-aware enhancement"""
        enhanced_features, context_scores = self.enhance_features_with_context(hidden_states, attention_mask)
        
        return {
            'enhanced_features': enhanced_features,
            'context_scores': context_scores,
            'esg_relevance': context_scores.max(dim=-1)[0]  # Overall ESG relevance
        }

class ESGSemanticEnhancer:
    """Semantic enhancement utilities for ESG text processing"""
    
    def __init__(self, model_name: str = "ProsusAI/finbert"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self._esg_domain_terms = self._load_esg_domain_terms()
    
    def _load_esg_domain_terms(self) -> Dict[str, List[str]]:
        """Load ESG domain terms for semantic enhancement (bias-free)"""
        # Minimal, unbiased ESG domain terms
        return {
            "environmental": ["environmental", "climate", "energy", "emissions", "sustainability"],
            "social": ["social", "employee", "community", "diversity", "safety"],
            "governance": ["governance", "board", "ethics", "compliance", "transparency"]
        }
    
    def initialize_model(self):
        """Initialize the base model for semantic analysis"""
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.eval()
    
    def compute_semantic_relevance(self, text: str) -> Dict[str, float]:
        """Compute semantic relevance to ESG domains without bias"""
        if not self.model:
            self.initialize_model()
        
        text_lower = text.lower()
        relevance_scores = {}
        
        for domain, terms in self._esg_domain_terms.items():
            # Simple term frequency approach (unbiased)
            score = sum(1 for term in terms if term in text_lower)
            # Normalize by text length to avoid length bias
            relevance_scores[domain] = score / max(len(text.split()), 1)
        
        return relevance_scores
    
    def enhance_text_representation(self, text: str, max_length: int = 512) -> Dict[str, torch.Tensor]:
        """Enhance text representation with ESG semantic information"""
        if not self.model:
            self.initialize_model()
        
        # Tokenize and encode
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True,
            padding=True, max_length=max_length
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            hidden_states = outputs.last_hidden_state
            pooled_output = outputs.pooler_output
        
        # Compute semantic relevance
        semantic_scores = self.compute_semantic_relevance(text)
        
        return {
            'hidden_states': hidden_states,
            'pooled_output': pooled_output,
            'attention_mask': inputs['attention_mask'],
            'semantic_scores': semantic_scores
        }

def create_context_aware_enhancement(hidden_size: int = 768, 
                                   context_dim: int = 128,
                                   dropout_rate: float = 0.1) -> ESGContextAwareModule:
    """Factory function to create context-aware enhancement module"""
    return ESGContextAwareModule(
        hidden_size=hidden_size,
        context_dim=context_dim,
        dropout_rate=dropout_rate
    )

def load_esg_indicators(indicators_file: str = "data/indicators/final_esg_indicators.csv") -> List[Dict]:
    """Load ESG indicators for context-aware training"""
    try:
        import pandas as pd
        df = pd.read_csv(indicators_file)
        indicators = []
        for _, row in df.iterrows():
            indicators.append({
                'id': row.get('indicator_id', ''),
                'name': row.get('indicator_name', ''),
                'category': row.get('category', ''),
                'description': row.get('description', '')
            })
        return indicators
    except Exception as e:
        print(f"Warning: Could not load ESG indicators: {e}")
        return []

class ContextAwareTrainingMixin:
    """Mixin class for adding context-aware capabilities to training"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.context_module = None
        self.semantic_enhancer = None
    
    def initialize_context_awareness(self, hidden_size: int = 768):
        """Initialize context-aware components"""
        self.context_module = create_context_aware_enhancement(hidden_size=hidden_size)
        self.semantic_enhancer = ESGSemanticEnhancer()
        
        if hasattr(self, 'device'):
            self.context_module = self.context_module.to(self.device)
    
    def enhance_batch_with_context(self, batch: Dict[str, torch.Tensor], 
                                 hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Enhance batch with context-aware features"""
        if self.context_module is None:
            return {'enhanced_features': hidden_states.mean(dim=1)}
        
        attention_mask = batch.get('attention_mask')
        if attention_mask is None:
            attention_mask = torch.ones(hidden_states.shape[:2], device=hidden_states.device)
        
        context_output = self.context_module(hidden_states, attention_mask)
        return context_output

# Export main components
__all__ = [
    'ESGContextAwareModule',
    'ESGSemanticEnhancer', 
    'ContextAwareTrainingMixin',
    'create_context_aware_enhancement',
    'load_esg_indicators'
]