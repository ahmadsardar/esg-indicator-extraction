#!/usr/bin/env python3
"""
Context-Aware ESG Indicator Extraction Model

This model is designed to:
1. Extract known ESG indicators from the expanded 46-indicator set
2. Recognize NEW/UNSEEN ESG indicators by understanding ESG context patterns
3. Use contextual embeddings and semantic similarity to identify ESG-related content
4. Classify new indicators into Environmental, Social, or Governance categories

Approach:
- Fine-tune FinBERT on known indicators
- Use contextual pattern recognition for unknown indicators
- Implement semantic similarity matching
- Apply ESG domain knowledge for classification

Author: Thesis Project
Date: 2024
"""

import json
import torch
import numpy as np
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, pipeline
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import pandas as pd
from typing import List, Dict, Tuple, Any
import re
from datetime import datetime

class ContextAwareESGExtractor:
    def __init__(self, model_name: str = "ProsusAI/finbert"):
        """
        Initialize the context-aware ESG extraction model
        
        Args:
            model_name: Base model to use (default: FinBERT)
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.known_indicators = []
        self.esg_context_patterns = []
        self.category_embeddings = {}
        
        # ESG context keywords for pattern recognition
        self.esg_keywords = {
            "Environmental": [
                "carbon", "emissions", "greenhouse gas", "climate", "energy", "renewable",
                "waste", "recycling", "water", "biodiversity", "pollution", "sustainability",
                "environmental impact", "carbon footprint", "scope 1", "scope 2", "scope 3",
                "energy efficiency", "renewable energy", "waste reduction", "circular economy",
                "deforestation", "land use", "air quality", "water consumption", "toxic"
            ],
            "Social": [
                "employee", "diversity", "inclusion", "safety", "health", "training",
                "human rights", "community", "stakeholder", "customer", "supplier",
                "workplace", "gender", "equality", "labor", "social impact", "wellbeing",
                "discrimination", "harassment", "child labor", "forced labor", "privacy",
                "data protection", "product safety", "consumer", "local community"
            ],
            "Governance": [
                "board", "governance", "ethics", "compliance", "transparency", "accountability",
                "risk management", "audit", "corruption", "bribery", "executive compensation",
                "shareholder", "disclosure", "reporting", "oversight", "independence",
                "anti-corruption", "whistleblower", "conflict of interest", "regulatory",
                "internal controls", "board diversity", "executive", "management"
            ]
        }
        
    def load_known_indicators(self, indicators_file: str = "data/expanded_esg_indicators.json"):
        """
        Load the expanded set of known ESG indicators
        """
        with open(indicators_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.known_indicators = data['indicators']
        
        print(f"Loaded {len(self.known_indicators)} known ESG indicators")
        
        # Create indicator lookup by name and keywords
        self.indicator_lookup = {}
        for ind in self.known_indicators:
            # Add main name
            self.indicator_lookup[ind['name'].lower()] = ind
            
            # Add variations and keywords from description
            desc_words = re.findall(r'\b\w+\b', ind['description'].lower())
            for word in desc_words:
                if len(word) > 3:  # Only meaningful words
                    if word not in self.indicator_lookup:
                        self.indicator_lookup[word] = []
                    if isinstance(self.indicator_lookup[word], list):
                        self.indicator_lookup[word].append(ind)
                    else:
                        self.indicator_lookup[word] = [self.indicator_lookup[word], ind]
        
    def initialize_model(self):
        """
        Initialize the FinBERT model and tokenizer
        """
        print(f"Initializing {self.model_name} model...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        
        # Set model to evaluation mode
        self.model.eval()
        
        print("Model initialized successfully")
        
    def get_text_embedding(self, text: str) -> np.ndarray:
        """
        Get contextual embedding for a text using FinBERT
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, 
                               padding=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use [CLS] token embedding as sentence representation
            embedding = outputs.last_hidden_state[:, 0, :].numpy()
            
        return embedding.flatten()
        
    def build_category_embeddings(self):
        """
        Build representative embeddings for each ESG category
        """
        print("Building category embeddings...")
        
        for category, keywords in self.esg_keywords.items():
            # Create representative text for each category
            category_text = f"{category} indicators include: " + ", ".join(keywords[:10])
            embedding = self.get_text_embedding(category_text)
            self.category_embeddings[category] = embedding
            
        print("Category embeddings built")
        
    def extract_known_indicators(self, text: str) -> List[Dict]:
        """
        Extract known ESG indicators from text using keyword matching and context
        """
        found_indicators = []
        text_lower = text.lower()
        
        # Direct name matching
        for indicator in self.known_indicators:
            indicator_name = indicator['name'].lower()
            
            # Check for direct mentions or variations
            patterns = [
                indicator_name,
                indicator_name.replace('indicator', ''),
                indicator_name.replace('emissions', 'emission'),
                # Add more pattern variations as needed
            ]
            
            for pattern in patterns:
                if pattern in text_lower and len(pattern) > 3:
                    # Extract surrounding context
                    start_idx = text_lower.find(pattern)
                    context_start = max(0, start_idx - 100)
                    context_end = min(len(text), start_idx + len(pattern) + 100)
                    context = text[context_start:context_end]
                    
                    found_indicators.append({
                        'indicator': indicator,
                        'matched_text': text[start_idx:start_idx + len(pattern)],
                        'context': context,
                        'confidence': 0.9,  # High confidence for direct matches
                        'extraction_type': 'known_direct'
                    })
                    break
                    
        return found_indicators
        
    def detect_esg_context(self, text: str) -> Dict[str, float]:
        """
        Detect ESG context in text and return category scores
        """
        text_lower = text.lower()
        category_scores = {"Environmental": 0, "Social": 0, "Governance": 0}
        
        for category, keywords in self.esg_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    # Weight by keyword importance (longer keywords = higher weight)
                    weight = len(keyword.split()) * 0.1 + 0.1
                    score += weight
                    
            # Normalize by text length
            category_scores[category] = score / (len(text.split()) + 1)
            
        return category_scores
        
    def extract_potential_new_indicators(self, text: str, threshold: float = 0.05) -> List[Dict]:
        """
        Extract potential new ESG indicators using context analysis
        """
        potential_indicators = []
        
        # Split text into sentences
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:  # Skip very short sentences
                continue
                
            # Check if sentence has ESG context
            category_scores = self.detect_esg_context(sentence)
            max_score = max(category_scores.values())
            
            if max_score > threshold:
                # Determine most likely category
                predicted_category = max(category_scores, key=category_scores.get)
                
                # Extract potential indicator phrases using NLP patterns
                indicator_phrases = self.extract_indicator_phrases(sentence)
                
                for phrase in indicator_phrases:
                    # Check if it's not already a known indicator
                    if not self.is_known_indicator(phrase):
                        potential_indicators.append({
                            'phrase': phrase,
                            'sentence': sentence,
                            'predicted_category': predicted_category,
                            'category_scores': category_scores,
                            'confidence': max_score,
                            'extraction_type': 'new_contextual'
                        })
                        
        return potential_indicators
        
    def extract_indicator_phrases(self, sentence: str) -> List[str]:
        """
        Extract potential indicator phrases from a sentence using NLP patterns
        """
        phrases = []
        
        # Pattern 1: Noun phrases with measurement context
        measurement_patterns = [
            r'\b([a-zA-Z\s]+)\s+(?:increased|decreased|improved|reduced|measured|reported)\b',
            r'\b(?:our|the|total|annual)\s+([a-zA-Z\s]+)\s+(?:is|was|reached|achieved)\b',
            r'\b([a-zA-Z\s]+)\s+(?:rate|ratio|percentage|level|amount|volume)\b',
            r'\b([a-zA-Z\s]+)\s+(?:performance|efficiency|intensity|footprint)\b'
        ]
        
        for pattern in measurement_patterns:
            matches = re.finditer(pattern, sentence, re.IGNORECASE)
            for match in matches:
                phrase = match.group(1).strip()
                if len(phrase.split()) <= 4 and len(phrase) > 3:  # Reasonable phrase length
                    phrases.append(phrase)
                    
        # Pattern 2: ESG-specific noun phrases
        esg_patterns = [
            r'\b(\w+\s+emissions?)\b',
            r'\b(\w+\s+consumption)\b',
            r'\b(\w+\s+diversity)\b',
            r'\b(\w+\s+safety)\b',
            r'\b(\w+\s+governance)\b',
            r'\b(\w+\s+compliance)\b'
        ]
        
        for pattern in esg_patterns:
            matches = re.finditer(pattern, sentence, re.IGNORECASE)
            for match in matches:
                phrase = match.group(1).strip()
                phrases.append(phrase)
                
        return list(set(phrases))  # Remove duplicates
        
    def is_known_indicator(self, phrase: str) -> bool:
        """
        Check if a phrase matches a known indicator
        """
        phrase_lower = phrase.lower()
        
        # Check direct matches
        for indicator in self.known_indicators:
            if phrase_lower in indicator['name'].lower() or indicator['name'].lower() in phrase_lower:
                return True
                
        return False
        
    def semantic_similarity_classification(self, text: str) -> Dict[str, float]:
        """
        Use semantic similarity to classify text into ESG categories
        """
        if not self.category_embeddings:
            self.build_category_embeddings()
            
        text_embedding = self.get_text_embedding(text)
        similarities = {}
        
        for category, cat_embedding in self.category_embeddings.items():
            similarity = cosine_similarity(
                text_embedding.reshape(1, -1),
                cat_embedding.reshape(1, -1)
            )[0][0]
            similarities[category] = float(similarity)
            
        return similarities
        
    def extract_all_indicators(self, text: str) -> Dict[str, List]:
        """
        Main extraction method that finds both known and potential new indicators
        """
        results = {
            'known_indicators': [],
            'potential_new_indicators': [],
            'text_esg_scores': {},
            'semantic_similarities': {}
        }
        
        # Extract known indicators
        results['known_indicators'] = self.extract_known_indicators(text)
        
        # Extract potential new indicators
        results['potential_new_indicators'] = self.extract_potential_new_indicators(text)
        
        # Get overall ESG context scores
        results['text_esg_scores'] = self.detect_esg_context(text)
        
        # Get semantic similarities
        results['semantic_similarities'] = self.semantic_similarity_classification(text)
        
        return results
        
    def save_extraction_results(self, results: Dict, filename: str = None):
        """
        Save extraction results to JSON file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"results/esg_extraction_{timestamp}.json"
            
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = self.make_json_serializable(results)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
            
        print(f"Extraction results saved to {filename}")
        
    def make_json_serializable(self, obj):
        """
        Convert numpy arrays and other non-serializable objects to JSON-compatible format
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self.make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        else:
            return obj
            
    def generate_summary_report(self, results: Dict) -> str:
        """
        Generate a human-readable summary of extraction results
        """
        report = "\n=== ESG Indicator Extraction Summary ===\n\n"
        
        # Known indicators summary
        known_count = len(results['known_indicators'])
        report += f"Known ESG Indicators Found: {known_count}\n"
        
        if known_count > 0:
            report += "\nKnown Indicators:\n"
            for item in results['known_indicators']:
                indicator = item['indicator']
                report += f"  - {indicator['name']} ({indicator['category']})\n"
                report += f"    Confidence: {item['confidence']:.2f}\n"
                
        # Potential new indicators summary
        new_count = len(results['potential_new_indicators'])
        report += f"\nPotential New ESG Indicators: {new_count}\n"
        
        if new_count > 0:
            report += "\nPotential New Indicators:\n"
            for item in results['potential_new_indicators']:
                report += f"  - '{item['phrase']}' (Predicted: {item['predicted_category']})\n"
                report += f"    Confidence: {item['confidence']:.3f}\n"
                
        # ESG context analysis
        report += "\nESG Context Analysis:\n"
        for category, score in results['text_esg_scores'].items():
            report += f"  {category}: {score:.3f}\n"
            
        # Semantic similarity
        report += "\nSemantic Similarity to ESG Categories:\n"
        for category, similarity in results['semantic_similarities'].items():
            report += f"  {category}: {similarity:.3f}\n"
            
        return report

def main():
    """Main execution function for testing"""
    
    # Initialize the extractor
    extractor = ContextAwareESGExtractor()
    
    # Load known indicators
    extractor.load_known_indicators()
    
    # Initialize model
    extractor.initialize_model()
    
    # Test text with both known and potential new indicators
    test_text = """
    Our company achieved a 25% reduction in carbon emissions this year, improving our 
    energy efficiency significantly. We also implemented new biodiversity protection 
    measures and enhanced our employee wellbeing programs. The board diversity ratio 
    increased to 40% women representation. Additionally, we introduced innovative 
    circular packaging solutions and improved our supply chain transparency metrics.
    Our new digital inclusion initiatives reached 10,000 community members.
    """
    
    print("Testing Context-Aware ESG Extraction...")
    print(f"Test text length: {len(test_text)} characters\n")
    
    # Extract indicators
    results = extractor.extract_all_indicators(test_text)
    
    # Generate and print summary
    summary = extractor.generate_summary_report(results)
    print(summary)
    
    # Save results
    extractor.save_extraction_results(results, "results/test_extraction_results.json")
    
if __name__ == "__main__":
    main()