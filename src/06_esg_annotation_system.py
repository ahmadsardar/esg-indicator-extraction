"""
ESG Annotation System
Enhances existing ESG datasets with comprehensive labeling and annotation
for training FinBERT model on 114 ESG indicators.
"""

import pandas as pd
import numpy as np
import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

class ESGAnnotationSystem:
    def __init__(self):
        self.data_dir = Path('data')
        self.indicators_dir = self.data_dir / 'indicators'
        self.processed_dir = self.data_dir / 'processed' / 'corporate_reports'
        self.annotations_dir = self.data_dir / 'annotations'
        
        # Create annotations directory if it doesn't exist
        self.annotations_dir.mkdir(exist_ok=True)
        
        # Load final ESG indicators
        self.esg_indicators = self.load_esg_indicators()
        
        # Enhanced TF-IDF vectorizer for better semantic matching
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 4),  # Include 4-grams for better phrase matching
            lowercase=True,
            min_df=1,  # Include rare terms that might be important
            max_df=0.95,  # Remove very common terms
            sublinear_tf=True,  # Use sublinear term frequency scaling
            norm='l2'  # L2 normalization
        )
        
        # Enhanced numerical patterns for better value extraction
        self.numerical_patterns = [
            # Percentage patterns
            r'(\d+(?:[.,]\d+)*)\s*(%|percent|percentage|per cent)',
            # Weight/mass patterns
            r'(\d+(?:[.,]\d+)*)\s*(tons?|tonnes?|kt|Mt|Gt|kg|g|mg|lbs?|pounds?)',
            # Energy patterns
            r'(\d+(?:[.,]\d+)*)\s*(GWh|MWh|kWh|TWh|TJ|GJ|MJ|kJ|BTU)',
            # Volume patterns
            r'(\d+(?:[.,]\d+)*)\s*(m³|cubic meters?|liters?|litres?|gallons?|barrels?)',
            # Emissions patterns
            r'(\d+(?:[.,]\d+)*)\s*(tCO2e?|CO2|CO₂|emissions?|carbon|greenhouse gas)',
            # Currency patterns
            r'(\d+(?:[.,]\d+)*)\s*(€|EUR|\$|USD|million|billion|trillion|thousand)',
            # People patterns
            r'(\d+(?:[.,]\d+)*)\s*(employees?|workers?|people|staff|workforce|FTE)',
            # Time patterns
            r'(\d+(?:[.,]\d+)*)\s*(hours?|days?|weeks?|months?|years?|minutes?)',
            # Area patterns
            r'(\d+(?:[.,]\d+)*)\s*(hectares?|km²|m²|acres?|square\s+(?:meters?|kilometres?|miles?))',
            # Water patterns
            r'(\d+(?:[.,]\d+)*)\s*(water|H2O|consumption|usage|withdrawal)',
            # Waste patterns
            r'(\d+(?:[.,]\d+)*)\s*(waste|recycled?|landfill|disposal)',
            # General numerical with units
            r'(\d+(?:[.,]\d+)*)\s*([A-Za-z]+(?:/[A-Za-z]+)?)',
        ]
        
    def load_esg_indicators(self) -> pd.DataFrame:
        """Load the final ESG indicators dataset"""
        indicators_path = self.indicators_dir / 'final_esg_indicators.csv'
        if not indicators_path.exists():
            raise FileNotFoundError(f"Final ESG indicators not found at {indicators_path}")
        
        df = pd.read_csv(indicators_path)
        print(f"Loaded {len(df)} ESG indicators for annotation")
        return df
    
    def load_individual_datasets(self) -> Dict[str, pd.DataFrame]:
        """Load individual ESG datasets from each corporate report"""
        individual_reports_dir = self.processed_dir / 'individual_reports'
        if not individual_reports_dir.exists():
            raise FileNotFoundError(f"Individual reports directory not found at {individual_reports_dir}")
        
        datasets = {}
        csv_files = list(individual_reports_dir.glob('*_esg_dataset.csv'))
        
        if not csv_files:
            raise FileNotFoundError(f"No individual ESG datasets found in {individual_reports_dir}")
        
        for csv_file in csv_files:
            # Extract report name from filename
            report_name = csv_file.stem.replace('_esg_dataset', '')
            df = pd.read_csv(csv_file)
            df['document_id'] = report_name  # Add document identifier
            datasets[report_name] = df
            print(f"Loaded {report_name}: {len(df)} text segments")
        
        print(f"Total datasets loaded: {len(datasets)}")
        return datasets
    
    def combine_datasets(self, datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Combine individual datasets while preserving document information"""
        combined_dfs = []
        for report_name, df in datasets.items():
            df_copy = df.copy()
            df_copy['document_id'] = report_name
            combined_dfs.append(df_copy)
        
        combined_df = pd.concat(combined_dfs, ignore_index=True)
        print(f"Combined dataset: {len(combined_df)} total text segments from {len(datasets)} documents")
        return combined_df
    
    def extract_numerical_values(self, text: str) -> List[Dict]:
        """Enhanced extraction of numerical values and their units from text"""
        values = []
        seen_positions = set()  # Avoid duplicate extractions
        
        for pattern in self.numerical_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # Skip if we've already extracted from this position
                if any(abs(match.start() - pos) < 10 for pos in seen_positions):
                    continue
                    
                value_str = match.group(1).replace(',', '.').replace(' ', '')
                unit = match.group(2) if len(match.groups()) > 1 else ''
                
                try:
                    value = float(value_str)
                    
                    # Get surrounding context for better understanding
                    start_context = max(0, match.start() - 50)
                    end_context = min(len(text), match.end() + 50)
                    context = text[start_context:end_context].strip()
                    
                    # Classify value type based on unit and context
                    value_type = self.classify_value_type(unit, context)
                    
                    values.append({
                        'value': value,
                        'unit': unit.lower(),
                        'value_type': value_type,
                        'raw_text': match.group(0),
                        'context': context,
                        'position': match.span(),
                        'confidence': self.calculate_extraction_confidence(value, unit, context)
                    })
                    
                    seen_positions.add(match.start())
                    
                except ValueError:
                    continue
        
        # Sort by confidence and remove low-confidence extractions
        values = [v for v in values if v['confidence'] > 0.3]
        values.sort(key=lambda x: x['confidence'], reverse=True)
        
        return values
    
    def classify_value_type(self, unit: str, context: str) -> str:
        """Classify the type of numerical value based on unit and context"""
        unit_lower = unit.lower()
        context_lower = context.lower()
        
        if any(keyword in unit_lower for keyword in ['%', 'percent', 'percentage']):
            return 'percentage'
        elif any(keyword in unit_lower for keyword in ['co2', 'emission', 'carbon', 'greenhouse']):
            return 'emissions'
        elif any(keyword in unit_lower for keyword in ['eur', '$', 'usd', 'million', 'billion']):
            return 'financial'
        elif any(keyword in unit_lower for keyword in ['employee', 'worker', 'people', 'staff', 'fte']):
            return 'workforce'
        elif any(keyword in unit_lower for keyword in ['gwh', 'mwh', 'kwh', 'energy', 'tj', 'gj']):
            return 'energy'
        elif any(keyword in unit_lower for keyword in ['ton', 'kg', 'g', 'weight', 'mass']):
            return 'mass'
        elif any(keyword in unit_lower for keyword in ['water', 'liter', 'gallon', 'volume']):
            return 'volume'
        elif any(keyword in unit_lower for keyword in ['waste', 'recycl', 'disposal']):
            return 'waste'
        elif any(keyword in context_lower for keyword in ['target', 'goal', 'objective']):
            return 'target'
        else:
            return 'general'
    
    def calculate_extraction_confidence(self, value: float, unit: str, context: str) -> float:
        """Calculate confidence score for numerical extraction"""
        confidence = 0.5  # Base confidence
        
        # Value reasonableness
        if 0 < value < 1e12:  # Reasonable range
            confidence += 0.2
        
        # Unit specificity
        if unit and len(unit) > 1:
            confidence += 0.2
        
        # Context relevance
        esg_keywords = ['sustainability', 'environmental', 'social', 'governance', 'emission', 
                       'energy', 'waste', 'water', 'employee', 'diversity', 'safety']
        if any(keyword in context.lower() for keyword in esg_keywords):
            confidence += 0.3
        
        return min(confidence, 1.0)
    
    def preprocess_text(self, text: str) -> str:
        """Enhanced text preprocessing for better similarity matching"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Expand common ESG abbreviations
        abbreviations = {
            'ghg': 'greenhouse gas',
            'co2': 'carbon dioxide',
            'esg': 'environmental social governance',
            'csr': 'corporate social responsibility',
            'sdg': 'sustainable development goal',
            'tcfd': 'task force climate financial disclosure',
            'sasb': 'sustainability accounting standards board',
            'gri': 'global reporting initiative'
        }
        
        for abbr, full in abbreviations.items():
            text = re.sub(r'\b' + abbr + r'\b', full, text)
        
        return text.strip()
    
    def calculate_semantic_similarity(self, text: str, indicator_descriptions: List[str]) -> np.ndarray:
        """Enhanced semantic similarity calculation with multiple measures"""
        try:
            # Preprocess text and descriptions
            processed_text = self.preprocess_text(text)
            processed_descriptions = [self.preprocess_text(desc) for desc in indicator_descriptions]
            
            # Combine text with indicator descriptions
            all_texts = [processed_text] + processed_descriptions
            
            # Fit and transform
            tfidf_matrix = self.vectorizer.fit_transform(all_texts)
            
            # Calculate cosine similarity
            cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
            
            # Calculate keyword overlap similarity
            keyword_similarities = self.calculate_keyword_similarity(processed_text, processed_descriptions)
            
            # Combine similarities with weights
            combined_similarities = (0.7 * cosine_similarities + 0.3 * keyword_similarities)
            
            return combined_similarities
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            return np.zeros(len(indicator_descriptions))
    
    def calculate_keyword_similarity(self, text: str, descriptions: List[str]) -> np.ndarray:
        """Calculate keyword-based similarity for ESG-specific terms"""
        # ESG-specific keywords with weights
        esg_keywords = {
            'environmental': 2.0, 'emission': 2.0, 'carbon': 2.0, 'energy': 2.0, 'water': 2.0,
            'waste': 2.0, 'biodiversity': 2.0, 'climate': 2.0, 'renewable': 2.0, 'pollution': 2.0,
            'social': 2.0, 'employee': 2.0, 'diversity': 2.0, 'safety': 2.0, 'human rights': 2.0,
            'community': 2.0, 'training': 1.5, 'health': 1.5, 'workplace': 1.5, 'labor': 1.5,
            'governance': 2.0, 'ethics': 2.0, 'compliance': 2.0, 'transparency': 2.0, 'board': 1.5,
            'risk': 1.5, 'audit': 1.5, 'stakeholder': 1.5, 'accountability': 1.5, 'integrity': 1.5,
            'sustainable': 2.0, 'responsibility': 1.5, 'target': 1.5, 'goal': 1.5, 'performance': 1.0
        }
        
        text_words = set(text.split())
        similarities = []
        
        for desc in descriptions:
            desc_words = set(desc.split())
            
            # Calculate weighted keyword overlap
            overlap_score = 0
            total_weight = 0
            
            for keyword, weight in esg_keywords.items():
                if keyword in text and keyword in desc:
                    overlap_score += weight
                total_weight += weight
            
            # Normalize by total possible weight
            similarity = overlap_score / total_weight if total_weight > 0 else 0
            similarities.append(similarity)
        
        return np.array(similarities)
    
    def enhance_indicator_matching(self, text: str, current_indicator_id: str) -> List[Dict]:
        """Enhanced indicator matching with multi-level thresholds and context awareness"""
        # Prepare indicator descriptions for similarity calculation
        indicator_descriptions = []
        indicator_info = []
        
        for _, indicator in self.esg_indicators.iterrows():
            # Enhanced description combining multiple fields with weights
            desc_parts = [
                indicator['name'] * 2,  # Name is most important
                indicator['description'],
                indicator['subcategory'],
                indicator.get('keywords', ''),  # If keywords field exists
                indicator.get('framework', '')   # Framework context
            ]
            desc = ' '.join(str(part) for part in desc_parts if pd.notna(part))
            indicator_descriptions.append(desc)
            
            indicator_info.append({
                'indicator_id': indicator['indicator_id'],
                'name': indicator['name'],
                'category': indicator['category'],
                'subcategory': indicator['subcategory'],
                'description': indicator['description'],
                'framework': indicator.get('framework', ''),
                'source': indicator.get('source', '')
            })
        
        # Calculate similarities
        similarities = self.calculate_semantic_similarity(text, indicator_descriptions)
        
        # Enhanced matching with multiple thresholds
        matches = []
        text_length = len(text)
        
        for i, similarity in enumerate(similarities):
            # Dynamic threshold based on text length and content
            base_threshold = 0.05  # Lower base threshold
            
            # Adjust threshold based on text characteristics
            if text_length > 500:
                threshold = base_threshold + 0.05  # Higher threshold for longer texts
            elif text_length < 200:
                threshold = base_threshold - 0.02  # Lower threshold for shorter texts
            else:
                threshold = base_threshold
            
            # Additional boost for exact keyword matches
            keyword_boost = self.calculate_keyword_boost(text, indicator_info[i])
            adjusted_similarity = similarity + keyword_boost
            
            if adjusted_similarity > threshold:
                match_info = indicator_info[i].copy()
                match_info['similarity_score'] = float(adjusted_similarity)
                match_info['base_similarity'] = float(similarity)
                match_info['keyword_boost'] = float(keyword_boost)
                match_info['is_primary'] = (match_info['indicator_id'] == current_indicator_id)
                match_info['match_quality'] = self.assess_match_quality(adjusted_similarity, keyword_boost)
                matches.append(match_info)
        
        # Sort by adjusted similarity score
        matches.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        # Return top matches with quality filtering
        high_quality_matches = [m for m in matches if m['match_quality'] in ['high', 'medium']]
        if len(high_quality_matches) >= 3:
            return high_quality_matches[:5]
        else:
            return matches[:7]  # Include more matches if few high-quality ones
    
    def calculate_keyword_boost(self, text: str, indicator_info: Dict) -> float:
        """Calculate additional boost for exact keyword matches"""
        boost = 0.0
        text_lower = text.lower()
        
        # Check for exact matches in indicator name
        name_words = indicator_info['name'].lower().split()
        for word in name_words:
            if len(word) > 3 and word in text_lower:
                boost += 0.1
        
        # Check for category/subcategory matches
        if indicator_info['category'].lower() in text_lower:
            boost += 0.05
        if indicator_info['subcategory'].lower() in text_lower:
            boost += 0.05
        
        return min(boost, 0.3)  # Cap the boost
    
    def assess_match_quality(self, similarity_score: float, keyword_boost: float) -> str:
        """Assess the quality of indicator match"""
        if similarity_score > 0.4 or keyword_boost > 0.15:
            return 'high'
        elif similarity_score > 0.2 or keyword_boost > 0.05:
            return 'medium'
        else:
            return 'low'
    
    def create_enhanced_annotations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create enhanced annotations for the dataset"""
        print("Creating enhanced annotations...")
        
        enhanced_data = []
        
        for idx, row in df.iterrows():
            if idx % 1000 == 0:
                print(f"Processing row {idx}/{len(df)}")
            
            text = str(row['text'])
            
            # Extract numerical values
            numerical_values = self.extract_numerical_values(text)
            
            # Enhanced indicator matching
            indicator_matches = self.enhance_indicator_matching(text, row.get('indicator_id', ''))
            
            # Calculate enhanced metrics
            annotation_quality = self.assess_annotation_quality(text, numerical_values, indicator_matches)
            
            # Extract value types and confidence scores
            value_types = list(set(v.get('value_type', 'general') for v in numerical_values))
            avg_numerical_confidence = np.mean([v.get('confidence', 0) for v in numerical_values]) if numerical_values else 0
            
            # Calculate text complexity metrics
            text_complexity = self.calculate_text_complexity(text)
            
            # Create enhanced annotation with comprehensive metadata
            annotation = {
                # Original data
                'text': text,
                'text_length': len(text),
                'text_complexity_score': text_complexity,
                'original_indicator_id': row.get('indicator_id', ''),
                'original_indicator_name': row.get('indicator_name', ''),
                'original_category': row.get('category', ''),
                'original_subcategory': row.get('subcategory', ''),
                'original_framework': row.get('framework', ''),
                'original_match_score': row.get('match_score', 0),
                'report_name': row.get('report_name', ''),
                'document_id': row.get('document_id', ''),  # Preserve document ID for document-level splits
                
                # Enhanced numerical annotations
                'numerical_values': json.dumps(numerical_values),
                'num_values_found': len(numerical_values),
                'has_numerical_data': len(numerical_values) > 0,
                'numerical_value_types': json.dumps(value_types),
                'avg_numerical_confidence': avg_numerical_confidence,
                'has_high_conf_numerical': any(v.get('confidence', 0) > 0.7 for v in numerical_values),
                
                # Enhanced indicator matches
                'indicator_matches': json.dumps(indicator_matches),
                'num_indicator_matches': len(indicator_matches),
                'best_match_score': indicator_matches[0]['similarity_score'] if indicator_matches else 0,
                'best_match_indicator': indicator_matches[0]['indicator_id'] if indicator_matches else '',
                'best_match_quality': indicator_matches[0].get('match_quality', 'none') if indicator_matches else 'none',
                'has_high_quality_match': any(m.get('match_quality') == 'high' for m in indicator_matches),
                
                # Multi-label classification preparation
                'primary_category': indicator_matches[0]['category'] if indicator_matches else row.get('category', ''),
                'primary_subcategory': indicator_matches[0]['subcategory'] if indicator_matches else row.get('subcategory', ''),
                'all_matched_categories': json.dumps(list(set(m['category'] for m in indicator_matches))),
                
                # Training readiness metrics
                'is_esg_relevant': len(indicator_matches) > 0,
                'confidence_score': indicator_matches[0]['similarity_score'] if indicator_matches else 0,
                'annotation_quality': annotation_quality,
                'training_weight': self.calculate_training_weight(annotation_quality, numerical_values, indicator_matches),
                'is_training_ready': annotation_quality in ['high', 'medium'] and (len(numerical_values) > 0 or len(indicator_matches) > 0)
            }
            
            enhanced_data.append(annotation)
        
        return pd.DataFrame(enhanced_data)
    
    def assess_annotation_quality(self, text: str, numerical_values: List, indicator_matches: List) -> str:
        """Enhanced quality assessment with multiple factors and weighted scoring"""
        score = 0.0
        
        # Text length factor (optimal range: 200-1000 characters)
        text_len = len(text)
        if 200 <= text_len <= 1000:
            score += 2.0
        elif 100 <= text_len < 200 or 1000 < text_len <= 2000:
            score += 1.5
        elif text_len > 100:
            score += 1.0
        
        # Numerical data quality factor
        if len(numerical_values) > 0:
            score += 2.0
            
            # Bonus for high-confidence numerical extractions
            high_conf_values = [v for v in numerical_values if v.get('confidence', 0) > 0.7]
            if high_conf_values:
                score += 1.0
            
            # Bonus for multiple numerical values
            if len(numerical_values) > 1:
                score += 0.5
            
            # Bonus for specific value types
            value_types = set(v.get('value_type', 'general') for v in numerical_values)
            if any(vt in ['emissions', 'energy', 'financial', 'workforce'] for vt in value_types):
                score += 0.5
        
        # Indicator matching quality factor
        if len(indicator_matches) > 0:
            best_match = indicator_matches[0]
            similarity_score = best_match.get('similarity_score', 0)
            match_quality = best_match.get('match_quality', 'low')
            
            # Base score for having matches
            score += 1.0
            
            # Quality-based scoring
            if match_quality == 'high':
                score += 2.0
            elif match_quality == 'medium':
                score += 1.0
            
            # Similarity threshold bonuses
            if similarity_score > 0.5:
                score += 1.5
            elif similarity_score > 0.3:
                score += 1.0
            elif similarity_score > 0.15:
                score += 0.5
            
            # Multiple high-quality matches bonus
            high_quality_matches = [m for m in indicator_matches if m.get('match_quality') == 'high']
            if len(high_quality_matches) > 1:
                score += 0.5
        
        # ESG relevance factor (keyword-based)
        esg_keywords = [
            'sustainability', 'environmental', 'social', 'governance', 'emission', 'carbon',
            'energy', 'waste', 'water', 'employee', 'diversity', 'safety', 'ethics',
            'compliance', 'transparency', 'stakeholder', 'renewable', 'climate', 'biodiversity'
        ]
        text_lower = text.lower()
        keyword_count = sum(1 for keyword in esg_keywords if keyword in text_lower)
        if keyword_count >= 3:
            score += 1.0
        elif keyword_count >= 1:
            score += 0.5
        
        # Context richness factor
        if any(phrase in text_lower for phrase in ['target', 'goal', 'objective', 'strategy', 'initiative']):
            score += 0.5
        
        if any(phrase in text_lower for phrase in ['performance', 'achievement', 'progress', 'improvement']):
            score += 0.5
        
        # Final quality assessment with enhanced thresholds
        if score >= 7.0:
            return 'high'
        elif score >= 4.0:
            return 'medium'
        else:
            return 'low'
    
    def calculate_text_complexity(self, text: str) -> float:
        """Calculate text complexity score based on various factors"""
        complexity = 0.0
        
        # Sentence count and average length
        sentences = re.split(r'[.!?]+', text)
        num_sentences = len([s for s in sentences if s.strip()])
        avg_sentence_length = len(text) / max(num_sentences, 1)
        
        # Word count and vocabulary diversity
        words = text.lower().split()
        num_words = len(words)
        unique_words = len(set(words))
        vocabulary_diversity = unique_words / max(num_words, 1)
        
        # Technical term density
        technical_terms = [
            'sustainability', 'environmental', 'governance', 'compliance', 'emissions',
            'biodiversity', 'stakeholder', 'materiality', 'transparency', 'accountability'
        ]
        technical_density = sum(1 for term in technical_terms if term in text.lower()) / max(num_words, 1)
        
        # Numerical content density
        numerical_density = len(re.findall(r'\d+', text)) / max(num_words, 1)
        
        # Calculate complexity score (0-1 scale)
        complexity = (
            min(avg_sentence_length / 50, 1.0) * 0.3 +  # Sentence complexity
            vocabulary_diversity * 0.3 +                  # Vocabulary richness
            technical_density * 10 * 0.3 +               # Technical content
            numerical_density * 5 * 0.1                  # Numerical content
        )
        
        return min(complexity, 1.0)
    
    def calculate_training_weight(self, quality: str, numerical_values: List, indicator_matches: List) -> float:
        """Calculate training weight for sample prioritization"""
        base_weights = {'high': 1.0, 'medium': 0.7, 'low': 0.3}
        weight = base_weights.get(quality, 0.1)
        
        # Boost for numerical data
        if numerical_values:
            weight *= 1.2
            # Additional boost for high-confidence numerical data
            if any(v.get('confidence', 0) > 0.7 for v in numerical_values):
                weight *= 1.1
        
        # Boost for high-quality indicator matches
        if indicator_matches:
            high_quality_matches = [m for m in indicator_matches if m.get('match_quality') == 'high']
            if high_quality_matches:
                weight *= 1.3
        
        return min(weight, 2.0)  # Cap at 2.0
    
    def create_document_level_splits(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Create document-level training, validation, and test splits to prevent data leakage"""
        # Get unique documents
        unique_documents = df['document_id'].unique()
        print(f"Found {len(unique_documents)} unique documents for splitting")
        
        # Calculate document-level statistics for stratification
        doc_stats = []
        for doc_id in unique_documents:
            doc_df = df[df['document_id'] == doc_id]
            
            # Calculate document characteristics for stratification
            high_quality_ratio = len(doc_df[doc_df['annotation_quality'] == 'high']) / len(doc_df)
            numerical_ratio = len(doc_df[doc_df['has_numerical_data']]) / len(doc_df)
            avg_confidence = doc_df['confidence_score'].mean()
            total_segments = len(doc_df)
            
            # Create stratification key based on document characteristics
            quality_tier = 'high' if high_quality_ratio > 0.6 else ('medium' if high_quality_ratio > 0.3 else 'low')
            numerical_tier = 'high' if numerical_ratio > 0.3 else ('medium' if numerical_ratio > 0.1 else 'low')
            size_tier = 'large' if total_segments > 500 else ('medium' if total_segments > 200 else 'small')
            
            doc_stats.append({
                'document_id': doc_id,
                'total_segments': total_segments,
                'high_quality_ratio': high_quality_ratio,
                'numerical_ratio': numerical_ratio,
                'avg_confidence': avg_confidence,
                'quality_tier': quality_tier,
                'numerical_tier': numerical_tier,
                'size_tier': size_tier,
                'stratify_key': f"{quality_tier}_{numerical_tier}_{size_tier}"
            })
        
        doc_stats_df = pd.DataFrame(doc_stats)
        
        # Sort documents by average confidence and quality for optimal distribution
        doc_stats_df = doc_stats_df.sort_values(['avg_confidence', 'high_quality_ratio'], ascending=[False, False])
        
        # Document-level split: 70% train, 15% validation, 15% test
        n_docs = len(unique_documents)
        n_train_docs = max(1, int(0.7 * n_docs))
        n_val_docs = max(1, int(0.15 * n_docs)) if n_docs > 2 else 0
        n_test_docs = n_docs - n_train_docs - n_val_docs
        
        # Ensure minimum documents in each split
        if n_docs < 3:
            print(f"Warning: Only {n_docs} documents available. Using all for training.")
            train_docs = doc_stats_df['document_id'].tolist()
            val_docs = []
            test_docs = []
        else:
            # Stratified document assignment to maintain distribution balance
            train_docs = []
            val_docs = []
            test_docs = []
            
            # Group documents by stratification key for balanced splitting
            for strat_key in doc_stats_df['stratify_key'].unique():
                strat_docs = doc_stats_df[doc_stats_df['stratify_key'] == strat_key]['document_id'].tolist()
                n_strat = len(strat_docs)
                
                if n_strat == 1:
                    train_docs.extend(strat_docs)
                elif n_strat == 2:
                    train_docs.append(strat_docs[0])
                    if len(val_docs) < n_val_docs:
                        val_docs.append(strat_docs[1])
                    else:
                        test_docs.append(strat_docs[1])
                else:
                    # Proportional split within stratification group
                    n_train_strat = max(1, int(0.7 * n_strat))
                    n_val_strat = int(0.15 * n_strat) if n_strat > 2 else 0
                    n_test_strat = n_strat - n_train_strat - n_val_strat
                    
                    train_docs.extend(strat_docs[:n_train_strat])
                    if n_val_strat > 0:
                        val_docs.extend(strat_docs[n_train_strat:n_train_strat + n_val_strat])
                    if n_test_strat > 0:
                        test_docs.extend(strat_docs[n_train_strat + n_val_strat:])
        
        # Create final splits based on document assignment
        train_df = df[df['document_id'].isin(train_docs)].copy()
        val_df = df[df['document_id'].isin(val_docs)].copy() if val_docs else pd.DataFrame()
        test_df = df[df['document_id'].isin(test_docs)].copy() if test_docs else pd.DataFrame()
        
        # Shuffle within each split while maintaining document integrity
        if len(train_df) > 0:
            train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
        if len(val_df) > 0:
            val_df = val_df.sample(frac=1, random_state=42).reset_index(drop=True)
        if len(test_df) > 0:
            test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Print split summary
        print(f"\nDocument-level split summary:")
        print(f"Training: {len(train_docs)} documents, {len(train_df)} segments")
        print(f"Validation: {len(val_docs)} documents, {len(val_df)} segments")
        print(f"Test: {len(test_docs)} documents, {len(test_df)} segments")
        
        print(f"\nTraining documents: {train_docs}")
        print(f"Validation documents: {val_docs}")
        print(f"Test documents: {test_docs}")
        
        return {
            'train': train_df,
            'validation': val_df,
            'test': test_df,
            'document_splits': {
                'train_docs': train_docs,
                'val_docs': val_docs,
                'test_docs': test_docs
            }
        }
    
    def generate_annotation_statistics(self, df: pd.DataFrame, splits: Dict[str, pd.DataFrame]) -> Dict:
        """Generate comprehensive statistics about annotations with enhanced metrics"""
        # Basic statistics
        total_samples = len(df)
        numerical_samples = len(df[df['has_numerical_data']])
        indicator_samples = len(df[df['is_esg_relevant']])
        
        # Quality distribution
        quality_dist = df['annotation_quality'].value_counts().to_dict()
        
        # Enhanced numerical statistics with safety checks
        numerical_stats = {
            'total_values_extracted': df['num_values_found'].sum() if 'num_values_found' in df.columns else 0,
            'samples_with_numerical': numerical_samples,
            'percentage_with_numerical': (numerical_samples / total_samples * 100) if total_samples > 0 else 0,
            'avg_values_per_sample': df['num_values_found'].mean() if 'num_values_found' in df.columns else 0,
            'samples_with_multiple_values': len(df[df['num_values_found'] > 1]) if 'num_values_found' in df.columns else 0,
            'average_numerical_confidence': df['avg_numerical_confidence'].mean() if 'avg_numerical_confidence' in df.columns else 0,
            'high_confidence_numerical': len(df[df['has_high_conf_numerical']]) if 'has_high_conf_numerical' in df.columns else 0,
            'value_type_distribution': {}
        }
        
        # Value type distribution analysis
        all_value_types = []
        for types_data in df['numerical_value_types']:
            try:
                if isinstance(types_data, str):
                    types = json.loads(types_data)
                elif isinstance(types_data, list):
                    types = types_data
                else:
                    continue
                    
                if isinstance(types, list):
                    all_value_types.extend(types)
            except (json.JSONDecodeError, TypeError):
                continue
        
        if all_value_types:
            numerical_stats['value_type_distribution'] = {vtype: all_value_types.count(vtype) for vtype in set(all_value_types)}
        
        # Enhanced indicator statistics with safety checks
        indicator_stats = {
            'samples_with_indicators': indicator_samples,
            'percentage_with_indicators': (indicator_samples / total_samples * 100) if total_samples > 0 else 0,
            'avg_matches_per_sample': df['num_indicator_matches'].mean() if 'num_indicator_matches' in df.columns else 0,
            'samples_with_multiple_matches': len(df[df['num_indicator_matches'] > 1]) if 'num_indicator_matches' in df.columns else 0,
            'unique_indicators_matched': df['best_match_indicator'].nunique() if 'best_match_indicator' in df.columns else 0,
            'average_best_match_score': df['best_match_score'].mean() if 'best_match_score' in df.columns else 0,
            'high_quality_matches': len(df[df['has_high_quality_match']]) if 'has_high_quality_match' in df.columns else 0,
            'match_quality_distribution': df['best_match_quality'].value_counts().to_dict() if 'best_match_quality' in df.columns else {}
        }
        
        # Training readiness metrics with safety checks
        training_stats = {
            'training_ready_samples': len(df[df['is_training_ready']]) if 'is_training_ready' in df.columns else 0,
            'percentage_training_ready': (len(df[df['is_training_ready']]) / total_samples * 100) if total_samples > 0 and 'is_training_ready' in df.columns else 0,
            'average_training_weight': df['training_weight'].mean() if 'training_weight' in df.columns else 0,
            'high_weight_samples': len(df[df['training_weight'] > 1.0]) if 'training_weight' in df.columns else 0
        }
        
        # Text complexity metrics with safety checks
        complexity_stats = {
            'average_text_length': df['text_length'].mean() if 'text_length' in df.columns else df['text'].str.len().mean(),
            'average_complexity_score': df['text_complexity_score'].mean() if 'text_complexity_score' in df.columns else 0,
            'high_complexity_samples': len(df[df['text_complexity_score'] > 0.7]) if 'text_complexity_score' in df.columns else 0,
            'text_length_distribution': {}
        }
        
        # Calculate text length distribution safely
        if 'text_length' in df.columns:
            complexity_stats['text_length_distribution'] = {
                'short_texts_(<200)': len(df[df['text_length'] < 200]),
                'medium_texts_(200-1000)': len(df[(df['text_length'] >= 200) & (df['text_length'] <= 1000)]),
                'long_texts_(>1000)': len(df[df['text_length'] > 1000])
            }
        else:
            text_lengths = df['text'].str.len()
            complexity_stats['text_length_distribution'] = {
                'short_texts_(<200)': len(text_lengths[text_lengths < 200]),
                'medium_texts_(200-1000)': len(text_lengths[(text_lengths >= 200) & (text_lengths <= 1000)]),
                'long_texts_(>1000)': len(text_lengths[text_lengths > 1000])
            }
        
        # Category analysis with safety checks
        category_stats = {
            'distribution': df['primary_category'].value_counts().to_dict() if 'primary_category' in df.columns else {},
            'subcategory_distribution': df['primary_subcategory'].value_counts().to_dict() if 'primary_subcategory' in df.columns else {},
            'unique_categories': df['primary_category'].nunique() if 'primary_category' in df.columns else 0,
            'unique_subcategories': df['primary_subcategory'].nunique() if 'primary_subcategory' in df.columns else 0
        }
        
        # Training splits analysis
        split_stats = {}
        for split_name, split_df in splits.items():
            if split_name == 'document_splits':  # Skip document splits metadata
                continue
            if len(split_df) > 0 and 'annotation_quality' in split_df.columns:
                split_stats[f'{split_name}_samples'] = len(split_df)
                split_stats[f'{split_name}_quality'] = split_df['annotation_quality'].value_counts().to_dict()
                split_stats[f'{split_name}_numerical'] = len(split_df[split_df['has_numerical_data']]) if 'has_numerical_data' in split_df.columns else 0
                split_stats[f'{split_name}_indicators'] = len(split_df[split_df['is_esg_relevant']]) if 'is_esg_relevant' in split_df.columns else 0
                split_stats[f'{split_name}_avg_confidence'] = split_df['confidence_score'].mean() if 'confidence_score' in split_df.columns else 0
            else:
                split_stats[f'{split_name}_samples'] = 0
                split_stats[f'{split_name}_quality'] = {}
                split_stats[f'{split_name}_numerical'] = 0
                split_stats[f'{split_name}_indicators'] = 0
                split_stats[f'{split_name}_avg_confidence'] = 0
        
        # Compile comprehensive statistics
        stats = {
            'dataset_overview': {
                'total_samples': total_samples,
                'samples_with_numerical_data': numerical_samples,
                'samples_with_indicators': indicator_samples,
                'average_confidence_score': df['confidence_score'].mean()
            },
            'quality_analysis': {
                'distribution': quality_dist,
                'high_quality_percentage': (quality_dist.get('high', 0) / total_samples * 100) if total_samples > 0 else 0,
                'training_ready_percentage': training_stats['percentage_training_ready']
            },
            'numerical_analysis': numerical_stats,
            'indicator_analysis': indicator_stats,
            'text_analysis': complexity_stats,
            'training_analysis': training_stats,
            'category_analysis': category_stats,
            'training_splits': split_stats
        }
        
        return stats
    
    def save_annotations(self, enhanced_df: pd.DataFrame, splits: Dict[str, pd.DataFrame], stats: Dict):
        """Save all annotation results with standardized filenames (no timestamps)"""
        
        # Save complete enhanced dataset with standardized name
        enhanced_path = self.annotations_dir / 'enhanced_esg_annotations.csv'
        enhanced_df.to_csv(enhanced_path, index=False)
        print(f"Enhanced annotations saved to: {enhanced_path}")
        
        # Save training splits with standardized names (only non-empty ones)
        files_created = {'enhanced_annotations': str(enhanced_path)}
        
        for split_name, split_df in splits.items():
            if split_name == 'document_splits':  # Skip the document split metadata
                continue
            if isinstance(split_df, pd.DataFrame) and len(split_df) > 0:
                split_path = self.annotations_dir / f'esg_{split_name}_set.csv'
                split_df.to_csv(split_path, index=False)
                print(f"{split_name.capitalize()} set saved to: {split_path}")
                files_created[f'{split_name}_set'] = str(split_path)
            else:
                print(f"Warning: {split_name} set is empty, skipping save")
        
        # Save document split information with standardized name
        if 'document_splits' in splits:
            doc_splits_path = self.annotations_dir / 'document_splits.json'
            with open(doc_splits_path, 'w') as f:
                json.dump(splits['document_splits'], f, indent=2)
            print(f"Document splits saved to: {doc_splits_path}")
            files_created['document_splits'] = str(doc_splits_path)
        
        # Save statistics with standardized name
        stats_path = self.annotations_dir / 'annotation_statistics.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        print(f"Statistics saved to: {stats_path}")
        files_created['statistics'] = str(stats_path)
        
        # Save annotation summary
        summary = {
            'annotation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_samples_annotated': len(enhanced_df),
            'training_samples': len(splits['train']) if 'train' in splits and len(splits['train']) > 0 else 0,
            'validation_samples': len(splits['validation']) if 'validation' in splits and len(splits['validation']) > 0 else 0,
            'test_samples': len(splits['test']) if 'test' in splits and len(splits['test']) > 0 else 0,
            'annotation_quality_high': len(enhanced_df[enhanced_df['annotation_quality'] == 'high']),
            'samples_with_numerical_data': len(enhanced_df[enhanced_df['has_numerical_data']]),
            'unique_indicators_found': enhanced_df['best_match_indicator'].nunique(),
            'unique_documents': enhanced_df['document_id'].nunique(),
            'document_level_splitting': True,
            'files_created': files_created
        }
        
        # Add document split summary if available
        if 'document_splits' in splits:
            doc_splits = splits['document_splits']
            summary['document_split_summary'] = {
                'train_documents': len(doc_splits['train_docs']),
                'validation_documents': len(doc_splits['val_docs']),
                'test_documents': len(doc_splits['test_docs']),
                'train_docs': doc_splits['train_docs'],
                'val_docs': doc_splits['val_docs'],
                'test_docs': doc_splits['test_docs']
            }
        
        summary_path = self.annotations_dir / 'annotation_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Annotation summary saved to: {summary_path}")
        files_created['summary'] = str(summary_path)
        
        return summary

def main():
    """Main annotation process with document-level splitting"""
    print("=== ESG ANNOTATION SYSTEM (Document-Level Splitting) ===")
    print(f"Starting annotation process at {datetime.now()}")
    
    # Initialize annotation system
    annotator = ESGAnnotationSystem()
    
    # Load individual datasets
    print("\n1. Loading individual ESG datasets...")
    individual_datasets = annotator.load_individual_datasets()
    
    # Combine datasets while preserving document information
    print("\n2. Combining datasets with document tracking...")
    combined_df = annotator.combine_datasets(individual_datasets)
    
    # Create enhanced annotations
    print("\n3. Creating enhanced annotations...")
    enhanced_df = annotator.create_enhanced_annotations(combined_df)
    
    # Create document-level training splits
    print("\n4. Creating document-level training splits...")
    splits = annotator.create_document_level_splits(enhanced_df)
    
    # Generate annotation statistics
    print("\n5. Generating annotation statistics...")
    stats = annotator.generate_annotation_statistics(enhanced_df, splits)
    
    # Save all results
    print("\n6. Saving annotation results...")
    summary = annotator.save_annotations(enhanced_df, splits, stats)
    
    # Print summary
    print("\n=== ANNOTATION SUMMARY ===")
    print(f"Total samples annotated: {summary['total_samples_annotated']:,}")
    print(f"Training samples: {summary['training_samples']:,}")
    print(f"Validation samples: {summary['validation_samples']:,}")
    print(f"Test samples: {summary['test_samples']:,}")
    print(f"High-quality annotations: {summary['annotation_quality_high']:,}")
    print(f"Samples with numerical data: {summary['samples_with_numerical_data']:,}")
    print(f"Unique indicators found: {summary['unique_indicators_found']}")
    
    print("\n=== ANNOTATION STATISTICS ===")
    print(f"Average confidence score: {stats['dataset_overview']['average_confidence_score']:.3f}")
    print(f"Total numerical values extracted: {stats['numerical_analysis']['total_values_extracted']:,}")
    print(f"Average values per sample: {stats['numerical_analysis']['avg_values_per_sample']:.2f}")
    print(f"Unique indicators matched: {stats['indicator_analysis']['unique_indicators_matched']:,}")
    print(f"High quality percentage: {stats['quality_analysis']['high_quality_percentage']:.1f}%")
    print(f"Training ready percentage: {stats['quality_analysis']['training_ready_percentage']:.1f}%")
    
    print("\nQuality Distribution:")
    for quality, count in stats['quality_analysis']['distribution'].items():
        print(f"  {quality}: {count:,} ({count/len(enhanced_df)*100:.1f}%)")
    
    print("\nCategory Distribution:")
    for category, count in stats['category_analysis']['distribution'].items():
        print(f"  {category}: {count:,} ({count/len(enhanced_df)*100:.1f}%)")
    
    print("\n=== ANNOTATION COMPLETE ===")
    print("Enhanced ESG dataset ready for FinBERT training!")
    
    return summary

if __name__ == "__main__":
    summary = main()