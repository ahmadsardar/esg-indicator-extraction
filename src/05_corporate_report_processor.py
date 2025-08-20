#!/usr/bin/env python3
"""
Corporate Report Processing Script

This script processes corporate ESG reports (PDFs) to extract relevant text segments
for our 89 comprehensive ESG indicators. It creates structured datasets for training
the FinBERT model to recognize and extract ESG information.

Steps:
1. Load comprehensive ESG indicators (89 indicators)
2. Extract text from corporate PDF reports
3. Segment text into meaningful chunks
4. Match text segments to ESG indicators using keyword matching and semantic similarity
5. Create labeled training dataset
6. Save processed datasets for model training
"""

import pandas as pd
import json
import PyPDF2
import fitz  # PyMuPDF for better PDF text extraction
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CorporateReportProcessor:
    def __init__(self, indicators_path: str, reports_dir: str, output_dir: str):
        """
        Initialize the Corporate Report Processor
        
        Args:
            indicators_path: Path to comprehensive ESG indicators JSON file
            reports_dir: Directory containing corporate PDF reports
            output_dir: Directory to save processed datasets
        """
        self.indicators_path = Path(indicators_path)
        self.reports_dir = Path(reports_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.indicators = []
        self.indicator_keywords = {}
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
    
    def load_indicators(self) -> None:
        """
        Load the comprehensive ESG indicators from JSON file
        """
        logger.info(f"Loading ESG indicators from {self.indicators_path}")
        
        with open(self.indicators_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.indicators = data.get('indicators', [])
        logger.info(f"Loaded {len(self.indicators)} ESG indicators")
        
        # Create keyword mappings for each indicator
        self._create_indicator_keywords()
    
    def _create_indicator_keywords(self) -> None:
        """
        Create keyword mappings for each ESG indicator to help with text matching
        """
        logger.info("Creating keyword mappings for ESG indicators")
        
        for indicator in self.indicators:
            indicator_id = indicator.get('indicator_id', '')
            name = indicator.get('name', '').lower()
            description = indicator.get('description', '').lower()
            category = indicator.get('category', '').lower()
            subcategory = indicator.get('subcategory', '').lower()
            
            # Extract keywords from name and description
            keywords = set()
            
            # Add words from name (remove common words)
            name_words = re.findall(r'\b\w{3,}\b', name)
            keywords.update(name_words)
            
            # Add key terms from description
            desc_words = re.findall(r'\b\w{4,}\b', description)
            keywords.update(desc_words[:10])  # Limit to top 10 words
            
            # Add category terms
            if category:
                keywords.add(category)
            if subcategory:
                keywords.add(subcategory)
            
            # Remove common stopwords
            stop_words = set(stopwords.words('english'))
            keywords = keywords - stop_words
            
            self.indicator_keywords[indicator_id] = {
                'keywords': list(keywords),
                'name': name,
                'description': description,
                'category': category
            }
        
        logger.info(f"Created keyword mappings for {len(self.indicator_keywords)} indicators")
    
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """
        Extract text from PDF file using PyMuPDF for better accuracy
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text content
        """
        try:
            doc = fitz.open(pdf_path)
            text = ""
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text += page.get_text()
            
            doc.close()
            return text
        
        except Exception as e:
            logger.warning(f"Failed to extract text from {pdf_path} using PyMuPDF: {e}")
            
            # Fallback to PyPDF2
            try:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    text = ""
                    
                    for page in pdf_reader.pages:
                        text += page.extract_text()
                    
                    return text
            
            except Exception as e2:
                logger.error(f"Failed to extract text from {pdf_path}: {e2}")
                return ""
    
    def segment_text(self, text: str, min_length: int = 50, max_length: int = 500) -> List[str]:
        """
        Segment text into meaningful chunks for processing
        
        Args:
            text: Input text to segment
            min_length: Minimum segment length
            max_length: Maximum segment length
            
        Returns:
            List of text segments
        """
        # Clean text
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = re.sub(r'[^\w\s.,;:!?()-]', '', text)  # Remove special characters
        
        # Split into sentences
        sentences = sent_tokenize(text)
        
        segments = []
        current_segment = ""
        
        for sentence in sentences:
            # Skip very short sentences
            if len(sentence.strip()) < 20:
                continue
            
            # If adding this sentence would exceed max_length, save current segment
            if len(current_segment) + len(sentence) > max_length and len(current_segment) >= min_length:
                segments.append(current_segment.strip())
                current_segment = sentence
            else:
                current_segment += " " + sentence if current_segment else sentence
        
        # Add the last segment if it meets minimum length
        if len(current_segment.strip()) >= min_length:
            segments.append(current_segment.strip())
        
        return segments
    
    def match_segments_to_indicators(self, segments: List[str]) -> List[Dict]:
        """
        Match text segments to ESG indicators using keyword matching and semantic similarity
        
        Args:
            segments: List of text segments
            
        Returns:
            List of matched segments with indicator information
        """
        matched_data = []
        
        for segment in segments:
            segment_lower = segment.lower()
            matches = []
            
            # Check each indicator for keyword matches
            for indicator in self.indicators:
                indicator_id = indicator.get('indicator_id', '')
                keywords = self.indicator_keywords.get(indicator_id, {}).get('keywords', [])
                
                # Count keyword matches
                keyword_matches = sum(1 for keyword in keywords if keyword in segment_lower)
                
                if keyword_matches > 0:
                    # Calculate match score
                    match_score = keyword_matches / max(len(keywords), 1)
                    
                    matches.append({
                        'indicator_id': indicator_id,
                        'indicator_name': indicator.get('name', ''),
                        'category': indicator.get('category', ''),
                        'subcategory': indicator.get('subcategory', ''),
                        'framework': indicator.get('framework', ''),
                        'keyword_matches': keyword_matches,
                        'match_score': match_score,
                        'matched_keywords': [kw for kw in keywords if kw in segment_lower]
                    })
            
            # Sort matches by score and keep top matches
            matches.sort(key=lambda x: x['match_score'], reverse=True)
            top_matches = matches[:3]  # Keep top 3 matches
            
            if top_matches:
                matched_data.append({
                    'text_segment': segment,
                    'segment_length': len(segment),
                    'matches': top_matches,
                    'best_match': top_matches[0],
                    'num_matches': len(matches)
                })
        
        return matched_data
    
    def process_single_report(self, pdf_path: Path) -> Dict:
        """
        Process a single corporate report
        
        Args:
            pdf_path: Path to PDF report
            
        Returns:
            Processing results dictionary
        """
        logger.info(f"Processing report: {pdf_path.name}")
        
        # Extract text
        text = self.extract_text_from_pdf(pdf_path)
        if not text:
            logger.warning(f"No text extracted from {pdf_path.name}")
            return {'error': 'No text extracted'}
        
        # Segment text
        segments = self.segment_text(text)
        logger.info(f"Created {len(segments)} text segments")
        
        # Match segments to indicators
        matched_data = self.match_segments_to_indicators(segments)
        logger.info(f"Found {len(matched_data)} segments with indicator matches")
        
        return {
            'report_name': pdf_path.name,
            'total_text_length': len(text),
            'total_segments': len(segments),
            'matched_segments': len(matched_data),
            'matched_data': matched_data
        }
    
    def process_all_reports(self) -> Dict:
        """
        Process all PDF reports in the reports directory
        
        Returns:
            Combined processing results
        """
        logger.info(f"Processing all reports in {self.reports_dir}")
        
        all_results = []
        all_matched_data = []
        
        # Find all PDF files
        pdf_files = list(self.reports_dir.rglob('*.pdf'))
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        for pdf_path in pdf_files:
            result = self.process_single_report(pdf_path)
            
            if 'error' not in result:
                all_results.append(result)
                all_matched_data.extend(result['matched_data'])
        
        return {
            'total_reports_processed': len(all_results),
            'total_matched_segments': len(all_matched_data),
            'reports': all_results,
            'all_matched_data': all_matched_data
        }
    
    def create_training_dataset(self, processing_results: Dict) -> pd.DataFrame:
        """
        Create a structured training dataset from processing results
        
        Args:
            processing_results: Results from process_all_reports
            
        Returns:
            Training dataset as DataFrame
        """
        logger.info("Creating training dataset")
        
        training_data = []
        
        # Process each report's data separately to track source
        for report_result in processing_results['reports']:
            report_name = report_result['report_name']
            
            for item in report_result['matched_data']:
                text_segment = item['text_segment']
                best_match = item['best_match']
                
                training_data.append({
                    'text': text_segment,
                    'text_length': len(text_segment),
                    'indicator_id': best_match['indicator_id'],
                    'indicator_name': best_match['indicator_name'],
                    'category': best_match['category'],
                    'subcategory': best_match['subcategory'],
                    'framework': best_match['framework'],
                    'match_score': best_match['match_score'],
                    'keyword_matches': best_match['keyword_matches'],
                    'matched_keywords': ', '.join(best_match['matched_keywords']),
                    'label': 1,  # Positive label for matched segments
                    'report_name': report_name  # Add report name for individual dataset creation
                })
        
        df = pd.DataFrame(training_data)
        logger.info(f"Created training dataset with {len(df)} samples")
        
        return df
    
    def save_datasets(self, processing_results: Dict, training_df: pd.DataFrame) -> None:
        """
        Save all generated datasets
        
        Args:
            processing_results: Processing results dictionary
            training_df: Training dataset DataFrame
        """
        logger.info(f"Saving datasets to {self.output_dir}")
        
        # Create individual reports directory
        individual_dir = self.output_dir / 'individual_reports'
        individual_dir.mkdir(parents=True, exist_ok=True)
        
        # Save combined training dataset
        training_csv = self.output_dir / 'esg_training_dataset.csv'
        training_df.to_csv(training_csv, index=False, encoding='utf-8')
        logger.info(f"Saved combined training dataset: {training_csv}")
        
        # Save individual datasets for each PDF
        logger.info("Creating individual datasets for each PDF...")
        individual_stats = {}
        
        for report_name in training_df['report_name'].unique():
            # Filter data for this specific report
            report_data = training_df[training_df['report_name'] == report_name].copy()
            
            # Clean report name for filename (remove .pdf extension and invalid characters)
            clean_name = report_name.replace('.pdf', '').replace(' ', '_').replace('-', '_')
            clean_name = ''.join(c for c in clean_name if c.isalnum() or c == '_')
            
            # Save individual CSV
            individual_csv_path = individual_dir / f'{clean_name}_esg_dataset.csv'
            report_data.to_csv(individual_csv_path, index=False, encoding='utf-8')
            
            # Calculate individual statistics
            individual_stats[report_name] = {
                'samples': len(report_data),
                'unique_indicators': report_data['indicator_id'].nunique(),
                'avg_match_score': report_data['match_score'].mean(),
                'avg_text_length': report_data['text_length'].mean(),
                'category_distribution': report_data['category'].value_counts().to_dict(),
                'framework_distribution': report_data['framework'].value_counts().to_dict(),
                'file_path': str(individual_csv_path)
            }
            
            logger.info(f"Saved {report_name}: {len(report_data)} samples → {individual_csv_path.name}")
        
        # Save individual statistics
        individual_stats_path = individual_dir / 'individual_report_statistics.json'
        with open(individual_stats_path, 'w', encoding='utf-8') as f:
            json.dump(individual_stats, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved individual statistics: {individual_stats_path}")
        
        # Save processing results as JSON
        results_json = self.output_dir / 'report_processing_results.json'
        with open(results_json, 'w', encoding='utf-8') as f:
            json.dump(processing_results, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved processing results: {results_json}")
        
        # Create summary statistics
        summary_stats = {
            'total_reports_processed': processing_results['total_reports_processed'],
            'total_matched_segments': processing_results['total_matched_segments'],
            'training_samples': len(training_df),
            'unique_indicators_found': training_df['indicator_id'].nunique(),
            'category_distribution': training_df['category'].value_counts().to_dict(),
            'framework_distribution': training_df['framework'].value_counts().to_dict(),
            'average_match_score': training_df['match_score'].mean(),
            'average_text_length': training_df['text_length'].mean(),
            'individual_datasets_created': len(individual_stats)
        }
        
        summary_json = self.output_dir / 'dataset_summary.json'
        with open(summary_json, 'w', encoding='utf-8') as f:
            json.dump(summary_stats, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved summary statistics: {summary_json}")
        
        # Save category-wise breakdown
        category_breakdown = training_df.groupby(['category', 'framework']).agg({
            'indicator_id': 'count',
            'match_score': 'mean',
            'text_length': 'mean'
        }).round(3)
        
        breakdown_csv = self.output_dir / 'category_framework_breakdown.csv'
        category_breakdown.to_csv(breakdown_csv)
        logger.info(f"Saved category breakdown: {breakdown_csv}")

def main():
    """
    Main execution function
    """
    print("=== Corporate Report Processing for ESG Dataset Creation ===")
    
    # Configuration
    indicators_path = 'data/indicators/comprehensive_esg_indicators.json'
    reports_dir = '../Datasets'  # Corporate reports directory
    output_dir = 'data/processed/corporate_reports'
    
    # Initialize processor
    processor = CorporateReportProcessor(
        indicators_path=indicators_path,
        reports_dir=reports_dir,
        output_dir=output_dir
    )
    
    try:
        # Step 1: Load ESG indicators
        print("\nStep 1: Loading comprehensive ESG indicators...")
        processor.load_indicators()
        print(f"✓ Loaded {len(processor.indicators)} ESG indicators")
        
        # Step 2: Process all corporate reports
        print("\nStep 2: Processing corporate PDF reports...")
        processing_results = processor.process_all_reports()
        print(f"✓ Processed {processing_results['total_reports_processed']} reports")
        print(f"✓ Found {processing_results['total_matched_segments']} text segments with ESG matches")
        
        # Step 3: Create training dataset
        print("\nStep 3: Creating structured training dataset...")
        training_df = processor.create_training_dataset(processing_results)
        print(f"✓ Created training dataset with {len(training_df)} samples")
        print(f"✓ Covers {training_df['indicator_id'].nunique()} unique ESG indicators")
        
        # Step 4: Save all datasets
        print("\nStep 4: Saving datasets and results...")
        processor.save_datasets(processing_results, training_df)
        print(f"✓ Saved combined dataset to: {processor.output_dir}")
        print(f"✓ Saved {training_df['report_name'].nunique()} individual datasets to: {processor.output_dir / 'individual_reports'}")
        
        # Display summary
        print("\n=== PROCESSING SUMMARY ===")
        print(f"Reports processed: {processing_results['total_reports_processed']}")
        print(f"Training samples created: {len(training_df)}")
        print(f"Unique ESG indicators found: {training_df['indicator_id'].nunique()}")
        print(f"Average match score: {training_df['match_score'].mean():.3f}")
        
        print("\nCategory distribution:")
        for category, count in training_df['category'].value_counts().items():
            percentage = (count / len(training_df)) * 100
            print(f"  {category}: {count} ({percentage:.1f}%)")
        
        print("\n=== DATASET CREATION COMPLETED ===")
        print("Ready for FinBERT model training!")
        
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        raise

if __name__ == "__main__":
    main()