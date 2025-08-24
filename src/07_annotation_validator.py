"""
ESG Annotation Validator
Analyzes and validates the quality of ESG annotations for FinBERT training.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class ESGAnnotationValidator:
    def __init__(self):
        self.annotations_dir = Path('data/annotations')
        self.output_dir = Path('data/validation')
        self.output_dir.mkdir(exist_ok=True)
        
    def load_latest_annotations(self) -> pd.DataFrame:
        """Load the standardized annotation file"""
        annotation_file = self.annotations_dir / 'enhanced_esg_annotations.csv'
        if not annotation_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {annotation_file}")
        
        print(f"Loading annotations from: {annotation_file}")
        
        df = pd.read_csv(annotation_file)
        print(f"Loaded {len(df)} annotated samples")
        return df
    
    def analyze_annotation_quality(self, df: pd.DataFrame) -> Dict:
        """Comprehensive analysis of annotation quality"""
        print("\nAnalyzing annotation quality...")
        
        analysis = {
            'overview': {
                'total_samples': len(df),
                'samples_with_numerical_data': len(df[df['has_numerical_data']]),
                'samples_with_indicators': len(df[df['is_esg_relevant']]),
                'high_quality_samples': len(df[df['annotation_quality'] == 'high']),
                'medium_quality_samples': len(df[df['annotation_quality'] == 'medium']),
                'low_quality_samples': len(df[df['annotation_quality'] == 'low'])
            }
        }
        
        # Quality distribution by category
        quality_by_category = df.groupby(['primary_category', 'annotation_quality']).size().unstack(fill_value=0)
        analysis['quality_by_category'] = quality_by_category.to_dict()
        
        # Numerical data analysis
        numerical_df = df[df['has_numerical_data']]
        if len(numerical_df) > 0:
            analysis['numerical_analysis'] = {
                'samples_with_numerical_data': len(numerical_df),
                'avg_values_per_sample': numerical_df['num_values_found'].mean(),
                'max_values_per_sample': numerical_df['num_values_found'].max(),
                'samples_with_multiple_values': len(numerical_df[numerical_df['num_values_found'] > 1])
            }
        
        # Indicator matching analysis
        relevant_df = df[df['is_esg_relevant']]
        if len(relevant_df) > 0:
            analysis['indicator_analysis'] = {
                'samples_with_indicators': len(relevant_df),
                'avg_confidence_score': relevant_df['confidence_score'].mean(),
                'high_confidence_samples': len(relevant_df[relevant_df['confidence_score'] > 0.3]),
                'unique_indicators_found': relevant_df['best_match_indicator'].nunique(),
                'most_common_indicators': relevant_df['best_match_indicator'].value_counts().head(10).to_dict()
            }
        
        return analysis
    
    def find_best_training_samples(self, df: pd.DataFrame, n_samples: int = 100) -> pd.DataFrame:
        """Find the best samples for training based on multiple criteria"""
        print(f"\nFinding top {n_samples} training samples...")
        
        # Create a composite score for sample quality
        df['training_score'] = 0
        
        # Add points for different quality factors
        df.loc[df['has_numerical_data'], 'training_score'] += 3
        df.loc[df['is_esg_relevant'], 'training_score'] += 2
        df.loc[df['confidence_score'] > 0.2, 'training_score'] += 2
        df.loc[df['confidence_score'] > 0.4, 'training_score'] += 1
        df.loc[df['text_length'] > 200, 'training_score'] += 1
        df.loc[df['text_length'] > 500, 'training_score'] += 1
        df.loc[df['num_values_found'] > 1, 'training_score'] += 1
        df.loc[df['annotation_quality'] == 'high', 'training_score'] += 3
        df.loc[df['annotation_quality'] == 'medium', 'training_score'] += 1
        
        # Get top samples
        best_samples = df.nlargest(n_samples, 'training_score')
        
        print(f"Selected {len(best_samples)} best training samples")
        print(f"Average training score: {best_samples['training_score'].mean():.2f}")
        print(f"Score range: {best_samples['training_score'].min()} - {best_samples['training_score'].max()}")
        
        return best_samples
    
    def analyze_text_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze text patterns in annotations"""
        print("\nAnalyzing text patterns...")
        
        # Text length analysis
        text_lengths = df['text_length']
        
        # Common words in high-quality samples
        high_quality_df = df[df['annotation_quality'] == 'high']
        
        patterns = {
            'text_length_stats': {
                'mean': text_lengths.mean(),
                'median': text_lengths.median(),
                'std': text_lengths.std(),
                'min': text_lengths.min(),
                'max': text_lengths.max(),
                'quartiles': text_lengths.quantile([0.25, 0.5, 0.75]).to_dict()
            },
            'length_distribution': {
                'very_short': len(df[df['text_length'] < 100]),
                'short': len(df[(df['text_length'] >= 100) & (df['text_length'] < 300)]),
                'medium': len(df[(df['text_length'] >= 300) & (df['text_length'] < 600)]),
                'long': len(df[(df['text_length'] >= 600) & (df['text_length'] < 1000)]),
                'very_long': len(df[df['text_length'] >= 1000])
            }
        }
        
        return patterns
    
    def validate_numerical_extractions(self, df: pd.DataFrame) -> Dict:
        """Validate the quality of numerical value extractions"""
        print("\nValidating numerical extractions...")
        
        numerical_df = df[df['has_numerical_data']].copy()
        
        if len(numerical_df) == 0:
            return {'error': 'No numerical data found'}
        
        # Parse numerical values
        all_values = []
        all_units = []
        
        for idx, row in numerical_df.iterrows():
            try:
                values_json = json.loads(row['numerical_values'])
                for value_info in values_json:
                    all_values.append(value_info['value'])
                    all_units.append(value_info['unit'])
            except (json.JSONDecodeError, KeyError):
                continue
        
        validation = {
            'total_values_extracted': len(all_values),
            'unique_units_found': len(set(all_units)),
            'common_units': Counter(all_units).most_common(10),
            'value_ranges': {
                'min': min(all_values) if all_values else 0,
                'max': max(all_values) if all_values else 0,
                'mean': np.mean(all_values) if all_values else 0,
                'median': np.median(all_values) if all_values else 0
            },
            'samples_by_value_count': numerical_df['num_values_found'].value_counts().to_dict()
        }
        
        return validation
    
    def create_training_recommendations(self, df: pd.DataFrame, analysis: Dict) -> Dict:
        """Create recommendations for FinBERT training"""
        print("\nCreating training recommendations...")
        
        recommendations = {
            'dataset_readiness': {
                'total_samples': len(df),
                'recommended_for_training': len(df[df['annotation_quality'].isin(['medium', 'high'])]),
                'high_quality_percentage': (len(df[df['annotation_quality'] == 'high']) / len(df)) * 100,
                'samples_with_numerical_data_percentage': (len(df[df['has_numerical_data']]) / len(df)) * 100
            },
            'training_strategy': {
                'recommended_approach': 'Multi-task learning with numerical value extraction',
                'primary_task': 'ESG indicator classification',
                'secondary_task': 'Numerical value extraction',
                'batch_size_recommendation': 16,
                'learning_rate_recommendation': 2e-5,
                'epochs_recommendation': 3
            },
            'data_augmentation': {
                'needed': len(df[df['annotation_quality'] == 'high']) < 1000,
                'techniques': ['Back-translation', 'Synonym replacement', 'Contextual augmentation'],
                'target_samples': 5000
            },
            'model_architecture': {
                'base_model': 'FinBERT',
                'classification_head': 'Multi-label classification for 114 ESG indicators',
                'regression_head': 'Numerical value extraction',
                'additional_features': ['Text length', 'Confidence scores', 'Category embeddings']
            }
        }
        
        return recommendations
    
    def generate_validation_report(self, df: pd.DataFrame) -> Dict:
        """Generate comprehensive validation report"""
        print("\n=== GENERATING VALIDATION REPORT ===")
        
        # Run all analyses
        quality_analysis = self.analyze_annotation_quality(df)
        text_patterns = self.analyze_text_patterns(df)
        numerical_validation = self.validate_numerical_extractions(df)
        training_recommendations = self.create_training_recommendations(df, quality_analysis)
        best_samples = self.find_best_training_samples(df, 200)
        
        # Compile report
        report = {
            'timestamp': datetime.now().isoformat(),
            'dataset_overview': {
                'total_samples': len(df),
                'annotation_file': 'enhanced_esg_annotations_latest.csv',
                'validation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            },
            'quality_analysis': quality_analysis,
            'text_patterns': text_patterns,
            'numerical_validation': numerical_validation,
            'training_recommendations': training_recommendations,
            'best_samples_count': len(best_samples)
        }
        
        return report, best_samples
    
    def save_validation_results(self, report: Dict, best_samples: pd.DataFrame):
        """Save validation results and best samples with standardized filenames"""
        
        # Save validation report with standardized name
        report_path = self.output_dir / 'validation_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"Validation report saved to: {report_path}")
        
        # Save best training samples with standardized name
        samples_path = self.output_dir / 'best_training_samples.csv'
        best_samples.to_csv(samples_path, index=False)
        print(f"Best training samples saved to: {samples_path}")
        
        # Save summary with standardized name
        summary = {
            'validation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_samples_analyzed': report['dataset_overview']['total_samples'],
            'high_quality_samples': report['quality_analysis']['overview']['high_quality_samples'],
            'samples_with_numerical_data': report['quality_analysis']['overview']['samples_with_numerical_data'],
            'best_training_samples_selected': len(best_samples),
            'training_readiness_score': self.calculate_readiness_score(report),
            'files_created': {
                'validation_report': str(report_path),
                'best_samples': str(samples_path)
            }
        }
        
        summary_path = self.output_dir / 'validation_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Validation summary saved to: {summary_path}")
        
        return summary
    
    def calculate_readiness_score(self, report: Dict) -> float:
        """Calculate a training readiness score (0-100)"""
        score = 0
        
        # Quality distribution (40 points max)
        quality_analysis = report['quality_analysis']['overview']
        total_samples = quality_analysis['total_samples']
        high_quality_pct = (quality_analysis['high_quality_samples'] / total_samples) * 100
        medium_quality_pct = (quality_analysis['medium_quality_samples'] / total_samples) * 100
        
        score += min(high_quality_pct * 2, 20)  # Up to 20 points for high quality
        score += min(medium_quality_pct * 0.5, 20)  # Up to 20 points for medium quality
        
        # Numerical data (20 points max)
        numerical_pct = (quality_analysis['samples_with_numerical_data'] / total_samples) * 100
        score += min(numerical_pct * 0.2, 20)
        
        # Indicator coverage (20 points max)
        indicator_pct = (quality_analysis['samples_with_indicators'] / total_samples) * 100
        score += min(indicator_pct * 0.2, 20)
        
        # Dataset size (20 points max)
        if total_samples >= 15000:
            score += 20
        elif total_samples >= 10000:
            score += 15
        elif total_samples >= 5000:
            score += 10
        else:
            score += 5
        
        return min(score, 100)

def main():
    """Main validation process"""
    print("=== ESG ANNOTATION VALIDATOR ===")
    print(f"Starting validation at {datetime.now()}")
    
    # Initialize validator
    validator = ESGAnnotationValidator()
    
    # Load annotations
    print("\n1. Loading latest annotations...")
    df = validator.load_latest_annotations()
    
    # Generate validation report
    print("\n2. Generating validation report...")
    report, best_samples = validator.generate_validation_report(df)
    
    # Save results
    print("\n3. Saving validation results...")
    summary = validator.save_validation_results(report, best_samples)
    
    # Print summary
    print("\n=== VALIDATION SUMMARY ===")
    print(f"Total samples analyzed: {summary['total_samples_analyzed']:,}")
    print(f"High-quality samples: {summary['high_quality_samples']:,}")
    print(f"Samples with numerical data: {summary['samples_with_numerical_data']:,}")
    print(f"Best training samples selected: {summary['best_training_samples_selected']:,}")
    print(f"Training readiness score: {summary['training_readiness_score']:.1f}/100")
    
    # Training recommendations
    recommendations = report['training_recommendations']
    print("\n=== TRAINING RECOMMENDATIONS ===")
    print(f"Recommended approach: {recommendations['training_strategy']['recommended_approach']}")
    print(f"Primary task: {recommendations['training_strategy']['primary_task']}")
    print(f"Secondary task: {recommendations['training_strategy']['secondary_task']}")
    print(f"Recommended batch size: {recommendations['training_strategy']['batch_size_recommendation']}")
    print(f"Recommended learning rate: {recommendations['training_strategy']['learning_rate_recommendation']}")
    print(f"Recommended epochs: {recommendations['training_strategy']['epochs_recommendation']}")
    
    if recommendations['data_augmentation']['needed']:
        print(f"\nData augmentation needed: Yes")
        print(f"Target samples: {recommendations['data_augmentation']['target_samples']:,}")
        print(f"Recommended techniques: {', '.join(recommendations['data_augmentation']['techniques'])}")
    else:
        print(f"\nData augmentation needed: No")
    
    print("\n=== VALIDATION COMPLETE ===")
    print("ESG annotations validated and ready for FinBERT training!")
    
    return summary

if __name__ == "__main__":
    summary = main()