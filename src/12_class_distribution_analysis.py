"""
Class Distribution Analysis for ESG Indicators

This script provides comprehensive analysis of class distribution across all 47 ESG indicators
and explains the macro vs micro F1 optimization strategy for multilabel classification.

Author: ESG Analysis System
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class ESGClassDistributionAnalyzer:
    """
    Comprehensive analyzer for ESG indicator class distribution and F1 optimization strategies.
    """
    
    def __init__(self, data_path: str = "data/annotations"):
        self.data_path = Path(data_path)
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.all_indicators = []
        self.class_stats = {}
        
    def load_datasets(self):
        """Load train, validation, and test datasets."""
        print("Loading datasets...")
        
        # Load datasets
        self.train_data = pd.read_csv(self.data_path / "esg_train_set.csv")
        self.val_data = pd.read_csv(self.data_path / "esg_validation_set.csv")
        self.test_data = pd.read_csv(self.data_path / "esg_test_set.csv")
        
        print(f"Train set: {len(self.train_data)} samples")
        print(f"Validation set: {len(self.val_data)} samples")
        print(f"Test set: {len(self.test_data)} samples")
        
        # Extract all unique indicators from the indicator_matches column
        self.all_indicators = self._extract_all_indicators()
        
        print(f"Found {len(self.all_indicators)} unique ESG indicators")
        
        if len(self.all_indicators) != 47:
            print(f"Warning: Expected 47 indicators, found {len(self.all_indicators)}")
            print("Sample indicators:", list(self.all_indicators)[:10], "...")
    
    def _extract_all_indicators(self) -> set:
        """Extract all unique indicator IDs from the indicator_matches column."""
        all_indicators = set()
        
        # Combine all datasets
        all_data = pd.concat([self.train_data, self.val_data, self.test_data], ignore_index=True)
        
        for _, row in all_data.iterrows():
            if pd.notna(row['indicator_matches']):
                try:
                    # Parse the JSON string
                    matches = json.loads(row['indicator_matches'])
                    if isinstance(matches, list):
                        for match in matches:
                            if isinstance(match, dict) and 'indicator_id' in match:
                                indicator_id = match['indicator_id']
                                if pd.notna(indicator_id) and indicator_id != 'NaN':
                                    all_indicators.add(indicator_id)
                except (json.JSONDecodeError, TypeError) as e:
                    continue
        
        return all_indicators
    
    def _get_indicator_presence(self, data: pd.DataFrame, indicator_id: str) -> pd.Series:
        """Get binary presence of an indicator across all samples in the dataset."""
        presence = pd.Series([False] * len(data), index=data.index)
        
        for idx, row in data.iterrows():
            if pd.notna(row['indicator_matches']):
                try:
                    matches = json.loads(row['indicator_matches'])
                    if isinstance(matches, list):
                        for match in matches:
                            if isinstance(match, dict) and match.get('indicator_id') == indicator_id:
                                presence[idx] = True
                                break
                except (json.JSONDecodeError, TypeError):
                    continue
        
        return presence
    
    def analyze_class_distribution(self) -> Dict[str, Any]:
        """Analyze class distribution across all indicators and datasets."""
        print("\nAnalyzing class distribution...")
        
        analysis_results = {
            'total_indicators': len(self.all_indicators),
            'dataset_sizes': {
                'train': len(self.train_data),
                'validation': len(self.val_data),
                'test': len(self.test_data),
                'total': len(self.train_data) + len(self.val_data) + len(self.test_data)
            },
            'indicator_statistics': {},
            'overall_statistics': {},
            'imbalance_analysis': {}
        }
        
        # Combine all datasets for overall analysis
        all_data = pd.concat([self.train_data, self.val_data, self.test_data], ignore_index=True)
        
        # Analyze each indicator
        indicator_stats = []
        
        for indicator in self.all_indicators:
            # Calculate statistics for this indicator using presence detection
            train_presence = self._get_indicator_presence(self.train_data, indicator)
            val_presence = self._get_indicator_presence(self.val_data, indicator)
            test_presence = self._get_indicator_presence(self.test_data, indicator)
            overall_presence = self._get_indicator_presence(all_data, indicator)
            
            train_pos = train_presence.sum()
            train_total = len(self.train_data)
            val_pos = val_presence.sum()
            val_total = len(self.val_data)
            test_pos = test_presence.sum()
            test_total = len(self.test_data)
            
            overall_pos = overall_presence.sum()
            overall_total = len(all_data)
            
            stats = {
                'indicator': indicator,
                'train_positive': int(train_pos),
                'train_negative': int(train_total - train_pos),
                'train_positive_rate': float(train_pos / train_total),
                'val_positive': int(val_pos),
                'val_negative': int(val_total - val_pos),
                'val_positive_rate': float(val_pos / val_total),
                'test_positive': int(test_pos),
                'test_negative': int(test_total - test_pos),
                'test_positive_rate': float(test_pos / test_total),
                'overall_positive': int(overall_pos),
                'overall_negative': int(overall_total - overall_pos),
                'overall_positive_rate': float(overall_pos / overall_total),
                'imbalance_ratio': float((overall_total - overall_pos) / max(overall_pos, 1))
            }
            
            indicator_stats.append(stats)
            analysis_results['indicator_statistics'][indicator] = stats
        
        # Overall statistics
        positive_rates = [stats['overall_positive_rate'] for stats in indicator_stats]
        imbalance_ratios = [stats['imbalance_ratio'] for stats in indicator_stats]
        
        analysis_results['overall_statistics'] = {
            'mean_positive_rate': float(np.mean(positive_rates)),
            'median_positive_rate': float(np.median(positive_rates)),
            'std_positive_rate': float(np.std(positive_rates)),
            'min_positive_rate': float(np.min(positive_rates)),
            'max_positive_rate': float(np.max(positive_rates)),
            'mean_imbalance_ratio': float(np.mean(imbalance_ratios)),
            'median_imbalance_ratio': float(np.median(imbalance_ratios)),
            'highly_imbalanced_indicators': len([r for r in imbalance_ratios if r > 10]),
            'moderately_imbalanced_indicators': len([r for r in imbalance_ratios if 3 < r <= 10]),
            'balanced_indicators': len([r for r in imbalance_ratios if r <= 3])
        }
        
        # Imbalance analysis
        analysis_results['imbalance_analysis'] = {
            'severe_imbalance': [stats['indicator'] for stats in indicator_stats if stats['imbalance_ratio'] > 20],
            'high_imbalance': [stats['indicator'] for stats in indicator_stats if 10 < stats['imbalance_ratio'] <= 20],
            'moderate_imbalance': [stats['indicator'] for stats in indicator_stats if 3 < stats['imbalance_ratio'] <= 10],
            'low_imbalance': [stats['indicator'] for stats in indicator_stats if stats['imbalance_ratio'] <= 3]
        }
        
        return analysis_results
    
    def explain_f1_optimization_strategy(self) -> Dict[str, Any]:
        """Explain macro vs micro F1 optimization strategies for multilabel classification."""
        
        strategy_explanation = {
            'multilabel_context': {
                'description': "ESG classification is a multilabel problem where each text segment can be associated with multiple ESG indicators simultaneously.",
                'challenges': [
                    "Class imbalance across different indicators",
                    "Varying difficulty of detecting different ESG aspects",
                    "Interdependencies between related indicators",
                    "Different business importance of various ESG factors"
                ]
            },
            'macro_f1': {
                'definition': "Macro F1 calculates F1 score for each class independently, then averages them (unweighted average).",
                'formula': "Macro F1 = (1/n) * Σ F1_i where n is number of classes",
                'advantages': [
                    "Treats all indicators equally regardless of frequency",
                    "Sensitive to performance on rare/minority indicators",
                    "Encourages balanced performance across all ESG aspects",
                    "Better for comprehensive ESG coverage assessment"
                ],
                'disadvantages': [
                    "May be dominated by performance on rare classes",
                    "Can be misleading if some indicators are inherently harder to detect",
                    "May not reflect real-world usage patterns"
                ],
                'best_for': "Research contexts where comprehensive ESG coverage is critical and all indicators are equally important."
            },
            'micro_f1': {
                'definition': "Micro F1 aggregates true positives, false positives, and false negatives across all classes before calculating F1.",
                'formula': "Micro F1 = 2 * (Micro Precision * Micro Recall) / (Micro Precision + Micro Recall)",
                'advantages': [
                    "Reflects overall classification performance",
                    "Naturally weighted by class frequency",
                    "More stable with class imbalance",
                    "Better represents practical utility"
                ],
                'disadvantages': [
                    "Can be dominated by frequent classes",
                    "May hide poor performance on rare but important indicators",
                    "Less sensitive to minority class performance"
                ],
                'best_for': "Production systems where overall accuracy matters more than balanced coverage."
            },
            'weighted_f1': {
                'definition': "Weighted F1 calculates F1 for each class and averages them weighted by class support (frequency).",
                'advantages': [
                    "Balances between macro and micro approaches",
                    "Accounts for class frequency while maintaining per-class perspective",
                    "More interpretable than micro F1"
                ],
                'best_for': "Balanced evaluation that considers both coverage and practical importance."
            },
            'recommendations': {
                'primary_metric': "Macro F1",
                'rationale': [
                    "ESG analysis requires comprehensive coverage of all sustainability aspects",
                    "Regulatory compliance often requires detection of specific rare indicators",
                    "Stakeholders need balanced assessment across E, S, and G dimensions",
                    "Academic research benefits from unbiased evaluation across all indicators"
                ],
                'secondary_metrics': [
                    "Micro F1 for overall performance assessment",
                    "Weighted F1 for practical deployment considerations",
                    "Per-class F1 scores for detailed analysis",
                    "Category-wise macro F1 (Environmental, Social, Governance)"
                ],
                'threshold_optimization': {
                    "strategy": "Optimize thresholds per indicator using validation set",
                    "objective": "Maximize macro F1 while maintaining minimum per-class recall",
                    "constraints": "Ensure no indicator falls below 0.3 F1 score"
                }
            }
        }
        
        return strategy_explanation
    
    def generate_visualizations(self, analysis_results: Dict[str, Any], output_dir: str = "results/analysis_plots"):
        """Generate comprehensive visualizations for class distribution analysis."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Overall positive rates distribution
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Positive rates histogram
        positive_rates = [stats['overall_positive_rate'] for stats in analysis_results['indicator_statistics'].values()]
        axes[0, 0].hist(positive_rates, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Distribution of Positive Rates Across Indicators')
        axes[0, 0].set_xlabel('Positive Rate')
        axes[0, 0].set_ylabel('Number of Indicators')
        axes[0, 0].axvline(np.mean(positive_rates), color='red', linestyle='--', label=f'Mean: {np.mean(positive_rates):.3f}')
        axes[0, 0].legend()
        
        # Imbalance ratios
        imbalance_ratios = [stats['imbalance_ratio'] for stats in analysis_results['indicator_statistics'].values()]
        axes[0, 1].hist(imbalance_ratios, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[0, 1].set_title('Distribution of Class Imbalance Ratios')
        axes[0, 1].set_xlabel('Imbalance Ratio (Negative/Positive)')
        axes[0, 1].set_ylabel('Number of Indicators')
        axes[0, 1].axvline(np.mean(imbalance_ratios), color='blue', linestyle='--', label=f'Mean: {np.mean(imbalance_ratios):.1f}')
        axes[0, 1].legend()
        
        # Imbalance categories pie chart
        imbalance_counts = [
            analysis_results['overall_statistics']['balanced_indicators'],
            analysis_results['overall_statistics']['moderately_imbalanced_indicators'],
            analysis_results['overall_statistics']['highly_imbalanced_indicators']
        ]
        labels = ['Balanced (≤3:1)', 'Moderate (3:1-10:1)', 'High (>10:1)']
        colors = ['lightgreen', 'orange', 'red']
        axes[1, 0].pie(imbalance_counts, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[1, 0].set_title('Imbalance Categories Distribution')
        
        # Top 10 most imbalanced indicators
        sorted_indicators = sorted(analysis_results['indicator_statistics'].items(), 
                                 key=lambda x: x[1]['imbalance_ratio'], reverse=True)[:10]
        indicator_names = [item[0][:15] + '...' if len(item[0]) > 15 else item[0] for item in sorted_indicators]
        ratios = [item[1]['imbalance_ratio'] for item in sorted_indicators]
        
        axes[1, 1].barh(range(len(indicator_names)), ratios, color='salmon')
        axes[1, 1].set_yticks(range(len(indicator_names)))
        axes[1, 1].set_yticklabels(indicator_names)
        axes[1, 1].set_xlabel('Imbalance Ratio')
        axes[1, 1].set_title('Top 10 Most Imbalanced Indicators')
        axes[1, 1].invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(output_path / 'class_distribution_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Detailed indicator analysis
        fig, ax = plt.subplots(figsize=(20, 8))
        
        indicators = list(analysis_results['indicator_statistics'].keys())
        positive_rates = [analysis_results['indicator_statistics'][ind]['overall_positive_rate'] for ind in indicators]
        
        bars = ax.bar(range(len(indicators)), positive_rates, alpha=0.7)
        ax.set_xlabel('ESG Indicators')
        ax.set_ylabel('Positive Rate')
        ax.set_title('Positive Rates Across All 47 ESG Indicators')
        ax.set_xticks(range(len(indicators)))
        ax.set_xticklabels([ind[:10] + '...' if len(ind) > 10 else ind for ind in indicators], 
                          rotation=45, ha='right')
        
        # Color bars by category
        for i, (ind, rate) in enumerate(zip(indicators, positive_rates)):
            if ind.startswith('E_'):
                bars[i].set_color('green')
            elif ind.startswith('S_'):
                bars[i].set_color('blue')
            elif ind.startswith('G_'):
                bars[i].set_color('orange')
            else:
                bars[i].set_color('gray')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='green', label='Environmental'),
                          Patch(facecolor='blue', label='Social'),
                          Patch(facecolor='orange', label='Governance'),
                          Patch(facecolor='gray', label='Other')]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(output_path / 'detailed_indicator_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved to {output_path}")
    
    def save_analysis_report(self, analysis_results: Dict[str, Any], 
                           f1_strategy: Dict[str, Any], 
                           output_path: str = "results/class_distribution_report.json"):
        """Save comprehensive analysis report."""
        
        report = {
            'analysis_metadata': {
                'total_indicators_analyzed': len(self.all_indicators),
                'expected_indicators': 47,
                'analysis_date': pd.Timestamp.now().isoformat(),
                'datasets_analyzed': ['train', 'validation', 'test']
            },
            'class_distribution_analysis': analysis_results,
            'f1_optimization_strategy': f1_strategy,
            'key_findings': {
                'most_imbalanced_indicators': analysis_results['imbalance_analysis']['severe_imbalance'][:5],
                'most_balanced_indicators': analysis_results['imbalance_analysis']['low_imbalance'][:5],
                'average_positive_rate': analysis_results['overall_statistics']['mean_positive_rate'],
                'median_imbalance_ratio': analysis_results['overall_statistics']['median_imbalance_ratio']
            },
            'recommendations': {
                'primary_evaluation_metric': 'Macro F1',
                'threshold_optimization_needed': True,
                'class_weighting_recommended': True,
                'additional_data_collection_priority': analysis_results['imbalance_analysis']['severe_imbalance']
            }
        }
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"Analysis report saved to {output_file}")
        return report
    
    def print_summary(self, analysis_results: Dict[str, Any]):
        """Print a comprehensive summary of the analysis."""
        print("\n" + "="*80)
        print("ESG CLASS DISTRIBUTION ANALYSIS SUMMARY")
        print("="*80)
        
        print(f"\nDataset Overview:")
        print(f"  Total Indicators Analyzed: {analysis_results['total_indicators']}")
        print(f"  Total Samples: {analysis_results['dataset_sizes']['total']:,}")
        print(f"    - Training: {analysis_results['dataset_sizes']['train']:,}")
        print(f"    - Validation: {analysis_results['dataset_sizes']['validation']:,}")
        print(f"    - Test: {analysis_results['dataset_sizes']['test']:,}")
        
        stats = analysis_results['overall_statistics']
        print(f"\nClass Distribution Statistics:")
        print(f"  Mean Positive Rate: {stats['mean_positive_rate']:.3f} ({stats['mean_positive_rate']*100:.1f}%)")
        print(f"  Median Positive Rate: {stats['median_positive_rate']:.3f} ({stats['median_positive_rate']*100:.1f}%)")
        print(f"  Standard Deviation: {stats['std_positive_rate']:.3f}")
        print(f"  Range: {stats['min_positive_rate']:.3f} - {stats['max_positive_rate']:.3f}")
        
        print(f"\nClass Imbalance Analysis:")
        print(f"  Mean Imbalance Ratio: {stats['mean_imbalance_ratio']:.1f}:1")
        print(f"  Median Imbalance Ratio: {stats['median_imbalance_ratio']:.1f}:1")
        print(f"  Balanced Indicators (≤3:1): {stats['balanced_indicators']}")
        print(f"  Moderately Imbalanced (3:1-10:1): {stats['moderately_imbalanced_indicators']}")
        print(f"  Highly Imbalanced (>10:1): {stats['highly_imbalanced_indicators']}")
        
        imbalance = analysis_results['imbalance_analysis']
        if imbalance['severe_imbalance']:
            print(f"\nSeverely Imbalanced Indicators (>20:1):")
            for indicator in imbalance['severe_imbalance'][:5]:
                ratio = analysis_results['indicator_statistics'][indicator]['imbalance_ratio']
                print(f"    - {indicator}: {ratio:.1f}:1")
        
        print(f"\nF1 Optimization Strategy:")
        print(f"  Recommended Primary Metric: Macro F1")
        print(f"  Rationale: Ensures balanced performance across all ESG indicators")
        print(f"  Secondary Metrics: Micro F1, Weighted F1, Per-class F1")
        print(f"  Threshold Optimization: Required for optimal performance")
        
        print("\n" + "="*80)

def main():
    """Main execution function."""
    print("Starting ESG Class Distribution Analysis...")
    
    # Initialize analyzer
    analyzer = ESGClassDistributionAnalyzer()
    
    try:
        # Load datasets
        analyzer.load_datasets()
        
        # Perform analysis
        analysis_results = analyzer.analyze_class_distribution()
        
        # Generate F1 strategy explanation
        f1_strategy = analyzer.explain_f1_optimization_strategy()
        
        # Print summary
        analyzer.print_summary(analysis_results)
        
        # Generate visualizations
        analyzer.generate_visualizations(analysis_results)
        
        # Save comprehensive report
        report = analyzer.save_analysis_report(analysis_results, f1_strategy)
        
        print("\nAnalysis completed successfully!")
        print("Check the following outputs:")
        print("  - results/class_distribution_report.json")
        print("  - results/analysis_plots/class_distribution_overview.png")
        print("  - results/analysis_plots/detailed_indicator_analysis.png")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()