#!/usr/bin/env python3
"""
Expanded ESG Indicators Analysis Script

This script analyzes the expanded ESG indicators derived from established frameworks:
1. GRI Standards - Global Reporting Initiative indicators
2. SASB Standards - Sustainability Accounting Standards Board indicators
3. TCFD Framework - Task Force on Climate-related Financial Disclosures
4. Additional ESG Frameworks - Other recognized standards

This complements the ontology-derived indicators with comprehensive framework-based metrics.
"""

import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
import re

def main():
    print("=== Expanded ESG Indicators Analysis ===")
    
    # Load the expanded ESG indicators
    indicators_path = Path('data/indicators/expanded_esg_indicators.csv')
    if not indicators_path.exists():
        print(f"Error: {indicators_path} not found!")
        return
    
    indicators_df = pd.read_csv(indicators_path)
    print(f"Loaded {len(indicators_df)} expanded ESG indicators")
    print(f"Columns: {list(indicators_df.columns)}")
    
    # Load the JSON version for detailed analysis
    json_path = Path('data/indicators/expanded_esg_indicators.json')
    if json_path.exists():
        with open(json_path, 'r') as f:
            indicators_json = json.load(f)
        print(f"JSON structure keys: {list(indicators_json.keys())}")
        if 'indicators' in indicators_json:
            print(f"Total indicators in JSON: {len(indicators_json['indicators'])}")
    
    # Framework Distribution Analysis
    print("\n=== Framework Distribution Analysis ===")
    framework_counts = indicators_df['source'].value_counts()
    print("Framework Distribution:")
    for framework, count in framework_counts.items():
        percentage = (count / len(indicators_df)) * 100
        print(f"  {framework}: {count} ({percentage:.1f}%)")
    
    # Category Analysis
    print("\n=== Category Analysis ===")
    category_counts = indicators_df['category'].value_counts()
    print("Category Distribution:")
    for category, count in category_counts.items():
        percentage = (count / len(indicators_df)) * 100
        print(f"  {category}: {count} ({percentage:.1f}%)")
    
    # Cross-tabulation of Framework vs Category
    print("\n=== Framework vs Category Cross-tabulation ===")
    cross_tab = pd.crosstab(indicators_df['source'], indicators_df['category'])
    print(cross_tab)
    
    # Detailed Framework Analysis
    print("\n=== Detailed Framework Analysis ===")
    
    # GRI indicators
    gri_indicators = indicators_df[indicators_df['source'] == 'GRI']
    print(f"\nGRI Indicators ({len(gri_indicators)}):")
    gri_categories = gri_indicators['category'].value_counts()
    for category, count in gri_categories.items():
        print(f"  {category}: {count}")
    
    print("\nSample GRI indicators:")
    for _, indicator in gri_indicators.head(3).iterrows():
        print(f"  - {indicator['name']}: {indicator['description'][:80]}...")
    
    # SASB indicators
    sasb_indicators = indicators_df[indicators_df['source'] == 'SASB']
    print(f"\nSASB Indicators ({len(sasb_indicators)}):")
    sasb_categories = sasb_indicators['category'].value_counts()
    for category, count in sasb_categories.items():
        print(f"  {category}: {count}")
    
    print("\nSample SASB indicators:")
    for _, indicator in sasb_indicators.head(3).iterrows():
        print(f"  - {indicator['name']}: {indicator['description'][:80]}...")
    
    # TCFD indicators
    tcfd_indicators = indicators_df[indicators_df['source'] == 'TCFD']
    print(f"\nTCFD Indicators ({len(tcfd_indicators)}):")
    tcfd_categories = tcfd_indicators['category'].value_counts()
    for category, count in tcfd_categories.items():
        print(f"  {category}: {count}")
    
    print("\nSample TCFD indicators:")
    for _, indicator in tcfd_indicators.head(3).iterrows():
        print(f"  - {indicator['name']}: {indicator['description'][:80]}...")
    
    # Indicator Complexity Analysis
    print("\n=== Indicator Complexity Analysis ===")
    indicators_df['description_length'] = indicators_df['description'].str.len()
    print("Description Length Statistics:")
    print(indicators_df['description_length'].describe())
    
    # Analyze common keywords in descriptions
    print("\n=== Common Keywords Analysis ===")
    all_descriptions = ' '.join(indicators_df['description'].astype(str))
    words = re.findall(r'\b\w{4,}\b', all_descriptions.lower())
    
    # Remove common stop words
    stop_words = {'that', 'with', 'from', 'they', 'been', 'have', 'this', 'will', 'would', 'could', 'should', 'their', 'there', 'where', 'when', 'what', 'which', 'while', 'such', 'these', 'those', 'than', 'more', 'most', 'some', 'other', 'also', 'only', 'each', 'both', 'many', 'much', 'well', 'make', 'made', 'used', 'using', 'include', 'includes', 'including'}
    filtered_words = [word for word in words if word not in stop_words]
    
    word_counts = Counter(filtered_words)
    top_keywords = word_counts.most_common(15)
    
    print("Top 15 Keywords in ESG Indicator Descriptions:")
    for word, count in top_keywords:
        print(f"  {word}: {count}")
    
    # Integration with Ontology Indicators
    print("\n=== Integration with Ontology Indicators ===")
    ontology_path = Path('data/indicators/esg_indicators_mapping.csv')
    if ontology_path.exists():
        ontology_df = pd.read_csv(ontology_path)
        
        print(f"Ontology indicators: {len(ontology_df)}")
        print(f"Expanded indicators: {len(indicators_df)}")
        print(f"Total combined indicators: {len(ontology_df) + len(indicators_df)}")
        
        print("\nOntology categories:")
        ontology_categories = ontology_df['Category'].value_counts()
        for category, count in ontology_categories.items():
            print(f"  {category}: {count}")
        
        print("\nExpanded framework categories:")
        for category, count in category_counts.items():
            print(f"  {category}: {count}")
    else:
        print("Ontology indicators file not found. Run the ontology analysis first.")
    
    # Create comprehensive summary
    print("\n=== Comprehensive Summary ===")
    summary = {
        'expanded_indicators': {
            'total': len(indicators_df),
            'by_framework': framework_counts.to_dict(),
            'by_category': category_counts.to_dict()
        }
    }
    
    if ontology_path.exists():
        summary['ontology_indicators'] = {
            'total': len(ontology_df),
            'by_category': ontology_df['Category'].value_counts().to_dict()
        }
        summary['combined_total'] = len(ontology_df) + len(indicators_df)
    
    print("Comprehensive ESG Indicator Summary:")
    print(json.dumps(summary, indent=2))
    
    # Save analysis results
    print("\n=== Saving Analysis Results ===")
    output_dir = Path('data/indicators')
    
    # Save summary statistics
    with open(output_dir / 'expanded_indicators_analysis.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save framework statistics
    framework_stats = pd.DataFrame({
        'Framework': framework_counts.index,
        'Count': framework_counts.values,
        'Percentage': (framework_counts.values / len(indicators_df) * 100).round(2)
    })
    framework_stats.to_csv(output_dir / 'framework_statistics.csv', index=False)
    
    # Save category statistics
    category_stats = pd.DataFrame({
        'Category': category_counts.index,
        'Count': category_counts.values,
        'Percentage': (category_counts.values / len(indicators_df) * 100).round(2)
    })
    category_stats.to_csv(output_dir / 'category_statistics.csv', index=False)
    
    print(f"Analysis results saved to: {output_dir}")
    print(f"  - Summary: expanded_indicators_analysis.json")
    print(f"  - Framework stats: framework_statistics.csv")
    print(f"  - Category stats: category_statistics.csv")
    
    print("\n=== Analysis Complete ===")
    print("\nKey Insights:")
    print("1. Comprehensive Coverage: Indicators span Environmental, Social, and Governance domains")
    print("2. Industry Relevance: SASB indicators provide sector-specific metrics")
    print("3. Climate Focus: TCFD indicators address climate-related risks and opportunities")
    print("4. Global Standards: GRI indicators ensure international comparability")
    print("\nIntegration with Ontology:")
    if ontology_path.exists():
        total_combined = len(ontology_df) + len(indicators_df)
        print(f"- Combined Total: ~{total_combined} indicators ({len(ontology_df)} ontology + {len(indicators_df)} framework-based)")
    print("- Complementary Coverage: Ontology provides semantic structure, frameworks provide practical metrics")
    print("- Model Training: Both sets can be used for comprehensive ESG extraction model training")

if __name__ == "__main__":
    main()