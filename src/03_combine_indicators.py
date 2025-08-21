"""
Combine ESG Indicators Script

This script combines the ontology-derived indicators and expanded framework-based indicators
into one comprehensive list without duplicates.

Input Files:
- Ontology indicators: From ontology analysis (51 indicators)
- Expanded indicators: From framework analysis (46 indicators)

Output:
- Combined comprehensive ESG indicators list
- Duplicate detection and removal
- Final count and statistics
"""

import pandas as pd
import json
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def main():
    # Set up paths
    data_dir = Path('data/indicators')
    output_dir = Path('data/indicators')
    
    print("=== COMBINING ESG INDICATORS ===")
    print("Loading data files...")
    
    # Load ontology indicators
    with open(data_dir / 'esg_ontology_analysis.json', 'r', encoding='utf-8') as f:
        ontology_data = json.load(f)
    
    # Load expanded indicators
    expanded_df = pd.read_csv(data_dir / 'expanded_esg_indicators.csv')
    
    # Load newly discovered indicators
    newly_discovered_df = pd.read_csv(data_dir / 'manually_discovered_indicators.csv')
    
    # Combine performance indicators and ESG categories from ontology
    all_ontology_indicators = ontology_data.get('performance_indicators', []) + ontology_data.get('esg_categories', [])
    
    print(f"Ontology indicators loaded: {len(all_ontology_indicators)}")
    print(f"Expanded indicators loaded: {len(expanded_df)}")
    print(f"Newly discovered indicators loaded: {len(newly_discovered_df)}")
    print(f"Total before deduplication: {len(all_ontology_indicators) + len(expanded_df) + len(newly_discovered_df)}")
    
    print("\nConverting ontology data to DataFrame...")
    
    # Convert ontology indicators to DataFrame format
    ontology_indicators = []
    
    # Create category mapping for better organization
    category_mapping = {
        'Energy': 'Environmental',
        'Water': 'Environmental', 
        'WaterEfficiency': 'Environmental',
        'WaterRecycling': 'Environmental',
        'Waste': 'Environmental',
        'WasteCategory': 'Environmental',
        'WasteOutput': 'Environmental',
        'WasteProcessing': 'Environmental',
        'WasteRecycling': 'Environmental',
        'WasteReduction': 'Environmental',
        'Hazardous_Waste': 'Environmental',
        'Emission': 'Environmental',
        'GHGEmission': 'Environmental',
        'Biodiversity': 'Environmental',
        'ClimateChange': 'Environmental',
        'EnvironmentalCompliance': 'Environmental',
        'EnvironmentalImpact': 'Environmental',
        'EnvironmentalManagement': 'Environmental',
        'EnvironmentalPolicy': 'Environmental',
        'EnvironmentalReporting': 'Environmental',
        'NaturalResource': 'Environmental',
        'PollutionPrevention': 'Environmental',
        'RenewableEnergy': 'Environmental',
        'SustainableSupplyChain': 'Environmental',
        'CarbonFootprint': 'Environmental',
        'Employee': 'Social',
        'EmployeeEngagement': 'Social',
        'EmployeeTraining': 'Social',
        'EmployeeWellbeing': 'Social',
        'WorkplaceSafety': 'Social',
        'DiversityInclusion': 'Social',
        'HumanRights': 'Social',
        'CommunityEngagement': 'Social',
        'CustomerSatisfaction': 'Social',
        'ProductSafety': 'Social',
        'SocialImpact': 'Social',
        'LaborPractices': 'Social',
        'HealthSafety': 'Social',
        'Stakeholder': 'Social',
        'CorporateGovernance': 'Governance',
        'BoardComposition': 'Governance',
        'ExecutiveCompensation': 'Governance',
        'RiskManagement': 'Governance',
        'Compliance': 'Governance',
        'Ethics': 'Governance',
        'Transparency': 'Governance',
        'DataPrivacy': 'Governance',
        'Cybersecurity': 'Governance',
        'BusinessEthics': 'Governance',
        'AntiCorruption': 'Governance'
    }
    
    for i, indicator in enumerate(all_ontology_indicators, 1):
        indicator_name = indicator.get('name', '')
        
        # Generate proper ID
        indicator_id = f"ONT_{i:03d}"
        
        # Determine category based on indicator name
        category = category_mapping.get(indicator_name, 'Environmental')  # Default to Environmental
        
        # Determine subcategory (use the indicator name as subcategory)
        subcategory = indicator_name
        
        # Create description from comment or generate one
        description = indicator.get('comment', '') or f"ESG indicator for {indicator_name.replace('_', ' ').lower()}"
        
        ontology_indicators.append({
            'indicator_id': indicator_id,
            'name': indicator_name,
            'description': description,
            'category': category,
            'subcategory': subcategory,
            'data_type': 'Quantitative',  # Default data type
            'unit': '',  # Will be determined during data extraction
            'source': 'ESG_ONTOLOGY',
            'framework': 'ONTOLOGY',
            'original_source': indicator.get('uri', '')
        })
    
    ontology_df = pd.DataFrame(ontology_indicators)
    print(f"Ontology DataFrame shape: {ontology_df.shape}")
    
    print("\nStandardizing expanded indicators DataFrame...")
    
    # Standardize expanded indicators DataFrame
    expanded_standardized = expanded_df.copy()
    
    # Ensure all required columns exist
    required_columns = ['indicator_id', 'name', 'description', 'category', 'subcategory', 
                       'data_type', 'unit', 'source', 'framework', 'original_source']
    
    for col in required_columns:
        if col not in expanded_standardized.columns:
            expanded_standardized[col] = ''
    
    # Reorder columns to match ontology DataFrame
    expanded_standardized = expanded_standardized[required_columns]
    
    print(f"Expanded DataFrame shape: {expanded_standardized.shape}")
    
    print("\nStandardizing newly discovered indicators DataFrame...")
    
    # Standardize newly discovered indicators DataFrame
    newly_discovered_standardized = newly_discovered_df.copy()
    
    # Ensure all required columns exist
    for col in required_columns:
        if col not in newly_discovered_standardized.columns:
            newly_discovered_standardized[col] = ''
    
    # Reorder columns to match ontology DataFrame
    newly_discovered_standardized = newly_discovered_standardized[required_columns]
    
    print(f"Newly discovered DataFrame shape: {newly_discovered_standardized.shape}")
    
    print("\nCombining and deduplicating...")
    
    # Combine all three DataFrames
    combined_df = pd.concat([ontology_df, expanded_standardized, newly_discovered_standardized], ignore_index=True)
    
    print(f"Combined DataFrame shape before deduplication: {combined_df.shape}")
    
    # Check for duplicates based on name (case-insensitive)
    combined_df['name_lower'] = combined_df['name'].str.lower().str.strip()
    
    # Find duplicates
    duplicates = combined_df[combined_df.duplicated(subset=['name_lower'], keep=False)]
    print(f"\nFound {len(duplicates)} duplicate entries:")
    
    if len(duplicates) > 0:
        print("Duplicate indicators:")
        for name in duplicates['name_lower'].unique():
            dup_entries = duplicates[duplicates['name_lower'] == name]
            print(f"  - {name}: {len(dup_entries)} entries from {dup_entries['source'].unique()}")
    
    # Remove duplicates - keep the first occurrence (ontology takes precedence)
    final_df = combined_df.drop_duplicates(subset=['name_lower'], keep='first')
    
    # Drop the helper column
    final_df = final_df.drop('name_lower', axis=1)
    
    # Reset index
    final_df = final_df.reset_index(drop=True)
    
    print(f"\nFinal DataFrame shape after deduplication: {final_df.shape}")
    print(f"Removed {len(combined_df) - len(final_df)} duplicate indicators")
    
    print("\n=== FINAL ESG INDICATORS ANALYSIS ===")
    print(f"Total unique indicators: {len(final_df)}")
    print()
    
    # By source
    print("By Source:")
    source_counts = final_df['source'].value_counts()
    for source, count in source_counts.items():
        percentage = (count / len(final_df)) * 100
        print(f"  {source}: {count} ({percentage:.1f}%)")
    print()
    
    # By framework
    print("By Framework:")
    framework_counts = final_df['framework'].value_counts()
    for framework, count in framework_counts.items():
        percentage = (count / len(final_df)) * 100
        print(f"  {framework}: {count} ({percentage:.1f}%)")
    print()
    
    # By category
    print("By Category:")
    category_counts = final_df['category'].value_counts()
    for category, count in category_counts.items():
        percentage = (count / len(final_df)) * 100
        print(f"  {category}: {count} ({percentage:.1f}%)")
    print()
    
    # By data type
    print("By Data Type:")
    datatype_counts = final_df['data_type'].value_counts()
    for dtype, count in datatype_counts.items():
        percentage = (count / len(final_df)) * 100
        print(f"  {dtype}: {count} ({percentage:.1f}%)")
    
    print("\nSaving combined dataset...")
    
    # Save as CSV
    output_csv = output_dir / 'final_esg_indicators.csv'
    final_df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"Saved final indicators to: {output_csv}")
    
    # Save as JSON for detailed structure
    output_json = output_dir / 'final_esg_indicators.json'
    
    final_data = {
        'metadata': {
            'total_indicators': len(final_df),
            'ontology_indicators': len(ontology_df),
            'expanded_indicators': len(expanded_standardized),
            'newly_discovered_indicators': len(newly_discovered_standardized),
            'duplicates_removed': len(combined_df) - len(final_df),
            'creation_date': pd.Timestamp.now().isoformat(),
            'description': 'Final comprehensive ESG indicators combining ontology-derived, framework-based, and newly discovered indicators from corporate reports',
            'ontology_performance_indicators': len(ontology_data.get('performance_indicators', [])),
            'ontology_esg_categories': len(ontology_data.get('esg_categories', []))
        },
        'statistics': {
            'by_source': source_counts.to_dict(),
            'by_framework': framework_counts.to_dict(),
            'by_category': category_counts.to_dict(),
            'by_data_type': datatype_counts.to_dict()
        },
        'indicators': final_df.to_dict('records')
    }
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved final indicators JSON to: {output_json}")
    
    # Save summary statistics
    summary_stats = pd.DataFrame([
        {'metric': 'Total Indicators', 'value': len(final_df)},
        {'metric': 'Ontology Indicators', 'value': len(ontology_df)},
        {'metric': 'Expanded Indicators', 'value': len(expanded_standardized)},
        {'metric': 'Duplicates Removed', 'value': len(combined_df) - len(final_df)},
        {'metric': 'Environmental Indicators', 'value': category_counts.get('Environmental', 0)},
        {'metric': 'Social Indicators', 'value': category_counts.get('Social', 0)},
        {'metric': 'Governance Indicators', 'value': category_counts.get('Governance', 0)}
    ])
    
    summary_stats.to_csv(output_dir / 'final_indicators_summary.csv', index=False)
    print(f"Saved summary statistics to: {output_dir / 'final_indicators_summary.csv'}")
    
    print("\n=== FINAL ESG INDICATORS DATASET ===")
    print(f"Shape: {final_df.shape}")
    print(f"Columns: {list(final_df.columns)}")
    print()
    
    # Show sample indicators from each source
    print("Sample indicators by source:")
    for source in final_df['source'].unique():
        sample = final_df[final_df['source'] == source].head(3)
        print(f"\n{source}:")
        for _, row in sample.iterrows():
            print(f"  - {row['name']} ({row['category']})")
    
    print("\n=== DATASET READY FOR MODEL TRAINING ===")
    print(f"Total unique ESG indicators: {len(final_df)}")
    print("Files created:")
    print(f"  - {output_csv}")
    print(f"  - {output_json}")
    print(f"  - {output_dir / 'final_indicators_summary.csv'}")
    
    return final_df

if __name__ == "__main__":
    main()