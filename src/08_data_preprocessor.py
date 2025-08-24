"""Data Preprocessor for ESG Model Evaluation

Converts the existing annotation format to the expected label format
for model evaluation and threshold optimization.
"""

import pandas as pd
import numpy as np
import json
import os
from sklearn.preprocessing import LabelEncoder

class ESGDataPreprocessor:
    """Preprocesses ESG annotation data for model evaluation"""
    
    def __init__(self):
        self.category_encoder = LabelEncoder()
        self.category_mapping = {
            'Environmental': 0,
            'Social': 1, 
            'Governance': 2
        }
        
    def process_dataset(self, input_path, output_path):
        """Process dataset to create required label columns"""
        print(f"Processing dataset: {input_path}")
        
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
            
        df = pd.read_csv(input_path)
        print(f"Loaded {len(df)} samples")
        
        # Create ESG indicator label (binary: is_esg_relevant)
        df['esg_indicator_label'] = df['is_esg_relevant'].astype(int)
        
        # Create numerical detection label (binary: has_numerical_data)
        df['numerical_label'] = df['has_numerical_data'].astype(int)
        
        # Create category classification label (multiclass: primary_category)
        # Map categories to integers
        df['category_label'] = df['primary_category'].map(self.category_mapping)
        
        # Handle missing category labels (set to 0 for Environmental as default)
        df['category_label'] = df['category_label'].fillna(0).astype(int)
        
        # Keep only necessary columns for evaluation
        eval_columns = [
            'text', 'esg_indicator_label', 'numerical_label', 'category_label',
            'is_esg_relevant', 'has_numerical_data', 'primary_category',
            'confidence_score', 'annotation_quality', 'training_weight'
        ]
        
        # Add columns that exist in the dataset
        available_columns = [col for col in eval_columns if col in df.columns]
        processed_df = df[available_columns].copy()
        
        # Save processed dataset
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        processed_df.to_csv(output_path, index=False)
        
        print(f"Processed dataset saved to: {output_path}")
        print(f"Dataset shape: {processed_df.shape}")
        
        # Print label distribution
        self.print_label_distribution(processed_df)
        
        return processed_df
    
    def print_label_distribution(self, df):
        """Print distribution of labels for analysis"""
        print("\nLabel Distribution:")
        print("=" * 30)
        
        # ESG Indicator Distribution
        esg_dist = df['esg_indicator_label'].value_counts().sort_index()
        print(f"ESG Indicator (Binary):")
        print(f"  Not ESG (0): {esg_dist.get(0, 0)} ({esg_dist.get(0, 0)/len(df)*100:.1f}%)")
        print(f"  ESG (1): {esg_dist.get(1, 0)} ({esg_dist.get(1, 0)/len(df)*100:.1f}%)")
        
        # Numerical Detection Distribution
        num_dist = df['numerical_label'].value_counts().sort_index()
        print(f"\nNumerical Detection (Binary):")
        print(f"  No Numbers (0): {num_dist.get(0, 0)} ({num_dist.get(0, 0)/len(df)*100:.1f}%)")
        print(f"  Has Numbers (1): {num_dist.get(1, 0)} ({num_dist.get(1, 0)/len(df)*100:.1f}%)")
        
        # Category Distribution
        cat_dist = df['category_label'].value_counts().sort_index()
        print(f"\nCategory Classification (Multiclass):")
        category_names = ['Environmental', 'Social', 'Governance']
        for i, name in enumerate(category_names):
            count = cat_dist.get(i, 0)
            print(f"  {name} ({i}): {count} ({count/len(df)*100:.1f}%)")
    
    def create_evaluation_splits(self, test_path, val_path, train_path=None):
        """Create evaluation-ready datasets from existing splits"""
        print("Creating evaluation-ready datasets...")
        
        # Process test set
        if os.path.exists(test_path):
            test_df = self.process_dataset(
                test_path, 
                test_path.replace('.csv', '_processed.csv')
            )
        else:
            print(f"Warning: Test set not found at {test_path}")
            test_df = None
        
        # Process validation set
        if os.path.exists(val_path):
            val_df = self.process_dataset(
                val_path,
                val_path.replace('.csv', '_processed.csv')
            )
        else:
            print(f"Warning: Validation set not found at {val_path}")
            val_df = None
        
        # Process training set if provided
        train_df = None
        if train_path and os.path.exists(train_path):
            train_df = self.process_dataset(
                train_path,
                train_path.replace('.csv', '_processed.csv')
            )
        
        return test_df, val_df, train_df
    
    def validate_processed_data(self, df):
        """Validate that processed data has required columns and valid labels"""
        required_columns = ['text', 'esg_indicator_label', 'numerical_label', 'category_label']
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Validate label ranges
        if not df['esg_indicator_label'].isin([0, 1]).all():
            raise ValueError("ESG indicator labels must be 0 or 1")
            
        if not df['numerical_label'].isin([0, 1]).all():
            raise ValueError("Numerical labels must be 0 or 1")
            
        if not df['category_label'].isin([0, 1, 2]).all():
            raise ValueError("Category labels must be 0, 1, or 2")
        
        print("✓ Data validation passed")
        return True

def main():
    """Main execution function"""
    print("ESG Data Preprocessor")
    print("=" * 30)
    
    # Initialize preprocessor
    preprocessor = ESGDataPreprocessor()
    
    # Define paths
    data_dir = 'data/annotations'
    test_path = f'{data_dir}/esg_test_set.csv'
    val_path = f'{data_dir}/esg_validation_set.csv'
    
    # Create evaluation-ready datasets
    test_df, val_df, train_df = preprocessor.create_evaluation_splits(
        test_path, val_path
    )
    
    # Validate processed data
    if test_df is not None:
        preprocessor.validate_processed_data(test_df)
        print(f"✓ Test set processed: {len(test_df)} samples")
    
    if val_df is not None:
        preprocessor.validate_processed_data(val_df)
        print(f"✓ Validation set processed: {len(val_df)} samples")
    
    print("\nData preprocessing completed successfully!")
    print("\nProcessed files:")
    if test_df is not None:
        print(f"  - {test_path.replace('.csv', '_processed.csv')}")
    if val_df is not None:
        print(f"  - {val_path.replace('.csv', '_processed.csv')}")
    
    return test_df, val_df

if __name__ == "__main__":
    main()