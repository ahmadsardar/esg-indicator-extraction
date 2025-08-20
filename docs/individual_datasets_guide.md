# Individual ESG Datasets Guide

## Overview
The corporate report processor now creates separate datasets for each PDF file, making it easier to track data sources and enable company-specific analysis.

## Dataset Structure

### Location
- **Combined Dataset**: `data/processed/corporate_reports/esg_training_dataset.csv`
- **Individual Datasets**: `data/processed/corporate_reports/individual_reports/`
- **Statistics**: `data/processed/corporate_reports/individual_reports/individual_report_statistics.json`

### Individual Dataset Files
Each PDF generates a separate CSV file named according to the pattern:
`{clean_pdf_name}_esg_dataset.csv`

Example files:
- `bayer_nachhaltigkeitsbericht_2023_esg_dataset.csv` (606 samples)
- `sap_2024_integrated_report_esg_dataset.csv` (2,018 samples)
- `DHL_Group_2023_Annual_Report_esg_dataset.csv` (1,473 samples)

## Using Multiple CSV Files for Model Training

### Yes, it's definitely possible!
Most machine learning frameworks support loading and combining multiple CSV files:

#### PyTorch Example
```python
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import glob

# Load all individual datasets
csv_files = glob.glob('data/processed/corporate_reports/individual_reports/*_esg_dataset.csv')
dataframes = [pd.read_csv(file) for file in csv_files]
combined_df = pd.concat(dataframes, ignore_index=True)

# Or load specific companies
bayer_df = pd.read_csv('data/processed/corporate_reports/individual_reports/bayer_nachhaltigkeitsbericht_2023_esg_dataset.csv')
sap_df = pd.read_csv('data/processed/corporate_reports/individual_reports/sap_2024_integrated_report_esg_dataset.csv')
```

#### TensorFlow/Keras Example
```python
import tensorflow as tf
import pandas as pd

# Load multiple datasets
datasets = []
for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    dataset = tf.data.Dataset.from_tensor_slices({
        'text': df['text'].values,
        'labels': df['label'].values
    })
    datasets.append(dataset)

# Combine datasets
combined_dataset = datasets[0]
for ds in datasets[1:]:
    combined_dataset = combined_dataset.concatenate(ds)
```

### Benefits of Individual Datasets

1. **Company-Specific Training**: Train models on specific companies
2. **Data Source Tracking**: Know exactly which company contributed which samples
3. **Balanced Sampling**: Control representation from different companies
4. **Cross-Validation**: Use company-based splits for validation
5. **Domain Adaptation**: Fine-tune models for specific industries

### Training Strategies

#### 1. Combined Training (Default)
```python
# Use all companies together
all_data = pd.concat([pd.read_csv(f) for f in csv_files])
```

#### 2. Company-Specific Training
```python
# Train separate models for each company
for csv_file in csv_files:
    company_data = pd.read_csv(csv_file)
    # Train model on company_data
```

#### 3. Stratified Training
```python
# Ensure balanced representation
from sklearn.model_selection import train_test_split

stratified_data = []
for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    # Sample equal amounts from each company
    sampled = df.sample(n=min(500, len(df)), random_state=42)
    stratified_data.append(sampled)

balanced_dataset = pd.concat(stratified_data)
```

#### 4. Leave-One-Company-Out Validation
```python
# Use for robust evaluation
for test_company in csv_files:
    test_data = pd.read_csv(test_company)
    train_files = [f for f in csv_files if f != test_company]
    train_data = pd.concat([pd.read_csv(f) for f in train_files])
    # Train on train_data, test on test_data
```

## Dataset Statistics Summary

- **Total Reports**: 20 companies
- **Total Samples**: 15,312 training examples
- **Unique Indicators**: 38 ESG indicators
- **Average Match Score**: 0.263
- **Companies Include**: Bayer, SAP, DHL, Deutsche Bank, E.ON, Siemens, Volkswagen, Nestlé, etc.

## Next Steps for Model Training

1. **Choose Training Strategy**: Combined vs. company-specific
2. **Data Preprocessing**: Tokenization, encoding
3. **Model Selection**: FinBERT fine-tuning
4. **Evaluation**: Company-based cross-validation
5. **Deployment**: Multi-company or specialized models

The individual datasets provide maximum flexibility for different training approaches while maintaining the option to use the combined dataset for traditional training.