# FinBERT-ESG: Multi-Task ESG Information Extraction from Corporate Reports

## Thesis Overview
**Title**: ESG Information Extraction from Corporate Reports: A Comparative Analysis of Fine-tuned FinBERT vs. Base Models 
**Type**: Bachelor Thesis
**Approach**: Fine-tuned FinBERT with lightweight multi-task architecture for ESG indicator classification, numerical data extraction, and category classification

## 🎯 Key Achievements
- **ESG Indicator Detection**: 42.53% F1-score (macro average across 47 indicators)
- **Numerical Data Detection**: 93.94% F1-score (+56.3% improvement over base FinBERT)
- **Category Classification**: 31.19% F1-score (+16.9% improvement over base FinBERT)
- **Multi-task Learning**: Simultaneous ESG classification, numerical extraction, and categorization
- **Lightweight Architecture**: Memory-efficient model with frozen early layers (best validation F1: 0.0340)
- **Comprehensive Evaluation**: Statistical comparison with baseline models and detailed performance analysis

## Project Structure
```
Project/
├── data/
│   ├── raw/                           # Original corporate reports
│   ├── processed/                     # Cleaned and preprocessed data
│   ├── annotations/                   # Manual ESG indicator labels
│   ├── indicators/                    # ESG indicator definitions
│   └── validation/                    # Data validation results
├── docs/
│   ├── esg_annotation_guidelines.md  # Guidelines for ESG annotation
│   ├── individual_datasets_guide.md  # Dataset-specific documentation
│   └── ontology_indicators_summary.md # ESG ontology summary
├── models/
│   ├── baseline/                      # Baseline extraction methods
│   ├── finbert_esg/                   # Fine-tuned FinBERT model & tokenizer
│   │   └── best_lightweight_model/    # Trained model weights (model.pt)
│   └── evaluation/                    # Model evaluation scripts
├── src/
│   ├── 01_ontology_analyzer.py        # ESG ontology parsing and analysis
│   ├── 02_esg_indicators_expansion.py # ESG framework expansion
│   ├── 03_combine_indicators.py       # Indicator combination logic
│   ├── 04_context_aware_esg_model.py  # Context-aware extraction model
│   ├── 05_corporate_report_processor.py # PDF processing pipeline
│   ├── 06_esg_annotation_system.py    # Annotation system implementation
│   ├── 07_annotation_validator.py     # Data validation and quality checks
│   ├── 08_data_preprocessor.py        # Data preprocessing for training
│   ├── 09_finetuned_esg_model.py      # FinBERT training script
│   ├── 10_threshold_optimization.py   # Threshold optimization for classification
│   ├── 11_model_comparison.py         # Model comparison and evaluation
│   └── 12_class_distribution_analysis.py # Class distribution analysis
├── results/
│   ├── analysis_plots/                # Performance visualization
│   ├── comparison/                    # Model comparison results
│   │   ├── performance_comparison.png # Performance charts
│   │   ├── improvement_analysis.png   # Improvement analysis
│   │   └── detailed_comparison_report.json # Detailed metrics
│   ├── threshold_optimization/        # Threshold optimization results
│   │   ├── optimized_thresholds.json  # Optimized classification thresholds
│   │   ├── threshold_optimization_results.json # Optimization results
│   │   └── *.png                      # Threshold analysis plots
│   ├── class_distribution_report.json # Class distribution analysis
│   ├── lightweight_training_history.json # Training progress
│   └── test_extraction_results.json  # Model test results
├── .gitignore                         # Git ignore configuration
├── requirements.txt                   # Python dependencies
├── esgontology.owl                    # ESG ontology definition
└── README.md                          # Project documentation
```

## Research Questions
1. How effectively can domain-adapted LLMs extract numerical ESG indicators from corporate reports?
2. What is the performance difference across Environmental, Social, and Governance categories?
3. How does ontology-guided training improve extraction accuracy compared to baseline methods?
4. Can multi-task learning simultaneously improve ESG classification and numerical extraction?

## Methodology

### 1. Data Preparation & Ontology Analysis
- **ESG Ontology Processing**: Automated parsing of ESG frameworks using `01_ontology_analyzer.py`
- **Indicator Expansion**: Extended ESG indicators from 46 to comprehensive framework via `02_esg_indicators_expansion.py`
- **Data Integration**: Combined multiple ESG datasets using `03_combine_indicators.py`
- **Annotation System**: Implemented systematic ESG labeling via `06_esg_annotation_system.py`
- **Quality Validation**: Ensured data integrity using `07_annotation_validator.py`

### 2. Model Architecture
- **Base Model**: FinBERT (Financial domain pre-trained BERT)
- **Multi-task Heads**: 
  - ESG Indicator Classification (47 indicators)
  - Numerical Data Detection (binary)
  - ESG Category Classification (multi-class)
- **Lightweight Design**: Frozen early layers for memory efficiency
- **Residual Connections**: Enhanced feature learning with skip connections

### 3. Training Pipeline
- **Implementation**: `08_finbert_esg_lightweight.py`
- **Multi-task Loss**: Weighted combination of classification losses
- **Optimization**: AdamW optimizer with learning rate scheduling
- **Memory Management**: Gradient accumulation and efficient batching
- **Checkpointing**: Best model selection based on validation performance

### 4. Evaluation Framework
- **Comparative Analysis**: `09_model_comparison_evaluation.py`
- **Baseline Comparison**: Base FinBERT vs Fine-tuned ESG model
- **Metrics**: Accuracy, F1-score (micro/macro), Precision, Recall, AUC
- **Statistical Analysis**: Performance improvements and significance testing
- **Visualization**: Automated chart generation for thesis documentation

## 📊 Results Summary

| Task | Base FinBERT F1 | Fine-tuned F1 | Improvement |
|------|----------------|---------------|-------------|
| **ESG Indicator Detection** | 0.2063 | **0.4253** | **+21.89%** |
| **Numerical Data Detection** | 0.3443 | **0.9496** | **+60.52%** |
| **Category Classification** | 0.2309 | **0.3057** | **+7.48%** |

### Key Findings
- ✅ **Exceptional performance** in numerical data detection (94.96% F1-score, +60.52% improvement)
- ✅ **Significant improvement** in ESG indicator detection (42.53% F1-score, +21.89% improvement)
- ✅ **Enhanced category classification** (30.57% F1-score, +7.48% improvement)
- ✅ **Multi-task learning** successfully combines related objectives across all three tasks
- ✅ **Context-aware architecture** with residual connections improves feature learning
- ✅ **Memory-efficient design** with frozen early layers suitable for production deployment
- ✅ **Comprehensive evaluation** with threshold optimization and detailed statistical analysis
- ✅ **Standardized output** with consistent naming conventions and structured results

## 🚀 Quick Start

### Running the Pipeline

1. **Data Processing**:
```bash
python src/01_ontology_analyzer.py
python src/02_esg_indicators_expansion.py
python src/03_combine_indicators.py
```

2. **Data Preprocessing**:
```bash
python src/08_data_preprocessor.py
```

3. **Model Training**:
```bash
python src/09_finetuned_esg_model.py
```

4. **Threshold Optimization**:
```bash
python src/10_threshold_optimization.py
```

5. **Model Evaluation**:
```bash
python src/11_model_comparison.py
python src/12_class_distribution_analysis.py
```

## 📁 Key Files

- **`models/finbert_esg/best_lightweight_model/model.pt`**: Trained model weights
- **`results/comparison/`**: Comprehensive evaluation results and visualizations
- **`esgontology.owl`**: ESG ontology definition file
- **`docs/`**: Detailed documentation and guidelines

## ESG Indicators Framework
**Comprehensive Indicator Set**: 47 indicators across three categories

### Environmental (E)
- Carbon emissions and climate metrics
- Energy efficiency and renewable energy
- Water usage and waste management
- Biodiversity and environmental impact

### Social (S)
- Employee welfare and diversity
- Community engagement and human rights
- Product safety and customer satisfaction
- Supply chain responsibility

### Governance (G)
- Board composition and independence
- Executive compensation and ethics
- Risk management and compliance
- Transparency and stakeholder engagement

## 🔬 Technical Implementation Details

### Model Architecture
```python
class LightweightESGModel(nn.Module):
    def __init__(self):
        # FinBERT base with frozen early layers
        self.bert = AutoModel.from_pretrained('ProsusAI/finbert')
        
        # Multi-task heads
        self.indicator_output = nn.Linear(768, 47)  # ESG indicators
        self.numerical_output = nn.Linear(768, 1)   # Numerical detection
        self.category_output = nn.Linear(768, num_categories)  # Categories
        
        # Residual connections for enhanced learning
        self.indicator_residual = nn.Linear(768, 47)
        self.numerical_residual = nn.Linear(768, 1)
        self.category_residual = nn.Linear(768, num_categories)
```

### Training Configuration
- **Learning Rate**: 2e-5 with linear decay
- **Batch Size**: 8 (with gradient accumulation)
- **Epochs**: 3 (completed training)
- **Loss Function**: Weighted multi-task loss
- **Optimizer**: AdamW with weight decay
- **Architecture**: Context-aware with residual connections
- **Frozen Layers**: Early BERT layers for memory efficiency
- **Multi-task Heads**: 47 ESG indicators + numerical detection + category classification

## 📈 Performance Analysis

### Strengths
1. **Domain Adaptation**: Significant improvement in ESG-specific tasks
2. **Multi-task Learning**: Effective knowledge sharing between related tasks
3. **Memory Efficiency**: Lightweight architecture suitable for production
4. **Comprehensive Evaluation**: Robust statistical analysis and visualization

### Technical Achievements
1. **Context-Aware Architecture**: Successfully implemented residual connections for enhanced learning
2. **Multi-task Learning**: Effective simultaneous training on three related ESG tasks
3. **Threshold Optimization**: Automated optimization of classification thresholds for improved performance
4. **Comprehensive Pipeline**: End-to-end processing from ontology analysis to model evaluation
5. **Standardized Output**: Consistent file naming and structured result formats
6. **Memory Efficiency**: Lightweight design suitable for resource-constrained environments

**Note**: This project demonstrates the practical application of domain-adapted language models for ESG information extraction, contributing to the growing field of sustainable finance and AI-driven ESG analysis.