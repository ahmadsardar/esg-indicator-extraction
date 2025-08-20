# FinBERT-ESG: Multi-Task ESG Information Extraction from Corporate Reports

## Thesis Overview
**Title**: Context-Aware ESG Information Extraction from Corporate Reports using Domain-Specific Language Models
**Type**: Bachelor Thesis (6000 words)
**Duration**: 6 weeks (accelerated timeline)
**Approach**: Fine-tuned FinBERT with lightweight multi-task architecture for ESG indicator classification, numerical data extraction, and category classification

## 🎯 Key Achievements
- **ESG Indicator Detection**: 47% F1-score (292% improvement over base FinBERT)
- **Category Classification**: 33% F1-score (51% improvement over base FinBERT)
- **Multi-task Learning**: Simultaneous ESG classification, numerical extraction, and categorization
- **Lightweight Architecture**: Memory-efficient model with frozen early layers
- **Comprehensive Evaluation**: Statistical comparison with baseline models and visualization

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
│   ├── 08_finbert_esg_lightweight.py  # FinBERT training script
│   └── 09_model_comparison_evaluation.py # Model comparison and evaluation
├── results/
│   ├── analysis_plots/                # Performance visualization
│   ├── comparison/                    # Model comparison results
│   │   ├── performance_comparison.png # Performance charts
│   │   ├── improvement_analysis.png   # Improvement analysis
│   │   └── detailed_comparison_report_*.json # Detailed metrics
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
  - ESG Indicator Classification (40 indicators)
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
| **ESG Indicator Detection** | 0.12 | **0.47** | **+292%** |
| **Category Classification** | 0.22 | **0.33** | **+51%** |
| **Numerical Detection** | 0.32 | 0.20 | -37% |

### Key Findings
- ✅ **Significant improvement** in ESG-specific tasks
- ✅ **Multi-task learning** effectively combines related objectives
- ✅ **Domain adaptation** substantially outperforms general models
- ⚠️ **Trade-off observed** in numerical detection (requires further investigation)

## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd Project

# Install dependencies
pip install -r requirements.txt

# Download required models (if not included)
python -c "from transformers import AutoModel; AutoModel.from_pretrained('ProsusAI/finbert')"
```

### Running the Pipeline

1. **Data Processing**:
```bash
python src/01_ontology_analyzer.py
python src/02_esg_indicators_expansion.py
python src/03_combine_indicators.py
```

2. **Model Training**:
```bash
python src/08_finbert_esg_lightweight.py
```

3. **Model Evaluation**:
```bash
python src/09_model_comparison_evaluation.py
```

## 📁 Key Files

- **`models/finbert_esg/best_lightweight_model/model.pt`**: Trained model weights
- **`results/comparison/`**: Comprehensive evaluation results and visualizations
- **`esgontology.owl`**: ESG ontology definition file
- **`docs/`**: Detailed documentation and guidelines

## ESG Indicators Framework
**Comprehensive Indicator Set**: 40 indicators across three categories

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
        self.indicator_output = nn.Linear(768, 40)  # ESG indicators
        self.numerical_output = nn.Linear(768, 1)   # Numerical detection
        self.category_output = nn.Linear(768, num_categories)  # Categories
        
        # Residual connections for enhanced learning
        self.indicator_residual = nn.Linear(768, 40)
        self.numerical_residual = nn.Linear(768, 1)
        self.category_residual = nn.Linear(768, num_categories)
```

### Training Configuration
- **Learning Rate**: 2e-5 with linear decay
- **Batch Size**: 8 (with gradient accumulation)
- **Epochs**: 10 with early stopping
- **Loss Function**: Weighted multi-task loss
- **Optimizer**: AdamW with weight decay

## 📈 Performance Analysis

### Strengths
1. **Domain Adaptation**: Significant improvement in ESG-specific tasks
2. **Multi-task Learning**: Effective knowledge sharing between related tasks
3. **Memory Efficiency**: Lightweight architecture suitable for production
4. **Comprehensive Evaluation**: Robust statistical analysis and visualization

### Areas for Improvement
1. **Numerical Detection**: Performance trade-off requires investigation
2. **Data Scale**: Larger annotated datasets could improve generalization
3. **Category Granularity**: More fine-grained ESG subcategories
4. **Cross-domain Evaluation**: Testing on different industry sectors

**Note**: This project demonstrates the practical application of domain-adapted language models for ESG information extraction, contributing to the growing field of sustainable finance and AI-driven ESG analysis.