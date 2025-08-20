# Context-Aware ESG Information Extraction from Corporate Reports

## Thesis Overview
**Title**: Context-Aware ESG Information Extraction from Corporate Reports using Domain-Specific Language Models
**Type**: Bachelor Thesis (6000 words)
**Duration**: 6 weeks (accelerated timeline)
**Approach**: Fine-tuning FinBERT with context-aware capabilities for comprehensive ESG indicator extraction and recognition

## Project Structure
```
Project/
├── data/
│   ├── raw/                           # Original corporate reports
│   ├── processed/                     # Cleaned and preprocessed data
│   ├── annotations/                   # Manual ESG indicator labels
│   ├── ontology/                      # ESG ontology analysis
│   ├── expanded_esg_indicators.json   # 46 comprehensive ESG indicators
│   └── expanded_esg_indicators.csv    # ESG indicators in CSV format
├── docs/
│   └── esg_annotation_guidelines.md  # Guidelines for ESG annotation
├── models/
│   ├── baseline/                      # Baseline extraction methods
│   ├── finbert_esg/                   # Fine-tuned FinBERT model
│   └── evaluation/                    # Model evaluation scripts
├── notebooks/
│   └── 01_ontology_analysis.ipynb     # ESG ontology analysis notebook
├── src/
│   ├── ontology_analyzer.py           # ESG ontology parsing and analysis
│   ├── esg_indicators_expansion.py    # ESG framework expansion script
│   └── context_aware_esg_model.py     # Context-aware extraction model
├── results/
│   ├── test_extraction_results.json   # Model test results
│   └── analysis_plots/                # Performance visualization
├── .gitignore                         # Git ignore configuration
├── requirements.txt                   # Python dependencies
└── README.md                          # Project documentation
```

## Research Questions
1. How effectively can domain-adapted LLMs extract numerical ESG indicators from corporate reports?
2. What is the performance difference across Environmental, Social, and Governance categories?
3. How does ontology-guided training improve extraction accuracy compared to baseline methods?

## Methodology
1. **Data Preparation**: Manual annotation of ESG indicators in corporate reports
2. **Model Selection**: Fine-tune FinBERT for ESG-specific Named Entity Recognition
3. **Training**: Domain adaptation using annotated ESG indicators
4. **Evaluation**: Performance comparison against baseline methods

## ESG Indicators Framework
**Comprehensive Indicator Set**: 46 indicators across three categories

### Environmental (18 indicators)
- GHG Emissions (Scope 1, 2, 3), Energy Efficiency, Renewable Energy Usage
- Water Usage, Waste Reduction, Biodiversity Impact, Carbon Footprint
- Environmental Compliance, Green Innovation, Resource Efficiency
- Climate Risk Assessment, Pollution Control, Sustainable Supply Chain
- Environmental Management Systems, Land Use, Circular Economy

### Social (16 indicators)
- Employee Diversity, Health & Safety, Training & Development
- Human Rights, Community Impact, Customer Satisfaction
- Labor Relations, Fair Wages, Work-Life Balance
- Product Safety, Social Innovation, Stakeholder Engagement
- Data Privacy, Supply Chain Labor, Accessibility, Local Employment

### Governance (12 indicators)
- Board Composition, Executive Compensation, Audit Quality
- Risk Management, Ethics & Compliance, Transparency
- Shareholder Rights, Cybersecurity, Anti-Corruption
- Regulatory Compliance, Stakeholder Governance, Business Ethics

**Sources**: GRI Standards, SASB Framework, TCFD Recommendations, EU Taxonomy, UN SDGs, Original Ontology

## Timeline (6 Weeks)
- **Week 1**: Literature review, ontology analysis, and ESG framework expansion ✅
- **Week 2**: Context-aware model development and dataset preparation 🔄
- **Week 3**: FinBERT fine-tuning and model training
- **Week 4**: Implementation and system integration
- **Week 5**: Evaluation, testing, and performance analysis
- **Week 6**: Thesis writing and documentation

## Key Deliverables
1. **Expanded ESG Framework**: 46 comprehensive indicators with academic sources
2. **Context-Aware Model**: Advanced extraction system recognizing both known and new ESG indicators
3. **Annotated Dataset**: Labeled training data from corporate reports
4. **Fine-tuned FinBERT**: Domain-adapted model for ESG-specific extraction
5. **Performance Evaluation**: Comprehensive analysis and baseline comparisons
6. **Complete Thesis**: 6000-word academic documentation

## Current Progress
### ✅ Completed
- Literature review and background research
- ESG ontology analysis (esgontology.owl)
- **ESG indicators expansion** (2 → 46 indicators with GRI, SASB, TCFD sources)
- **Context-aware model development** with semantic similarity and pattern recognition
- Project structure and documentation setup
- Git repository initialization

### 🔄 In Progress
- Dataset preparation from corporate reports
- Manual annotation for training data

### 📋 Upcoming
- FinBERT fine-tuning implementation
- Model training and optimization
- Evaluation and performance analysis
- Thesis writing and final documentation