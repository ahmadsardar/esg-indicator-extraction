# ESG Indicator Extraction using Specialized LLMs

## Thesis Overview
**Title**: The need of automatic extraction of ESG indicators in documents using specialised LLMs
**Type**: Bachelor Thesis (6000 words)
**Approach**: Fine-tuning existing language models for ESG-specific indicator extraction

## Project Structure
```
Project/
├── data/
│   ├── raw/                    # Original corporate reports
│   ├── processed/              # Cleaned and preprocessed data
│   ├── annotations/            # Manual ESG indicator labels
│   └── ontology/              # ESG ontology analysis
├── models/
│   ├── baseline/              # Baseline extraction methods
│   ├── finbert_esg/           # Fine-tuned FinBERT model
│   └── evaluation/            # Model evaluation scripts
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── ontology_analysis.ipynb
│   └── model_training.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── evaluation.py
│   └── extraction_pipeline.py
└── results/
    ├── performance_metrics.json
    ├── extracted_indicators.csv
    └── analysis_plots/
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

## ESG Indicators from Ontology
Based on esgontology.owl analysis:
- **Environmental**: EnergyEfficiency, WasteReduction, WaterUsage
- **Social**: Employee metrics, safety indicators
- **Governance**: Board composition, compliance metrics

## Timeline
- **Weeks 1-2**: Literature review and background research
- **Weeks 3-4**: Data preparation and ontology analysis
- **Weeks 5-7**: Model development and training
- **Weeks 8-10**: Implementation and evaluation
- **Weeks 11-12**: Thesis writing and documentation

## Key Deliverables
1. Annotated dataset of ESG indicators from corporate reports
2. Fine-tuned language model for ESG indicator extraction
3. Performance evaluation and comparison study
4. Complete thesis documentation (6000 words)