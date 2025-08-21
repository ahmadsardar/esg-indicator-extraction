"""
Model Comparison Evaluation Script
Compares base FinBERT model vs fine-tuned ESG FinBERT model performance

This script evaluates both models on the same test dataset to provide
comparative analysis for thesis research on ESG indicator extraction.
"""

import os
import json
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    precision_recall_fscore_support, roc_auc_score,
    accuracy_score, f1_score
)
from transformers import AutoTokenizer, AutoModel, AutoConfig
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class BaseFinBERTModel(torch.nn.Module):
    """Base FinBERT model for ESG evaluation"""
    
    def __init__(self, model_name='ProsusAI/finbert'):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Simple classification heads for comparison
        self.esg_indicator_head = torch.nn.Linear(self.config.hidden_size, 2)  # Binary
        self.numerical_head = torch.nn.Linear(self.config.hidden_size, 2)      # Binary
        self.category_head = torch.nn.Linear(self.config.hidden_size, 3)       # E, S, G
        
        self.dropout = torch.nn.Dropout(0.1)
        
    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        esg_logits = self.esg_indicator_head(pooled_output)
        numerical_logits = self.numerical_head(pooled_output)
        category_logits = self.category_head(pooled_output)
        
        return {
            'esg_indicator_logits': esg_logits,
            'numerical_logits': numerical_logits,
            'category_logits': category_logits
        }

class LightweightESGModel(torch.nn.Module):
    """Enhanced ESG model (fine-tuned version)"""
    
    def __init__(self, model_name='ProsusAI/finbert'):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Freeze early layers for efficiency
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        for i in range(6):
            for param in self.bert.encoder.layer[i].parameters():
                param.requires_grad = False
        
        # Match the exact architecture from training script
        self.hidden_size = self.config.hidden_size
        self.dropout = torch.nn.Dropout(0.3)
        
        # ESG Indicator Classification (matching training architecture)
        self.indicator_proj = torch.nn.Linear(self.hidden_size, 256)
        self.indicator_bn1 = torch.nn.BatchNorm1d(256)
        self.indicator_hidden = torch.nn.Linear(256, 128)
        self.indicator_bn2 = torch.nn.BatchNorm1d(128)
        self.indicator_output = torch.nn.Linear(128, 47)  # 47 indicators
        self.indicator_residual = torch.nn.Linear(self.hidden_size, 128)
        
        # Numerical Detection (matching training architecture)
        self.numerical_proj = torch.nn.Linear(self.hidden_size, 128)
        self.numerical_bn1 = torch.nn.BatchNorm1d(128)
        self.numerical_hidden = torch.nn.Linear(128, 64)
        self.numerical_bn2 = torch.nn.BatchNorm1d(64)
        self.numerical_output = torch.nn.Linear(64, 1)
        self.numerical_residual = torch.nn.Linear(self.hidden_size, 64)
        
        # ESG Category Classification (matching training architecture)
        self.category_proj = torch.nn.Linear(self.hidden_size, 64)
        self.category_bn1 = torch.nn.BatchNorm1d(64)
        self.category_hidden = torch.nn.Linear(64, 32)
        self.category_bn2 = torch.nn.BatchNorm1d(32)
        self.category_output = torch.nn.Linear(32, 3)
        self.category_residual = torch.nn.Linear(self.hidden_size, 32)
        
    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        # Task-specific predictions with residual connections (matching training architecture)
        # Indicator classification
        x1 = torch.nn.functional.relu(self.indicator_bn1(self.indicator_proj(pooled_output)))
        x1 = torch.nn.functional.dropout(x1, p=self.dropout.p, training=self.training)
        x1 = torch.nn.functional.relu(self.indicator_bn2(self.indicator_hidden(x1)))
        x1 = torch.nn.functional.dropout(x1, p=self.dropout.p, training=self.training)
        residual1 = self.indicator_residual(pooled_output)
        x1 = x1 + residual1
        indicator_logits = self.indicator_output(x1)
        
        # Numerical detection
        x2 = torch.nn.functional.relu(self.numerical_bn1(self.numerical_proj(pooled_output)))
        x2 = torch.nn.functional.dropout(x2, p=0.2, training=self.training)
        x2 = torch.nn.functional.relu(self.numerical_bn2(self.numerical_hidden(x2)))
        x2 = torch.nn.functional.dropout(x2, p=0.2, training=self.training)
        residual2 = self.numerical_residual(pooled_output)
        x2 = x2 + residual2
        numerical_logits = self.numerical_output(x2)
        
        # Category classification
        x3 = torch.nn.functional.relu(self.category_bn1(self.category_proj(pooled_output)))
        x3 = torch.nn.functional.dropout(x3, p=0.2, training=self.training)
        x3 = torch.nn.functional.relu(self.category_bn2(self.category_hidden(x3)))
        x3 = torch.nn.functional.dropout(x3, p=0.2, training=self.training)
        residual3 = self.category_residual(pooled_output)
        x3 = x3 + residual3
        category_logits = self.category_output(x3)
        
        # Convert outputs to match expected format for comparison
        # Convert multi-label indicator to binary ESG relevance
        esg_indicator_logits = torch.cat([torch.zeros_like(indicator_logits[:, :1]), torch.max(indicator_logits, dim=1, keepdim=True)[0]], dim=1)
        # Convert single output numerical to binary
        numerical_binary_logits = torch.cat([torch.zeros_like(numerical_logits), numerical_logits], dim=1)
        
        return {
            'esg_indicator_logits': esg_indicator_logits,
            'numerical_logits': numerical_binary_logits,
            'category_logits': category_logits
        }

class ModelComparator:
    """Handles model comparison and evaluation"""
    
    def __init__(self, test_data_path, results_dir='results/comparison'):
        self.test_data_path = test_data_path
        self.results_dir = results_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create results directory
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
        
        print(f"Using device: {self.device}")
        
    def load_test_data(self):
        """Load and prepare test dataset"""
        print("Loading test dataset...")
        
        # Load the test data
        df = pd.read_csv(self.test_data_path)
        
        # Prepare labels using correct column names
        df['esg_indicator_label'] = df['is_esg_relevant'].astype(int)
        df['numerical_label'] = df['has_numerical_data'].astype(int)
        
        # Map categories to integers using correct column name
        category_map = {'Environmental': 0, 'Social': 1, 'Governance': 2}
        df['category_label'] = df['primary_category'].map(category_map)
        
        # Filter out rows with missing category labels
        df = df.dropna(subset=['category_label'])
        
        # Limit dataset size for memory efficiency
        original_size = len(df)
        if len(df) > 100:
            df = df.sample(n=100, random_state=42)
            print(f"Sampled 100 test samples from {original_size} total samples for memory efficiency")
        
        print(f"Loaded {len(df)} test samples")
        print(f"ESG Indicator distribution: {df['esg_indicator_label'].value_counts().to_dict()}")
        print(f"Numerical data distribution: {df['numerical_label'].value_counts().to_dict()}")
        print(f"Category distribution: {df['category_label'].value_counts().to_dict()}")
        return df
    
    def tokenize_data(self, texts):
        """Tokenize text data"""
        return self.tokenizer(
            texts.tolist(),
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
    
    def evaluate_model(self, model, test_df, model_name):
        """Evaluate a single model on test data"""
        print(f"\nEvaluating {model_name}...")
        
        model.eval()
        model.to(self.device)
        
        # Tokenize test data
        encoded = self.tokenize_data(test_df['text'])
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        # Get predictions
        all_esg_preds = []
        all_numerical_preds = []
        all_category_preds = []
        
        batch_size = 1  # Minimal batch size for memory efficiency
        with torch.no_grad():
            for i in range(0, len(input_ids), batch_size):
                batch_input_ids = input_ids[i:i+batch_size]
                batch_attention_mask = attention_mask[i:i+batch_size]
                
                outputs = model(batch_input_ids, batch_attention_mask)
                
                esg_preds = torch.softmax(outputs['esg_indicator_logits'], dim=1)
                numerical_preds = torch.softmax(outputs['numerical_logits'], dim=1)
                category_preds = torch.softmax(outputs['category_logits'], dim=1)
                
                all_esg_preds.extend(esg_preds.cpu().numpy())
                all_numerical_preds.extend(numerical_preds.cpu().numpy())
                all_category_preds.extend(category_preds.cpu().numpy())
                
                # Clear GPU memory
                del outputs, esg_preds, numerical_preds, category_preds, batch_input_ids, batch_attention_mask
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                import gc
                gc.collect()
        
        # Convert to numpy arrays
        esg_probs = np.array(all_esg_preds)
        numerical_probs = np.array(all_numerical_preds)
        category_probs = np.array(all_category_preds)
        
        # Get predicted classes
        esg_pred_classes = np.argmax(esg_probs, axis=1)
        numerical_pred_classes = np.argmax(numerical_probs, axis=1)
        category_pred_classes = np.argmax(category_probs, axis=1)
        
        # Calculate metrics
        results = {
            'model_name': model_name,
            'esg_indicator': {
                'accuracy': accuracy_score(test_df['esg_indicator_label'], esg_pred_classes),
                'f1_micro': f1_score(test_df['esg_indicator_label'], esg_pred_classes, average='micro'),
                'f1_macro': f1_score(test_df['esg_indicator_label'], esg_pred_classes, average='macro'),
                'precision': precision_recall_fscore_support(test_df['esg_indicator_label'], esg_pred_classes, average='macro')[0],
                'recall': precision_recall_fscore_support(test_df['esg_indicator_label'], esg_pred_classes, average='macro')[1],
                'auc': roc_auc_score(test_df['esg_indicator_label'], esg_probs[:, 1]) if len(np.unique(test_df['esg_indicator_label'])) > 1 else 0.0
            },
            'numerical_detection': {
                'accuracy': accuracy_score(test_df['numerical_label'], numerical_pred_classes),
                'f1_micro': f1_score(test_df['numerical_label'], numerical_pred_classes, average='micro'),
                'f1_macro': f1_score(test_df['numerical_label'], numerical_pred_classes, average='macro'),
                'precision': precision_recall_fscore_support(test_df['numerical_label'], numerical_pred_classes, average='macro')[0],
                'recall': precision_recall_fscore_support(test_df['numerical_label'], numerical_pred_classes, average='macro')[1],
                'auc': roc_auc_score(test_df['numerical_label'], numerical_probs[:, 1]) if len(np.unique(test_df['numerical_label'])) > 1 else 0.0
            },
            'category_classification': {
                'accuracy': accuracy_score(test_df['category_label'], category_pred_classes),
                'f1_micro': f1_score(test_df['category_label'], category_pred_classes, average='micro'),
                'f1_macro': f1_score(test_df['category_label'], category_pred_classes, average='macro'),
                'precision': precision_recall_fscore_support(test_df['category_label'], category_pred_classes, average='macro')[0],
                'recall': precision_recall_fscore_support(test_df['category_label'], category_pred_classes, average='macro')[1]
            },
            'predictions': {
                'esg_indicator': esg_pred_classes.tolist(),
                'numerical_detection': numerical_pred_classes.tolist(),
                'category_classification': category_pred_classes.tolist()
            },
            'probabilities': {
                'esg_indicator': esg_probs.tolist(),
                'numerical_detection': numerical_probs.tolist(),
                'category_classification': category_probs.tolist()
            }
        }
        
        print(f"{model_name} Results:")
        print(f"  ESG Indicator F1: {results['esg_indicator']['f1_macro']:.4f}")
        print(f"  Numerical Detection F1: {results['numerical_detection']['f1_macro']:.4f}")
        print(f"  Category Classification F1: {results['category_classification']['f1_macro']:.4f}")
        
        return results
    
    def create_comparison_visualizations(self, base_results, finetuned_results, test_df):
        """Create comprehensive comparison visualizations"""
        print("\nCreating comparison visualizations...")
        
        # Set up the plotting style
        plt.rcParams['figure.figsize'] = (15, 10)
        plt.rcParams['font.size'] = 12
        
        # 1. Overall Performance Comparison
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('FinBERT vs Fine-tuned ESG Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # F1 Score Comparison
        tasks = ['ESG Indicator', 'Numerical Detection', 'Category Classification']
        base_f1 = [
            base_results['esg_indicator']['f1_macro'],
            base_results['numerical_detection']['f1_macro'],
            base_results['category_classification']['f1_macro']
        ]
        finetuned_f1 = [
            finetuned_results['esg_indicator']['f1_macro'],
            finetuned_results['numerical_detection']['f1_macro'],
            finetuned_results['category_classification']['f1_macro']
        ]
        
        x = np.arange(len(tasks))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, base_f1, width, label='Base FinBERT', alpha=0.8, color='skyblue')
        axes[0, 0].bar(x + width/2, finetuned_f1, width, label='Fine-tuned ESG', alpha=0.8, color='lightcoral')
        axes[0, 0].set_xlabel('Tasks')
        axes[0, 0].set_ylabel('F1 Score (Macro)')
        axes[0, 0].set_title('F1 Score Comparison')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(tasks, rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy Comparison
        base_acc = [
            base_results['esg_indicator']['accuracy'],
            base_results['numerical_detection']['accuracy'],
            base_results['category_classification']['accuracy']
        ]
        finetuned_acc = [
            finetuned_results['esg_indicator']['accuracy'],
            finetuned_results['numerical_detection']['accuracy'],
            finetuned_results['category_classification']['accuracy']
        ]
        
        axes[0, 1].bar(x - width/2, base_acc, width, label='Base FinBERT', alpha=0.8, color='skyblue')
        axes[0, 1].bar(x + width/2, finetuned_acc, width, label='Fine-tuned ESG', alpha=0.8, color='lightcoral')
        axes[0, 1].set_xlabel('Tasks')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Accuracy Comparison')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(tasks, rotation=45, ha='right')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Precision-Recall Comparison
        base_precision = [
            base_results['esg_indicator']['precision'],
            base_results['numerical_detection']['precision'],
            base_results['category_classification']['precision']
        ]
        finetuned_precision = [
            finetuned_results['esg_indicator']['precision'],
            finetuned_results['numerical_detection']['precision'],
            finetuned_results['category_classification']['precision']
        ]
        
        axes[1, 0].bar(x - width/2, base_precision, width, label='Base FinBERT', alpha=0.8, color='skyblue')
        axes[1, 0].bar(x + width/2, finetuned_precision, width, label='Fine-tuned ESG', alpha=0.8, color='lightcoral')
        axes[1, 0].set_xlabel('Tasks')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].set_title('Precision Comparison')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(tasks, rotation=45, ha='right')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # AUC Comparison (for binary tasks only)
        binary_tasks = ['ESG Indicator', 'Numerical Detection']
        base_auc = [
            base_results['esg_indicator']['auc'],
            base_results['numerical_detection']['auc']
        ]
        finetuned_auc = [
            finetuned_results['esg_indicator']['auc'],
            finetuned_results['numerical_detection']['auc']
        ]
        
        x_binary = np.arange(len(binary_tasks))
        axes[1, 1].bar(x_binary - width/2, base_auc, width, label='Base FinBERT', alpha=0.8, color='skyblue')
        axes[1, 1].bar(x_binary + width/2, finetuned_auc, width, label='Fine-tuned ESG', alpha=0.8, color='lightcoral')
        axes[1, 1].set_xlabel('Tasks')
        axes[1, 1].set_ylabel('AUC Score')
        axes[1, 1].set_title('AUC Score Comparison')
        axes[1, 1].set_xticks(x_binary)
        axes[1, 1].set_xticklabels(binary_tasks)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Improvement Analysis
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        improvements = {
            'ESG Indicator F1': finetuned_f1[0] - base_f1[0],
            'Numerical Detection F1': finetuned_f1[1] - base_f1[1],
            'Category Classification F1': finetuned_f1[2] - base_f1[2],
            'ESG Indicator Accuracy': finetuned_acc[0] - base_acc[0],
            'Numerical Detection Accuracy': finetuned_acc[1] - base_acc[1],
            'Category Classification Accuracy': finetuned_acc[2] - base_acc[2]
        }
        
        metrics = list(improvements.keys())
        values = list(improvements.values())
        colors = ['green' if v > 0 else 'red' for v in values]
        
        bars = ax.barh(metrics, values, color=colors, alpha=0.7)
        ax.set_xlabel('Improvement (Fine-tuned - Base)')
        ax.set_title('Performance Improvement Analysis\n(Positive = Better, Negative = Worse)', fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            ax.text(value + (0.001 if value >= 0 else -0.001), bar.get_y() + bar.get_height()/2, 
                   f'{value:.4f}', ha='left' if value >= 0 else 'right', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/improvement_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved to {self.results_dir}/")
    
    def generate_detailed_report(self, base_results, finetuned_results, test_df):
        """Generate detailed comparison report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        report = {
            'evaluation_timestamp': timestamp,
            'test_dataset_size': len(test_df),
            'base_model_results': base_results,
            'finetuned_model_results': finetuned_results,
            'comparison_summary': {
                'esg_indicator_improvement': {
                    'f1_improvement': finetuned_results['esg_indicator']['f1_macro'] - base_results['esg_indicator']['f1_macro'],
                    'accuracy_improvement': finetuned_results['esg_indicator']['accuracy'] - base_results['esg_indicator']['accuracy'],
                    'auc_improvement': finetuned_results['esg_indicator']['auc'] - base_results['esg_indicator']['auc']
                },
                'numerical_detection_improvement': {
                    'f1_improvement': finetuned_results['numerical_detection']['f1_macro'] - base_results['numerical_detection']['f1_macro'],
                    'accuracy_improvement': finetuned_results['numerical_detection']['accuracy'] - base_results['numerical_detection']['accuracy'],
                    'auc_improvement': finetuned_results['numerical_detection']['auc'] - base_results['numerical_detection']['auc']
                },
                'category_classification_improvement': {
                    'f1_improvement': finetuned_results['category_classification']['f1_macro'] - base_results['category_classification']['f1_macro'],
                    'accuracy_improvement': finetuned_results['category_classification']['accuracy'] - base_results['category_classification']['accuracy']
                }
            }
        }
        
        # Save detailed report
        with open(f'{self.results_dir}/detailed_comparison_report_{timestamp}.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nDetailed report saved to {self.results_dir}/detailed_comparison_report_{timestamp}.json")
        return report
    
    def run_comparison(self):
        """Run complete model comparison"""
        print("Starting Model Comparison Evaluation")
        print("=" * 50)
        
        # Load test data
        test_df = self.load_test_data()
        
        # Initialize models
        print("\nInitializing models...")
        base_model = BaseFinBERTModel()
        
        # Load fine-tuned model
        finetuned_model = LightweightESGModel()
        finetuned_model_path = 'models/finbert_esg/best_lightweight_model/model.pt'
        
        if os.path.exists(finetuned_model_path):
            print(f"Loading fine-tuned model weights from {finetuned_model_path}")
            checkpoint = torch.load(finetuned_model_path, map_location=self.device, weights_only=False)
            finetuned_model.load_state_dict(checkpoint['model_state_dict'])
            print("✓ Fine-tuned model weights loaded successfully")
        else:
            print(f"⚠️  Warning: Trained model weights not found at {finetuned_model_path}")
            print("   Using randomly initialized fine-tuned architecture for comparison")
            print("   This comparison shows architectural differences, not training benefits")
        
        # Evaluate both models
        base_results = self.evaluate_model(base_model, test_df, "Base FinBERT")
        finetuned_results = self.evaluate_model(finetuned_model, test_df, "Fine-tuned ESG FinBERT")
        
        # Create visualizations
        self.create_comparison_visualizations(base_results, finetuned_results, test_df)
        
        # Generate detailed report
        report = self.generate_detailed_report(base_results, finetuned_results, test_df)
        
        print("\n" + "=" * 50)
        print("COMPARISON SUMMARY")
        print("=" * 50)
        
        print(f"\nESG Indicator Detection:")
        print(f"  Base F1: {base_results['esg_indicator']['f1_macro']:.4f}")
        print(f"  Fine-tuned F1: {finetuned_results['esg_indicator']['f1_macro']:.4f}")
        print(f"  Improvement: {report['comparison_summary']['esg_indicator_improvement']['f1_improvement']:.4f}")
        
        print(f"\nNumerical Data Detection:")
        print(f"  Base F1: {base_results['numerical_detection']['f1_macro']:.4f}")
        print(f"  Fine-tuned F1: {finetuned_results['numerical_detection']['f1_macro']:.4f}")
        print(f"  Improvement: {report['comparison_summary']['numerical_detection_improvement']['f1_improvement']:.4f}")
        
        print(f"\nESG Category Classification:")
        print(f"  Base F1: {base_results['category_classification']['f1_macro']:.4f}")
        print(f"  Fine-tuned F1: {finetuned_results['category_classification']['f1_macro']:.4f}")
        print(f"  Improvement: {report['comparison_summary']['category_classification_improvement']['f1_improvement']:.4f}")
        
        return report

def main():
    """
    Main execution function
    """
    # Configuration - use the available test set
    test_data_path = 'data/annotations/esg_test_set.csv'
    
    # Check if test data exists
    if not os.path.exists(test_data_path):
        print(f"Test data not found at {test_data_path}")
        print("Please ensure the test dataset from training is available.")
        return
    
    # Run comparison
    comparator = ModelComparator(test_data_path)
    report = comparator.run_comparison()
    
    print("\nModel comparison evaluation completed successfully!")
    print(f"Results saved to: {comparator.results_dir}")

if __name__ == "__main__":
    main()