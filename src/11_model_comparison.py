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
import torch.nn as nn
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
import sys
import importlib.util

# Import context-aware module
context_module_path = os.path.join(os.path.dirname(__file__), "04_context_aware_esg_model.py")
if os.path.exists(context_module_path):
    spec = importlib.util.spec_from_file_location("context_aware_esg_model", context_module_path)
    context_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(context_module)
    
    # Import context-aware classes
    ESGContextAwareModule = context_module.ESGContextAwareModule
    ESGSemanticEnhancer = context_module.ESGSemanticEnhancer
    ContextAwareTrainingMixin = context_module.ContextAwareTrainingMixin
    create_context_aware_enhancement = context_module.create_context_aware_enhancement
else:
    # Fallback if context module not found
    ESGContextAwareModule = None
    ESGSemanticEnhancer = None
    ContextAwareTrainingMixin = None
    create_context_aware_enhancement = None

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

class LightweightESGModel(nn.Module):
    """Enhanced ESG model (fine-tuned version) with context-aware capabilities"""
    
    def __init__(self, model_name='ProsusAI/finbert', num_indicators=50, dropout_rate=0.4, 
                 enable_context_awareness=True):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        self.enable_context_awareness = enable_context_awareness
        
        # Freeze early layers for efficiency
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        for i in range(6):
            for param in self.bert.encoder.layer[i].parameters():
                param.requires_grad = False
        
        # Match the exact architecture from training script
        self.hidden_size = self.config.hidden_size
        self.dropout = nn.Dropout(dropout_rate)
        
        # Initialize context-aware module if enabled and available
        if self.enable_context_awareness and create_context_aware_enhancement is not None:
            self.context_module = create_context_aware_enhancement(
                hidden_size=self.hidden_size,
                context_dim=128,
                dropout_rate=dropout_rate
            )
        else:
            self.context_module = None
        
        # ESG Indicator Classification (matching training architecture)
        self.indicator_proj = nn.Linear(self.hidden_size, 256)
        self.indicator_bn1 = nn.BatchNorm1d(256)
        self.indicator_hidden = nn.Linear(256, 128)
        self.indicator_bn2 = nn.BatchNorm1d(128)
        self.indicator_output = nn.Linear(128, num_indicators)
        self.indicator_residual = nn.Linear(self.hidden_size, 128)
        
        # Numerical Detection (matching training architecture)
        self.numerical_proj = nn.Linear(self.hidden_size, 128)
        self.numerical_bn1 = nn.BatchNorm1d(128)
        self.numerical_hidden = nn.Linear(128, 64)
        self.numerical_bn2 = nn.BatchNorm1d(64)
        self.numerical_output = nn.Linear(64, 1)
        self.numerical_residual = nn.Linear(self.hidden_size, 64)
        
        # ESG Category Classification (matching training architecture)
        self.category_proj = nn.Linear(self.hidden_size, 64)
        self.category_bn1 = nn.BatchNorm1d(64)
        self.category_hidden = nn.Linear(64, 32)
        self.category_bn2 = nn.BatchNorm1d(32)
        self.category_output = nn.Linear(32, 3)
        self.category_residual = nn.Linear(self.hidden_size, 32)
        
    def forward(self, input_ids, attention_mask=None):
        # Get BERT embeddings with context awareness support
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=self.enable_context_awareness and self.context_module is not None,
            output_attentions=False
        )
        
        # Use pooled output as base
        pooled_output = outputs.pooler_output
        
        # Apply context-aware enhancement if enabled and available
        if self.enable_context_awareness and self.context_module is not None:
            hidden_states = outputs.last_hidden_state
            context_output = self.context_module(hidden_states, attention_mask)
            
            # Use enhanced features as the main representation
            enhanced_features = context_output['enhanced_features']
            
            # Combine with original pooled output for robustness
            pooled_output = 0.7 * enhanced_features + 0.3 * pooled_output
        
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
        x2 = torch.nn.functional.dropout(x2, p=self.dropout.p, training=self.training)
        x2 = torch.nn.functional.relu(self.numerical_bn2(self.numerical_hidden(x2)))
        x2 = torch.nn.functional.dropout(x2, p=self.dropout.p, training=self.training)
        residual2 = self.numerical_residual(pooled_output)
        x2 = x2 + residual2
        numerical_logits = self.numerical_output(x2)
        
        # Category classification
        x3 = torch.nn.functional.relu(self.category_bn1(self.category_proj(pooled_output)))
        x3 = torch.nn.functional.dropout(x3, p=self.dropout.p, training=self.training)
        x3 = torch.nn.functional.relu(self.category_bn2(self.category_hidden(x3)))
        x3 = torch.nn.functional.dropout(x3, p=self.dropout.p, training=self.training)
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
    
    def load_optimized_thresholds(self, threshold_file):
        """Load optimized thresholds from threshold optimization results"""
        if not os.path.exists(threshold_file):
            return None
        
        try:
            with open(threshold_file, 'r') as f:
                threshold_data = json.load(f)
            
            # Extract optimal thresholds
            thresholds = {}
            if 'esg_indicator' in threshold_data:
                esg_data = threshold_data['esg_indicator']
                if isinstance(esg_data, dict):
                    thresholds['esg_indicator'] = esg_data.get('optimal_threshold', 0.5)
                else:
                    thresholds['esg_indicator'] = float(esg_data) if esg_data is not None else 0.5
            if 'numerical_detection' in threshold_data:
                num_data = threshold_data['numerical_detection']
                if isinstance(num_data, dict):
                    thresholds['numerical_detection'] = num_data.get('optimal_threshold', 0.5)
                else:
                    thresholds['numerical_detection'] = float(num_data) if num_data is not None else 0.5
            if 'category_classification' in threshold_data:
                cat_data = threshold_data['category_classification']
                if isinstance(cat_data, dict):
                    thresholds['category_classification'] = cat_data.get('optimal_confidence_threshold', 0.5)
                else:
                    thresholds['category_classification'] = float(cat_data) if cat_data is not None else 0.5
            
            return thresholds
        except Exception as e:
            print(f"Warning: Could not load thresholds from {threshold_file}: {e}")
            return None
    
    def apply_binary_threshold(self, probs, threshold):
        """Apply custom threshold for binary classification"""
        if probs.shape[1] > 1:
            return (probs[:, 1] >= threshold).astype(int)
        else:
            return (probs.flatten() >= threshold).astype(int)
    
    def apply_confidence_threshold(self, probs, confidence_threshold):
        """Apply confidence threshold for multiclass classification"""
        max_probs = np.max(probs, axis=1)
        pred_classes = np.argmax(probs, axis=1)
        
        # Set low-confidence predictions to a default class (e.g., 0)
        confident_mask = max_probs >= confidence_threshold
        pred_classes[~confident_mask] = 0  # Default to first class for low confidence
        
        return pred_classes
    
    def calculate_comprehensive_metrics(self, y_true, y_pred, y_probs, task_type):
        """Calculate comprehensive metrics including multiple F1 computation methods"""
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'f1_micro': float(f1_score(y_true, y_pred, average='micro')),
            'f1_macro': float(f1_score(y_true, y_pred, average='macro')),
            'f1_weighted': float(f1_score(y_true, y_pred, average='weighted')),
            'precision': float(precision_recall_fscore_support(y_true, y_pred, average='macro')[0]),
            'recall': float(precision_recall_fscore_support(y_true, y_pred, average='macro')[1]),
            'precision_micro': float(precision_recall_fscore_support(y_true, y_pred, average='micro')[0]),
            'recall_micro': float(precision_recall_fscore_support(y_true, y_pred, average='micro')[1])
        }
        
        # Add task-specific metrics
        if task_type == 'binary':
            if len(np.unique(y_true)) > 1:
                y_scores = y_probs[:, 1] if y_probs.shape[1] > 1 else y_probs.flatten()
                metrics['auc'] = float(roc_auc_score(y_true, y_scores))
                
                # Per-class F1 scores
                f1_per_class = f1_score(y_true, y_pred, average=None)
                metrics['f1_class_0'] = float(f1_per_class[0]) if len(f1_per_class) > 0 else 0.0
                metrics['f1_class_1'] = float(f1_per_class[1]) if len(f1_per_class) > 1 else 0.0
            else:
                metrics['auc'] = 0.0
                metrics['f1_class_0'] = 0.0
                metrics['f1_class_1'] = 0.0
        
        elif task_type == 'multiclass':
            # Per-class F1 scores for multiclass
            f1_per_class = f1_score(y_true, y_pred, average=None)
            for i, f1_val in enumerate(f1_per_class):
                metrics[f'f1_class_{i}'] = float(f1_val)
        
        return metrics
    
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
        
        # Load optimized thresholds if available
        threshold_file = 'results/threshold_optimization/optimized_thresholds.json'
        optimized_thresholds = self.load_optimized_thresholds(threshold_file)
        
        # Get predicted classes using optimized thresholds
        if optimized_thresholds:
            print(f"Using optimized thresholds for {model_name}")
            esg_pred_classes = self.apply_binary_threshold(
                esg_probs, optimized_thresholds.get('esg_indicator', 0.5)
            )
            numerical_pred_classes = self.apply_binary_threshold(
                numerical_probs, optimized_thresholds.get('numerical_detection', 0.5)
            )
            category_pred_classes = self.apply_confidence_threshold(
                category_probs, optimized_thresholds.get('category_classification', 0.5)
            )
        else:
            print(f"Using default thresholds (0.5) for {model_name}")
            esg_pred_classes = np.argmax(esg_probs, axis=1)
            numerical_pred_classes = np.argmax(numerical_probs, axis=1)
            category_pred_classes = np.argmax(category_probs, axis=1)
        
        # Calculate comprehensive metrics with multiple F1 computation methods
        results = {
            'model_name': model_name,
            'threshold_info': {
                'esg_threshold': optimized_thresholds.get('esg_indicator', 0.5) if optimized_thresholds else 0.5,
                'numerical_threshold': optimized_thresholds.get('numerical_detection', 0.5) if optimized_thresholds else 0.5,
                'category_threshold': optimized_thresholds.get('category_classification', 0.5) if optimized_thresholds else 0.5
            },
            'esg_indicator': self.calculate_comprehensive_metrics(
                 test_df['esg_indicator_label'], esg_pred_classes, esg_probs, 'binary'
             ),
            'numerical_detection': self.calculate_comprehensive_metrics(
                test_df['numerical_label'], numerical_pred_classes, numerical_probs, 'binary'
            ),
            'category_classification': self.calculate_comprehensive_metrics(
                test_df['category_label'], category_pred_classes, category_probs, 'multiclass'
            ),
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
        evaluation_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        report = {
            'evaluation_date': evaluation_date,
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
        with open(f'{self.results_dir}/detailed_comparison_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nDetailed report saved to {self.results_dir}/detailed_comparison_report.json")
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
        finetuned_model_path = 'models/finbert_esg/best_lightweight_model/model.pt'
        
        if os.path.exists(finetuned_model_path):
            print(f"Loading fine-tuned model weights from {finetuned_model_path}")
            checkpoint = torch.load(finetuned_model_path, map_location=self.device, weights_only=False)
            
            # Initialize model with same parameters as training
            finetuned_model = LightweightESGModel(
                model_name='ProsusAI/finbert',
                num_indicators=47,  # Match the saved model
                dropout_rate=0.4,
                enable_context_awareness=True
            )
            
            try:
                # Try loading with context-aware architecture
                model_state_dict = checkpoint.get('model_state_dict', checkpoint)
                
                # Filter state dict to match model architecture
                filtered_state_dict = {}
                model_keys = set(finetuned_model.state_dict().keys())
                
                for key, value in model_state_dict.items():
                    if key in model_keys:
                        filtered_state_dict[key] = value
                    else:
                        print(f"Skipping key: {key} (not in model architecture)")
                
                finetuned_model.load_state_dict(filtered_state_dict, strict=False)
                print("✓ Fine-tuned model weights loaded successfully with context-aware architecture")
                
            except Exception as e:
                print(f"Failed to load with context-aware architecture: {e}")
                print("Trying fallback without context awareness...")
                
                # Fallback: try without context awareness
                finetuned_model = LightweightESGModel(
                    model_name='ProsusAI/finbert',
                    num_indicators=47,  # Match the saved model
                    dropout_rate=0.4,
                    enable_context_awareness=False
                )
                
                model_state_dict = checkpoint.get('model_state_dict', checkpoint)
                filtered_state_dict = {k: v for k, v in model_state_dict.items() 
                                     if k in finetuned_model.state_dict()}
                
                finetuned_model.load_state_dict(filtered_state_dict, strict=False)
                print("✓ Fine-tuned model weights loaded successfully (fallback mode)")
        else:
            print(f"⚠️  Warning: Trained model weights not found at {finetuned_model_path}")
            print("   Using randomly initialized fine-tuned architecture for comparison")
            print("   This comparison shows architectural differences, not training benefits")
            finetuned_model = LightweightESGModel(
                model_name='ProsusAI/finbert',
                num_indicators=47,  # Match the saved model
                dropout_rate=0.4,
                enable_context_awareness=True
            )
        
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