"""Threshold Optimization for Multilabel ESG Classification

This script implements threshold tuning on validation set to optimize F1 performance
for multilabel ESG indicator classification and binary tasks.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import importlib.util
warnings.filterwarnings('ignore')

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoConfig, AutoModel

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

class ThresholdOptimizer:
    """Optimizes classification thresholds for multilabel and binary tasks"""
    
    def __init__(self, validation_data_path, model_path=None):
        self.validation_data_path = validation_data_path
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results_dir = 'results/threshold_optimization'
        os.makedirs(self.results_dir, exist_ok=True)
        
    def load_validation_data(self):
        """Load validation dataset for threshold optimization"""
        print(f"Loading validation data from {self.validation_data_path}")
        
        if not os.path.exists(self.validation_data_path):
            raise FileNotFoundError(f"Validation data not found: {self.validation_data_path}")
            
        df = pd.read_csv(self.validation_data_path)
        print(f"Loaded {len(df)} validation samples")
        
        # Verify required columns
        required_cols = ['text', 'esg_indicator_label', 'numerical_label', 'category_label']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        return df
    
    def get_model_predictions(self, model, val_df):
        """Get model predictions and probabilities on validation set"""
        print("Getting model predictions on validation set...")
        
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
        
        model.eval()
        model.to(self.device)
        
        # Tokenize validation data
        encoded = tokenizer(
            val_df['text'].tolist(),
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        # Get predictions in batches
        all_esg_probs = []
        all_numerical_probs = []
        all_category_probs = []
        
        batch_size = 8  # Adjust based on GPU memory
        
        with torch.no_grad():
            for i in range(0, len(input_ids), batch_size):
                batch_input_ids = input_ids[i:i+batch_size]
                batch_attention_mask = attention_mask[i:i+batch_size]
                
                outputs = model(batch_input_ids, batch_attention_mask)
                
                # Get probabilities
                esg_probs = torch.softmax(outputs['esg_indicator_logits'], dim=1)
                numerical_probs = torch.softmax(outputs['numerical_logits'], dim=1)
                category_probs = torch.softmax(outputs['category_logits'], dim=1)
                
                all_esg_probs.extend(esg_probs.cpu().numpy())
                all_numerical_probs.extend(numerical_probs.cpu().numpy())
                all_category_probs.extend(category_probs.cpu().numpy())
                
                # Clear memory
                del outputs, esg_probs, numerical_probs, category_probs
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return {
            'esg_indicator_probs': np.array(all_esg_probs),
            'numerical_probs': np.array(all_numerical_probs),
            'category_probs': np.array(all_category_probs)
        }
    
    def optimize_binary_threshold(self, y_true, y_probs, task_name):
        """Optimize threshold for binary classification task"""
        print(f"Optimizing threshold for {task_name}...")
        
        # Use positive class probabilities
        y_scores = y_probs[:, 1] if y_probs.shape[1] > 1 else y_probs.flatten()
        
        # Get precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        
        # Calculate F1 scores for each threshold
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        # Find optimal threshold
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
        optimal_f1 = f1_scores[optimal_idx]
        
        # Calculate metrics at optimal threshold
        y_pred_optimal = (y_scores >= optimal_threshold).astype(int)
        
        results = {
            'task_name': task_name,
            'optimal_threshold': float(optimal_threshold),
            'optimal_f1_score': float(optimal_f1),
            'precision_at_optimal': float(precision[optimal_idx]),
            'recall_at_optimal': float(recall[optimal_idx]),
            'average_precision': float(average_precision_score(y_true, y_scores)),
            'default_threshold_f1': float(f1_score(y_true, (y_scores >= 0.5).astype(int), average='binary')),
            'improvement': float(optimal_f1 - f1_score(y_true, (y_scores >= 0.5).astype(int), average='binary'))
        }
        
        # Create visualization
        self.plot_threshold_analysis(y_true, y_scores, precision, recall, thresholds, 
                                   optimal_threshold, task_name)
        
        return results
    
    def optimize_multiclass_threshold(self, y_true, y_probs, task_name):
        """Optimize threshold for multiclass classification"""
        print(f"Optimizing threshold for {task_name}...")
        
        # For multiclass, we optimize the confidence threshold
        # Predictions with max probability below threshold are rejected
        
        max_probs = np.max(y_probs, axis=1)
        y_pred_default = np.argmax(y_probs, axis=1)
        
        # Test different confidence thresholds
        thresholds = np.arange(0.1, 1.0, 0.05)
        f1_scores = []
        
        for threshold in thresholds:
            # Mask predictions below confidence threshold
            confident_mask = max_probs >= threshold
            
            if confident_mask.sum() == 0:
                f1_scores.append(0.0)
                continue
                
            # Calculate F1 only on confident predictions
            y_true_confident = y_true[confident_mask]
            y_pred_confident = y_pred_default[confident_mask]
            
            f1 = f1_score(y_true_confident, y_pred_confident, average='macro')
            f1_scores.append(f1)
        
        # Find optimal threshold
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        optimal_f1 = f1_scores[optimal_idx]
        
        # Calculate coverage at optimal threshold
        coverage = (max_probs >= optimal_threshold).mean()
        
        results = {
            'task_name': task_name,
            'optimal_confidence_threshold': float(optimal_threshold),
            'optimal_f1_score': float(optimal_f1),
            'coverage_at_optimal': float(coverage),
            'default_f1': float(f1_score(y_true, y_pred_default, average='macro')),
            'improvement': float(optimal_f1 - f1_score(y_true, y_pred_default, average='macro'))
        }
        
        # Create visualization
        self.plot_confidence_analysis(thresholds, f1_scores, optimal_threshold, task_name)
        
        return results
    
    def plot_threshold_analysis(self, y_true, y_scores, precision, recall, thresholds, 
                              optimal_threshold, task_name):
        """Plot threshold analysis for binary classification"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Precision-Recall curve
        ax1.plot(recall, precision, 'b-', linewidth=2)
        ax1.set_xlabel('Recall')
        ax1.set_ylabel('Precision')
        ax1.set_title(f'Precision-Recall Curve - {task_name}')
        ax1.grid(True, alpha=0.3)
        
        # F1 vs Threshold
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        ax2.plot(thresholds, f1_scores[:-1], 'g-', linewidth=2)
        ax2.axvline(x=optimal_threshold, color='red', linestyle='--', 
                   label=f'Optimal: {optimal_threshold:.3f}')
        ax2.set_xlabel('Threshold')
        ax2.set_ylabel('F1 Score')
        ax2.set_title(f'F1 Score vs Threshold - {task_name}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Score distribution
        ax3.hist(y_scores[y_true == 0], bins=50, alpha=0.7, label='Negative', density=True)
        ax3.hist(y_scores[y_true == 1], bins=50, alpha=0.7, label='Positive', density=True)
        ax3.axvline(x=optimal_threshold, color='red', linestyle='--', 
                   label=f'Optimal Threshold')
        ax3.set_xlabel('Prediction Score')
        ax3.set_ylabel('Density')
        ax3.set_title(f'Score Distribution - {task_name}')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Precision and Recall vs Threshold
        ax4.plot(thresholds, precision[:-1], 'b-', label='Precision', linewidth=2)
        ax4.plot(thresholds, recall[:-1], 'r-', label='Recall', linewidth=2)
        ax4.axvline(x=optimal_threshold, color='green', linestyle='--', 
                   label=f'Optimal: {optimal_threshold:.3f}')
        ax4.set_xlabel('Threshold')
        ax4.set_ylabel('Score')
        ax4.set_title(f'Precision & Recall vs Threshold - {task_name}')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/threshold_analysis_{task_name.lower().replace(" ", "_")}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_confidence_analysis(self, thresholds, f1_scores, optimal_threshold, task_name):
        """Plot confidence threshold analysis for multiclass"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # F1 vs Confidence Threshold
        ax1.plot(thresholds, f1_scores, 'b-', linewidth=2)
        ax1.axvline(x=optimal_threshold, color='red', linestyle='--', 
                   label=f'Optimal: {optimal_threshold:.3f}')
        ax1.set_xlabel('Confidence Threshold')
        ax1.set_ylabel('F1 Score (Macro)')
        ax1.set_title(f'F1 Score vs Confidence Threshold - {task_name}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Coverage vs Threshold (placeholder - would need actual coverage data)
        ax2.plot(thresholds, np.linspace(1.0, 0.1, len(thresholds)), 'g-', linewidth=2)
        ax2.axvline(x=optimal_threshold, color='red', linestyle='--', 
                   label=f'Optimal: {optimal_threshold:.3f}')
        ax2.set_xlabel('Confidence Threshold')
        ax2.set_ylabel('Coverage')
        ax2.set_title(f'Coverage vs Confidence Threshold - {task_name}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/confidence_analysis_{task_name.lower().replace(" ", "_")}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_threshold_optimization(self):
        """Run complete threshold optimization pipeline"""
        print("Starting Threshold Optimization")
        print("=" * 50)
        
        # Load validation data
        val_df = self.load_validation_data()
        
        # Load model if path provided
        if self.model_path and os.path.exists(self.model_path):
            print(f"Loading model from {self.model_path}")
            
            # Check if model file exists
            if not os.path.exists(self.model_path):
                print(f"Model file not found at {self.model_path}")
                print("Available files in model directory:")
                model_dir = os.path.dirname(self.model_path)
                if os.path.exists(model_dir):
                    for file in os.listdir(model_dir):
                        print(f"  - {file}")
                # Use dummy predictions
                n_samples = len(val_df)
                predictions = {
                    'esg_indicator_probs': np.random.rand(n_samples, 2),
                    'numerical_probs': np.random.rand(n_samples, 2),
                    'category_probs': np.random.rand(n_samples, 3)
                }
            else:
                # Initialize model with correct parameters
                model = LightweightESGModel(
                    model_name='ProsusAI/finbert',
                    num_indicators=50,
                    dropout_rate=0.4,
                    enable_context_awareness=True
                )
                
                try:
                    checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
                    
                    # Handle different checkpoint formats
                    if 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                    elif 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    else:
                        # Assume the checkpoint is the state dict itself
                        state_dict = checkpoint
                    
                    # Handle potential key mismatches by filtering compatible keys
                    model_dict = model.state_dict()
                    filtered_dict = {}
                    
                    for k, v in state_dict.items():
                        if k in model_dict:
                            if model_dict[k].shape == v.shape:
                                filtered_dict[k] = v
                            else:
                                print(f"Shape mismatch for {k}: model {model_dict[k].shape} vs checkpoint {v.shape}")
                        else:
                            print(f"Key {k} not found in model")
                    
                    # Update model dict and load
                    model_dict.update(filtered_dict)
                    model.load_state_dict(model_dict, strict=False)
                    print(f"Loaded {len(filtered_dict)}/{len(state_dict)} parameters")
                    
                except Exception as e:
                    print(f"Failed to load model: {e}")
                    print("Trying fallback without context awareness...")
                    
                    # Fallback: try without context awareness
                    model = LightweightESGModel(
                        model_name='ProsusAI/finbert',
                        num_indicators=47,  # Try original number
                        dropout_rate=0.3,   # Try original dropout
                        enable_context_awareness=False
                    )
                    
                    checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
                    
                    if 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                    elif 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    else:
                        state_dict = checkpoint
                    
                    # Handle potential key mismatches
                    model_dict = model.state_dict()
                    filtered_dict = {k: v for k, v in state_dict.items() 
                                   if k in model_dict and model_dict[k].shape == v.shape}
                    model_dict.update(filtered_dict)
                    model.load_state_dict(model_dict, strict=False)
                    print(f"Fallback loaded {len(filtered_dict)}/{len(state_dict)} parameters")
                
                # Get predictions
                predictions = self.get_model_predictions(model, val_df)
        else:
            print("No model path provided or model not found. Using dummy predictions for demonstration.")
            # Create dummy predictions for demonstration
            n_samples = len(val_df)
            predictions = {
                'esg_indicator_probs': np.random.rand(n_samples, 2),
                'numerical_probs': np.random.rand(n_samples, 2),
                'category_probs': np.random.rand(n_samples, 3)
            }
        
        # Optimize thresholds for each task
        results = {}
        
        # ESG Indicator Detection (Binary)
        results['esg_indicator'] = self.optimize_binary_threshold(
            val_df['esg_indicator_label'].values,
            predictions['esg_indicator_probs'],
            'ESG Indicator Detection'
        )
        
        # Numerical Detection (Binary)
        results['numerical_detection'] = self.optimize_binary_threshold(
            val_df['numerical_label'].values,
            predictions['numerical_probs'],
            'Numerical Detection'
        )
        
        # Category Classification (Multiclass)
        results['category_classification'] = self.optimize_multiclass_threshold(
            val_df['category_label'].values,
            predictions['category_probs'],
            'Category Classification'
        )
        
        # Save results with standardized name
        results_file = f'{self.results_dir}/threshold_optimization_results.json'
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save optimized thresholds in the format expected by model comparison
        optimized_thresholds = {
            "esg_indicator": results["esg_indicator"]["optimal_threshold"],
            "numerical_detection": results["numerical_detection"]["optimal_threshold"],
            "category_classification": results["category_classification"]["optimal_confidence_threshold"]
        }
        
        thresholds_file = f'{self.results_dir}/optimized_thresholds.json'
        with open(thresholds_file, 'w') as f:
            json.dump(optimized_thresholds, f, indent=2)
        
        # Print summary
        print("\n" + "=" * 50)
        print("THRESHOLD OPTIMIZATION RESULTS")
        print("=" * 50)
        
        for task, result in results.items():
            print(f"\n{result['task_name']}:")
            if 'optimal_threshold' in result:
                print(f"  Optimal Threshold: {result['optimal_threshold']:.4f}")
                print(f"  F1 Improvement: {result['improvement']:.4f}")
                print(f"  Optimal F1: {result['optimal_f1_score']:.4f}")
            else:
                print(f"  Optimal Confidence Threshold: {result['optimal_confidence_threshold']:.4f}")
                print(f"  F1 Improvement: {result['improvement']:.4f}")
                print(f"  Coverage: {result['coverage_at_optimal']:.4f}")
        
        print(f"\nResults saved to: {results_file}")
        print(f"Optimized thresholds saved to: {thresholds_file}")
        print(f"Visualizations saved to: {self.results_dir}/")
        
        return results

def main():
    """Main execution function"""
    # Configuration - use processed validation data
    validation_data_path = 'data/annotations/esg_validation_set_processed.csv'
    model_path = 'models/finbert_esg/best_lightweight_model/model.pt'
    
    # Check if processed validation data exists
    if not os.path.exists(validation_data_path):
        print(f"Processed validation data not found at {validation_data_path}")
        print("Running data preprocessor first...")
        
        # Run data preprocessor
        from data_preprocessor import ESGDataPreprocessor
        preprocessor = ESGDataPreprocessor()
        
        original_val_path = 'data/annotations/esg_validation_set.csv'
        if os.path.exists(original_val_path):
            preprocessor.process_dataset(original_val_path, validation_data_path)
            print(f"Created processed validation set")
        else:
            print(f"No validation data found at {original_val_path}")
            print("Please ensure validation dataset is available.")
            return
    
    # Run threshold optimization
    optimizer = ThresholdOptimizer(validation_data_path, model_path)
    results = optimizer.run_threshold_optimization()
    
    print("\nThreshold optimization completed successfully!")

if __name__ == "__main__":
    main()