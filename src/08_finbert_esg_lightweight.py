#!/usr/bin/env python3
"""
Lightweight FinBERT ESG Model for Memory-Constrained Training

This script implements a memory-efficient FinBERT model for ESG analysis:
1. Simplified multi-task architecture
2. Memory-optimized training pipeline
3. Gradient accumulation for effective large batch training
4. Efficient data loading and preprocessing

Author: ESG Analysis System
Date: 2025-01-20
"""

import os
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    AdamW, get_linear_schedule_with_warmup
)
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score, 
    precision_score, recall_score, roc_auc_score
)
from sklearn.utils.class_weight import compute_class_weight
import gc
import warnings
import random
from collections import Counter
warnings.filterwarnings('ignore')

class LightweightESGModel(nn.Module):
    """
    Lightweight ESG model with minimal memory footprint
    """
    
    def __init__(self, model_name='ProsusAI/finbert', num_indicators=50, dropout_rate=0.3):
        super(LightweightESGModel, self).__init__()
        
        # Load FinBERT with minimal configuration
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Freeze early layers to reduce memory
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        
        # Freeze first 6 encoder layers (keep last 6 trainable)
        for i in range(6):
            for param in self.bert.encoder.layer[i].parameters():
                param.requires_grad = False
        
        self.hidden_size = self.config.hidden_size
        self.num_indicators = num_indicators
        self.dropout = nn.Dropout(dropout_rate)
        
        # Enhanced task heads with residual connections
        # 1. ESG Indicator Classification (primary task)
        self.indicator_proj = nn.Linear(self.hidden_size, 256)
        self.indicator_bn1 = nn.BatchNorm1d(256)
        self.indicator_hidden = nn.Linear(256, 128)
        self.indicator_bn2 = nn.BatchNorm1d(128)
        self.indicator_output = nn.Linear(128, num_indicators)
        self.indicator_residual = nn.Linear(self.hidden_size, 128)  # Residual connection
        
        # 2. Numerical Detection (secondary task)
        self.numerical_proj = nn.Linear(self.hidden_size, 128)
        self.numerical_bn1 = nn.BatchNorm1d(128)
        self.numerical_hidden = nn.Linear(128, 64)
        self.numerical_bn2 = nn.BatchNorm1d(64)
        self.numerical_output = nn.Linear(64, 1)
        self.numerical_residual = nn.Linear(self.hidden_size, 64)  # Residual connection
        
        # 3. ESG Category Classification (tertiary task)
        self.category_proj = nn.Linear(self.hidden_size, 64)
        self.category_bn1 = nn.BatchNorm1d(64)
        self.category_hidden = nn.Linear(64, 32)
        self.category_bn2 = nn.BatchNorm1d(32)
        self.category_output = nn.Linear(32, 3)  # E, S, G
        self.category_residual = nn.Linear(self.hidden_size, 32)  # Residual connection
    
    def forward(self, input_ids, attention_mask):
        """
        Forward pass with memory optimization
        """
        # Get BERT embeddings with autocast for mixed precision
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            output_attentions=False
        )
        
        # Use pooled output
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        # Task-specific predictions with residual connections
        # Indicator classification
        x1 = F.relu(self.indicator_bn1(self.indicator_proj(pooled_output)))
        x1 = F.dropout(x1, p=self.dropout.p, training=self.training)
        x1 = F.relu(self.indicator_bn2(self.indicator_hidden(x1)))
        x1 = F.dropout(x1, p=self.dropout.p, training=self.training)
        residual1 = self.indicator_residual(pooled_output)
        x1 = x1 + residual1  # Residual connection
        indicator_logits = self.indicator_output(x1)
        
        # Numerical detection
        x2 = F.relu(self.numerical_bn1(self.numerical_proj(pooled_output)))
        x2 = F.dropout(x2, p=self.dropout.p, training=self.training)
        x2 = F.relu(self.numerical_bn2(self.numerical_hidden(x2)))
        x2 = F.dropout(x2, p=self.dropout.p, training=self.training)
        residual2 = self.numerical_residual(pooled_output)
        x2 = x2 + residual2  # Residual connection
        numerical_logits = self.numerical_output(x2)
        
        # Category classification
        x3 = F.relu(self.category_bn1(self.category_proj(pooled_output)))
        x3 = F.dropout(x3, p=self.dropout.p, training=self.training)
        x3 = F.relu(self.category_bn2(self.category_hidden(x3)))
        x3 = F.dropout(x3, p=self.dropout.p, training=self.training)
        residual3 = self.category_residual(pooled_output)
        x3 = x3 + residual3  # Residual connection
        category_logits = self.category_output(x3)
        
        return {
            'indicator_logits': indicator_logits,
            'numerical_logits': numerical_logits,
            'category_logits': category_logits
        }

class MemoryEfficientDataset(Dataset):
    """
    Memory-efficient dataset with lazy loading
    """
    
    def __init__(self, texts, indicator_labels, numerical_labels, 
                 category_labels, tokenizer, max_length=128):
        self.texts = texts
        self.indicator_labels = indicator_labels
        self.numerical_labels = numerical_labels
        self.category_labels = category_labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])[:600]  # Increased text length
        
        # Enhanced text preprocessing with data augmentation
        text = text.strip().replace('\n', ' ').replace('\t', ' ')
        text = ' '.join(text.split())
        
        # Simple data augmentation during training (random synonym replacement could be added)
        if hasattr(self, 'augment') and self.augment and random.random() < 0.1:
            # Simple augmentation: randomly duplicate some words for emphasis
            words = text.split()
            if len(words) > 10:
                # Randomly select a word to emphasize (duplicate)
                idx_to_dup = random.randint(0, min(len(words)-1, 20))
                words.insert(idx_to_dup + 1, words[idx_to_dup])
                text = ' '.join(words)
        
        # Tokenize with improved settings
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
            add_special_tokens=True
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'indicator_labels': torch.FloatTensor(self.indicator_labels[idx]),
            'numerical_labels': torch.FloatTensor([self.numerical_labels[idx]]),
            'category_labels': torch.LongTensor([self.category_labels[idx]])
        }

class LightweightTrainer:
    """
    Memory-efficient trainer for ESG model
    """
    
    def __init__(self, model_name='ProsusAI/finbert', max_length=128, 
                 learning_rate=2e-5, batch_size=8, num_epochs=3,
                 gradient_accumulation_steps=2, early_stopping_patience=3):
        
        self.model_name = model_name
        self.max_length = max_length
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.early_stopping_patience = early_stopping_patience
        self.best_val_f1 = 0.0
        self.patience_counter = 0
        
        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_amp = torch.cuda.is_available()  # Enable mixed precision for GPU
        self.scaler = GradScaler() if self.use_amp else None
        print(f"Using device: {self.device}")
        if self.use_amp:
            print(f"Mixed precision training enabled")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Label encoders
        self.indicator_encoder = MultiLabelBinarizer()
        self.category_encoder = LabelEncoder()
        
        # Model components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        
    def load_data(self, train_path, val_path, test_path):
        """
        Load and preprocess data efficiently with error handling
        """
        print("\n=== LOADING DATA ===")
        
        try:
            # Load datasets with error handling
            if not os.path.exists(train_path):
                raise FileNotFoundError(f"Training data not found: {train_path}")
            if not os.path.exists(val_path):
                raise FileNotFoundError(f"Validation data not found: {val_path}")
            if not os.path.exists(test_path):
                raise FileNotFoundError(f"Test data not found: {test_path}")
                
            self.train_df = pd.read_csv(train_path)
            self.val_df = pd.read_csv(val_path)
            self.test_df = pd.read_csv(test_path)
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
        
        print(f"Train: {len(self.train_df)}, Val: {len(self.val_df)}, Test: {len(self.test_df)}")
        
        # Use more training data for better performance
        if len(self.train_df) > 8000:
            self.train_df = self.train_df.sample(n=8000, random_state=42)
            print(f"Sampled training data to {len(self.train_df)} samples for improved training")
        
        if len(self.val_df) > 1500:
            self.val_df = self.val_df.sample(n=1500, random_state=42)
            print(f"Sampled validation data to {len(self.val_df)} samples")
        
        # Prepare labels
        self._prepare_labels()
        
        # Create datasets
        self._create_datasets()
        
        print("Data loading completed!")
    
    def _prepare_labels(self):
        """
        Prepare labels efficiently
        """
        print("Preparing labels...")
        
        # Combine data for consistent encoding
        all_df = pd.concat([self.train_df, self.val_df, self.test_df], ignore_index=True)
        
        # Get unique indicators
        indicators = all_df['best_match_indicator'].dropna().unique().tolist()
        # Use top 60 indicators as fallback (compromise between coverage and performance)
        self.num_indicators = min(len(indicators), 60)  # Use top 60 indicators
        top_indicators = all_df['best_match_indicator'].value_counts().head(self.num_indicators).index.tolist()
        
        print(f"Using top {self.num_indicators} indicators for training")
        
        # Create indicator lists
        def create_indicator_list(df):
            indicator_lists = []
            for _, row in df.iterrows():
                indicators = []
                if pd.notna(row['best_match_indicator']) and row['best_match_indicator'] in top_indicators:
                    indicators.append(row['best_match_indicator'])
                indicator_lists.append(indicators)
            return indicator_lists
        
        # Fit encoders
        all_indicator_lists = create_indicator_list(all_df)
        self.indicator_encoder.fit(all_indicator_lists)
        
        categories = ['Environmental', 'Social', 'Governance']
        self.category_encoder.fit(categories)
        
        # Encode each split
        for df, split_name in [(self.train_df, 'train'), (self.val_df, 'val'), (self.test_df, 'test')]:
            indicator_lists = create_indicator_list(df)
            indicator_labels = self.indicator_encoder.transform(indicator_lists)
            
            numerical_labels = df['has_numerical_data'].fillna(0).astype(int).values
            
            # Handle category labels more carefully
            category_values = df['primary_category'].fillna('Environmental').values
            # Map any unknown categories to Environmental
            category_values = [cat if cat in categories else 'Environmental' for cat in category_values]
            category_labels = self.category_encoder.transform(category_values)
            
            setattr(self, f'{split_name}_indicator_labels', indicator_labels)
            setattr(self, f'{split_name}_numerical_labels', numerical_labels)
            setattr(self, f'{split_name}_category_labels', category_labels)
        
        print("Label preparation completed!")
    
    def _create_datasets(self):
        """
        Create PyTorch datasets
        """
        print("Creating datasets...")
        
        self.train_dataset = MemoryEfficientDataset(
            texts=self.train_df['text'].values,
            indicator_labels=self.train_indicator_labels,
            numerical_labels=self.train_numerical_labels,
            category_labels=self.train_category_labels,
            tokenizer=self.tokenizer,
            max_length=self.max_length
        )
        self.train_dataset.augment = True  # Enable augmentation for training
        
        self.val_dataset = MemoryEfficientDataset(
            texts=self.val_df['text'].values,
            indicator_labels=self.val_indicator_labels,
            numerical_labels=self.val_numerical_labels,
            category_labels=self.val_category_labels,
            tokenizer=self.tokenizer,
            max_length=self.max_length
        )
        
        print("Dataset creation completed!")
    
    def initialize_model(self):
        """
        Initialize the lightweight model with error handling
        """
        print("\n=== INITIALIZING MODEL ===")
        
        try:
            # Clear GPU cache before model initialization
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            self.model = LightweightESGModel(
                model_name=self.model_name,
                num_indicators=self.num_indicators
            ).to(self.device)
            
            # Count trainable parameters
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            
            print(f"Total parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
            
            # Check GPU memory usage
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
                memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
                print(f"GPU Memory - Allocated: {memory_allocated:.2f} GB, Reserved: {memory_reserved:.2f} GB")
            
            # Initialize optimizer with improved settings
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=0.02,  # Increased weight decay
                eps=1e-8,
                betas=(0.9, 0.999)
            )
            
        except Exception as e:
            print(f"Error initializing model: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise
        
        # Calculate training steps
        total_steps = len(self.train_dataset) // (self.batch_size * self.gradient_accumulation_steps) * self.num_epochs
        
        # Initialize scheduler with longer warmup
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(0.15 * total_steps),  # Increased warmup
            num_training_steps=total_steps
        )
        
        print(f"Model initialized successfully!")
        print(f"Total training steps: {total_steps}")
    
    def compute_loss(self, outputs, batch):
        """
        Compute multi-task loss
        """
        # Loss functions
        bce_loss = nn.BCEWithLogitsLoss()
        ce_loss = nn.CrossEntropyLoss()
        
        # Individual losses
        indicator_loss = bce_loss(
            outputs['indicator_logits'],
            batch['indicator_labels']
        )
        
        numerical_loss = bce_loss(
            outputs['numerical_logits'],
            batch['numerical_labels']
        )
        
        category_loss = ce_loss(
            outputs['category_logits'],
            batch['category_labels'].view(-1)
        )
        
        # Dynamic weighted total loss with focal loss for hard examples
        # Apply focal loss to indicator classification for hard examples
        indicator_probs = torch.sigmoid(outputs['indicator_logits'])
        focal_weight = (1 - indicator_probs) ** 2  # Focus on hard examples
        weighted_indicator_loss = (focal_weight * F.binary_cross_entropy_with_logits(
            outputs['indicator_logits'], 
            batch['indicator_labels'], 
            reduction='none'
        )).mean()
        
        # Optimized weighted total loss
        total_loss = (
            2.0 * weighted_indicator_loss +  # Primary task with focal loss
            1.5 * numerical_loss +           # Secondary task
            1.0 * category_loss              # Tertiary task
        )
        
        return {
            'total_loss': total_loss,
            'indicator_loss': indicator_loss,
            'numerical_loss': numerical_loss,
            'category_loss': category_loss
        }
    
    def train_epoch(self, dataloader):
        """
        Train for one epoch with mixed precision and memory management
        """
        self.model.train()
        total_losses = {'total_loss': 0, 'indicator_loss': 0, 'numerical_loss': 0, 'category_loss': 0}
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(dataloader):
            # Move to device
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask']
                    )
                    losses = self.compute_loss(outputs, batch)
                    scaled_loss = losses['total_loss'] / self.gradient_accumulation_steps
                
                # Backward pass with gradient scaling
                self.scaler.scale(scaled_loss).backward()
            else:
                # CPU fallback
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
                losses = self.compute_loss(outputs, batch)
                scaled_loss = losses['total_loss'] / self.gradient_accumulation_steps
                scaled_loss.backward()
            
            # Accumulate losses
            for key, value in losses.items():
                total_losses[key] += value.item()
            
            # Update weights
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.use_amp:
                    # Gradient clipping with scaler
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                # Memory cleanup
                if batch_idx % 20 == 0:  # More frequent cleanup for GPU
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            # Progress reporting
            if (batch_idx + 1) % (20 * self.gradient_accumulation_steps) == 0:
                print(f"  Batch {batch_idx + 1}/{len(dataloader)}, Loss: {losses['total_loss'].item():.4f}")
        
        # Final update
        if len(dataloader) % self.gradient_accumulation_steps != 0:
            if self.use_amp:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
            
            self.scheduler.step()
            self.optimizer.zero_grad()
        
        # Average losses
        for key in total_losses:
            total_losses[key] /= len(dataloader)
        
        return total_losses
    
    def evaluate(self, dataloader):
        """
        Evaluate the model with mixed precision
        """
        self.model.eval()
        total_losses = {'total_loss': 0, 'indicator_loss': 0, 'numerical_loss': 0, 'category_loss': 0}
        
        all_indicator_preds = []
        all_indicator_labels = []
        all_numerical_preds = []
        all_numerical_labels = []
        all_category_preds = []
        all_category_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
                
                # Forward pass with mixed precision
                if self.use_amp:
                    with autocast():
                        outputs = self.model(
                            input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask']
                        )
                        losses = self.compute_loss(outputs, batch)
                else:
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask']
                    )
                    losses = self.compute_loss(outputs, batch)
                
                for key, value in losses.items():
                    total_losses[key] += value.item()
                
                # Collect predictions
                indicator_preds = torch.sigmoid(outputs['indicator_logits']).cpu().numpy()
                numerical_preds = torch.sigmoid(outputs['numerical_logits']).cpu().numpy()
                category_preds = torch.softmax(outputs['category_logits'], dim=1).cpu().numpy()
                
                all_indicator_preds.extend(indicator_preds)
                all_indicator_labels.extend(batch['indicator_labels'].cpu().numpy())
                all_numerical_preds.extend(numerical_preds)
                all_numerical_labels.extend(batch['numerical_labels'].cpu().numpy())
                all_category_preds.extend(category_preds.argmax(axis=1))
                all_category_labels.extend(batch['category_labels'].cpu().numpy())
        
        # Average losses
        for key in total_losses:
            total_losses[key] /= len(dataloader)
        
        # Calculate metrics
        metrics = self._calculate_metrics(
            all_indicator_preds, all_indicator_labels,
            all_numerical_preds, all_numerical_labels,
            all_category_preds, all_category_labels
        )
        
        return total_losses, metrics
    
    def _calculate_metrics(self, indicator_preds, indicator_labels,
                          numerical_preds, numerical_labels,
                          category_preds, category_labels):
        """
        Calculate evaluation metrics
        """
        # Convert to numpy
        indicator_preds = np.array(indicator_preds)
        indicator_labels = np.array(indicator_labels)
        numerical_preds = np.array(numerical_preds).flatten()
        numerical_labels = np.array(numerical_labels).flatten()
        category_preds = np.array(category_preds).flatten()
        category_labels = np.array(category_labels).flatten()
        
        # Binary predictions
        indicator_preds_binary = (indicator_preds > 0.5).astype(int)
        numerical_preds_binary = (numerical_preds > 0.5).astype(int)
        
        metrics = {
            'indicator_f1_micro': f1_score(indicator_labels, indicator_preds_binary, average='micro', zero_division=0),
            'indicator_f1_macro': f1_score(indicator_labels, indicator_preds_binary, average='macro', zero_division=0),
            'numerical_accuracy': accuracy_score(numerical_labels, numerical_preds_binary),
            'numerical_f1': f1_score(numerical_labels, numerical_preds_binary, zero_division=0),
            'category_accuracy': accuracy_score(category_labels, category_preds),
            'category_f1': f1_score(category_labels, category_preds, average='macro', zero_division=0)
        }
        
        return metrics
    
    def train(self):
        """
        Main training loop
        """
        print("\n=== STARTING TRAINING ===")
        
        # Create balanced data loaders
        # Calculate class weights for category balancing
        category_counts = Counter(self.train_category_labels)
        total_samples = len(self.train_category_labels)
        category_weights = {}
        for cat, count in category_counts.items():
            category_weights[cat] = total_samples / (len(category_counts) * count)
        
        # Create sample weights for balanced sampling
        sample_weights = [category_weights[label] for label in self.train_category_labels]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,  # Use weighted sampler instead of shuffle
            num_workers=0,  # Avoid multiprocessing issues
            pin_memory=False
        )
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
        
        best_val_f1 = 0
        training_history = []
        patience_counter = 0
        
        for epoch in range(self.num_epochs):
            print(f"\n--- Epoch {epoch + 1}/{self.num_epochs} ---")
            
            try:
                # Training
                train_losses = self.train_epoch(train_loader)
                print(f"Training Loss: {train_losses['total_loss']:.4f}")
                
                # Validation
                val_losses, val_metrics = self.evaluate(val_loader)
                print(f"Validation Loss: {val_losses['total_loss']:.4f}")
                print(f"Validation F1 (Indicators): {val_metrics['indicator_f1_micro']:.4f}")
                print(f"Validation Accuracy (Numerical): {val_metrics['numerical_accuracy']:.4f}")
                print(f"Validation Accuracy (Category): {val_metrics['category_accuracy']:.4f}")
                
                # Early stopping and best model saving
                current_val_f1 = val_metrics['indicator_f1_micro']
                if current_val_f1 > best_val_f1:
                    best_val_f1 = current_val_f1
                    patience_counter = 0
                    self.save_model('best_lightweight_model')
                    print(f"New best model saved! F1: {best_val_f1:.4f}")
                else:
                    patience_counter += 1
                    print(f"No improvement. Patience: {patience_counter}/{self.early_stopping_patience}")
                
                # Record history
                training_history.append({
                    'epoch': epoch + 1,
                    'train_loss': train_losses['total_loss'],
                    'val_loss': val_losses['total_loss'],
                    'val_f1': val_metrics['indicator_f1_micro'],
                    'val_numerical_acc': val_metrics['numerical_accuracy'],
                    'val_category_acc': val_metrics['category_accuracy']
                })
                
                # Early stopping check
                if patience_counter >= self.early_stopping_patience:
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs!")
                    break
                
            except Exception as e:
                print(f"Error in epoch {epoch + 1}: {e}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print("Attempting to continue training...")
                continue
            
            # Memory cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        print("\n=== TRAINING COMPLETED ===")
        print(f"Best validation F1: {best_val_f1:.4f}")
        
        return training_history
    
    def save_model(self, model_name):
        """
        Save the trained model
        """
        model_dir = f"models/finbert_esg/{model_name}"
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model components
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'indicator_encoder': self.indicator_encoder,
            'category_encoder': self.category_encoder,
            'num_indicators': self.num_indicators,
            'config': {
                'model_name': self.model_name,
                'max_length': self.max_length,
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'num_epochs': self.num_epochs
            }
        }, f"{model_dir}/model.pt")
        
        # Save tokenizer
        self.tokenizer.save_pretrained(model_dir)
        
        print(f"Model saved to {model_dir}")

def main():
    """
    Main execution function
    """
    print("=== Lightweight FinBERT ESG Training ===")
    
    # Initialize trainer with GPU-optimized settings
    trainer = LightweightTrainer(
        model_name='ProsusAI/finbert',
        max_length=128,           # Optimal sequence length
        learning_rate=2e-5,       # Optimized learning rate
        batch_size=8,             # GPU-optimized batch size
        num_epochs=3,             # Extended training
        gradient_accumulation_steps=2,  # Effective batch size of 16
        early_stopping_patience=3  # Early stopping patience
    )
    
    # Load data with standardized filenames
    trainer.load_data(
        train_path='data/annotations/esg_train_set.csv',
        val_path='data/annotations/esg_validation_set.csv',
        test_path='data/annotations/esg_test_set.csv'
    )
    
    # Initialize model
    trainer.initialize_model()
    
    # Train model
    training_history = trainer.train()
    
    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/lightweight_training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)
    
    print("\n=== TRAINING COMPLETE ===")
    print("Lightweight model trained successfully!")
    print("Check 'models/finbert_esg/best_lightweight_model/' for saved model")

if __name__ == "__main__":
    main()