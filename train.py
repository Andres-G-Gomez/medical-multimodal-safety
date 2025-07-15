"""
Enhanced Training script for Medical Multimodal Safety System with CheXpert Multi-Label Support
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm
import os
from typing import Dict, Tuple, List
import time
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, average_precision_score

from config import get_config
from model import create_model, MedicalMultimodalModel
from data_loader import create_data_loaders
from utils import setup_logging, save_checkpoint, load_checkpoint

class MultiLabelUncertaintyLoss(nn.Module):
    """Enhanced loss for multi-label classification with uncertainty estimation"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Multi-label classification loss
        if config.loss_function == "bce_with_logits":
            # Get class weights if available
            if hasattr(config, 'class_weights') and config.class_weights:
                weights = config.get_class_weight_tensor()
                self.classification_loss = nn.BCEWithLogitsLoss(pos_weight=weights)
            else:
                self.classification_loss = nn.BCEWithLogitsLoss()
        elif config.loss_function == "focal_loss":
            self.classification_loss = self._focal_loss
        else:
            self.classification_loss = nn.BCEWithLogitsLoss()
        
        # Loss weights
        self.classification_weight = 1.0
        self.uncertainty_weight = 0.1
        
    def _focal_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Focal loss for handling class imbalance"""
        alpha = self.config.focal_loss_alpha
        gamma = self.config.focal_loss_gamma
        
        # Convert logits to probabilities
        probs = torch.sigmoid(logits)
        
        # Calculate focal loss
        ce_loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_loss = alpha * (1 - p_t) ** gamma * ce_loss
        
        return focal_loss.mean()
        
    def forward(self, logits: torch.Tensor, labels: torch.Tensor, 
                uncertainty_mean: torch.Tensor = None, uncertainty_var: torch.Tensor = None) -> Tuple[torch.Tensor, Dict]:
        
        # Multi-label classification loss
        cls_loss = self.classification_loss(logits, labels)
        
        # Initialize uncertainty loss
        uncertainty_loss = torch.tensor(0.0, device=logits.device)
        
        # Uncertainty loss (if uncertainty estimates are provided)
        if uncertainty_mean is not None and uncertainty_var is not None:
            # Get predictions
            predictions = torch.sigmoid(logits)
            
            # Calculate per-class accuracy
            pred_binary = (predictions > self.config.multi_label_threshold).float()
            correct_mask = (pred_binary == labels).float()
            
            # Uncertainty regularization - penalize high uncertainty on correct predictions
            uncertainty_loss = (uncertainty_var * correct_mask).mean()
        
        # Total loss
        total_loss = (self.classification_weight * cls_loss + 
                     self.uncertainty_weight * uncertainty_loss)
        
        loss_dict = {
            'total_loss': total_loss,
            'classification_loss': cls_loss,
            'uncertainty_loss': uncertainty_loss
        }
        
        return total_loss, loss_dict

class MultiLabelTrainer:
    """Enhanced training manager for multi-label medical classification"""
    
    def __init__(self):
        self.config = get_config()
        self.device = torch.device(self.config.device)
        
        # Initialize logging
        self.logger = setup_logging()
        
        # Create model
        self.model = create_model()
        
        # Create data loaders
        self.train_loader, self.val_loader, self.test_loader = create_data_loaders()
        
        # Loss function
        self.criterion = MultiLabelUncertaintyLoss(self.config)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Learning rate scheduler
        total_steps = len(self.train_loader) * self.config.num_epochs
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=self.config.learning_rate * 0.01
        )
        
        # Mixed precision training
        self.scaler = GradScaler() if self.config.mixed_precision else None
        
        # Tracking
        self.best_val_score = 0.0  # Using F1 score for multi-label
        self.train_losses = []
        self.val_losses = []
        
        self.logger.info("Trainer initialized successfully")
        
    def calculate_multi_label_metrics(self, predictions: np.ndarray, targets: np.ndarray, 
                                    threshold: float = None) -> Dict[str, float]:
        """Calculate comprehensive metrics for multi-label classification"""
        if threshold is None:
            threshold = self.config.multi_label_threshold
            
        # Convert predictions to binary
        pred_binary = (predictions > threshold).astype(int)
        
        # Overall metrics
        subset_accuracy = accuracy_score(targets, pred_binary)
        
        # Per-class metrics (micro/macro averaging)
        precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
            targets, pred_binary, average='micro', zero_division=0
        )
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            targets, pred_binary, average='macro', zero_division=0
        )
        
        # ROC AUC (handle potential errors with constant classes)
        try:
            auc_roc_micro = roc_auc_score(targets, predictions, average='micro')
            auc_roc_macro = roc_auc_score(targets, predictions, average='macro')
        except ValueError:
            auc_roc_micro = auc_roc_macro = 0.0
        
        # Average Precision (AP)
        try:
            ap_micro = average_precision_score(targets, predictions, average='micro')
            ap_macro = average_precision_score(targets, predictions, average='macro')
        except ValueError:
            ap_micro = ap_macro = 0.0
        
        # Hamming loss (fraction of incorrect labels)
        hamming_loss = np.mean(pred_binary != targets)
        
        metrics = {
            'subset_accuracy': subset_accuracy,
            'precision_micro': precision_micro,
            'recall_micro': recall_micro,
            'f1_micro': f1_micro,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'auc_roc_micro': auc_roc_micro,
            'auc_roc_macro': auc_roc_macro,
            'ap_micro': ap_micro,
            'ap_macro': ap_macro,
            'hamming_loss': hamming_loss
        }
        
        return metrics
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch with multi-label support"""
        self.model.train()
        
        total_loss = 0
        total_cls_loss = 0
        total_unc_loss = 0
        all_predictions = []
        all_targets = []
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            images = batch['image'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # Use CheXpert multi-labels
            if self.config.use_multi_label:
                labels = batch['chexpert_labels'].to(self.device)  # [batch_size, 14]
            else:
                labels = batch['binary_label'].to(self.device)    # [batch_size]
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.config.mixed_precision:
                with autocast():
                    outputs = self.model(images, input_ids, attention_mask)
                    
                    # Handle uncertainty outputs if available
                    uncertainty_mean = outputs.get('uncertainty_mean', None)
                    uncertainty_var = outputs.get('uncertainty_var', None)
                    
                    loss, loss_dict = self.criterion(
                        outputs['logits'], labels, uncertainty_mean, uncertainty_var
                    )
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images, input_ids, attention_mask)
                
                uncertainty_mean = outputs.get('uncertainty_mean', None)
                uncertainty_var = outputs.get('uncertainty_var', None)
                
                loss, loss_dict = self.criterion(
                    outputs['logits'], labels, uncertainty_mean, uncertainty_var
                )
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                
                # Optimizer step
                self.optimizer.step()
            
            # Update scheduler
            self.scheduler.step()
            
            # Collect predictions and targets for metrics
            if self.config.use_multi_label:
                predictions = torch.sigmoid(outputs['logits']).detach().cpu().numpy()
                targets = labels.detach().cpu().numpy()
            else:
                predictions = torch.softmax(outputs['logits'], dim=-1)[:, 1].detach().cpu().numpy()
                targets = labels.detach().cpu().numpy()
            
            all_predictions.append(predictions)
            all_targets.append(targets)
            
            # Accumulate losses
            total_loss += loss.item()
            total_cls_loss += loss_dict['classification_loss'].item()
            total_unc_loss += loss_dict['uncertainty_loss'].item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'LR': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
        
        # Calculate epoch metrics
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        if self.config.use_multi_label:
            metrics = self.calculate_multi_label_metrics(all_predictions, all_targets)
            # Use macro F1 as primary metric
            primary_metric = metrics['f1_macro']
        else:
            # Binary classification metrics
            pred_binary = (all_predictions > 0.5).astype(int)
            accuracy = accuracy_score(all_targets, pred_binary)
            metrics = {'accuracy': accuracy}
            primary_metric = accuracy
        
        avg_loss = total_loss / len(self.train_loader)
        avg_cls_loss = total_cls_loss / len(self.train_loader)
        avg_unc_loss = total_unc_loss / len(self.train_loader)
        
        epoch_metrics = {
            'train_loss': avg_loss,
            'train_cls_loss': avg_cls_loss,
            'train_unc_loss': avg_unc_loss,
            'train_primary_metric': primary_metric,
            **{f'train_{k}': v for k, v in metrics.items()}
        }
        
        self.train_losses.append(avg_loss)
        return epoch_metrics
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Validate for one epoch with multi-label support"""
        self.model.eval()
        
        total_loss = 0
        total_cls_loss = 0
        total_unc_loss = 0
        all_predictions = []
        all_targets = []
        all_uncertainties = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                # Move batch to device
                images = batch['image'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                if self.config.use_multi_label:
                    labels = batch['chexpert_labels'].to(self.device)
                else:
                    labels = batch['binary_label'].to(self.device)
                
                # Forward pass
                outputs = self.model(images, input_ids, attention_mask)
                
                uncertainty_mean = outputs.get('uncertainty_mean', None)
                uncertainty_var = outputs.get('uncertainty_var', None)
                
                loss, loss_dict = self.criterion(
                    outputs['logits'], labels, uncertainty_mean, uncertainty_var
                )
                
                # Collect predictions and targets
                if self.config.use_multi_label:
                    predictions = torch.sigmoid(outputs['logits']).cpu().numpy()
                    targets = labels.cpu().numpy()
                else:
                    predictions = torch.softmax(outputs['logits'], dim=-1)[:, 1].cpu().numpy()
                    targets = labels.cpu().numpy()
                
                all_predictions.append(predictions)
                all_targets.append(targets)
                
                # Collect uncertainties if available
                if uncertainty_var is not None:
                    uncertainties = uncertainty_var.cpu().numpy()
                    all_uncertainties.extend(uncertainties.flatten())
                
                # Accumulate losses
                total_loss += loss.item()
                total_cls_loss += loss_dict['classification_loss'].item()
                total_unc_loss += loss_dict['uncertainty_loss'].item()
        
        # Calculate epoch metrics
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        if self.config.use_multi_label:
            metrics = self.calculate_multi_label_metrics(all_predictions, all_targets)
            primary_metric = metrics['f1_macro']
        else:
            pred_binary = (all_predictions > 0.5).astype(int)
            accuracy = accuracy_score(all_targets, pred_binary)
            metrics = {'accuracy': accuracy}
            primary_metric = accuracy
        
        avg_loss = total_loss / len(self.val_loader)
        avg_cls_loss = total_cls_loss / len(self.val_loader)
        avg_unc_loss = total_unc_loss / len(self.val_loader)
        avg_uncertainty = np.mean(all_uncertainties) if all_uncertainties else 0.0
        
        epoch_metrics = {
            'val_loss': avg_loss,
            'val_cls_loss': avg_cls_loss,
            'val_unc_loss': avg_unc_loss,
            'val_primary_metric': primary_metric,
            'val_avg_uncertainty': avg_uncertainty,
            **{f'val_{k}': v for k, v in metrics.items()}
        }
        
        self.val_losses.append(avg_loss)
        
        # Save best model (based on primary metric, not loss)
        if primary_metric > self.best_val_score:
            self.best_val_score = primary_metric
            self.save_best_model(epoch, epoch_metrics)
        
        return epoch_metrics
    
    def save_best_model(self, epoch: int, metrics: Dict[str, float]):
        """Save the best model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_score': self.best_val_score,
            'metrics': metrics,
            'config': vars(self.config)
        }
        
        save_path = os.path.join(self.config.model_dir, 'best_model.pt')
        save_checkpoint(checkpoint, save_path)
        
        metric_name = 'F1-Macro' if self.config.use_multi_label else 'Accuracy'
        self.logger.info(f"Saved best model at epoch {epoch+1} with {metric_name}: {self.best_val_score:.4f}")
    
    def train(self) -> Dict[str, float]:
        """Full training loop"""
        self.logger.info("Starting multi-label training...")
        self.logger.info(f"Training on device: {self.device}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")
        self.logger.info(f"Multi-label mode: {self.config.use_multi_label}")
        self.logger.info(f"Number of classes: {self.config.num_classes}")
        
        start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            # Train epoch
            train_metrics = self.train_epoch(epoch)
            
            # Validate epoch
            val_metrics = self.validate_epoch(epoch)
            
            # Combine metrics
            epoch_metrics = {**train_metrics, **val_metrics}
            
            # Log epoch results
            if self.config.use_multi_label:
                self.logger.info(
                    f"Epoch {epoch+1}/{self.config.num_epochs} - "
                    f"Train Loss: {train_metrics['train_loss']:.4f}, "
                    f"Train F1: {train_metrics['train_f1_macro']:.3f}, "
                    f"Val Loss: {val_metrics['val_loss']:.4f}, "
                    f"Val F1: {val_metrics['val_f1_macro']:.3f}"
                )
            else:
                self.logger.info(
                    f"Epoch {epoch+1}/{self.config.num_epochs} - "
                    f"Train Loss: {train_metrics['train_loss']:.4f}, "
                    f"Train Acc: {train_metrics['train_accuracy']:.3f}, "
                    f"Val Loss: {val_metrics['val_loss']:.4f}, "
                    f"Val Acc: {val_metrics['val_accuracy']:.3f}"
                )
        
        training_time = time.time() - start_time
        self.logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Final evaluation
        final_metrics = self.evaluate_final_model()
        
        return final_metrics
    
    def evaluate_final_model(self) -> Dict[str, float]:
        """Evaluate the final model on test set"""
        # Load best model
        checkpoint_path = os.path.join(self.config.model_dir, 'best_model.pt')
        if os.path.exists(checkpoint_path):
            checkpoint = load_checkpoint(checkpoint_path, self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.logger.info("Loaded best model for final evaluation")
        
        self.model.eval()
        
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Final Evaluation"):
                images = batch['image'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                if self.config.use_multi_label:
                    labels = batch['chexpert_labels'].to(self.device)
                else:
                    labels = batch['binary_label'].to(self.device)
                
                outputs = self.model(images, input_ids, attention_mask)
                
                uncertainty_mean = outputs.get('uncertainty_mean', None)
                uncertainty_var = outputs.get('uncertainty_var', None)
                
                loss, _ = self.criterion(
                    outputs['logits'], labels, uncertainty_mean, uncertainty_var
                )
                
                # Collect predictions
                if self.config.use_multi_label:
                    predictions = torch.sigmoid(outputs['logits']).cpu().numpy()
                    targets = labels.cpu().numpy()
                else:
                    predictions = torch.softmax(outputs['logits'], dim=-1)[:, 1].cpu().numpy()
                    targets = labels.cpu().numpy()
                
                all_predictions.append(predictions)
                all_targets.append(targets)
                total_loss += loss.item()
        
        # Calculate final metrics
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        if self.config.use_multi_label:
            metrics = self.calculate_multi_label_metrics(all_predictions, all_targets)
            
            # Log per-pathology performance
            self.logger.info("Per-pathology test performance:")
            for i, pathology in enumerate(self.config.chexpert_observations):
                if i < all_targets.shape[1]:
                    pathology_targets = all_targets[:, i]
                    pathology_preds = all_predictions[:, i]
                    
                    if len(np.unique(pathology_targets)) > 1:  # Avoid division by zero
                        try:
                            auc = roc_auc_score(pathology_targets, pathology_preds)
                            self.logger.info(f"  {pathology}: AUC = {auc:.3f}")
                        except ValueError:
                            pass
        else:
            pred_binary = (all_predictions > 0.5).astype(int)
            accuracy = accuracy_score(all_targets, pred_binary)
            metrics = {'accuracy': accuracy}
        
        test_loss = total_loss / len(self.test_loader)
        
        final_metrics = {
            'test_loss': test_loss,
            **{f'test_{k}': v for k, v in metrics.items()}
        }
        
        if self.config.use_multi_label:
            self.logger.info(f"Final Test Results - Loss: {test_loss:.4f}, Macro F1: {metrics['f1_macro']:.3f}")
        else:
            self.logger.info(f"Final Test Results - Loss: {test_loss:.4f}, Accuracy: {metrics['accuracy']:.3f}")
        
        return final_metrics

def main():
    """Main training function"""
    trainer = MultiLabelTrainer()
    final_metrics = trainer.train()
    
    print("\nTraining completed!")
    
    # Print final results
    config = get_config()
    if config.use_multi_label:
        print(f"Final test Macro F1: {final_metrics.get('test_f1_macro', 0.0):.3f}")
        print(f"Final test Micro F1: {final_metrics.get('test_f1_micro', 0.0):.3f}")
        print(f"Final test AUC (Macro): {final_metrics.get('test_auc_roc_macro', 0.0):.3f}")
    else:
        print(f"Final test accuracy: {final_metrics.get('test_accuracy', 0.0):.3f}")
    
    print(f"Final test loss: {final_metrics.get('test_loss', 0.0):.4f}")

if __name__ == "__main__":
    main()