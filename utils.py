"""
Utility functions for Medical Multimodal Safety System
"""

import torch
import logging
import os
import json
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from config import get_config

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration"""
    config = get_config()
    
    # Create logs directory if it doesn't exist
    os.makedirs(config.log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('medical_multimodal_safety')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # File handler
    log_file = os.path.join(config.log_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger

def save_checkpoint(checkpoint: Dict[str, Any], filepath: str):
    """Save model checkpoint"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")

def load_checkpoint(filepath: str, device: torch.device) -> Dict[str, Any]:
    """Load model checkpoint"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    
    checkpoint = torch.load(filepath, map_location=device)
    print(f"Checkpoint loaded from {filepath}")
    return checkpoint

def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count model parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params
    }

def save_results(results: Dict[str, Any], filename: str):
    """Save results to JSON file"""
    config = get_config()
    filepath = os.path.join(config.log_dir, filename)
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    
    results_json = convert_numpy(results)
    
    with open(filepath, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"Results saved to {filepath}")

def load_results(filename: str) -> Dict[str, Any]:
    """Load results from JSON file"""
    config = get_config()
    filepath = os.path.join(config.log_dir, filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Results file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        results = json.load(f)
    
    return results

def create_confusion_matrix_plot(y_true: np.ndarray, y_pred: np.ndarray, 
                                class_names: list = None) -> plt.Figure:
    """Create confusion matrix visualization"""
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(cm))]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    
    plt.tight_layout()
    return fig

def create_uncertainty_distribution_plot(uncertainties: np.ndarray, 
                                        errors: np.ndarray = None) -> plt.Figure:
    """Create uncertainty distribution visualization"""
    fig, axes = plt.subplots(1, 2 if errors is not None else 1, figsize=(12, 5))
    
    if errors is None:
        axes = [axes]
    
    # Uncertainty histogram
    axes[0].hist(uncertainties, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0].set_xlabel('Uncertainty')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Uncertainty Distribution')
    axes[0].grid(alpha=0.3)
    
    # Uncertainty vs errors (if provided)
    if errors is not None:
        correct_unc = uncertainties[errors == 0]
        incorrect_unc = uncertainties[errors == 1]
        
        axes[1].hist(correct_unc, bins=20, alpha=0.7, color='green', 
                    label='Correct Predictions', density=True)
        axes[1].hist(incorrect_unc, bins=20, alpha=0.7, color='red', 
                    label='Incorrect Predictions', density=True)
        axes[1].set_xlabel('Uncertainty')
        axes[1].set_ylabel('Density')
        axes[1].set_title('Uncertainty by Prediction Correctness')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_calibration_plot(confidences: np.ndarray, accuracies: np.ndarray, 
                           n_bins: int = 10) -> plt.Figure:
    """Create calibration plot"""
    from sklearn.calibration import calibration_curve
    
    fraction_of_positives, mean_predicted_value = calibration_curve(
        accuracies, confidences, n_bins=n_bins
    )
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    
    # Model calibration
    ax.plot(mean_predicted_value, fraction_of_positives, 'o-', 
            label='Model Calibration', linewidth=2, markersize=8)
    
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title('Calibration Plot (Reliability Diagram)')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_roc_curve_plot(y_true: np.ndarray, y_scores: np.ndarray) -> plt.Figure:
    """Create ROC curve visualization"""
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
            label='Random Classifier')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig

def print_model_summary(model: torch.nn.Module):
    """Print comprehensive model summary"""
    param_counts = count_parameters(model)
    
    print("="*60)
    print("MODEL SUMMARY")
    print("="*60)
    print(f"Total Parameters: {param_counts['total_parameters']:,}")
    print(f"Trainable Parameters: {param_counts['trainable_parameters']:,}")
    print(f"Non-trainable Parameters: {param_counts['non_trainable_parameters']:,}")
    print(f"Model Size (MB): {param_counts['total_parameters'] * 4 / 1024 / 1024:.2f}")
    print("="*60)

def format_time(seconds: float) -> str:
    """Format time in human readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

def get_device_info() -> Dict[str, Any]:
    """Get information about available devices"""
    device_info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'current_device': torch.cuda.current_device() if torch.cuda.is_available() else None
    }
    
    if torch.cuda.is_available():
        device_info['device_name'] = torch.cuda.get_device_name(0)
        device_info['memory_total'] = torch.cuda.get_device_properties(0).total_memory
        device_info['memory_allocated'] = torch.cuda.memory_allocated(0)
        device_info['memory_cached'] = torch.cuda.memory_reserved(0)
    
    return device_info

def seed_everything(seed: int = 42):
    """Set random seeds for reproducibility"""
    import random
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Random seed set to {seed}")

def calculate_model_flops(model: torch.nn.Module, input_shape: tuple) -> int:
    """Calculate approximate FLOPs for model (simplified)"""
    # This is a simplified FLOP calculation
    # For more accurate calculation, use libraries like ptflops or fvcore
    
    total_params = sum(p.numel() for p in model.parameters())
    # Rough approximation: 2 FLOPs per parameter per forward pass
    approx_flops = total_params * 2
    
    return approx_flops

def create_training_curves_plot(train_losses: list, val_losses: list, 
                               train_accs: list = None, val_accs: list = None) -> plt.Figure:
    """Create training curves visualization"""
    fig, axes = plt.subplots(1, 2 if train_accs is not None else 1, figsize=(15, 5))
    
    if train_accs is None:
        axes = [axes]
    
    # Loss curves
    epochs = range(1, len(train_losses) + 1)
    axes[0].plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    axes[0].plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Accuracy curves (if provided)
    if train_accs is not None:
        axes[1].plot(epochs, train_accs, 'b-', label='Training Accuracy', linewidth=2)
        axes[1].plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    return fig

def validate_config():
    """Validate configuration settings"""
    config = get_config()
    
    # Check device availability
    if config.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Using CPU.")
        config.device = "cpu"
    
    # Check batch size
    if config.device == "cuda" and config.batch_size > 16:
        print("Warning: Large batch size on GPU may cause memory issues.")
    
    # Check paths
    required_dirs = [config.data_dir, config.model_dir, config.log_dir]
    for dir_path in required_dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    print("Configuration validated successfully!")

def print_system_info():
    """Print comprehensive system information"""
    device_info = get_device_info()
    config = get_config()
    
    print("="*60)
    print("SYSTEM INFORMATION")
    print("="*60)
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {device_info['cuda_available']}")
    
    if device_info['cuda_available']:
        print(f"CUDA Version: {device_info['cuda_version']}")
        print(f"Device Name: {device_info['device_name']}")
        print(f"Total GPU Memory: {device_info['memory_total'] / 1024**3:.1f} GB")
        print(f"Current Device: {device_info['current_device']}")
    
    print(f"Selected Device: {config.device}")
    print(f"Mixed Precision: {config.mixed_precision}")
    print(f"Batch Size: {config.batch_size}")
    print("="*60)

def memory_cleanup():
    """Clean up GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU memory cache cleared")

def create_safety_summary_plot(safety_results: Dict[str, Any]) -> plt.Figure:
    """Create comprehensive safety summary visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Safety metrics overview
    metrics = ['Uncertainty Correlation', 'Failure Detection AUROC', 
               'Calibration Error', 'Safety Score']
    values = [
        safety_results.get('uncertainty_error_correlation', 0),
        safety_results.get('failure_detection_auroc', 0),
        1 - safety_results.get('expected_calibration_error', 1),  # Higher is better
        safety_results.get('safety_score_mean', 0)
    ]
    
    colors = ['green' if v > 0.7 else 'orange' if v > 0.5 else 'red' for v in values]
    
    bars = axes[0, 0].bar(metrics, values, color=colors, alpha=0.7)
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].set_title('Safety Metrics Overview')
    axes[0, 0].set_ylabel('Score')
    plt.setp(axes[0, 0].xaxis.get_majorticklabels(), rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{value:.3f}', ha='center', va='bottom')
    
    # Uncertainty distribution (placeholder)
    uncertainties = np.random.beta(2, 5, 1000)  # Example distribution
    axes[0, 1].hist(uncertainties, bins=30, alpha=0.7, color='skyblue')
    axes[0, 1].set_title('Uncertainty Distribution')
    axes[0, 1].set_xlabel('Uncertainty')
    axes[0, 1].set_ylabel('Frequency')
    
    # Coverage vs accuracy trade-off
    coverage = np.linspace(0.1, 1.0, 10)
    accuracy = 0.6 + 0.3 * (1 - coverage)  # Example relationship
    axes[1, 0].plot(coverage, accuracy, 'o-', linewidth=2, markersize=6)
    axes[1, 0].set_xlabel('Coverage (Fraction of Cases Handled)')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_title('Coverage vs Accuracy Trade-off')
    axes[1, 0].grid(alpha=0.3)
    
    # Safety recommendation distribution
    recommendations = ['High Confidence', 'Moderate', 'Low Confidence', 'Requires Review']
    counts = [40, 30, 20, 10]  # Example distribution
    colors_pie = ['green', 'yellow', 'orange', 'red']
    
    axes[1, 1].pie(counts, labels=recommendations, colors=colors_pie, autopct='%1.1f%%')
    axes[1, 1].set_title('Safety Recommendation Distribution')
    
    plt.tight_layout()
    return fig

def export_model_for_deployment(model: torch.nn.Module, 
                               example_input: tuple,
                               export_path: str):
    """Export model for deployment using TorchScript"""
    model.eval()
    
    try:
        # Create example inputs
        example_image, example_input_ids, example_attention_mask = example_input
        
        # Trace the model
        traced_model = torch.jit.trace(
            model, 
            (example_image, example_input_ids, example_attention_mask)
        )
        
        # Save traced model
        traced_model.save(export_path)
        print(f"Model exported for deployment: {export_path}")
        
        # Verify the traced model works
        with torch.no_grad():
            original_output = model(example_image, example_input_ids, example_attention_mask)
            traced_output = traced_model(example_image, example_input_ids, example_attention_mask)
            
            # Check if outputs are close
            logits_close = torch.allclose(original_output['logits'], traced_output['logits'], atol=1e-5)
            print(f"Model tracing verification: {'✅ Passed' if logits_close else '❌ Failed'}")
            
    except Exception as e:
        print(f"Model export failed: {e}")

def benchmark_model_performance(model: torch.nn.Module, 
                               data_loader,
                               num_warmup: int = 10,
                               num_benchmark: int = 100) -> Dict[str, float]:
    """Benchmark model inference performance"""
    import time
    
    model.eval()
    device = next(model.parameters()).device
    
    # Get a sample batch for benchmarking
    sample_batch = next(iter(data_loader))
    images = sample_batch['image'][:1].to(device)  # Single sample
    input_ids = sample_batch['input_ids'][:1].to(device)
    attention_mask = sample_batch['attention_mask'][:1].to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(images, input_ids, attention_mask)
    
    # Benchmark
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_benchmark):
            _ = model(images, input_ids, attention_mask)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_inference_time = total_time / num_benchmark
    throughput = 1.0 / avg_inference_time
    
    return {
        'average_inference_time_ms': avg_inference_time * 1000,
        'throughput_samples_per_sec': throughput,
        'total_benchmark_time_s': total_time
    }

# Test utility functions
def test_utils():
    """Test utility functions"""
    print("Testing utility functions...")
    
    # Test logging setup
    logger = setup_logging()
    logger.info("Testing logging functionality")
    
    # Test device info
    device_info = get_device_info()
    print(f"Device info: {device_info}")
    
    # Test configuration validation
    validate_config()
    
    # Test system info
    print_system_info()
    
    print("Utility functions test completed!")

if __name__ == "__main__":
    test_utils()