"""
Safety evaluation framework for medical multimodal AI
"""

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
from config import get_config
from model import MedicalMultimodalModel
from data_loader import create_data_loaders
import seaborn as sns

class SafetyEvaluator:
    """Comprehensive safety evaluation for medical AI"""
    
    def __init__(self, model: MedicalMultimodalModel):
        self.model = model
        self.config = get_config()
        self.results = {}
        
    def evaluate_uncertainty_calibration(self, data_loader) -> Dict[str, float]:
        """Benchmark 1: How well does uncertainty align with actual errors?"""
        print("Evaluating uncertainty-risk alignment...")
        
        all_predictions = []
        all_uncertainties = []
        all_labels = []
        all_confidences = []
        
        self.model.eval()
        with torch.no_grad():
            for batch in data_loader:
                images = batch['image'].to(self.config.device)
                input_ids = batch['input_ids'].to(self.config.device) 
                attention_mask = batch['attention_mask'].to(self.config.device)
                labels = batch['label'].to(self.config.device)
                
                # Get predictions with uncertainty
                uncertainty_outputs = self.model.predict_with_uncertainty(
                    images, input_ids, attention_mask, num_samples=20
                )
                
                predictions = uncertainty_outputs['predictions'].argmax(dim=-1)
                confidences = uncertainty_outputs['predictions'].max(dim=-1)[0]
                uncertainties = uncertainty_outputs['total_uncertainty'].squeeze()
                
                all_predictions.extend(predictions.cpu().numpy())
                all_uncertainties.extend(uncertainties.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_confidences.extend(confidences.cpu().numpy())
        
        # Calculate metrics
        predictions = np.array(all_predictions)
        uncertainties = np.array(all_uncertainties)
        labels = np.array(all_labels)
        confidences = np.array(all_confidences)
        
        # Prediction errors
        errors = predictions != labels
        
        # Uncertainty-error correlation
        uncertainty_error_corr = np.corrcoef(uncertainties, errors.astype(float))[0, 1]
        
        # Confidence-accuracy correlation  
        confidence_accuracy_corr = np.corrcoef(confidences, (predictions == labels).astype(float))[0, 1]
        
        # Expected Calibration Error (ECE)
        ece = self._calculate_ece(confidences, predictions == labels)
        
        results = {
            'uncertainty_error_correlation': uncertainty_error_corr,
            'confidence_accuracy_correlation': confidence_accuracy_corr,
            'expected_calibration_error': ece,
            'average_uncertainty': uncertainties.mean(),
            'uncertainty_std': uncertainties.std()
        }
        
        self.results['uncertainty_calibration'] = results
        return results
    
    def evaluate_failure_detection(self, clean_loader, corrupted_loader=None) -> Dict[str, float]:
        """Benchmark 2: Can uncertainty detect corrupted/adversarial inputs?"""
        print("Evaluating failure detection capabilities...")
        
        if corrupted_loader is None:
            # Create corrupted data by adding noise
            corrupted_loader = self._create_corrupted_data(clean_loader)
        
        # Get uncertainties for clean data
        clean_uncertainties = self._get_uncertainties(clean_loader)
        
        # Get uncertainties for corrupted data  
        corrupted_uncertainties = self._get_uncertainties(corrupted_loader)
        
        # AUROC for detecting corruption using uncertainty
        labels = np.concatenate([
            np.zeros(len(clean_uncertainties)),  # Clean = 0
            np.ones(len(corrupted_uncertainties))  # Corrupted = 1
        ])
        scores = np.concatenate([clean_uncertainties, corrupted_uncertainties])
        
        try:
            auroc = roc_auc_score(labels, scores)
        except ValueError:
            auroc = 0.5  # Random performance if all same label
        
        # Detection threshold optimization
        thresholds = np.linspace(scores.min(), scores.max(), 100)
        best_threshold = self._find_best_threshold(labels, scores, thresholds)
        
        results = {
            'failure_detection_auroc': auroc,
            'best_threshold': best_threshold,
            'clean_uncertainty_mean': clean_uncertainties.mean(),
            'corrupted_uncertainty_mean': corrupted_uncertainties.mean(),
            'separation_gap': corrupted_uncertainties.mean() - clean_uncertainties.mean()
        }
        
        self.results['failure_detection'] = results
        return results
    
    def evaluate_distribution_shift_robustness(self, source_loader, target_loader) -> Dict[str, float]:
        """Benchmark 3: Performance degradation across different distributions"""
        print("Evaluating robustness under distribution shift...")
        
        # Evaluate on source domain
        source_metrics = self._evaluate_performance(source_loader)
        
        # Evaluate on target domain  
        target_metrics = self._evaluate_performance(target_loader)
        
        # Calculate performance drops
        accuracy_drop = source_metrics['accuracy'] - target_metrics['accuracy']
        f1_drop = source_metrics['f1'] - target_metrics['f1']
        
        # Uncertainty increase in target domain
        source_uncertainty = self._get_uncertainties(source_loader).mean()
        target_uncertainty = self._get_uncertainties(target_loader).mean()
        uncertainty_increase = target_uncertainty - source_uncertainty
        
        results = {
            'source_accuracy': source_metrics['accuracy'],
            'target_accuracy': target_metrics['accuracy'], 
            'accuracy_drop': accuracy_drop,
            'source_f1': source_metrics['f1'],
            'target_f1': target_metrics['f1'],
            'f1_drop': f1_drop,
            'source_uncertainty': source_uncertainty,
            'target_uncertainty': target_uncertainty,
            'uncertainty_increase': uncertainty_increase
        }
        
        self.results['distribution_shift'] = results
        return results
    
    def evaluate_human_ai_collaboration(self, data_loader) -> Dict[str, float]:
        """Benchmark 4: Decision accuracy when AI provides uncertainty estimates"""
        print("Evaluating human-AI collaboration safety...")
        
        all_predictions = []
        all_uncertainties = []
        all_labels = []
        all_safety_scores = []
        
        self.model.eval()
        with torch.no_grad():
            for batch in data_loader:
                images = batch['image'].to(self.config.device)
                input_ids = batch['input_ids'].to(self.config.device)
                attention_mask = batch['attention_mask'].to(self.config.device)
                labels = batch['label'].to(self.config.device)
                
                # Get model outputs
                outputs = self.model(images, input_ids, attention_mask)
                safety_assessment = self.model.get_safety_assessment(outputs)
                
                predictions = outputs['logits'].argmax(dim=-1)
                uncertainties = outputs['uncertainty_var'].squeeze()
                
                all_predictions.extend(predictions.cpu().numpy())
                all_uncertainties.extend(uncertainties.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_safety_scores.extend(safety_assessment['safety_score'].cpu().numpy())
        
        predictions = np.array(all_predictions)
        uncertainties = np.array(all_uncertainties)
        labels = np.array(all_labels)
        safety_scores = np.array(all_safety_scores)
        
        # Cases flagged for human review
        high_uncertainty_cases = uncertainties > self.config.uncertainty_threshold
        flagged_for_review = high_uncertainty_cases
        
        # Accuracy on high-confidence cases
        high_confidence_cases = ~flagged_for_review
        if high_confidence_cases.sum() > 0:
            high_conf_accuracy = accuracy_score(
                labels[high_confidence_cases], 
                predictions[high_confidence_cases]
            )
        else:
            high_conf_accuracy = 0.0
        
        # Coverage (percentage of cases handled by AI)
        coverage = high_confidence_cases.mean()
        
        results = {
            'high_confidence_accuracy': high_conf_accuracy,
            'coverage': coverage,
            'review_rate': flagged_for_review.mean(),
            'safety_score_mean': safety_scores.mean(),
            'safety_score_std': safety_scores.std()
        }
        
        self.results['human_ai_collaboration'] = results
        return results
    
    def _calculate_ece(self, confidences: np.ndarray, accuracies: np.ndarray, n_bins: int = 10) -> float:
        """Calculate Expected Calibration Error"""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def _get_uncertainties(self, data_loader) -> np.ndarray:
        """Extract uncertainties from data loader"""
        uncertainties = []
        
        self.model.eval()
        with torch.no_grad():
            for batch in data_loader:
                images = batch['image'].to(self.config.device)
                input_ids = batch['input_ids'].to(self.config.device)
                attention_mask = batch['attention_mask'].to(self.config.device)
                
                uncertainty_outputs = self.model.predict_with_uncertainty(
                    images, input_ids, attention_mask, num_samples=10
                )
                
                batch_uncertainties = uncertainty_outputs['total_uncertainty'].squeeze()
                uncertainties.extend(batch_uncertainties.cpu().numpy())
        
        return np.array(uncertainties)
    
    def _create_corrupted_data(self, clean_loader):
        """Create corrupted version of data by adding noise"""
        corrupted_batches = []
        
        for batch in clean_loader:
            corrupted_batch = batch.copy()
            
            # Add Gaussian noise to images
            noise = torch.randn_like(batch['image']) * 0.1
            corrupted_batch['image'] = batch['image'] + noise
            
            # Clip to valid range
            corrupted_batch['image'] = torch.clamp(corrupted_batch['image'], 0, 1)
            
            corrupted_batches.append(corrupted_batch)
            
            # Only corrupt first few batches for speed
            if len(corrupted_batches) >= 10:
                break
        
        return corrupted_batches
    
    def _find_best_threshold(self, labels: np.ndarray, scores: np.ndarray, thresholds: np.ndarray) -> float:
        """Find optimal threshold for binary classification"""
        best_f1 = 0
        best_threshold = thresholds[0]
        
        for threshold in thresholds:
            predictions = scores >= threshold
            if len(np.unique(predictions)) > 1:  # Avoid all same prediction
                _, _, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
        
        return best_threshold
    
    def _evaluate_performance(self, data_loader) -> Dict[str, float]:
        """Evaluate standard performance metrics"""
        all_predictions = []
        all_labels = []
        
        self.model.eval()
        with torch.no_grad():
            for batch in data_loader:
                images = batch['image'].to(self.config.device)
                input_ids = batch['input_ids'].to(self.config.device)
                attention_mask = batch['attention_mask'].to(self.config.device)
                labels = batch['label'].to(self.config.device)
                
                outputs = self.model(images, input_ids, attention_mask)
                predictions = outputs['logits'].argmax(dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        predictions = np.array(all_predictions)
        labels = np.array(all_labels)
        
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def run_comprehensive_evaluation(self, test_loader) -> Dict[str, Any]:
        """Run all safety benchmarks"""
        print("Starting comprehensive safety evaluation...")
        
        # Split test loader for different evaluations
        test_batches = list(test_loader)
        clean_batches = test_batches[:len(test_batches)//2]
        target_batches = test_batches[len(test_batches)//2:]
        
        # Create smaller loaders
        from torch.utils.data import DataLoader
        from torch.utils.data import TensorDataset
        
        # Run all benchmarks
        results = {}
        
        # Benchmark 1: Uncertainty Calibration
        results.update(self.evaluate_uncertainty_calibration(test_loader))
        
        # Benchmark 2: Failure Detection
        results.update(self.evaluate_failure_detection(test_loader))
        
        # Benchmark 3: Distribution Shift (using splits as proxy)
        if len(target_batches) > 0:
            # Create loaders from batches
            clean_loader_subset = self._batches_to_loader(clean_batches)
            target_loader_subset = self._batches_to_loader(target_batches)
            results.update(self.evaluate_distribution_shift_robustness(clean_loader_subset, target_loader_subset))
        
        # Benchmark 4: Human-AI Collaboration
        results.update(self.evaluate_human_ai_collaboration(test_loader))
        
        self.results['comprehensive'] = results
        
        # Print summary
        self.print_evaluation_summary()
        
        return results
    
    def _batches_to_loader(self, batches):
        """Convert list of batches back to a data loader"""
        # Simple implementation - just return the batches as an iterable
        return batches
    
    def print_evaluation_summary(self):
        """Print a comprehensive evaluation summary"""
        print("\n" + "="*60)
        print("MEDICAL AI SAFETY EVALUATION SUMMARY")
        print("="*60)
        
        if 'uncertainty_calibration' in self.results:
            uc = self.results['uncertainty_calibration']
            print(f"\n🎯 UNCERTAINTY CALIBRATION:")
            print(f"   Uncertainty-Error Correlation: {uc['uncertainty_error_correlation']:.3f}")
            print(f"   Confidence-Accuracy Correlation: {uc['confidence_accuracy_correlation']:.3f}")
            print(f"   Expected Calibration Error: {uc['expected_calibration_error']:.3f}")
        
        if 'failure_detection' in self.results:
            fd = self.results['failure_detection']
            print(f"\n🛡️ FAILURE DETECTION:")
            print(f"   Detection AUROC: {fd['failure_detection_auroc']:.3f}")
            print(f"   Uncertainty Separation: {fd['separation_gap']:.3f}")
        
        if 'distribution_shift' in self.results:
            ds = self.results['distribution_shift']
            print(f"\n📊 DISTRIBUTION SHIFT ROBUSTNESS:")
            print(f"   Accuracy Drop: {ds['accuracy_drop']:.3f}")
            print(f"   F1 Drop: {ds['f1_drop']:.3f}")
            print(f"   Uncertainty Increase: {ds['uncertainty_increase']:.3f}")
        
        if 'human_ai_collaboration' in self.results:
            hac = self.results['human_ai_collaboration']
            print(f"\n👥 HUMAN-AI COLLABORATION:")
            print(f"   High-Confidence Accuracy: {hac['high_confidence_accuracy']:.3f}")
            print(f"   Coverage: {hac['coverage']:.3f}")
            print(f"   Review Rate: {hac['review_rate']:.3f}")
        
        print("\n" + "="*60)
    
    def save_results(self, filename: str = "safety_evaluation_results.json"):
        """Save evaluation results to file"""
        import json
        
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
            else:
                return obj
        
        results_json = convert_numpy(self.results)
        
        with open(filename, 'w') as f:
            json.dump(results_json, f, indent=2)
        
        print(f"Results saved to {filename}")

def run_safety_evaluation():
    """Main function to run safety evaluation"""
    print("Loading model and data...")
    
    # Create model and data loaders
    from model import create_model
    model = create_model()
    
    _, _, test_loader = create_data_loaders()
    
    # Create evaluator
    evaluator = SafetyEvaluator(model)
    
    # Run comprehensive evaluation
    results = evaluator.run_comprehensive_evaluation(test_loader)
    
    # Save results
    evaluator.save_results()
    
    return evaluator, results

if __name__ == "__main__":
    evaluator, results = run_safety_evaluation()