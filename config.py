"""
Enhanced configuration for Medical Multimodal Safety System with CheXpert Labels
"""

import torch
from dataclasses import dataclass
from typing import List, Tuple, Dict

@dataclass
class Config:
    # Dataset
    dataset_name: str = "itsanmolgupta/mimic-cxr-dataset"
    subset_size: int = 5000  
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    
    # Data processing
    image_size: Tuple[int, int] = (224, 224)
    max_text_length: int = 512
    batch_size: int = 8
    num_workers: int = 0  # Set to 0 for Windows compatibility
    
    # Model architecture - Updated for multi-label classification
    vision_model: str = "vit_base_patch16_224"
    text_model: str = "emilyalsentzer/Bio_ClinicalBERT"
    hidden_dim: int = 768
    
    # CheXpert Labels Configuration
    num_classes: int = 14  # 14 CheXpert observations
    num_binary_classes: int = 2  # For overall normal/abnormal classification
    chexpert_observations: List[str] = None  # Will be set in __post_init__
    
    # Multi-label classification settings
    use_multi_label: bool = True  # Enable multi-label mode
    multi_label_threshold: float = 0.5  # Threshold for positive prediction
    class_weights: Dict[str, float] = None  # For handling class imbalance
    
    # Label handling for uncertain labels (-1 in CheXpert)
    uncertain_label_strategy: str = "positive"  # "positive", "negative", "ignore", "separate_class"
    
    dropout: float = 0.1
    
    # Uncertainty quantification
    mc_dropout_samples: int = 100
    uncertainty_threshold: float = 0.7
    confidence_threshold: float = 0.8
    
    # Training - Updated for multi-label
    learning_rate: float = 3e-5  # Slightly lower for multi-label
    weight_decay: float = 0.01
    num_epochs: int = 25  # More epochs for multi-label learning
    warmup_steps: int = 100
    gradient_clip: float = 1.0
    
    # Loss function configuration
    loss_function: str = "bce_with_logits"  # For multi-label: "bce_with_logits", "focal_loss"
    focal_loss_alpha: float = 0.25  # For focal loss
    focal_loss_gamma: float = 2.0   # For focal loss
    
    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True  # For memory efficiency
    
    # Paths
    data_dir: str = "./data"
    model_dir: str = "./models"
    log_dir: str = "./logs"
    
    # Safety thresholds - Per pathology
    ood_threshold: float = 0.5
    failure_detection_threshold: float = 0.6
    
    # Pathology-specific thresholds (can be tuned per observation)
    pathology_thresholds: Dict[str, float] = None  # Will be set in __post_init__
    
    # Evaluation metrics for multi-label
    eval_metrics: List[str] = None  # Will be set in __post_init__
    
    # Demo settings
    demo_port: int = 8501
    demo_title: str = "Medical AI Safety Demo - CheXpert Multi-Label"
    
    # Dataset options (try in order of preference)
    dataset_options: List[str] = None
    
    # Safety-first specific settings
    enable_safety_mode: bool = True
    safety_override_threshold: float = 0.9  # High confidence needed to override safety
    human_review_threshold: float = 0.3     # Low confidence triggers human review
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        assert self.train_split + self.val_split + self.test_split == 1.0
        assert 0 < self.learning_rate < 1
        assert self.batch_size > 0
        
        # Set CheXpert observations
        self.chexpert_observations = [
            "No Finding",
            "Enlarged Cardiomediastinum", 
            "Cardiomegaly",
            "Lung Opacity", 
            "Lung Lesion",
            "Edema",
            "Consolidation",
            "Pneumonia", 
            "Atelectasis",
            "Pneumothorax",
            "Pleural Effusion",
            "Pleural Other",
            "Fracture",
            "Support Devices"
        ]
        
        # Ensure num_classes matches CheXpert observations
        assert self.num_classes == len(self.chexpert_observations)
        
        # Set default pathology thresholds (can be tuned)
        if self.pathology_thresholds is None:
            self.pathology_thresholds = {
                obs: self.multi_label_threshold for obs in self.chexpert_observations
            }
        
        # Set evaluation metrics for multi-label classification
        if self.eval_metrics is None:
            self.eval_metrics = [
                "accuracy", "precision", "recall", "f1",
                "auc_roc", "auc_pr", "hamming_loss",
                "subset_accuracy", "micro_f1", "macro_f1"
            ]
        
        # Set class weights for imbalanced dataset (optional)
        if self.class_weights is None:
            # These can be computed from your dataset statistics
            # Based on your output, Support Devices is very common (~56%), 
            # while some pathologies are rare
            self.class_weights = {
                "No Finding": 1.0,
                "Enlarged Cardiomediastinum": 5.0,  # Very rare
                "Cardiomegaly": 2.0,
                "Lung Opacity": 1.0,
                "Lung Lesion": 3.0,  # Rare but important
                "Edema": 1.5,
                "Consolidation": 2.0,
                "Pneumonia": 2.0,  # Important to catch
                "Atelectasis": 1.0,
                "Pneumothorax": 2.5,  # Critical to catch
                "Pleural Effusion": 1.0,
                "Pleural Other": 5.0,  # Very rare
                "Fracture": 3.0,
                "Support Devices": 0.7  # Very common, lower weight
            }
        
        # Create directories if they don't exist
        import os
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Validate uncertain label strategy
        valid_strategies = ["positive", "negative", "ignore", "separate_class"]
        assert self.uncertain_label_strategy in valid_strategies, \
            f"uncertain_label_strategy must be one of {valid_strategies}"
        
        # Adjust num_classes if treating uncertain as separate class
        if self.uncertain_label_strategy == "separate_class":
            # Each observation becomes 3-class: negative, uncertain, positive
            self.num_classes = len(self.chexpert_observations) * 3

    def get_pathology_index(self, pathology_name: str) -> int:
        """Get the index of a pathology in the label tensor"""
        try:
            return self.chexpert_observations.index(pathology_name)
        except ValueError:
            raise ValueError(f"Pathology '{pathology_name}' not found in CheXpert observations")
    
    def get_class_weight_tensor(self):
        """Get class weights as a tensor for loss computation"""
        import torch
        weights = [self.class_weights[obs] for obs in self.chexpert_observations]
        return torch.tensor(weights, dtype=torch.float32, device=self.device)
    
    def print_config_summary(self):
        """Print a summary of the configuration"""
        print("=" * 60)
        print("MEDICAL AI SAFETY CONFIGURATION SUMMARY")
        print("=" * 60)
        print(f"📊 Dataset: {self.dataset_name}")
        print(f"📏 Subset Size: {self.subset_size:,}")
        print(f"🏷️  Classification: {self.num_classes}-class multi-label")
        print(f"🎯 Observations: {', '.join(self.chexpert_observations[:3])}... (+{len(self.chexpert_observations)-3} more)")
        print(f"🧠 Vision Model: {self.vision_model}")
        print(f"📝 Text Model: {self.text_model}")
        print(f"⚙️  Batch Size: {self.batch_size}")
        print(f"🔧 Device: {self.device}")
        print(f"🛡️  Safety Mode: {'Enabled' if self.enable_safety_mode else 'Disabled'}")
        print(f"📈 Mixed Precision: {'Enabled' if self.mixed_precision else 'Disabled'}")
        print(f"🎲 Uncertain Labels: Treated as '{self.uncertain_label_strategy}'")
        print("=" * 60)

# Global config instance
config = Config()

def get_config():
    """Get the global configuration"""
    return config

def update_config(**kwargs):
    """Update configuration parameters"""
    global config
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            print(f"Warning: {key} is not a valid configuration parameter")

def setup_for_multi_label():
    """Setup configuration specifically for multi-label classification"""
    update_config(
        use_multi_label=True,
        loss_function="bce_with_logits",
        learning_rate=3e-5,
        num_epochs=25
    )
    print("✅ Configuration updated for multi-label classification")

def setup_for_binary():
    """Setup configuration for binary normal/abnormal classification"""
    update_config(
        use_multi_label=False,
        num_classes=2,
        loss_function="cross_entropy",
        learning_rate=5e-5,
        num_epochs=20
    )
    print("✅ Configuration updated for binary classification")

if __name__ == "__main__":
    config.print_config_summary()
    
    print(f"\n🔍 Example pathology indices:")
    print(f"   Pneumonia: {config.get_pathology_index('Pneumonia')}")
    print(f"   Cardiomegaly: {config.get_pathology_index('Cardiomegaly')}")
    
    print(f"\n⚖️  Class weights tensor shape: {config.get_class_weight_tensor().shape}")
    print(f"   Sample weights: {config.get_class_weight_tensor()[:5].tolist()}")