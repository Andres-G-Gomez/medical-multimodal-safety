"""
Multimodal medical model with uncertainty quantification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from transformers import AutoModel
import numpy as np
from typing import Dict, Tuple, Optional
from config import get_config

class CrossModalAttention(nn.Module):
    """Cross-attention mechanism between vision and text features"""
    
    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, vision_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        # Cross-attention: vision attends to text
        attended_features, attention_weights = self.multihead_attn(
            query=vision_features,
            key=text_features, 
            value=text_features
        )
        
        # Residual connection and normalization
        output = self.norm(vision_features + attended_features)
        return output, attention_weights

class UncertaintyHead(nn.Module):
    """Head for uncertainty estimation using learned variance"""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.mean_head = nn.Linear(hidden_dim, 1)
        self.var_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.mean_head(x)
        # Ensure positive variance using softplus
        variance = F.softplus(self.var_head(x)) + 1e-6
        return mean, variance

class MedicalMultimodalModel(nn.Module):
    """Main multimodal model with safety mechanisms"""
    
    def __init__(self):
        super().__init__()
        self.config = get_config()
        
        # Vision encoder
        self.vision_encoder = timm.create_model(
            self.config.vision_model,
            pretrained=True,
            num_classes=0  # Remove classification head
        )
        
        # Text encoder  
        self.text_encoder = AutoModel.from_pretrained(
            self.config.text_model,
            use_safetensors=True  # Force safetensors to avoid torch.load security issue
        )
        
        # Cross-modal fusion
        self.cross_attention = CrossModalAttention(self.config.hidden_dim)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim // 2, self.config.num_classes)
        )
        
        # Uncertainty estimation head
        self.uncertainty_head = UncertaintyHead(self.config.hidden_dim)
        
        # Feature projector to ensure same dimensionality
        vision_dim = self.vision_encoder.num_features
        text_dim = self.text_encoder.config.hidden_size
        
        self.vision_projector = nn.Linear(vision_dim, self.config.hidden_dim)
        self.text_projector = nn.Linear(text_dim, self.config.hidden_dim)
        
    def forward(self, images: torch.Tensor, input_ids: torch.Tensor, 
                attention_mask: torch.Tensor, return_attention: bool = False) -> Dict[str, torch.Tensor]:
        
        # Extract vision features
        vision_features = self.vision_encoder(images)  # [batch_size, vision_dim]
        vision_features = self.vision_projector(vision_features)  # [batch_size, hidden_dim]
        vision_features = vision_features.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        # Extract text features
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state  # [batch_size, seq_len, text_dim]
        text_features = self.text_projector(text_features)  # [batch_size, seq_len, hidden_dim]
        
        # Cross-modal attention
        fused_features, attention_weights = self.cross_attention(vision_features, text_features)
        
        # Global pooling
        pooled_features = fused_features.mean(dim=1)  # [batch_size, hidden_dim]
        
        # Classification
        logits = self.classifier(pooled_features)
        
        # Uncertainty estimation
        uncertainty_mean, uncertainty_var = self.uncertainty_head(pooled_features)
        
        outputs = {
            'logits': logits,
            'uncertainty_mean': uncertainty_mean,
            'uncertainty_var': uncertainty_var,
            'features': pooled_features
        }
        
        if return_attention:
            outputs['attention_weights'] = attention_weights
            
        return outputs
    
    def predict_with_uncertainty(self, images: torch.Tensor, input_ids: torch.Tensor, 
                                attention_mask: torch.Tensor, num_samples: int = None) -> Dict[str, torch.Tensor]:
        """Monte Carlo Dropout for uncertainty estimation"""
        if num_samples is None:
            num_samples = self.config.mc_dropout_samples
            
        self.train()  # Enable dropout during inference
        
        predictions = []
        uncertainties = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                outputs = self.forward(images, input_ids, attention_mask)
                predictions.append(F.softmax(outputs['logits'], dim=-1))
                uncertainties.append(outputs['uncertainty_var'])
        
        # Stack predictions
        predictions = torch.stack(predictions)  # [num_samples, batch_size, num_classes]
        uncertainties = torch.stack(uncertainties)  # [num_samples, batch_size, 1]
        
        # Calculate epistemic uncertainty (variance across predictions)
        mean_prediction = predictions.mean(dim=0)
        epistemic_uncertainty = predictions.var(dim=0).mean(dim=-1, keepdim=True)
        
        # Calculate aleatoric uncertainty (average of learned variances)
        aleatoric_uncertainty = uncertainties.mean(dim=0)
        
        # Total uncertainty
        total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
        
        return {
            'predictions': mean_prediction,
            'epistemic_uncertainty': epistemic_uncertainty,
            'aleatoric_uncertainty': aleatoric_uncertainty,
            'total_uncertainty': total_uncertainty
        }
    
    def get_safety_assessment(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Assess safety based on predictions and uncertainty"""
        predictions = F.softmax(outputs['logits'], dim=-1)
        max_confidence = predictions.max(dim=-1)[0]
        
        # Simple uncertainty proxy (can be enhanced)
        uncertainty_proxy = outputs['uncertainty_var'].squeeze()
        
        # Safety flags
        high_uncertainty = uncertainty_proxy > self.config.uncertainty_threshold
        low_confidence = max_confidence < self.config.confidence_threshold
        
        safety_score = 1.0 - (uncertainty_proxy + (1 - max_confidence)) / 2
        
        return {
            'confidence': max_confidence,
            'uncertainty': uncertainty_proxy,
            'high_uncertainty': high_uncertainty,
            'low_confidence': low_confidence,
            'safety_score': safety_score,
            'requires_review': high_uncertainty | low_confidence
        }

def create_model() -> MedicalMultimodalModel:
    """Create and initialize the model"""
    model = MedicalMultimodalModel()
    config = get_config()
    
    # Move to device
    model = model.to(config.device)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Model moved to device: {config.device}")
    
    return model

def test_model():
    """Test model functionality"""
    print("Testing model...")
    
    config = get_config()
    model = create_model()
    
    # Create dummy input
    batch_size = 2
    images = torch.randn(batch_size, 3, *config.image_size).to(config.device)
    input_ids = torch.randint(0, 1000, (batch_size, config.max_text_length)).to(config.device)
    attention_mask = torch.ones(batch_size, config.max_text_length).to(config.device)
    
    # Test forward pass
    with torch.no_grad():
        outputs = model(images, input_ids, attention_mask, return_attention=True)
        print(f"Logits shape: {outputs['logits'].shape}")
        print(f"Uncertainty mean shape: {outputs['uncertainty_mean'].shape}")
        print(f"Attention weights shape: {outputs['attention_weights'].shape}")
        
        # Test uncertainty estimation
        uncertainty_outputs = model.predict_with_uncertainty(images, input_ids, attention_mask, num_samples=10)
        print(f"Predictions shape: {uncertainty_outputs['predictions'].shape}")
        print(f"Total uncertainty shape: {uncertainty_outputs['total_uncertainty'].shape}")
        
        # Test safety assessment
        safety = model.get_safety_assessment(outputs)
        print(f"Safety score: {safety['safety_score']}")
        print(f"Requires review: {safety['requires_review']}")
    
    print("Model test successful!")

if __name__ == "__main__":
    test_model()