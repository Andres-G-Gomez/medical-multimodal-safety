"""
Streamlit demo for Medical Multimodal Safety System
"""

import streamlit as st
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pandas as pd
from typing import Dict, Any
import io
import base64

from config import get_config
from model import create_model, MedicalMultimodalModel
from utils import load_checkpoint
import os

# Page configuration
st.set_page_config(
    page_title="Medical AI Safety Demo",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_demo_model():
    """Load the trained model (cached for performance)"""
    config = get_config()
    model = create_model()
    
    # Try to load trained model, fallback to untrained
    checkpoint_path = os.path.join(config.model_dir, 'best_model.pt')
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = load_checkpoint(checkpoint_path, config.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            st.success("✅ Loaded trained model")
        except Exception as e:
            st.warning(f"⚠️ Using untrained model: {e}")
    else:
        st.info("ℹ️ Using untrained model (train the model first for better results)")
    
    model.eval()
    return model

def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Preprocess uploaded image for model input"""
    from torchvision import transforms
    
    config = get_config()
    
    transform = transforms.Compose([
        transforms.Resize(config.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Apply transforms and add batch dimension
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor

def preprocess_text(text: str) -> Dict[str, torch.Tensor]:
    """Preprocess clinical text for model input"""
    from transformers import AutoTokenizer
    
    config = get_config()
    tokenizer = AutoTokenizer.from_pretrained(config.text_model)
    
    encoding = tokenizer(
        text,
        max_length=config.max_text_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    return {
        'input_ids': encoding['input_ids'],
        'attention_mask': encoding['attention_mask']
    }

def predict_with_safety(model: MedicalMultimodalModel, image: torch.Tensor, 
                       text_encoding: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """Make prediction with comprehensive safety assessment"""
    config = get_config()
    device = torch.device(config.device)
    
    # Move to device
    image = image.to(device)
    input_ids = text_encoding['input_ids'].to(device)
    attention_mask = text_encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        # Get standard prediction
        outputs = model(image, input_ids, attention_mask, return_attention=True)
        
        # Get uncertainty estimation
        uncertainty_outputs = model.predict_with_uncertainty(
            image, input_ids, attention_mask, num_samples=50
        )
        
        # Get safety assessment
        safety_assessment = model.get_safety_assessment(outputs)
        
        # Process results
        probabilities = torch.softmax(outputs['logits'], dim=-1).cpu().numpy()[0]
        predicted_class = probabilities.argmax()
        confidence = probabilities.max()
        
        # Extract uncertainties
        epistemic_unc = uncertainty_outputs['epistemic_uncertainty'].cpu().numpy()[0][0]
        aleatoric_unc = uncertainty_outputs['aleatoric_uncertainty'].cpu().numpy()[0][0]
        total_unc = uncertainty_outputs['total_uncertainty'].cpu().numpy()[0][0]
        
        # Extract safety metrics
        safety_score = safety_assessment['safety_score'].cpu().numpy()[0]
        requires_review = safety_assessment['requires_review'].cpu().numpy()[0]
        
        # Attention weights for visualization
        attention_weights = outputs['attention_weights'].cpu().numpy()[0]
        
        return {
            'prediction': {
                'class': int(predicted_class),
                'class_name': 'Abnormal' if predicted_class == 1 else 'Normal',
                'confidence': float(confidence),
                'probabilities': probabilities.tolist()
            },
            'uncertainty': {
                'epistemic': float(epistemic_unc),
                'aleatoric': float(aleatoric_unc),
                'total': float(total_unc)
            },
            'safety': {
                'safety_score': float(safety_score),
                'requires_review': bool(requires_review),
                'recommendation': get_safety_recommendation(safety_score, total_unc, confidence)
            },
            'attention_weights': attention_weights
        }

def get_safety_recommendation(safety_score: float, uncertainty: float, confidence: float) -> str:
    """Generate safety recommendation based on metrics"""
    if safety_score > 0.8 and uncertainty < 0.3 and confidence > 0.9:
        return "✅ HIGH CONFIDENCE - Safe for automated decision"
    elif safety_score > 0.6 and uncertainty < 0.5:
        return "⚠️ MODERATE CONFIDENCE - Consider expert review"
    elif safety_score > 0.4:
        return "🔶 LOW CONFIDENCE - Recommend expert review"
    else:
        return "🚨 VERY LOW CONFIDENCE - Requires immediate expert review"

def create_uncertainty_plot(uncertainty_data: Dict[str, float]) -> plt.Figure:
    """Create uncertainty breakdown visualization"""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    uncertainties = ['Epistemic\n(Model)', 'Aleatoric\n(Data)', 'Total']
    values = [
        uncertainty_data['epistemic'],
        uncertainty_data['aleatoric'], 
        uncertainty_data['total']
    ]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    bars = ax.bar(uncertainties, values, color=colors, alpha=0.8)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel('Uncertainty Score', fontsize=12)
    ax.set_title('Uncertainty Breakdown', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(values) * 1.2)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_confidence_gauge(confidence: float, safety_score: float) -> plt.Figure:
    """Create confidence and safety gauge visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Confidence gauge
    theta = np.linspace(0, np.pi, 100)
    r = 1
    
    ax1.plot(r * np.cos(theta), r * np.sin(theta), 'k-', linewidth=3)
    ax1.fill_between(r * np.cos(theta), 0, r * np.sin(theta), alpha=0.1)
    
    # Add confidence needle
    conf_angle = np.pi * (1 - confidence)
    ax1.plot([0, r * np.cos(conf_angle)], [0, r * np.sin(conf_angle)], 
             'r-', linewidth=4, label=f'Confidence: {confidence:.1%}')
    
    ax1.set_xlim(-1.2, 1.2)
    ax1.set_ylim(-0.2, 1.2)
    ax1.set_aspect('equal')
    ax1.set_title('Model Confidence', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.axis('off')
    
    # Safety score gauge
    ax2.plot(r * np.cos(theta), r * np.sin(theta), 'k-', linewidth=3)
    ax2.fill_between(r * np.cos(theta), 0, r * np.sin(theta), alpha=0.1)
    
    # Add safety needle
    safety_angle = np.pi * (1 - safety_score)
    color = 'green' if safety_score > 0.7 else 'orange' if safety_score > 0.4 else 'red'
    ax2.plot([0, r * np.cos(safety_angle)], [0, r * np.sin(safety_angle)], 
             color=color, linewidth=4, label=f'Safety Score: {safety_score:.1%}')
    
    ax2.set_xlim(-1.2, 1.2)
    ax2.set_ylim(-0.2, 1.2)
    ax2.set_aspect('equal')
    ax2.set_title('Safety Assessment', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.axis('off')
    
    plt.tight_layout()
    return fig

def create_attention_heatmap(attention_weights: np.ndarray, text: str) -> plt.Figure:
    """Create attention visualization heatmap"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Take average across attention heads and first few tokens
    attention_avg = attention_weights.mean(axis=0)[0, :min(20, attention_weights.shape[-1])]
    
    # Create mock text tokens (in real implementation, would use actual tokenizer)
    text_tokens = text.split()[:len(attention_avg)]
    if len(text_tokens) < len(attention_avg):
        text_tokens.extend([f'token_{i}' for i in range(len(text_tokens), len(attention_avg))])
    
    # Create heatmap
    attention_data = attention_avg.reshape(1, -1)
    sns.heatmap(attention_data, xticklabels=text_tokens, yticklabels=['Image'],
                cmap='Blues', annot=False, cbar=True, ax=ax)
    
    ax.set_title('Cross-Modal Attention: Image ← Text', fontsize=14, fontweight='bold')
    ax.set_xlabel('Text Tokens', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    return fig

def main():
    """Main Streamlit app"""
    st.title("🏥 Medical AI Safety Demo")
    st.markdown("**Vision-Language Medical AI with Safety Mechanisms**")
    
    # Sidebar for configuration
    st.sidebar.header("🔧 Configuration")
    
    # Model loading
    with st.spinner("Loading model..."):
        model = load_demo_model()
    
    # Demo options
    demo_mode = st.sidebar.radio(
        "Demo Mode",
        ["Upload Files", "Example Cases", "Safety Analysis"]
    )
    
    if demo_mode == "Upload Files":
        st.header("📤 Upload Medical Data")
        
        # File upload section
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🖼️ Chest X-Ray Image")
            uploaded_image = st.file_uploader(
                "Upload chest X-ray image", 
                type=['png', 'jpg', 'jpeg'],
                help="Upload a chest X-ray image for analysis"
            )
            
            if uploaded_image is not None:
                image = Image.open(uploaded_image)
                st.image(image, caption="Uploaded X-ray", use_column_width=True)
        
        with col2:
            st.subheader("📝 Clinical Notes")
            clinical_text = st.text_area(
                "Enter clinical notes",
                value="Chest radiograph shows normal heart size and clear lung fields. No acute cardiopulmonary process.",
                height=200,
                help="Enter clinical notes or radiology report text"
            )
        
        # Analysis button
        if st.button("🔍 Analyze", type="primary") and uploaded_image is not None:
            with st.spinner("Analyzing..."):
                # Preprocess inputs
                image_tensor = preprocess_image(image)
                text_encoding = preprocess_text(clinical_text)
                
                # Make prediction
                results = predict_with_safety(model, image_tensor, text_encoding)
                
                # Display results
                st.header("📊 Analysis Results")
                
                # Main prediction
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Prediction", results['prediction']['class_name'])
                    st.metric("Confidence", f"{results['prediction']['confidence']:.1%}")
                
                with col2:
                    st.metric("Safety Score", f"{results['safety']['safety_score']:.1%}")
                    st.metric("Total Uncertainty", f"{results['uncertainty']['total']:.3f}")
                
                with col3:
                    if results['safety']['requires_review']:
                        st.error("⚠️ Requires Review")
                    else:
                        st.success("✅ Safe Decision")
                
                # Safety recommendation
                st.info(f"**Recommendation:** {results['safety']['recommendation']}")
                
                # Detailed visualizations
                st.subheader("📈 Detailed Analysis")
                
                tab1, tab2, tab3 = st.tabs(["Uncertainty Analysis", "Confidence Gauges", "Attention Visualization"])
                
                with tab1:
                    fig_uncertainty = create_uncertainty_plot(results['uncertainty'])
                    st.pyplot(fig_uncertainty)
                    
                    st.write("**Uncertainty Breakdown:**")
                    st.write(f"- **Epistemic (Model Uncertainty):** {results['uncertainty']['epistemic']:.3f}")
                    st.write(f"- **Aleatoric (Data Uncertainty):** {results['uncertainty']['aleatoric']:.3f}")
                    st.write(f"- **Total Uncertainty:** {results['uncertainty']['total']:.3f}")
                
                with tab2:
                    fig_gauges = create_confidence_gauge(
                        results['prediction']['confidence'],
                        results['safety']['safety_score']
                    )
                    st.pyplot(fig_gauges)
                
                with tab3:
                    fig_attention = create_attention_heatmap(
                        results['attention_weights'],
                        clinical_text
                    )
                    st.pyplot(fig_attention)
                    st.write("Heatmap shows which text tokens the model focuses on when analyzing the image.")
    
    elif demo_mode == "Example Cases":
        st.header("📋 Example Cases")
        st.write("Explore pre-defined cases demonstrating different safety scenarios")
        
        example_cases = {
            "Normal Case (High Confidence)": {
                "description": "Clear normal chest X-ray with high model confidence",
                "image_path": "examples/normal_clear.jpg",
                "text": "Chest radiograph shows normal heart size and clear lung fields. No acute cardiopulmonary process.",
                "expected": "Normal with high confidence"
            },
            "Abnormal Case (High Uncertainty)": {
                "description": "Subtle abnormality that should trigger uncertainty",
                "image_path": "examples/subtle_abnormal.jpg", 
                "text": "Chest radiograph with possible subtle opacity in left lower lobe. Clinical correlation recommended.",
                "expected": "High uncertainty, requires review"
            },
            "Poor Quality Image": {
                "description": "Low quality image that should be flagged",
                "image_path": "examples/poor_quality.jpg",
                "text": "Chest radiograph, limited study due to patient motion and poor inspiration.",
                "expected": "Low confidence, requires review"
            }
        }
        
        selected_case = st.selectbox("Select example case:", list(example_cases.keys()))
        
        case = example_cases[selected_case]
        st.write(f"**Description:** {case['description']}")
        st.write(f"**Expected Outcome:** {case['expected']}")
        
        if st.button("Run Example", type="primary"):
            st.info("Example cases would run here with pre-loaded data")
            # In a real implementation, you would load the example images and run the analysis
    
    elif demo_mode == "Safety Analysis":
        st.header("🛡️ Safety Framework Analysis")
        
        st.write("""
        This medical AI system implements multiple safety mechanisms:
        
        ### 🎯 Uncertainty Quantification
        - **Epistemic Uncertainty**: Model's knowledge uncertainty
        - **Aleatoric Uncertainty**: Inherent data uncertainty  
        - **Monte Carlo Dropout**: Multiple forward passes for robust estimation
        
        ### 🔍 Failure Detection
        - **Out-of-distribution detection**: Identifies unusual inputs
        - **Confidence calibration**: Ensures probabilities are meaningful
        - **Adversarial robustness**: Detects corrupted or manipulated inputs
        
        ### 👥 Human-in-the-Loop Integration
        - **Safety thresholds**: Automatic flagging of uncertain cases
        - **Attention visualization**: Shows model reasoning
        - **Uncertainty communication**: Clear uncertainty reporting
        
        ### 📊 Safety Benchmarks
        - **Uncertainty-Risk Alignment**: How well uncertainty predicts errors
        - **Failure Detection Efficacy**: Ability to catch mistakes
        - **Distribution Shift Robustness**: Performance across different datasets
        - **Human-AI Collaboration**: Effectiveness of AI assistance
        """)
        
        # Show safety configuration
        st.subheader("⚙️ Current Safety Configuration")
        config = get_config()
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Uncertainty Threshold:** {config.uncertainty_threshold}")
            st.write(f"**Confidence Threshold:** {config.confidence_threshold}")
        with col2:
            st.write(f"**OOD Threshold:** {config.ood_threshold}")
            st.write(f"**MC Dropout Samples:** {config.mc_dropout_samples}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **⚠️ Important Notice:** This is a research prototype for demonstration purposes only. 
    Do not use for actual medical diagnosis. Always consult qualified healthcare professionals.
    """)

if __name__ == "__main__":
    main()