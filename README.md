# Medical Multimodal Safety System

A vision-language AI system for medical imaging analysis with robust safety mechanisms, uncertainty quantification, and failure detection for high-stakes medical decisions.

## 🎯 Project Overview

This project implements a multimodal AI system that analyzes medical images and clinical text with a focus on safety-first design principles. Unlike traditional medical AI that optimizes for accuracy, this system prioritizes uncertainty quantification, out-of-distribution detection, and graceful failure handling.

### Key Features

- **Multimodal Architecture**: Combines vision (chest X-rays) and language (clinical reports)
- **Uncertainty Quantification**: Monte Carlo dropout and ensemble methods
- **Safety Mechanisms**: Out-of-distribution detection and confidence thresholding
- **Failure Detection**: Automated detection of model failures and edge cases
- **Human-in-the-Loop**: Integration points for expert review

## 🚀 Quick Start

### Prerequisites

- Python 3.11+ (tested with Python 3.11)
- Git
- 6GB+ RAM
- NVIDIA GPU with CUDA support (recommended for training)

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/medical-multimodal-safety.git
cd medical-multimodal-safety
```

2. **Create virtual environment with Python 3.11**

```bash
# Windows
py -3.11 -m venv venv
venv\Scripts\activate

# Linux/Mac
python3.11 -m venv venv
source venv/bin/activate
```

3. **Install PyTorch with CUDA support (for GPU acceleration)**

```bash
# For NVIDIA GPU users (recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CPU-only users
pip install torch torchvision torchaudio
```

4. **Install remaining dependencies**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

5. **Verify GPU installation (optional)**

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

### Quick Demo

```bash
# Run the interactive demo
streamlit run demo.py
```

## 📊 Dataset

This project uses the MIMIC-CXR dataset:

- **Primary**: [HuggingFace processed version](https://huggingface.co/datasets/StanfordAIMI/mimic-cxr-images-512)
- **Size**: 30,633 chest X-rays with clinical reports
- **Labels**: Automatically generated CheXpert labels for 14 pathologies
- **Text**: Combined findings + impression reports

Label Distribution (Training Set)
| Pathology | Positive Rate | Clinical Importance |
|-----------|---------------|-------------------|
| Support Devices | 56.6% | Common (monitoring) |
| Lung Opacity | 28.5% | Moderate (investigation needed) |
| Atelectasis | 28.5% | Moderate (follow-up) |
| Pleural Effusion | 24.7% | High (treatment often needed) |
| Edema | 20.2% | High (cardiac/pulmonary concern) |
| No Finding | 16.5% | Normal studies |
| Cardiomegaly | 12.2% | High (cardiac evaluation) |
| Pneumonia | 9.5% | Critical (immediate treatment) |
| Pneumothorax | 6.2% | Emergency (life-threatening) |

## 🏗️ Architecture

```
Input: Chest X-ray (224×224) + Clinical Text (findings + impression)
    ↓
Vision Encoder (ViT) + Text Encoder (ClinicalBERT)
    ↓
Cross-Modal Attention Fusion (768-dim)
    ↓
Multi-Head Output:
├── CheXpert Classification (14-class multi-label)
├── Uncertainty Estimation (Epistemic + Aleatoric)
└── Safety Assessment (Human Review Flags)
```

## 🛡️ Safety Features

### 1. Uncertainty Quantification

- **Epistemic**: Model uncertainty via Monte Carlo dropout
- **Aleatoric**: Data uncertainty via learned variance
- **Calibration**: Temperature scaling for probability calibration

### 2. Safety Benchmarks

- Uncertainty-risk alignment correlation
- Failure detection efficacy (AUROC)
- Robustness under distribution shift
- Human-AI collaboration safety metrics

### 3. Automated Safety Flags

```
# Human review triggers
if confidence < 0.3:                              # Low confidence
    flag_for_review = True
if pathology == "Pneumothorax" and uncertainty > 0.4:  # Critical findings
    flag_for_review = True
if model_disagreement > 0.5:                      # High epistemic uncertainty
    flag_for_review = True
```

## 📁 Project Structure

```
medical-multimodal-safety/
├── config.py              # Enhanced configuration with CheXpert
├── data_loader.py          # Data loading with CheXpert labeling
├── train.py               # Multi-label training script
├── model.py               # Vision-language architecture
├── utils.py               # Logging and checkpointing utilities
├── requirements.txt       # Python dependencies
└── README.md             # Documentation
```

## 🔬 Usage

### Training

```bash
python train.py --config configs/default.yaml
```

### Evaluation

```bash
python safety_eval.py --model_path models/best_model.pt
```

### Demo

```bash
streamlit run demo.py
```

## 📈 Benchmarks

This project introduces novel safety benchmarks for medical AI:

1. **Uncertainty-Risk Alignment**: Correlation between model uncertainty and clinical risk
2. **Failure Detection Efficacy**: AUROC for detecting incorrect predictions
3. **Distribution Shift Robustness**: Performance across different hospitals/scanners
4. **Human-AI Collaboration Safety**: Decision accuracy with uncertainty estimates

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [MIMIC-CXR](https://physionet.org/content/mimic-cxr/) dataset from MIT-LCP
- [CheXpert labeler](https://github.com/stanfordmlgroup/chexpert-labeler) for extracting labels from radiology reports
- [ClinicalBERT](https://github.com/EmilyAlsentzer/clinicalBERT) for medical text encoding
- Stanford AIMI for preprocessed datasets

## 📞 Contact

- **Author**: Andres Gomez
- **Email**: andres.gab(dot)gomez AT gmail.com
- **Project Link**: https://github.com/Andres-G-Gomez/medical-multimodal-safety

---

**⚠️ Important**: This is a research prototype and should not be used for actual medical diagnosis. Always consult qualified healthcare professionals for medical decisions.
