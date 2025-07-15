import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from torchvision import transforms
from PIL import Image
from typing import Dict, Any, Tuple, List, Optional
from datasets import load_dataset
import random
import numpy as np
import re

class CheXpertLabeler:
    """
    CheXpert labeling implementation for generating structured labels
    """
    
    def __init__(self):
        self.observations = [
            "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
            "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
            "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
            "Pleural Other", "Fracture", "Support Devices"
        ]
        
        # Simplified mention patterns for key observations
        self.mention_patterns = {
            "No Finding": [
                r"no\s+(?:acute\s+)?(?:cardiopulmonary\s+)?(?:abnormalit|finding|disease)",
                r"normal\s+(?:study|exam|chest|heart|lungs)",
                r"unremarkable", r"clear\s+lungs?", r"negative\s+(?:study|exam)"
            ],
            "Cardiomegaly": [
                r"cardiomegaly", r"enlarged\s+heart", r"cardiac\s+enlargement",
                r"heart\s+size\s+(?:is\s+)?enlarged", r"increased\s+heart\s+size"
            ],
            "Lung Opacity": [
                r"opacity", r"opacities", r"infiltrate", r"infiltration",
                r"dense", r"density", r"densities"
            ],
            "Lung Lesion": [
                r"lesion", r"mass", r"masses", r"nodule", r"nodules",
                r"tumor", r"neoplasm", r"metastas"
            ],
            "Edema": [
                r"edema", r"pulmonary\s+edema", r"fluid\s+overload",
                r"vascular\s+congestion", r"congestion"
            ],
            "Consolidation": [r"consolidation", r"consolidate", r"consolidating"],
            "Pneumonia": [r"pneumonia", r"pneumonic", r"infection"],
            "Atelectasis": [r"atelectasis", r"atelectatic", r"collapse", r"volume\s+loss"],
            "Pneumothorax": [r"pneumothorax"],
            "Pleural Effusion": [r"pleural\s+effusion", r"effusion", r"pleural\s+fluid"],
            "Fracture": [r"fracture", r"fractured", r"break", r"broken"],
            "Support Devices": [
                r"tube", r"line", r"catheter", r"pacemaker", r"device",
                r"clip", r"clips", r"wire", r"lead", r"stent"
            ]
        }
        
        self.negation_patterns = [
            r"no\s+(?:evidence\s+of\s+)?(?:acute\s+)?", r"absent", r"negative\s+for",
            r"rule\s+out", r"without", r"free\s+of", r"clear\s+of"
        ]
        
        self.uncertainty_patterns = [
            r"possible", r"likely", r"probable", r"suggest", r"suspicious",
            r"concern(?:ing)?(?:\s+for)?", r"may\s+(?:be|represent)",
            r"cannot\s+(?:rule\s+out|exclude)", r"borderline", r"compatible\s+with"
        ]

    def preprocess_text(self, text: str) -> str:
        if not text:
            return ""
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\.]', ' ', text)
        return text.strip()

    def label_report(self, text: str) -> Dict[str, Optional[int]]:
        """Generate CheXpert labels for a report"""
        text = self.preprocess_text(text)
        labels = {}
        
        for observation in self.observations:
            labels[observation] = None  # Default: blank
            
            patterns = self.mention_patterns.get(observation, [])
            
            for pattern in patterns:
                matches = list(re.finditer(pattern, text, re.IGNORECASE))
                
                if matches:
                    # Check context for negation/uncertainty
                    for match in matches:
                        start = match.start()
                        context_start = max(0, start - 50)
                        context_end = min(len(text), match.end() + 30)
                        
                        preceding = text[context_start:start]
                        context = text[context_start:context_end]
                        
                        # Check uncertainty
                        is_uncertain = any(re.search(up, context, re.IGNORECASE) 
                                         for up in self.uncertainty_patterns)
                        if is_uncertain:
                            labels[observation] = -1
                            break
                        
                        # Check negation
                        is_negated = any(re.search(np, preceding, re.IGNORECASE) 
                                       for np in self.negation_patterns)
                        if is_negated:
                            labels[observation] = 0
                            break
                        
                        # Default to positive
                        labels[observation] = 1
                        break
                    break
        
        # Special handling for "No Finding"
        other_positives = sum(1 for obs, label in labels.items() 
                            if obs != "No Finding" and label == 1)
        
        if labels["No Finding"] is None and other_positives == 0:
            no_finding_patterns = [
                r"no\s+acute\s+cardiopulmonary\s+process",
                r"no\s+acute\s+findings?", r"normal\s+(?:study|exam|chest)"
            ]
            if any(re.search(pattern, text) for pattern in no_finding_patterns):
                labels["No Finding"] = 1
        
        return labels


class MIMICCXRDataset(Dataset):
    """Enhanced Dataset class for MIMIC-CXR images and reports with CheXpert labels"""
    
    def __init__(self, split: str = "train"):
        self.config = get_config()
        self.split = split
        
        print(f"Loading MIMIC-CXR dataset ({split} split)...")
        
        # Load full dataset and create balanced subset
        full_dataset = load_dataset(self.config.dataset_name, split="train")
        
        # For better balance, sample from different parts of the dataset
        total_size = len(full_dataset)
        subset_size = min(self.config.subset_size, total_size)
        
        random.seed(42)  # For reproducibility
        
        # Sample from different segments to ensure diversity
        segment_size = total_size // 10  # Divide dataset into 10 segments
        indices = []
        samples_per_segment = subset_size // 10
        
        for i in range(10):
            start_idx = i * segment_size
            end_idx = min((i + 1) * segment_size, total_size)
            segment_indices = list(range(start_idx, end_idx))
            
            # Sample from this segment
            segment_sample_size = min(samples_per_segment, len(segment_indices))
            if segment_sample_size > 0:
                selected = random.sample(segment_indices, segment_sample_size)
                indices.extend(selected)
        
        # If we need more samples, fill from remaining
        remaining_needed = subset_size - len(indices)
        if remaining_needed > 0:
            all_indices = set(range(total_size))
            used_indices = set(indices)
            remaining_indices = list(all_indices - used_indices)
            if remaining_indices:
                additional = random.sample(remaining_indices, min(remaining_needed, len(remaining_indices)))
                indices.extend(additional)
        
        # Select the balanced subset
        self.dataset = full_dataset.select(indices[:subset_size])
        print(f"✅ Successfully loaded {self.config.dataset_name} with stratified sampling")
        print(f"   Selected {len(self.dataset)} samples from {total_size} total samples")
        
        # Initialize CheXpert labeler
        self.chexpert_labeler = CheXpertLabeler()
        print("🏷️  Initializing CheXpert labeler...")
        
        # Pre-compute CheXpert labels for the subset
        print("🔍 Generating CheXpert labels...")
        self._precompute_chexpert_labels()
        
        # Split data
        total_size = len(self.dataset)
        train_size = int(self.config.train_split * total_size)
        val_size = int(self.config.val_split * total_size)
        
        if split == "train":
            self.data = self.dataset.select(range(train_size))
        elif split == "val":
            self.data = self.dataset.select(range(train_size, train_size + val_size))
        else:  # test
            self.data = self.dataset.select(range(train_size + val_size, total_size))
        
        print(f"Loaded {len(self.data)} samples for {split} split")
        
        # Initialize tokenizer for clinical text
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.text_model)
        
        # Image transforms
        self.image_transforms = transforms.Compose([
            transforms.Resize(self.config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Print label statistics
        self._print_label_statistics()
    
    def _precompute_chexpert_labels(self):
        """Pre-compute CheXpert labels for all samples"""
        self.chexpert_labels = []
        
        for i, item in enumerate(self.dataset):
            if i % 1000 == 0:
                print(f"   Processing labels: {i}/{len(self.dataset)}")
            
            # Combine findings and impression
            combined_text = ""
            if item.get('findings'):
                combined_text += item['findings'] + " "
            if item.get('impression'):
                combined_text += item['impression']
            
            # Generate CheXpert labels
            chexpert_labels = self.chexpert_labeler.label_report(combined_text)
            self.chexpert_labels.append(chexpert_labels)
        
        print(f"✅ Generated CheXpert labels for {len(self.chexpert_labels)} samples")
    
    def _print_label_statistics(self):
        """Print statistics about the generated labels"""
        print("\n📊 CheXpert Label Statistics:")
        print("-" * 50)
        
        # Get indices for current split
        total_size = len(self.dataset)
        train_size = int(self.config.train_split * total_size)
        val_size = int(self.config.val_split * total_size)
        
        if self.split == "train":
            start_idx, end_idx = 0, train_size
        elif self.split == "val":
            start_idx, end_idx = train_size, train_size + val_size
        else:  # test
            start_idx, end_idx = train_size + val_size, total_size
        
        # Calculate statistics for current split
        split_labels = self.chexpert_labels[start_idx:end_idx]
        
        for obs in self.chexpert_labeler.observations:
            labels = [sample_labels.get(obs) for sample_labels in split_labels]
            
            blank = sum(1 for l in labels if l is None)
            negative = sum(1 for l in labels if l == 0)
            uncertain = sum(1 for l in labels if l == -1)
            positive = sum(1 for l in labels if l == 1)
            total = len(labels)
            
            print(f"{obs:25} | Pos: {positive:4d} ({positive/total:.1%}) | "
                  f"Neg: {negative:4d} ({negative/total:.1%}) | "
                  f"Unc: {uncertain:4d} ({uncertain/total:.1%}) | "
                  f"Blank: {blank:4d} ({blank/total:.1%})")
        
        print("-" * 50)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        
        # Calculate the global index for accessing pre-computed labels
        total_size = len(self.dataset)
        train_size = int(self.config.train_split * total_size)
        val_size = int(self.config.val_split * total_size)
        
        if self.split == "train":
            global_idx = idx
        elif self.split == "val":
            global_idx = train_size + idx
        else:  # test
            global_idx = train_size + val_size + idx
        
        # Get pre-computed CheXpert labels
        chexpert_labels = self.chexpert_labels[global_idx]
        
        # Process image
        image = item['image']
        if isinstance(image, str):  # If path string
            image = Image.open(image).convert('RGB')
        image_tensor = self.image_transforms(image)
        
        # Process text - combine findings and impression
        text_parts = []
        if item.get('findings'):
            text_parts.append(item['findings'])
        if item.get('impression'):
            text_parts.append(item['impression'])
        
        if text_parts:
            text = " ".join(text_parts)
        else:
            # Fallback text if no clinical text available
            text = "Chest radiograph examination. Medical imaging study for clinical evaluation."
        
        # Tokenize text
        text_encoding = self.tokenizer(
            text,
            max_length=self.config.max_text_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Create multi-label tensor from CheXpert labels (14 observations)
        chexpert_tensor = torch.zeros(len(self.chexpert_labeler.observations), dtype=torch.float32)
        for i, obs in enumerate(self.chexpert_labeler.observations):
            label_value = chexpert_labels.get(obs)
            if label_value == 1:  # Positive
                chexpert_tensor[i] = 1.0
            elif label_value == -1:  # Uncertain - treat as positive for training
                chexpert_tensor[i] = 1.0
            # else: 0.0 for negative or blank
        
        # Create binary classification label (any positive finding = abnormal)
        has_positive_finding = any(label == 1 for label in chexpert_labels.values() 
                                 if label is not None)
        no_finding_positive = chexpert_labels.get("No Finding") == 1
        
        if no_finding_positive and not has_positive_finding:
            binary_label = 0  # Normal
        elif has_positive_finding:
            binary_label = 1  # Abnormal
        else:
            # Fallback to text-based classification
            text_lower = text.lower()
            abnormal_keywords = [
                'pneumonia', 'effusion', 'consolidation', 'atelectasis',
                'pneumothorax', 'cardiomegaly', 'edema', 'opacity',
                'infiltrate', 'mass', 'nodule', 'fracture', 'abnormal'
            ]
            normal_keywords = [
                'normal', 'clear', 'no acute', 'no evidence', 'unremarkable'
            ]
            
            has_abnormal = any(keyword in text_lower for keyword in abnormal_keywords)
            has_normal = any(keyword in text_lower for keyword in normal_keywords)
            
            binary_label = 1 if has_abnormal and not has_normal else 0
        
        # Convert individual CheXpert labels to individual tensors for easy access
        individual_labels = {}
        for obs in self.chexpert_labeler.observations:
            label_value = chexpert_labels.get(obs)
            # Convert to binary (positive/uncertain = 1, negative/blank = 0)
            individual_labels[obs.lower().replace(' ', '_')] = torch.tensor(
                1.0 if label_value in [1, -1] else 0.0, dtype=torch.float32
            )
        
        return {
            'image': image_tensor,
            'input_ids': text_encoding['input_ids'].squeeze(),
            'attention_mask': text_encoding['attention_mask'].squeeze(),
            'binary_label': torch.tensor(binary_label, dtype=torch.long),
            'chexpert_labels': chexpert_tensor,  # 14-dim multi-label tensor
            'chexpert_raw': chexpert_labels,     # Raw CheXpert dict for analysis
            'text': text,  # Keep original text for visualization
            'path': item.get('path', f'sample_{idx}'),
            **individual_labels  # Individual observation labels
        }


def custom_collate_fn(batch):
    """Custom collate function to handle None values in chexpert_raw"""
    # Separate out the problematic chexpert_raw field
    chexpert_raw_list = [item.pop('chexpert_raw') for item in batch]
    
    # Use default collate for everything else
    from torch.utils.data._utils.collate import default_collate
    collated_batch = default_collate(batch)
    
    # Add back chexpert_raw as a list (don't try to tensor it)
    collated_batch['chexpert_raw'] = chexpert_raw_list
    
    return collated_batch


def create_data_loaders() -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test data loaders with CheXpert labels"""
    config = get_config()
    
    print("🚀 Creating enhanced data loaders with CheXpert labels...")
    
    # Create datasets
    train_dataset = MIMICCXRDataset("train")
    val_dataset = MIMICCXRDataset("val") 
    test_dataset = MIMICCXRDataset("test")
    
    # Create data loaders with custom collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True if config.device == "cuda" else False,
        collate_fn=custom_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True if config.device == "cuda" else False,
        collate_fn=custom_collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True if config.device == "cuda" else False,
        collate_fn=custom_collate_fn
    )
    
    print("✅ Data loaders created successfully!")
    return train_loader, val_loader, test_loader


def test_enhanced_data_loading():
    """Test the enhanced data loading functionality with CheXpert labels"""
    print("🧪 Testing enhanced data loading with CheXpert labels...")
    
    train_loader, val_loader, test_loader = create_data_loaders()
    
    # Test a batch
    batch = next(iter(train_loader))
    print(f"\n📦 Batch Information:")
    print(f"   Batch keys: {list(batch.keys())}")
    print(f"   Image shape: {batch['image'].shape}")
    print(f"   Input IDs shape: {batch['input_ids'].shape}")
    print(f"   Binary labels shape: {batch['binary_label'].shape}")
    print(f"   CheXpert labels shape: {batch['chexpert_labels'].shape}")
    
    print(f"\n📋 Sample Analysis:")
    print(f"   Sample text: {batch['text'][0][:200]}...")
    print(f"   Binary label: {batch['binary_label'][0].item()}")
    print(f"   CheXpert raw: {batch['chexpert_raw'][0]}")
    
    # Show which CheXpert observations are positive
    chexpert_observations = [
        "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
        "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
        "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
        "Pleural Other", "Fracture", "Support Devices"
    ]
    
    positive_obs = []
    for i, obs in enumerate(chexpert_observations):
        if batch['chexpert_labels'][0][i] > 0.5:
            positive_obs.append(obs)
    
    print(f"   Positive observations: {positive_obs}")
    
    # Test individual label access
    print(f"   Pneumonia label: {batch.get('pneumonia', [torch.tensor(0)])[0].item()}")
    print(f"   Cardiomegaly label: {batch.get('cardiomegaly', [torch.tensor(0)])[0].item()}")
    
    print("\n✅ Enhanced data loading test successful!")
    
    return train_loader, val_loader, test_loader


# Move Config class to module level to make it pickleable
class Config:
    dataset_name = "itsanmolgupta/mimic-cxr-dataset"
    subset_size = 10000
    train_split = 0.7
    val_split = 0.15
    # test_split = 0.15 (remaining)
    image_size = (224, 224)
    max_text_length = 512
    text_model = "emilyalsentzer/Bio_ClinicalBERT"
    batch_size = 16
    num_workers = 0  # Set to 0 for Windows compatibility
    device = "cuda" if torch.cuda.is_available() else "cpu"

def get_config():
    """Get configuration object"""
    return Config()


if __name__ == "__main__":
    test_enhanced_data_loading()