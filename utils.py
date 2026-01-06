"""
Olive Disease Detection - Utility Functions
Data validation, model evaluation, and visualization tools
"""

import os
import yaml
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
from collections import defaultdict
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import cv2
from tqdm import tqdm


class Config:
    """Load configuration from YAML file"""
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def __getitem__(self, key):
        return self.config.get(key)


class DataValidator:
    """Comprehensive dataset validation and analysis"""
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.class_counts = defaultdict(int)
        self.image_sizes = defaultdict(list)
        self.corrupted_images = []
    
    def validate_dataset(self):
        """Check dataset structure and integrity"""
        print("="*70)
        print("DATASET VALIDATION REPORT")
        print("="*70)
        
        if not self.dataset_path.exists():
            print(f"ERROR: Dataset path not found: {self.dataset_path}")
            return False
        
        # Check class directories
        class_dirs = [d for d in self.dataset_path.iterdir() if d.is_dir()]
        print(f"\nFound {len(class_dirs)} classes:")
        
        for class_dir in sorted(class_dirs):
            images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
            self.class_counts[class_dir.name] = len(images)
            print(f"  {class_dir.name}: {len(images)} images", end="")
            
            # Check for corrupted images
            corrupted = 0
            for img_path in images:
                try:
                    img = Image.open(img_path)
                    img.verify()
                    self.image_sizes[class_dir.name].append(img.size)
                except Exception as e:
                    corrupted += 1
                    self.corrupted_images.append((img_path, str(e)))
            
            if corrupted > 0:
                print(f" [{corrupted} CORRUPTED]")
            else:
                print(" [OK]")
        
        # Print statistics
        total_images = sum(self.class_counts.values())
        print(f"\nTotal Images: {total_images}")
        
        if len(self.corrupted_images) > 0:
            print(f"\n⚠️  WARNING: Found {len(self.corrupted_images)} corrupted images")
        else:
            print(f"\n✓ All images valid")
        
        # Check class balance
        print("\n" + "-"*70)
        print("CLASS BALANCE ANALYSIS")
        print("-"*70)
        
        min_count = min(self.class_counts.values()) if self.class_counts else 0
        max_count = max(self.class_counts.values()) if self.class_counts else 0
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        print(f"Min images per class: {min_count}")
        print(f"Max images per class: {max_count}")
        print(f"Imbalance ratio: {imbalance_ratio:.2f}:1")
        
        if imbalance_ratio > 3:
            print("⚠️  WARNING: High class imbalance detected!")
        else:
            print("✓ Reasonable class balance")
        
        # Image size analysis
        print("\n" + "-"*70)
        print("IMAGE SIZE ANALYSIS")
        print("-"*70)
        
        for class_name, sizes in self.image_sizes.items():
            if sizes:
                widths = [s[0] for s in sizes]
                heights = [s[1] for s in sizes]
                print(f"{class_name}:")
                print(f"  Avg size: {np.mean(widths):.0f}x{np.mean(heights):.0f}")
                print(f"  Min size: {min(widths)}x{min(heights)}")
                print(f"  Max size: {max(widths)}x{max(heights)}")
        
        print("\n" + "="*70)
        return True
    
    def visualize_class_distribution(self, save_path='results/class_distribution.png'):
        """Plot class distribution"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        classes = list(self.class_counts.keys())
        counts = list(self.class_counts.values())
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(classes)))
        bars = ax.bar(classes, counts, color=colors, edgecolor='black', linewidth=1.5)
        
        ax.set_xlabel('Disease Class', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Images', fontsize=12, fontweight='bold')
        ax.set_title('Olive Disease Dataset Distribution', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Distribution plot saved to {save_path}")
        plt.close()


class ModelEvaluator:
    """Comprehensive model evaluation and metrics"""
    def __init__(self, model, test_loader, class_names, device):
        self.model = model
        self.test_loader = test_loader
        self.class_names = class_names
        self.device = device
    
    def evaluate(self):
        """Full model evaluation"""
        print("\n" + "="*70)
        print("MODEL EVALUATION REPORT")
        print("="*70)
        
        all_preds = []
        all_targets = []
        all_probs = []
        
        self.model.eval()
        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc="Evaluating"):
                images = images.to(self.device)
                outputs = self.model(images)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                
                all_preds.extend(outputs.argmax(1).cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        all_probs = np.array(all_probs)
        
        # Classification report
        print("\nCLASSIFICATION REPORT:")
        print(classification_report(
            all_targets, all_preds,
            target_names=self.class_names,
            digits=4
        ))
        
        # Confusion matrix
        cm = confusion_matrix(all_targets, all_preds)
        self._plot_confusion_matrix(cm)
        
        # Per-class metrics
        print("\n" + "-"*70)
        print("PER-CLASS PERFORMANCE")
        print("-"*70)
        
        for i, class_name in enumerate(self.class_names):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"{class_name}:")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            print(f"  Support: {(all_targets == i).sum()}")
        
        return all_targets, all_preds, all_probs
    
    def _plot_confusion_matrix(self, cm, save_path='results/confusion_matrix.png'):
        """Plot and save confusion matrix"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   ax=ax, cbar_kws={'label': 'Count'})
        
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax.set_title('Confusion Matrix - Olive Disease Detection', 
                    fontsize=14, fontweight='bold')
        
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
        plt.close()


class AugmentationVisualizer:
    """Visualize data augmentation effects"""
    @staticmethod
    def visualize_augmentation(image_path, augmentation_transform, 
                              num_augmented=6, save_path='results/augmentation_demo.png'):
        """Show original and augmented versions of image"""
        from torchvision import transforms
        
        image = Image.open(image_path).convert('RGB')
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original image
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original Image', fontweight='bold', fontsize=12)
        axes[0, 0].axis('off')
        
        # Augmented versions
        for i in range(1, num_augmented):
            row = i // 3
            col = i % 3
            
            try:
                augmented = augmentation_transform(image)
                if isinstance(augmented, torch.Tensor):
                    # Denormalize and convert to PIL
                    augmented = augmented * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + \
                               torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                    augmented = augmented.clamp(0, 1)
                    augmented = transforms.ToPILImage()(augmented)
                
                axes[row, col].imshow(augmented)
                axes[row, col].set_title(f'Augmented v{i}', fontweight='bold', fontsize=12)
                axes[row, col].axis('off')
            except:
                pass
        
        plt.suptitle('Data Augmentation - Middle East Optimized', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Augmentation visualization saved to {save_path}")
        plt.close()


class MetricsVisualizer:
    """Visualize training metrics"""
    @staticmethod
    def plot_training_history(history_path, save_path='results/training_curves.png'):
        """Plot training and validation curves"""
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss curves
        axes[0].plot(history['epochs'], history['train_loss'], 'o-', label='Train', linewidth=2)
        axes[0].plot(history['epochs'], history['val_loss'], 's-', label='Validation', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Loss', fontsize=12, fontweight='bold')
        axes[0].set_title('Training & Validation Loss', fontsize=13, fontweight='bold')
        axes[0].legend(fontsize=11)
        axes[0].grid(alpha=0.3)
        
        # Accuracy curves
        axes[1].plot(history['epochs'], history['train_acc'], 'o-', label='Train', linewidth=2)
        axes[1].plot(history['epochs'], history['val_acc'], 's-', label='Validation', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        axes[1].set_title('Training & Validation Accuracy', fontsize=13, fontweight='bold')
        axes[1].legend(fontsize=11)
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
        plt.close()


def print_system_info():
    """Print system and model information"""
    print("\n" + "="*70)
    print("SYSTEM INFORMATION")
    print("="*70)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("="*70 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Olive Disease Detection - Utilities')
    parser.add_argument('--validate-data', type=str, default=None, help='Validate dataset at path')
    parser.add_argument('--plot-history', type=str, default='results/training_history.json', 
                       help='Plot training history')
    parser.add_argument('--system-info', action='store_true', help='Print system information')
    args = parser.parse_args()
    
    if args.system_info:
        print_system_info()
    
    if args.validate_data:
        validator = DataValidator(args.validate_data)
        validator.validate_dataset()
        validator.visualize_class_distribution()
    
    if os.path.exists(args.plot_history):
        MetricsVisualizer.plot_training_history(args.plot_history)
    
    if not any([args.system_info, args.validate_data, os.path.exists(args.plot_history)]):
        print("Please specify --validate-data, --plot-history, or --system-info")