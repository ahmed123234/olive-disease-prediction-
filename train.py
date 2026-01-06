"""
Olive Disease Detection - Complete Training Pipeline
Implements transfer learning with MobileNetV2/EfficientNet-B0
Optimized for Middle Eastern olive grove conditions
"""

import os
import yaml
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from datetime import datetime


class Config:
    """Load configuration from YAML file"""
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def __getitem__(self, key):
        return self.config.get(key)
    
    def __repr__(self):
        return json.dumps(self.config, indent=2)


class OliveDataAugmentation:
    """Custom data augmentation optimized for Middle Eastern olive conditions"""
    def __init__(self, cfg):
        self.cfg = cfg
        img_size = cfg['dataset']['image_size']
        
        # Training transforms with aggressive augmentation
        self.train_transforms = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(cfg['augmentation']['rotation_range']),
            transforms.ColorJitter(
                brightness=cfg['augmentation']['brightness_range'],
                contrast=cfg['augmentation']['contrast_range'],
                saturation=0.2,
                hue=0.1
            ),
            transforms.GaussianBlur(
                kernel_size=tuple(cfg['augmentation']['blur_kernel_range']),
                sigma=(0.1, 1.5)
            ),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Validation/Test transforms (minimal augmentation)
        self.val_transforms = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


class OliveModel(nn.Module):
    """Transfer Learning Model for Olive Disease Detection"""
    def __init__(self, cfg):
        super(OliveModel, self).__init__()
        num_classes = cfg['model']['num_classes']
        arch = cfg['model']['architecture']
        
        # Load backbone architecture
        if arch == "MobileNetV2":
            self.backbone = models.mobilenet_v2(pretrained=cfg['model']['pretrained'])
            num_features = self.backbone.classifier[1].in_features
            self.feature_extractor = self.backbone.features
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            
        elif arch == "EfficientNet-B0":
            self.backbone = models.efficientnet_b0(pretrained=cfg['model']['pretrained'])
            num_features = self.backbone.classifier[1].in_features
            self.feature_extractor = self.backbone.features
            self.avg_pool = self.backbone.avgpool
        else:
            raise ValueError(f"Unknown architecture: {arch}")
        
        # Freeze backbone layers if specified
        if cfg['model']['freeze_backbone']:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(cfg['model']['dropout_rate']),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg['model']['dropout_rate']),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg['model']['dropout_rate']),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.avg_pool(x)
        x = x.flatten(1)
        return self.classifier(x)


class OliveTrainer:
    """Complete training pipeline with validation and checkpointing"""
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.setup_directories()
        self.writer = SummaryWriter(cfg['logging']['log_dir'])
        
        print(f"\n{'='*70}")
        print(f"Training on device: {self.device}")
        print(f"{'='*70}\n")
    
    def setup_directories(self):
        """Create necessary directories"""
        for path_key in ['models', 'logs', 'results']:
            Path(self.cfg['paths'][path_key]).mkdir(parents=True, exist_ok=True)
    
    def load_data(self):
        """Load and split dataset"""
        aug = OliveDataAugmentation(self.cfg)
        
        print("Loading dataset...")
        dataset = datasets.ImageFolder(
            self.cfg['dataset']['path'],
            transform=aug.train_transforms
        )
        
        print(f"Found {len(dataset)} images")
        
        # Calculate split sizes
        train_size = int(len(dataset) * self.cfg['dataset']['train_test_split'])
        val_size = int(train_size * self.cfg['dataset']['val_split'])
        train_size = train_size - val_size
        test_size = len(dataset) - train_size - val_size
        
        train_ds, val_ds, test_ds = random_split(
            dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Apply appropriate transforms to each split
        for ds in [val_ds, test_ds]:
            ds.dataset.transform = aug.val_transforms
        
        self.train_loader = DataLoader(
            train_ds,
            batch_size=self.cfg['training']['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_ds,
            batch_size=self.cfg['validation']['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        self.test_loader = DataLoader(
            test_ds,
            batch_size=self.cfg['validation']['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        self.class_names = dataset.classes
        
        print(f"  Train: {len(train_ds)} samples")
        print(f"  Val: {len(val_ds)} samples")
        print(f"  Test: {len(test_ds)} samples")
        print(f"  Classes: {self.class_names}\n")
        
        return self.train_loader, self.val_loader, self.test_loader
    
    def train(self):
        """Main training loop with early stopping"""
        model = OliveModel(self.cfg).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.cfg['training']['learning_rate'],
            weight_decay=self.cfg['training']['weight_decay']
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=self.cfg['training']['epochs'])
        
        best_val_loss = float('inf')
        patience_counter = 0
        training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'epochs': []
        }
        
        print(f"\n{'='*70}")
        print("Starting Training...")
        print(f"{'='*70}\n")
        
        for epoch in range(self.cfg['training']['epochs']):
            # Training phase
            model.train()
            train_loss = 0.0
            train_preds, train_targets = [], []
            
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1:3d}/{self.cfg['training']['epochs']} Train")
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    self.cfg['training']['gradient_clipping']
                )
                optimizer.step()
                
                train_loss += loss.item()
                train_preds.extend(outputs.argmax(1).cpu().numpy())
                train_targets.extend(labels.cpu().numpy())
                
                pbar.set_postfix({'loss': loss.item()})
            
            # Validation phase
            val_loss, val_acc, val_preds, val_targets = self.validate(model, criterion)
            
            # Metrics
            train_loss /= len(self.train_loader)
            train_acc = accuracy_score(train_targets, train_preds)
            
            # Logging
            self.writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
            self.writer.add_scalars('Accuracy', {'train': train_acc, 'val': val_acc}, epoch)
            
            # Store history
            training_history['epochs'].append(epoch + 1)
            training_history['train_loss'].append(train_loss)
            training_history['val_loss'].append(val_loss)
            training_history['train_acc'].append(train_acc)
            training_history['val_acc'].append(val_acc)
            
            print(f"Epoch {epoch+1:3d}: Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f} | Val Acc={val_acc:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss - self.cfg['training']['early_stopping']['min_delta']:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), f"{self.cfg['paths']['models']}/best_model.pth")
                print(f"  ✓ Model saved (best validation loss: {best_val_loss:.4f})")
            else:
                patience_counter += 1
            
            if patience_counter >= self.cfg['training']['early_stopping']['patience']:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
            
            scheduler.step()
        
        # Save training history
        with open(f"{self.cfg['paths']['results']}/training_history.json", 'w') as f:
            json.dump(training_history, f, indent=4)
        
        self.writer.close()
        
        # Load best model and test
        print(f"\n{'='*70}")
        print("Loading best model and evaluating on test set...")
        print(f"{'='*70}\n")
        
        model.load_state_dict(torch.load(f"{self.cfg['paths']['models']}/best_model.pth"))
        self.evaluate_on_test(model)
        
        return model
    
    def validate(self, model, criterion):
        """Validation loop"""
        model.eval()
        val_loss = 0.0
        preds, targets = [], []
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                preds.extend(outputs.argmax(1).cpu().numpy())
                targets.extend(labels.cpu().numpy())
        
        val_loss /= len(self.val_loader)
        val_acc = accuracy_score(targets, preds)
        
        return val_loss, val_acc, preds, targets
    
    def evaluate_on_test(self, model):
        """Final evaluation on test set"""
        model.eval()
        all_preds, all_targets = [], []
        
        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                outputs = model(images)
                all_preds.extend(outputs.argmax(1).cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_preds)
        precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
        cm = confusion_matrix(all_targets, all_preds)
        
        print(f"Test Accuracy:  {accuracy:.4f}")
        print(f"Test Precision: {precision:.4f}")
        print(f"Test Recall:    {recall:.4f}")
        print(f"Test F1-Score:  {f1:.4f}")
        
        # Per-class metrics
        print(f"\nPer-Class Metrics:")
        for i, class_name in enumerate(self.class_names):
            class_mask = np.array(all_targets) == i
            if class_mask.sum() > 0:
                class_acc = accuracy_score(
                    np.array(all_targets)[class_mask],
                    np.array(all_preds)[class_mask]
                )
                print(f"  {class_name}: {class_acc:.4f}")
        
        # Save results
        results = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'confusion_matrix': cm.tolist(),
            'class_names': self.class_names
        }
        
        with open(f"{self.cfg['paths']['results']}/test_results.json", 'w') as f:
            json.dump(results, f, indent=4)
        
        # Plot confusion matrix
        self._plot_confusion_matrix(cm)
    
    def _plot_confusion_matrix(self, cm):
        """Plot and save confusion matrix"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=self.class_names,
               yticklabels=self.class_names,
               ylabel='True label',
               xlabel='Predicted label')
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
        
        fig.tight_layout()
        plt.savefig(f"{self.cfg['paths']['results']}/confusion_matrix.png", dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved")
        plt.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Olive Disease Detection Model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    cfg = Config(args.config)
    trainer = OliveTrainer(cfg)
    
    print("Loading dataset...")
    trainer.load_data()
    
    print("Starting training...")
    model = trainer.train()
    
    print("\n✓ Training complete!")