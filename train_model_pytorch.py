import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from pathlib import Path
import json
from tqdm import tqdm

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
DATA_DIR = Path('/app/backend/dataset')
MODEL_PATH = Path('/app/backend/food_model.pth')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Food categories
food_categories = [
    'Aloo_Gobi', 'Aloo_Paratha', 'Biryani', 'Butter_Chicken', 'Chole_Bhature',
    'Dal_Makhani', 'Dhokla', 'Dosa', 'Gulab_Jamun', 'Halwa',
    'Idli', 'Jalebi', 'Kadhi', 'Kheer', 'Masala_Dosa',
    'Medu_Vada', 'Mutter_Paneer', 'Naan', 'Paneer_Butter_Masala', 'Paneer_Tikka',
    'Pani_Puri', 'Papdi_Chaat', 'Pav_Bhaji', 'Poha', 'Pulao',
    'Rajma_Chawal', 'Rasgulla', 'Roti', 'Samosa', 'Sev_Puri',
    'Shahi_Paneer', 'Tandoori_Chicken', 'Upma', 'Vada_Pav', 'Veg_Biryani',
    'Veg_Pulao', 'Vegetable_Kurma', 'Vegetable_Soup'
]

class FoodClassifier(nn.Module):
    """Food classification model using MobileNetV2"""
    def __init__(self, num_classes):
        super(FoodClassifier, self).__init__()
        # Load pre-trained MobileNetV2
        self.base_model = models.mobilenet_v2(pretrained=True)
        
        # Freeze base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Replace classifier
        num_features = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.base_model(x)

def get_data_transforms():
    """Get data augmentation transforms"""
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), shear=0.2),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': running_loss / (pbar.n + 1),
            'acc': 100. * correct / total
        })
    
    return running_loss / len(dataloader), 100. * correct / total

def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': running_loss / (pbar.n + 1),
                'acc': 100. * correct / total
            })
    
    return running_loss / len(dataloader), 100. * correct / total

def train_model():
    """Train the food classification model"""
    print("Starting PyTorch model training...")
    
    # Check if dataset exists
    if not DATA_DIR.exists():
        print(f"Dataset not found at {DATA_DIR}")
        print("Please download the dataset from Kaggle first.")
        return False
    
    # Get transforms
    train_transform, val_transform = get_data_transforms()
    
    # Load datasets
    full_dataset = datasets.ImageFolder(DATA_DIR, transform=train_transform)
    
    # Split into train and validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Apply different transforms to validation
    val_dataset.dataset.transform = val_transform
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    num_classes = len(full_dataset.classes)
    print(f"Found {num_classes} classes")
    print(f"Training samples: {train_size}")
    print(f"Validation samples: {val_size}")
    
    # Save class indices
    class_to_idx = full_dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    with open('/app/backend/class_indices.json', 'w') as f:
        json.dump(idx_to_class, f)
    print("Class indices saved")
    
    # Create model
    model = FoodClassifier(num_classes).to(device)
    print("\nModel created successfully")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Training loop
    best_val_acc = 0.0
    patience = 5
    patience_counter = 0
    
    print("\nStarting training...\n")
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print("-" * 50)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'class_to_idx': class_to_idx
            }, MODEL_PATH)
            print(f"✅ Best model saved with Val Acc: {val_acc:.2f}%")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
        
        print()
    
    print("\n" + "="*50)
    print(f"Training completed! Best Val Acc: {best_val_acc:.2f}%")
    print(f"Model saved to {MODEL_PATH}")
    print("="*50)
    
    return True

if __name__ == '__main__':
    success = train_model()
    if success:
        print("\n✅ Training completed successfully!")
    else:
        print("\n❌ Training failed. Please check the error messages above.")