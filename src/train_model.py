import json
import os
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def get_device() -> torch.device:
    """Return the best available device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_transforms(image_size: int = 224) -> transforms.Compose:
    """Create image transforms for Food-101 dataset."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def load_data(data_dir: str, batch_size: int = 32) -> Tuple[DataLoader, DataLoader, dict]:
    """Load Food-101 dataset with train and test splits."""
    transform = build_transforms()
    train_dataset = datasets.Food101(root=data_dir, split="train", download=True, transform=transform)
    test_dataset = datasets.Food101(root=data_dir, split="test", download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, test_loader, train_dataset.class_to_idx


def build_model(num_classes: int = 101, use_resnet: bool = False) -> nn.Module:
    """Load a pretrained EfficientNet_B0 (default) or ResNet50 model and replace the classifier head."""
    if use_resnet:
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        for param in model.parameters():
            param.requires_grad = False
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        for param in model.features.parameters():
            param.requires_grad = False
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    return model


def train_one_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, device: torch.device) -> float:
    model.train()
    running_loss = 0.0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    return running_loss / len(dataloader.dataset)


def evaluate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()

    total = len(dataloader.dataset)
    return running_loss / total, correct / total


def main():
    data_dir = os.environ.get("FOOD101_DATA", "data")
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")

    train_loader, test_loader, class_to_idx = load_data(data_dir)

    model = build_model(num_classes=101)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

    num_epochs = int(os.environ.get("EPOCHS", 3))
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")

    model_path = os.path.join(models_dir, "food_classifier.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Saved model weights to {model_path}")

    classes_path = os.path.join(models_dir, "classes.json")
    with open(classes_path, "w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, indent=2)
    print(f"Saved class mapping to {classes_path}")


if __name__ == "__main__":
    main()
