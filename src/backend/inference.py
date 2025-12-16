import io
import json
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms


class CaloriePredictor:
    def __init__(self, models_dir: str = "models", calorie_lookup_path: str | None = None) -> None:
        self.root_dir = Path(__file__).resolve().parents[2]
        self.device = self._get_device()
        self.transform = self._build_transforms()

        models_path = self._resolve_path(models_dir)
        weights_path = models_path / "food_classifier.pth"
        classes_path = models_path / "classes.json"
        calories_path = (
            Path(calorie_lookup_path)
            if calorie_lookup_path
            else self.root_dir / "src" / "backend" / "calorie_lookup.json"
        )

        self.idx_to_class = self._load_class_mapping(classes_path)
        self.calorie_lookup = self._load_calorie_lookup(calories_path)

        self.model = self._build_model(num_classes=len(self.idx_to_class))
        state_dict = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def _resolve_path(self, path: str | Path) -> Path:
        path = Path(path)
        return path if path.is_absolute() else self.root_dir / path

    @staticmethod
    def _get_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @staticmethod
    def _build_transforms(image_size: int = 224) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _build_model(self, num_classes: int = 101) -> nn.Module:
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        for param in model.features.parameters():
            param.requires_grad = False
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    @staticmethod
    def _load_class_mapping(classes_path: Path) -> Dict[int, str]:
        with classes_path.open("r", encoding="utf-8") as f:
            class_to_idx = json.load(f)
        return {int(idx): cls for cls, idx in class_to_idx.items()}

    @staticmethod
    def _load_calorie_lookup(calories_path: Path) -> Dict[str, Dict[str, Any]]:
        with calories_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def predict(self, image_bytes: bytes) -> Dict[str, Any]:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)

        confidence, predicted_idx = torch.max(probabilities, dim=1)
        class_idx = predicted_idx.item()
        confidence_score = confidence.item()
        food_name = self.idx_to_class.get(class_idx, "unknown")
        calorie_info = self.calorie_lookup.get(food_name, {})

        return {
            "food_name": food_name,
            "calories": {
                "value": calorie_info.get("calories"),
                "unit": calorie_info.get("unit"),
            },
            "confidence": confidence_score,
        }


__all__ = ["CaloriePredictor"]
