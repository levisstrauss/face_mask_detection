import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
import torch.nn as nn
from torchvision.models import ResNet18_Weights
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MaskDetector:
    def __init__(self, model_path):
        logger.info("Initializing MaskDetector...")
        self.device = torch.device('cpu')
        self.model = self.load_model(model_path)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def load_model(self, model_path):
        try:
            model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
            model.eval()
            
            num_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Linear(num_features, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 2)
            )
            
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            return model
        except Exception as e:
            logger.error(f"Error in model loading: {str(e)}")
            raise

    def predict_image(self, image):
        try:
            if isinstance(image, str):
                image = Image.open(image).convert('RGB')
            elif not isinstance(image, Image.Image):
                raise ValueError("Input must be a PIL Image or path to image")

            transformed_image = self.transform(image).unsqueeze(0)
            
            with torch.no_grad():
                outputs = self.model(transformed_image)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                confidence, prediction = probs.max(1)

            return {
                'prediction': 'Mask' if prediction.item() == 0 else 'No Mask',
                'confidence': float(confidence.item()) * 100,
                'status': 'HIGH_CONFIDENCE' if confidence.item() >= 0.9 else 'LOW_CONFIDENCE'
            }
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            raise