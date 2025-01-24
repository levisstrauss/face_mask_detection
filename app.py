import gradio as gr
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
import torch.nn as nn


class MaskDetector:
    def __init__(self, model_path):
        self.device = torch.device('cpu')
        self.model = self.load_model(model_path)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def load_model(self, model_path):
        model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model

    def predict(self, image):
        img = Image.fromarray(image).convert('RGB')
        img = self.transform(img).unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(img)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, prediction = probs.max(1)

        return "Mask" if prediction.item() == 0 else "No Mask"


detector = MaskDetector('models/face_mask_detector_final.pth')


def detect_mask(image):
    return detector.predict(image)


demo = gr.Interface(
    fn=detect_mask,
    inputs=gr.Image(),
    outputs="text"
)

demo.launch()
