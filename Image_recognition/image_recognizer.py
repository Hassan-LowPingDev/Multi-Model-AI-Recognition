import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image


class ImageCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 5))

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

classes = ["airplane", "automobile", "bird", "cat", "dog"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ImageCNN().to(device)
checkpoint = torch.load("image_recognition_model.pth", map_location=device, weights_only=True)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print("Trained model loaded successfully!")