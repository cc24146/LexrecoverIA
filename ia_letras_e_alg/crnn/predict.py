import os
import torch
from PIL import Image
import torchvision.transforms as transforms
from model import CRNN

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(PROJECT_ROOT, "crnn", "best_model.pth")

model = CRNN()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

image = Image.open(os.path.join(PROJECT_ROOT, "digit.png"))
image = transform(image)
image = image.unsqueeze(0)

emnist_labels = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt"

with torch.no_grad():
    output = model(image)
    predicted = output.argmax(dim=1).item()

print("Caractere previsto:", emnist_labels[predicted])