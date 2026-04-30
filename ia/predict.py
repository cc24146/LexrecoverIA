import torch
from PIL import Image
import torchvision.transforms as transforms
from model import CNN

model = CNN()
model.load_state_dict(torch.load("model.pth"))
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

image = Image.open("digit.png")

image = transform(image)

image = image.unsqueeze(0)

with torch.no_grad():
    output = model(image)
    predicted = output.argmax(dim=1).item()

print("Número previsto:", predicted)