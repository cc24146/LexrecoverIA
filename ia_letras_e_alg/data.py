# pega os dados
import torchvision
import torchvision.transforms as transforms
import torch
from PIL import Image  # Import necessário para a transposição

def get_data():
    transform = transforms.Compose([
        transforms.Lambda(lambda img: img.transpose(Image.TRANSPOSE)),
        transforms.ToTensor()
    ])

    # Mudamos para EMNIST e adicionamos o split="balanced"
    train_dataset = torchvision.datasets.EMNIST(
        root="./data",
        split="balanced",
        train=True,
        transform=transform,
        download=True
    )

    test_dataset = torchvision.datasets.EMNIST(
        root="./data",
        split="balanced",
        train=False,
        transform=transform,
        download=True
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64)

    return train_loader, test_loader