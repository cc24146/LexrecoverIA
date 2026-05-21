# pega os dados
import torchvision
import torchvision.transforms as transforms
import torch

def get_data():
    transform = transforms.ToTensor()   # converte a imagem de pixels para binário

    train_dataset = torchvision.datasets.MNIST(
        root="./data",
        train=True,
        transform=transform,
        download=True
    )   # baixa os dados do dataset

    test_dataset = torchvision.datasets.MNIST(
        root="./data",
        train=False,
        transform=transform
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True) # pega 64 imagens por vez e embaralha elas para a ia não decorar a ordem
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64)

    return train_loader, test_loader