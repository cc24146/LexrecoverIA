from model import CNN
from data import get_data
from train import train, evaluate
import torch

model = CNN()

train_loader, test_loader = get_data()

train(model, train_loader)
evaluate(model, test_loader)
torch.save(model.state_dict(), "model.pth")