import os
import torch
from model import CRNN
from data import get_data
from train import train
from plot import plot_history

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(PROJECT_ROOT, "crnn", "model_last.pth")

model = CRNN()

train_loader, test_loader = get_data()

train_losses, train_accs, val_losses, val_accs = train(
    model,
    train_loader,
    test_loader
)
plot_history(
    train_losses,
    train_accs,
    val_losses,
    val_accs
)
torch.save(model.state_dict(), MODEL_PATH)