from model import CNN
from data import get_data
from train import train, evaluate
import torch
from plot import plot_history

model = CNN()

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
torch.save(model.state_dict(), "model_last.pth")
