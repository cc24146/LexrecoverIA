import matplotlib.pyplot as plt


def plot_history(train_losses, train_accs, val_losses, val_accs):

    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12,5))

    # -----------------------
    # Loss
    # -----------------------
    plt.subplot(1,2,1)

    plt.plot(epochs, train_losses, label="Treino")
    plt.plot(epochs, val_losses, label="Validação")

    plt.title("Loss")
    plt.xlabel("Épocas")
    plt.ylabel("Loss")

    plt.legend()

    # -----------------------
    # Accuracy
    # -----------------------
    plt.subplot(1,2,2)

    plt.plot(epochs, train_accs, label="Treino")
    plt.plot(epochs, val_accs, label="Validação")

    plt.title("Accuracy")

    plt.xlabel("Épocas")
    plt.ylabel("Accuracy (%)")

    plt.legend()

    plt.tight_layout()
    plt.savefig("training_history.png", dpi=300)
    plt.show()