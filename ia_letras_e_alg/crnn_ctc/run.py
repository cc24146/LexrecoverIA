import os
import torch

from crnn_ctc.model import CRNNCTC
from crnn_ctc.data import get_data
from crnn_ctc.train import train


PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

BRESSAY_ROOT = os.path.join(
    PROJECT_ROOT,
    "dataset_bressay"
)

BEST_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "crnn_ctc",
    "best_model.pth"
)


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    "Dispositivo:",
    device
)

print(
    "Arquivo run.py:",
    os.path.abspath(__file__)
)

print(
    "Raiz do projeto:",
    PROJECT_ROOT
)

print(
    "Procurando modelo em:",
    BEST_MODEL_PATH
)

print(
    "Modelo encontrado:",
    os.path.exists(BEST_MODEL_PATH)
)


(
    train_loader,
    val_loader,
    test_loader
) = get_data(
    BRESSAY_ROOT,
    batch_size=2
)


model = CRNNCTC().to(
    device
)


if not os.path.exists(
    BEST_MODEL_PATH
):
    raise FileNotFoundError(
        f"best_model.pth não encontrado em: "
        f"{BEST_MODEL_PATH}"
    )


model.load_state_dict(
    torch.load(
        BEST_MODEL_PATH,
        map_location=device
    )
)

print()
print(
    "Continuando treinamento a partir do best_model.pth"
)
print()


train(
    model,
    train_loader,
    val_loader,
    device,
    epochs=3
)