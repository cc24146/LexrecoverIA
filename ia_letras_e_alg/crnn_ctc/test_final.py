import os

import torch
import torch.nn as nn

from crnn_ctc.model import CRNNCTC
from crnn_ctc.data import get_data
from crnn_ctc.train import avaliar


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

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "crnn_ctc",
    "final_model.pth"
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

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()


ctc_loss = nn.CTCLoss(
    blank=0,
    reduction="mean",
    zero_infinity=True
)


print()
print("==============================")
print("TESTE FINAL")
print("==============================")


test_loss, test_cer = avaliar(
    model,
    test_loader,
    ctc_loss,
    device
)


print()
print("==============================")
print("RESULTADO FINAL")
print("==============================")

print(
    f"Test Loss: "
    f"{test_loss:.4f}"
)

print(
    f"Test CER: "
    f"{test_cer * 100:.2f}%"
)