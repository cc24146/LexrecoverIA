import torch

from crnn_ctc.model import CRNNCTC
from crnn_ctc.config import NUM_CLASSES


model = CRNNCTC()


imagem = torch.randn(
    1,
    1,
    32,
    160
)


saida = model(
    imagem
)


print(
    "Formato da entrada:",
    imagem.shape
)

print(
    "Formato da saída:",
    saida.shape
)

print(
    "Número de classes:",
    NUM_CLASSES
)