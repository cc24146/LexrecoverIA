import torch

from crnn_ctc.decoder import (
    greedy_decode
)

from crnn_ctc.config import (
    CHAR_TO_INDEX,
    NUM_CLASSES
)


sequencia = [
    CHAR_TO_INDEX["c"],
    0,
    CHAR_TO_INDEX["a"],
    0,
    CHAR_TO_INDEX["r"],
    0,
    CHAR_TO_INDEX["r"],
    0,
    CHAR_TO_INDEX["o"]
]

outputs = torch.full(
    (
        1,
        len(sequencia),
        NUM_CLASSES
    ),
    -10.0
)


for tempo, indice in enumerate(
    sequencia
):
    outputs[
        0,
        tempo,
        indice
    ] = 10.0


comprimentos = torch.tensor(
    [
        len(sequencia)
    ]
)


resultado = greedy_decode(
    outputs,
    comprimentos
)


print(
    resultado
)