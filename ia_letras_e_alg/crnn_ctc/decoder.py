import torch

from .config import (
    BLANK_INDEX,
    INDEX_TO_CHAR
)


def greedy_decode(
    outputs,
    comprimentos
):
    probabilidades = torch.softmax(
        outputs,
        dim=2
    )

    indices = torch.argmax(
        probabilidades,
        dim=2
    )

    resultados = []

    for batch_index in range(
        indices.size(0)
    ):
        comprimento = int(
            comprimentos[
                batch_index
            ]
        )

        sequencia = indices[
            batch_index,
            :comprimento
        ].tolist()

        texto = []

        anterior = None

        for indice in sequencia:

            if (
                indice != BLANK_INDEX
                and indice != anterior
            ):
                caractere = (
                    INDEX_TO_CHAR.get(
                        indice
                    )
                )

                if caractere is not None:
                    texto.append(
                        caractere
                    )

            anterior = indice

        resultados.append(
            "".join(
                texto
            )
        )

    return resultados