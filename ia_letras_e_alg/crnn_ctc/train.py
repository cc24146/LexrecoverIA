import os

import torch
import torch.nn as nn

from .decoder import greedy_decode
from .metrics import calcular_cer


PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

BEST_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "crnn_ctc",
    "best_model.pth"
)

LAST_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "crnn_ctc",
    "model_last.pth"
)


def treinar_epoca(
    model,
    loader,
    optimizer,
    ctc_loss,
    device
):
    model.train()

    loss_total = 0.0

    for numero_batch, batch in enumerate(
        loader,
        start=1
    ):
        imagens = batch[
            "imagens"
        ].to(device)

        alvos = batch[
            "alvos"
        ].to(device)

        larguras = batch[
            "larguras"
        ]

        comprimentos_alvos = batch[
            "comprimentos_alvos"
        ]

        optimizer.zero_grad()

        outputs = model(
            imagens
        )

        input_lengths = (
            model.calcular_comprimentos_saida(
                larguras
            )
        )

        log_probs = torch.log_softmax(
            outputs,
            dim=2
        )

        log_probs = log_probs.permute(
            1,
            0,
            2
        )

        loss = ctc_loss(
            log_probs,
            alvos,
            input_lengths,
            comprimentos_alvos
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0
        )

        optimizer.step()

        loss_total += loss.item()

        if numero_batch % 100 == 0:
            print(
                f"  Lote "
                f"{numero_batch}/{len(loader)} | "
                f"Loss: {loss.item():.4f}"
            )

    return (
        loss_total
        / numero_batch
    )


def avaliar(
    model,
    loader,
    ctc_loss,
    device
):
    model.eval()

    loss_total = 0.0

    referencias = []
    previsoes = []

    with torch.no_grad():

        for numero_batch, batch in enumerate(
            loader,
            start=1
        ):
            imagens = batch[
                "imagens"
            ].to(device)

            alvos = batch[
                "alvos"
            ].to(device)

            larguras = batch[
                "larguras"
            ]

            comprimentos_alvos = batch[
                "comprimentos_alvos"
            ]

            outputs = model(
                imagens
            )

            input_lengths = (
                model.calcular_comprimentos_saida(
                    larguras
                )
            )

            if numero_batch == 1:

                indices = outputs.argmax(
                    dim=2
                )

                quantidade_blank = 0
                total_posicoes = 0

                for indice_amostra in range(
                    indices.size(0)
                ):
                    comprimento = int(
                        input_lengths[
                            indice_amostra
                        ]
                    )

                    sequencia = indices[
                        indice_amostra,
                        :comprimento
                    ]

                    quantidade_blank += (
                        sequencia == 0
                    ).sum().item()

                    total_posicoes += (
                        comprimento
                    )

                percentual_blank = (
                    quantidade_blank
                    / total_posicoes
                    * 100
                )

                print()
                print(
                    f"Blank nas previsões: "
                    f"{percentual_blank:.2f}%"
                )

                print(
                    "Input lengths:",
                    input_lengths
                )

                print(
                    "Target lengths:",
                    comprimentos_alvos
                )

            log_probs = torch.log_softmax(
                outputs,
                dim=2
            )

            loss = ctc_loss(
                log_probs.permute(
                    1,
                    0,
                    2
                ),
                alvos,
                input_lengths,
                comprimentos_alvos
            )

            loss_total += loss.item()

            textos_previstos = greedy_decode(
                outputs,
                input_lengths
            )

            if numero_batch == 1:

                print()
                print(
                    "EXEMPLOS DE VALIDACAO"
                )

                print(
                    "=============================="
                )

                for referencia, previsao in zip(
                    batch["textos"][:3],
                    textos_previstos[:3]
                ):
                    print(
                        "Correto :",
                        repr(referencia)
                    )

                    print(
                        "Previsto:",
                        repr(previsao)
                    )

                    print()

            referencias.extend(
                batch["textos"]
            )

            previsoes.extend(
                textos_previstos
            )

    cer = calcular_cer(
        referencias,
        previsoes
    )

    return (
        loss_total / numero_batch,
        cer
    )


def train(
    model,
    train_loader,
    val_loader,
    device,
    epochs=3
):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0001
    )

    ctc_loss = nn.CTCLoss(
        blank=0,
        reduction="mean",
        zero_infinity=True
    )

    (
        val_loss_inicial,
        melhor_cer
    ) = avaliar(
        model,
        val_loader,
        ctc_loss,
        device
    )

    print()
    print(
        f"CER inicial: "
        f"{melhor_cer * 100:.2f}%"
    )

    for epoch in range(
        1,
        epochs + 1
    ):
        print()
        print(
            f"=============================="
        )

        print(
            f"ÉPOCA {epoch}/{epochs}"
        )

        print(
            f"=============================="
        )

        train_loss = treinar_epoca(
            model,
            train_loader,
            optimizer,
            ctc_loss,
            device
        )

        (
            val_loss,
            val_cer
        ) = avaliar(
            model,
            val_loader,
            ctc_loss,
            device
        )

        print()
        print(
            f"Época {epoch:02d} | "
            f"Treino Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val CER: {val_cer * 100:.2f}%"
        )

        torch.save(
            model.state_dict(),
            LAST_MODEL_PATH
        )

        print(
            "Último modelo salvo em:",
            LAST_MODEL_PATH
        )

        if val_cer < melhor_cer:

            melhor_cer = val_cer

            torch.save(
                model.state_dict(),
                BEST_MODEL_PATH
            )

            print(
                f"Novo melhor modelo! "
                f"CER = {melhor_cer * 100:.2f}%"
            )