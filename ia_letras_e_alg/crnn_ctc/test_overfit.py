import os
import torch
import torch.nn as nn

from crnn_ctc.model import CRNNCTC
from crnn_ctc.data import get_data
from crnn_ctc.decoder import greedy_decode


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


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


(
    train_loader,
    _,
    _
) = get_data(
    BRESSAY_ROOT,
    batch_size=2
)


batch = next(
    iter(train_loader)
)


imagens = batch[
    "imagens"
].to(device)

alvos = batch[
    "alvos"
].to(device)

larguras = batch[
    "larguras"
]

target_lengths = batch[
    "comprimentos_alvos"
]


model = CRNNCTC().to(
    device
)


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


ctc_loss = nn.CTCLoss(
    blank=0,
    reduction="mean",
    zero_infinity=True
)


print()
print("TEXTOS QUE O MODELO DEVE DECORAR")
print("================================")

for texto in batch["textos"]:
    print(
        repr(texto)
    )


for passo in range(
    1,
    601
):

    model.train()

    optimizer.zero_grad()

    outputs = model(
        imagens
    )

    input_lengths = (
        model
        .calcular_comprimentos_saida(
            larguras
        )
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
        target_lengths
    )

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        5.0
    )

    optimizer.step()


    if passo % 50 == 0:

        model.eval()

        with torch.no_grad():

            outputs_teste = model(
                imagens
            )

            previsoes = greedy_decode(
                outputs_teste,
                input_lengths
            )

            probabilidades = torch.softmax(
                outputs_teste,
                dim=2
            )

            blank_medio = (
                probabilidades[
                    :,
                    :,
                    0
                ]
                .mean()
                .item()
                * 100
            )

            indices = outputs_teste.argmax(
                dim=2
            )

            blank_argmax = (
                (
                    indices == 0
                )
                .float()
                .mean()
                .item()
                * 100
            )


        print()
        print(
            f"Passo {passo:03d} | "
            f"Loss: {loss.item():.4f}"
        )

        print(
            f"Probabilidade média blank: "
            f"{blank_medio:.2f}%"
        )

        print(
            f"Blank como maior classe: "
            f"{blank_argmax:.2f}%"
        )

        print()

        for correto, previsto in zip(
            batch["textos"][:2],
            previsoes[:2]
        ):

            print(
                "Correto :",
                repr(correto)
            )

            print(
                "Previsto:",
                repr(previsto)
            )

            print()