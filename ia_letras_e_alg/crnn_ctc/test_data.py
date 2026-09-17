import os

from crnn_ctc.data import (
    get_data,
    comprimento_minimo_ctc
)



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


(
    train_loader,
    val_loader,
    test_loader
) = get_data(
    BRESSAY_ROOT,
    batch_size=2
)


batch = next(
    iter(train_loader)
)

print()
print("VERIFICACAO CTC")
print("==============================")

for largura, texto in zip(
    batch["larguras"],
    batch["textos"]
):

    passos_saida = (
        int(largura)
        // 4
    )

    minimo = comprimento_minimo_ctc(
        texto
    )

    print(
        f"Saida: {passos_saida:3d} | "
        f"Minimo: {minimo:3d} | "
        f"Valido: {passos_saida >= minimo}"
    )


print()
print(
    "Formato das imagens:"
)

print(
    batch["imagens"].shape
)


print()
print(
    "Larguras originais:"
)

print(
    batch["larguras"]
)


print()
print(
    "Comprimentos dos textos:"
)

print(
    batch[
        "comprimentos_alvos"
    ]
)


print()
print(
    "Textos:"
)

for texto in batch[
    "textos"
]:
    print(
        repr(texto)
    )