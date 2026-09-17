import os
from collections import Counter


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

LINES_PATH = os.path.join(
    BRESSAY_ROOT,
    "data",
    "lines"
)


def possui_anotacao_especial(texto):
    marcadores = (
        "##",
        "$$",
        "--",
        "@@"
    )

    return any(
        marcador in texto
        for marcador in marcadores
    )


def analisar_dataset():

    if not os.path.exists(LINES_PATH):
        print("Pasta não encontrada:")
        print(LINES_PATH)
        return

    arquivos_txt = []

    for raiz, _, arquivos in os.walk(
        LINES_PATH
    ):
        for arquivo in arquivos:

            if arquivo.lower().endswith(
                ".txt"
            ):
                arquivos_txt.append(
                    os.path.join(
                        raiz,
                        arquivo
                    )
                )

    caracteres_limpos = Counter()

    total = 0
    limpas = 0
    especiais = 0

    comprimentos = []

    for caminho in arquivos_txt:

        with open(
            caminho,
            "r",
            encoding="utf-8"
        ) as arquivo:

            texto = arquivo.read().strip()

        if not texto:
            continue

        total += 1

        if possui_anotacao_especial(texto):
            especiais += 1
            continue

        limpas += 1

        caracteres_limpos.update(
            texto
        )

        comprimentos.append(
            len(texto)
        )

    print()
    print("TOTAL")
    print("==============================")

    print(
        "Linhas encontradas:",
        total
    )

    print(
        "Linhas limpas:",
        limpas
    )

    print(
        "Linhas ignoradas:",
        especiais
    )

    print()
    print("CARACTERES DAS LINHAS LIMPAS")
    print("==============================")

    for caractere in sorted(
        caracteres_limpos
    ):

        print(
            f"{repr(caractere):8} "
            f"{caracteres_limpos[caractere]}"
        )

    print()
    print(
        "Caracteres distintos:",
        len(caracteres_limpos)
    )

    if comprimentos:

        print()
        print(
            "Menor linha:",
            min(comprimentos),
            "caracteres"
        )

        print(
            "Maior linha:",
            max(comprimentos),
            "caracteres"
        )

        media = (
            sum(comprimentos)
            / len(comprimentos)
        )

        print(
            f"Média: {media:.2f} caracteres"
        )

    print()
    print("CARACTERES RAROS")
    print("==============================")

    for caractere, quantidade in sorted(
        caracteres_limpos.items(),
        key=lambda item: item[1]
    ):

        if quantidade <= 20:

            print(
                f"{repr(caractere):8} "
                f"{quantidade}"
            )


if __name__ == "__main__":
    analisar_dataset()