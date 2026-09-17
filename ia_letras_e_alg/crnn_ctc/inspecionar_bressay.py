import os


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

SETS_PATH = os.path.join(
    BRESSAY_ROOT,
    "sets"
)


def mostrar_arquivo(nome):

    caminho = os.path.join(
        SETS_PATH,
        nome
    )

    print()
    print("=" * 50)
    print(nome)
    print("=" * 50)

    if not os.path.exists(caminho):
        print("Arquivo não encontrado:")
        print(caminho)
        return

    with open(
        caminho,
        "r",
        encoding="utf-8"
    ) as arquivo:

        for indice, linha in enumerate(
            arquivo
        ):
            print(
                repr(
                    linha.rstrip("\n")
                )
            )

            if indice >= 9:
                break


def mostrar_estrutura_lines():

    pasta = os.path.join(
        BRESSAY_ROOT,
        "data",
        "lines"
    )

    print()
    print("=" * 50)
    print("ARQUIVOS DE data/lines")
    print("=" * 50)

    contador = 0

    for raiz, _, arquivos in os.walk(
        pasta
    ):

        for arquivo in arquivos:

            caminho = os.path.join(
                raiz,
                arquivo
            )

            relativo = os.path.relpath(
                caminho,
                BRESSAY_ROOT
            )

            print(relativo)

            contador += 1

            if contador >= 20:
                return


if __name__ == "__main__":

    mostrar_arquivo(
        "training.txt"
    )

    mostrar_arquivo(
        "validation.txt"
    )

    mostrar_arquivo(
        "test.txt"
    )

    mostrar_estrutura_lines()