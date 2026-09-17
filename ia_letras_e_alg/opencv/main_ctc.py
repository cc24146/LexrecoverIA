import os
import sys

import cv2
import numpy as np


PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(
        0,
        PROJECT_ROOT
    )


from opencv.reconhecimento_ctc import ReconhecedorCTC


def detectar_linhas(
    imagem
):

    cinza = cv2.cvtColor(
        imagem,
        cv2.COLOR_BGR2GRAY
    )

    _, binaria = cv2.threshold(
        cinza,
        0,
        255,
        cv2.THRESH_BINARY_INV
        + cv2.THRESH_OTSU
    )

    possui_tinta = (
        np.any(
            binaria > 0,
            axis=1
        )
        .astype(np.uint8)
    )

    altura = imagem.shape[0]

    gap_maximo = max(
        4,
        int(
            altura * 0.01
        )
    )

    intervalos = []

    inicio = None
    ultimo = None

    for y, ativo in enumerate(
        possui_tinta
    ):

        if ativo:

            if inicio is None:

                inicio = y

            elif (
                ultimo is not None
                and y - ultimo - 1
                > gap_maximo
            ):

                intervalos.append(
                    (
                        inicio,
                        ultimo
                    )
                )

                inicio = y

            ultimo = y

    if inicio is not None:

        intervalos.append(
            (
                inicio,
                ultimo
            )
        )

    linhas = []

    margem_y = 4
    margem_x = 8

    for y1, y2 in intervalos:

        y1 = max(
            0,
            y1 - margem_y
        )

        y2 = min(
            imagem.shape[0] - 1,
            y2 + margem_y
        )

        trecho = binaria[
            y1:y2 + 1
        ]

        colunas = np.where(
            np.any(
                trecho > 0,
                axis=0
            )
        )[0]

        if len(colunas) == 0:
            continue

        x1 = max(
            0,
            int(colunas[0])
            - margem_x
        )

        x2 = min(
            imagem.shape[1] - 1,
            int(colunas[-1])
            + margem_x
        )

        largura = (
            x2 - x1 + 1
        )

        altura_linha = (
            y2 - y1 + 1
        )

        if (
            largura < 15
            or altura_linha < 5
        ):
            continue

        linhas.append(
            (
                x1,
                y1,
                largura,
                altura_linha
            )
        )

    linhas.sort(
        key=lambda caixa: caixa[1]
    )

    return linhas


def main():

    if len(sys.argv) > 1:

        caminho_imagem = (
            sys.argv[1]
        )

    else:

        caminho_imagem = os.path.join(
            PROJECT_ROOT,
            "opencv",
            "imagens",
            "dificil.png"
        )

    imagem = cv2.imread(
        caminho_imagem
    )

    if imagem is None:

        raise FileNotFoundError(
            f"Imagem não encontrada: "
            f"{caminho_imagem}"
        )

    reconhecedor = (
        ReconhecedorCTC()
    )

    caixas = detectar_linhas(
        imagem
    )

    imagem_debug = (
        imagem.copy()
    )

    pasta_debug = os.path.join(
        PROJECT_ROOT,
        "opencv",
        "debug_linhas_ctc"
    )

    os.makedirs(
        pasta_debug,
        exist_ok=True
    )

    textos = []

    print()
    print(
        "=============================="
    )

    print(
        "RECONHECIMENTO CTC"
    )

    print(
        "=============================="
    )

    for indice, caixa in enumerate(
        caixas,
        start=1
    ):

        x, y, w, h = caixa

        linha = imagem[
            y:y + h,
            x:x + w
        ]

        caminho_debug = os.path.join(
            pasta_debug,
            f"linha_{indice:02d}.png"
        )

        cv2.imwrite(
            caminho_debug,
            linha
        )

        texto = reconhecedor.reconhecer(
            linha
        )

        textos.append(
            texto
        )

        print(
            f"Linha {indice}: "
            f"{texto}"
        )

        cv2.rectangle(
            imagem_debug,
            (x, y),
            (
                x + w,
                y + h
            ),
            (
                0,
                255,
                0
            ),
            2
        )

        cv2.putText(
            imagem_debug,
            str(indice),
            (
                x,
                max(
                    20,
                    y - 5
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (
                0,
                0,
                255
            ),
            2
        )

    print()
    print(
        "TEXTO FINAL"
    )

    print(
        "=============================="
    )

    print(
        "\n".join(
            textos
        )
    )

    cv2.imshow(
        "Linhas detectadas - CTC",
        imagem_debug
    )

    cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()