import os
import cv2
import torch

from torch.utils.data import (
    Dataset,
    DataLoader
)

from .config import (
    CHAR_TO_INDEX
)


ALTURA_IMAGEM = 32


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


def texto_suportado(texto):
    return all(
        caractere in CHAR_TO_INDEX
        for caractere in texto
    )


def codificar_texto(texto):
    return torch.tensor(
        [
            CHAR_TO_INDEX[caractere]
            for caractere in texto
        ],
        dtype=torch.long
    )


def comprimento_minimo_ctc(texto):
    comprimento = len(texto)

    for i in range(
        1,
        len(texto)
    ):
        if texto[i] == texto[i - 1]:
            comprimento += 1

    return comprimento


def preparar_imagem(
    caminho_imagem,
    texto
):
    imagem = cv2.imread(
        caminho_imagem,
        cv2.IMREAD_GRAYSCALE
    )

    if imagem is None:
        raise FileNotFoundError(
            f"Imagem não encontrada: {caminho_imagem}"
        )

    _, imagem = cv2.threshold(
        imagem,
        0,
        255,
        cv2.THRESH_BINARY_INV
        + cv2.THRESH_OTSU
    )

    pontos = cv2.findNonZero(
        imagem
    )

    if pontos is not None:
        x, y, w, h = cv2.boundingRect(
            pontos
        )

        imagem = imagem[
            y:y + h,
            x:x + w
        ]

    altura_original, largura_original = (
        imagem.shape
    )

    escala = (
        ALTURA_IMAGEM
        / altura_original
    )

    nova_largura = max(
        1,
        round(
            largura_original
            * escala
        )
    )

    imagem = cv2.resize(
        imagem,
        (
            nova_largura,
            ALTURA_IMAGEM
        ),
        interpolation=cv2.INTER_AREA
    )

    margem_horizontal = 4

    imagem = cv2.copyMakeBorder(
        imagem,
        0,
        0,
        margem_horizontal,
        margem_horizontal,
        cv2.BORDER_CONSTANT,
        value=0
    )

    passos_minimos = (
        comprimento_minimo_ctc(
            texto
        )
    )

    largura_minima = (
        passos_minimos
        * 4
    )

    largura_atual = (
        imagem.shape[1]
    )

    if largura_atual < largura_minima:
        faltam = (
            largura_minima
            - largura_atual
        )

        imagem = cv2.copyMakeBorder(
            imagem,
            0,
            0,
            0,
            faltam,
            cv2.BORDER_CONSTANT,
            value=0
        )

    imagem = (
        imagem.astype("float32")
        / 255.0
    )

    return torch.from_numpy(
        imagem
    ).unsqueeze(0)


class BressayDataset(Dataset):

    def __init__(
        self,
        pasta_lines,
        arquivo_split
    ):
        self.pasta_lines = (
            pasta_lines
        )

        self.amostras = []

        ids = self._carregar_ids(
            arquivo_split
        )

        ignoradas_anotacao = 0
        ignoradas_caractere = 0

        for identificador in ids:

            pasta = os.path.join(
                self.pasta_lines,
                identificador
            )

            if not os.path.isdir(
                pasta
            ):
                continue

            arquivos_txt = sorted(
                arquivo
                for arquivo in os.listdir(
                    pasta
                )
                if arquivo.lower().endswith(
                    ".txt"
                )
            )

            for arquivo_txt in arquivos_txt:

                caminho_txt = os.path.join(
                    pasta,
                    arquivo_txt
                )

                with open(
                    caminho_txt,
                    "r",
                    encoding="utf-8"
                ) as arquivo:

                    texto = (
                        arquivo
                        .read()
                        .strip()
                    )

                if not texto:
                    continue

                if possui_anotacao_especial(
                    texto
                ):
                    ignoradas_anotacao += 1
                    continue

                if not texto_suportado(
                    texto
                ):
                    ignoradas_caractere += 1
                    continue

                nome_base = os.path.splitext(
                    arquivo_txt
                )[0]

                caminho_imagem = os.path.join(
                    pasta,
                    nome_base + ".png"
                )

                if not os.path.exists(
                    caminho_imagem
                ):
                    continue

                self.amostras.append(
                    (
                        caminho_imagem,
                        texto
                    )
                )

        print(
            f"{os.path.basename(arquivo_split)}: "
            f"{len(self.amostras)} linhas válidas"
        )

        print(
            f"  ignoradas por anotação: "
            f"{ignoradas_anotacao}"
        )

        print(
            f"  ignoradas por caractere: "
            f"{ignoradas_caractere}"
        )

    def _carregar_ids(
        self,
        arquivo_split
    ):
        with open(
            arquivo_split,
            "r",
            encoding="utf-8"
        ) as arquivo:

            return [
                linha.strip()
                for linha in arquivo
                if linha.strip()
            ]

    def __len__(self):
        return len(
            self.amostras
        )

    def __getitem__(
        self,
        indice
    ):
        (
            caminho_imagem,
            texto
        ) = self.amostras[
            indice
        ]

        imagem = preparar_imagem(
            caminho_imagem,
            texto
        )

        alvo = codificar_texto(
            texto
        )

        return {
            "imagem": imagem,
            "alvo": alvo,
            "texto": texto,
            "largura": imagem.shape[2]
        }


def collate_ctc(batch):

    maior_largura = max(
        item["largura"]
        for item in batch
    )

    if maior_largura % 4 != 0:
        maior_largura += (
            4
            - maior_largura % 4
        )

    imagens = []
    larguras = []
    alvos = []
    comprimentos_alvos = []
    textos = []

    for item in batch:

        imagem = item[
            "imagem"
        ]

        largura = item[
            "largura"
        ]

        faltam = (
            maior_largura
            - largura
        )

        if faltam > 0:
            imagem = torch.nn.functional.pad(
                imagem,
                (
                    0,
                    faltam,
                    0,
                    0
                ),
                value=0
            )

        imagens.append(
            imagem
        )

        larguras.append(
            largura
        )

        alvos.append(
            item["alvo"]
        )

        comprimentos_alvos.append(
            len(
                item["alvo"]
            )
        )

        textos.append(
            item["texto"]
        )

    return {
        "imagens": torch.stack(
            imagens
        ),

        "alvos": torch.cat(
            alvos
        ),

        "larguras": torch.tensor(
            larguras,
            dtype=torch.long
        ),

        "comprimentos_alvos":
            torch.tensor(
                comprimentos_alvos,
                dtype=torch.long
            ),

        "textos": textos
    }


def get_data(
    raiz_bressay,
    batch_size=8
):

    pasta_lines = os.path.join(
        raiz_bressay,
        "data",
        "lines"
    )

    pasta_sets = os.path.join(
        raiz_bressay,
        "sets"
    )

    train_dataset = BressayDataset(
        pasta_lines,
        os.path.join(
            pasta_sets,
            "training.txt"
        )
    )

    val_dataset = BressayDataset(
        pasta_lines,
        os.path.join(
            pasta_sets,
            "validation.txt"
        )
    )

    test_dataset = BressayDataset(
        pasta_lines,
        os.path.join(
            pasta_sets,
            "test.txt"
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_ctc
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_ctc
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_ctc
    )

    return (
        train_loader,
        val_loader,
        test_loader
    )