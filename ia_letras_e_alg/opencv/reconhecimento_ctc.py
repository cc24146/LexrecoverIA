import os

import cv2
import numpy as np
import torch

from crnn_ctc.model import CRNNCTC
from crnn_ctc.decoder import greedy_decode


PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "crnn_ctc",
    "final_model.pth"
)


class ReconhecedorCTC:

    def __init__(self):

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.model = CRNNCTC().to(
            self.device
        )

        self.model.load_state_dict(
            torch.load(
                MODEL_PATH,
                map_location=self.device
            )
        )

        self.model.eval()

        print(
            "Modelo CTC carregado em:",
            self.device
        )


    def preparar_linha(
        self,
        imagem
    ):

        if len(imagem.shape) == 3:

            imagem = cv2.cvtColor(
                imagem,
                cv2.COLOR_BGR2GRAY
            )

        _, binaria = cv2.threshold(
            imagem,
            0,
            255,
            cv2.THRESH_BINARY_INV
            + cv2.THRESH_OTSU
        )

        pontos = cv2.findNonZero(
            binaria
        )

        if pontos is None:
            return None

        x, y, w, h = cv2.boundingRect(
            pontos
        )

        binaria = binaria[
            y:y + h,
            x:x + w
        ]

        nova_altura = 32

        escala = (
            nova_altura
            / binaria.shape[0]
        )

        nova_largura = max(
            1,
            int(
                binaria.shape[1]
                * escala
            )
        )

        binaria = cv2.resize(
            binaria,
            (
                nova_largura,
                nova_altura
            ),
            interpolation=cv2.INTER_AREA
        )

        margem = 4

        binaria = cv2.copyMakeBorder(
            binaria,
            0,
            0,
            margem,
            margem,
            cv2.BORDER_CONSTANT,
            value=0
        )

        imagem_tensor = (
            torch
            .from_numpy(binaria)
            .float()
            / 255.0
        )

        imagem_tensor = (
            imagem_tensor
            .unsqueeze(0)
            .unsqueeze(0)
        )

        return imagem_tensor


    def reconhecer(
        self,
        imagem
    ):

        imagem_tensor = (
            self.preparar_linha(
                imagem
            )
        )

        if imagem_tensor is None:
            return ""

        largura = (
            imagem_tensor.shape[3]
        )

        imagem_tensor = (
            imagem_tensor
            .to(self.device)
        )

        with torch.no_grad():

            outputs = self.model(
                imagem_tensor
            )

        larguras = torch.tensor(
            [largura],
            dtype=torch.long
        )

        comprimentos = (
            self.model
            .calcular_comprimentos_saida(
                larguras
            )
        )

        texto = greedy_decode(
            outputs,
            comprimentos
        )[0]

        return texto