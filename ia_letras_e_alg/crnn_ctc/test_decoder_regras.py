import unittest

import torch

from crnn_ctc.config import (
    BLANK_INDEX,
    CHAR_TO_INDEX,
    NUM_CLASSES,
)
from crnn_ctc.decoder import greedy_decode


class TesteDecodificador(unittest.TestCase):
    def decodificar(self, sequencia, comprimento=None):
        indices = [
            BLANK_INDEX
            if caractere is None
            else CHAR_TO_INDEX[caractere]
            for caractere in sequencia
        ]

        saidas = torch.full(
            (1, len(indices), NUM_CLASSES),
            -10.0
        )

        for posicao, indice in enumerate(indices):
            saidas[0, posicao, indice] = 10.0

        comprimentos = torch.tensor([
            len(indices)
            if comprimento is None
            else comprimento
        ])

        return greedy_decode(
            saidas,
            comprimentos
        )[0]

    def test_agrupa_repeticoes_consecutivas(self):
        self.assertEqual(
            self.decodificar(["a", "a", "a"]),
            "a"
        )

    def test_blank_separa_letras_repetidas(self):
        self.assertEqual(
            self.decodificar([
                "c", "a", "r", None, "r", "o"
            ]),
            "carro"
        )

    def test_ignora_posicoes_apos_comprimento(self):
        self.assertEqual(
            self.decodificar(
                ["o", None, "x"],
                comprimento=2
            ),
            "o"
        )

    def test_somente_blank_retorna_vazio(self):
        self.assertEqual(
            self.decodificar([None, None]),
            ""
        )

    def test_preserva_texto_bruto(self):
        texto = "Roma, ações 2026! aguandar"

        sequencia = [
            item
            for caractere in texto
            for item in (caractere, None)
        ]

        self.assertEqual(
            self.decodificar(sequencia),
            texto
        )


if __name__ == "__main__":
    unittest.main()