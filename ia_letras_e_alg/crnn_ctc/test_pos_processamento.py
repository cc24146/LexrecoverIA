import unittest
from unittest.mock import patch

from crnn_ctc import corretor
from crnn_ctc.pos_processamento import (
    analisar_texto,
    corrigir_texto,
)


class TestePosProcessamento(unittest.TestCase):
    def test_preserva_expressoes_protegidas(self):
        expressoes = [
            "abcd-x",
            "abcd‐x",
            "abcd‑x",
            "abcd–x",
            "abcd'x",
            "abcd’x",
            "abcd123",
        ]

        with patch.object(
            corretor,
            "DICIONARIO",
            {"abce"}
        ):
            for expressao in expressoes:
                with self.subTest(expressao=expressao):
                    self.assertEqual(
                        corrigir_texto(expressao),
                        expressao
                    )

                    analise = analisar_texto(expressao)

                    self.assertEqual(
                        len(analise["palavras"]),
                        1
                    )

                    self.assertEqual(
                        analise["palavras"][0]["motivo"],
                        "expressao_protegida"
                    )

    def test_preserva_formatacao_e_posicoes(self):
        texto = "abcd‑x,  abcd123!\n  abcd’x."

        with patch.object(
            corretor,
            "DICIONARIO",
            {"abce"}
        ):
            self.assertEqual(
                corrigir_texto(texto),
                texto
            )

            analise = analisar_texto(texto)

        self.assertEqual(
            analise["texto_bruto"],
            texto
        )

        self.assertEqual(
            [
                palavra["original"]
                for palavra in analise["palavras"]
            ],
            ["abcd‑x", "abcd123", "abcd’x"]
        )

        for palavra in analise["palavras"]:
            self.assertEqual(
                texto[
                    palavra["inicio"]:palavra["fim"]
                ],
                palavra["original"]
            )

    def test_correcao_preserva_maiusculas(self):
        with patch.object(
            corretor,
            "DICIONARIO",
            {"rosa"}
        ):
            self.assertEqual(
                corrigir_texto("RosaX"),
                "RosaX"
            )


if __name__ == "__main__":
    unittest.main()