import json
import re
import sys
import unicodedata
from pathlib import Path
from collections import Counter


PASTA = Path(__file__).resolve().parent
PROJETO = PASTA.parent

sys.path.insert(0, str(PROJETO))

from crnn_ctc.pos_processamento import analisar_texto


PADRAO = r"\w+(?:[-‐‑–'’]\w+)*"


def normalizar(texto):
    return unicodedata.normalize(
        "NFC",
        texto
    ).lower()


def alinhar(referencia, previsao):
    n = len(referencia)
    m = len(previsao)

    custos = [
        [0] * (m + 1)
        for _ in range(n + 1)
    ]

    for i in range(n + 1):
        custos[i][0] = i

    for j in range(m + 1):
        custos[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diferentes = (
                referencia[i - 1]
                != previsao[j - 1]
            )

            custos[i][j] = min(
                custos[i - 1][j] + 1,
                custos[i][j - 1] + 1,
                custos[i - 1][j - 1] + diferentes
            )

    pares = []
    i, j = n, m

    while i > 0 or j > 0:
        if i > 0 and j > 0:
            diferentes = (
                referencia[i - 1]
                != previsao[j - 1]
            )

            if custos[i][j] == (
                custos[i - 1][j - 1]
                + diferentes
            ):
                pares.append((i - 1, j - 1))
                i -= 1
                j -= 1
                continue

        if i > 0 and custos[i][j] == (
            custos[i - 1][j] + 1
        ):
            pares.append((i - 1, None))
            i -= 1
        else:
            pares.append((None, j - 1))
            j -= 1

    return pares[::-1]


def main():
    caminho = PASTA / (
        "resultado_recortes_"
        "20260919_215750_708565.json"
    )

    dados = json.loads(
        caminho.read_text(encoding="utf-8")
    )

    totais = {
        "corretas": 0,
        "corretas_com_alternativas": 0,
        "substituicoes": 0,
        "esperada_entre_candidatos": 0,
        "substituicoes_sem_candidatos": 0,
        "substituicoes_com_candidatos_inadequados": 0,
        "ausentes": 0,
        "extras": 0,
    }

    exemplos = []

    tamanhos_listas = []
    esperada_primeira = 0
    esperada_top5 = 0

    motivos_sem_candidatos = Counter()

    for numero, linha in enumerate(
        dados["linhas"],
        start=1
    ):
        referencia = re.findall(
            PADRAO,
            normalizar(linha["referencia"])
        )

        analise = analisar_texto(
            unicodedata.normalize(
                "NFC",
                linha["bruto"]
            )
        )

        palavras = analise["palavras"]

        previsao = [
            normalizar(p["original"])
            for p in palavras
        ]

        for indice_ref, indice_pred in alinhar(
            referencia,
            previsao
        ):
            if indice_ref is None:
                totais["extras"] += 1
                continue

            if indice_pred is None:
                totais["ausentes"] += 1
                continue

            esperada = referencia[indice_ref]
            obtida = previsao[indice_pred]
            palavra = palavras[indice_pred]

            candidatos_ordenados = list(
                dict.fromkeys(
                    normalizar(c)
                    for c in palavra["candidatos"]
                )
            )

            candidatos = set(candidatos_ordenados)

            if candidatos_ordenados:
                tamanhos_listas.append(
                    len(candidatos_ordenados)
                )

            if esperada == obtida:
                totais["corretas"] += 1

                if candidatos:
                    totais[
                        "corretas_com_alternativas"
                    ] += 1

                continue

            totais["substituicoes"] += 1

            if esperada in candidatos:
                totais[
                    "esperada_entre_candidatos"
                ] += 1

                posicao = (
                    candidatos_ordenados.index(esperada)
                    + 1
                )

                if posicao == 1:
                    esperada_primeira += 1

                if posicao <= 5:
                    esperada_top5 += 1

                situacao = (
                    f"esperada disponível na posição {posicao}"
                )

            elif not candidatos:
                totais[
                    "substituicoes_sem_candidatos"
                ] += 1

                motivos_sem_candidatos[
                    palavra["motivo"]
                ] += 1

                situacao = "sem candidatos"

            else:
                totais[
                    "substituicoes_com_candidatos_inadequados"
                ] += 1
                situacao = "esperada ausente dos candidatos"

            if len(exemplos) < 15:
                exemplos.append(
                    f"Linha {numero}: "
                    f"{palavra['original']!r} → "
                    f"referência {esperada!r} | "
                    f"{situacao} | "
                    f"motivo: {palavra['motivo']}"
                )

        print(
            f"Analisadas {numero}/{len(dados['linhas'])} linhas",
            flush=True
        )

    print("\nRESULTADOS")

    for nome, quantidade in totais.items():
        print(f"{nome}: {quantidade}")

    if totais["substituicoes"]:
        cobertura = (
            totais["esperada_entre_candidatos"]
            / totais["substituicoes"]
        )

        print(
            f"\nCobertura das substituições: "
            f"{cobertura:.2%}"
        )

    if totais["corretas"]:
        alternativas_desnecessarias = (
            totais["corretas_com_alternativas"]
            / totais["corretas"]
        )

        print(
            "Palavras corretas com alternativas: "
            f"{alternativas_desnecessarias:.2%}"
        )

        print("\nMOTIVOS DOS ERROS SEM CANDIDATOS")

    for motivo, quantidade in (
        motivos_sem_candidatos.most_common()
    ):
        print(f"{motivo}: {quantidade}")

    print("\nPOSIÇÃO E QUANTIDADE DE ALTERNATIVAS")

    print(
        "Resposta esperada em primeiro:",
        esperada_primeira
    )

    print(
        "Resposta esperada entre as cinco primeiras:",
        esperada_top5
    )

    if tamanhos_listas:
        print(
            "Média de candidatos por lista não vazia:",
            round(
                sum(tamanhos_listas)
                / len(tamanhos_listas),
                2
            )
        )

        print(
            "Maior lista:",
            max(tamanhos_listas)
        )

    print("\nEXEMPLOS")

    for exemplo in exemplos:
        print(exemplo)


if __name__ == "__main__":
    main()