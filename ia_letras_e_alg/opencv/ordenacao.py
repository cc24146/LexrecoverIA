from statistics import median
import os

DIRETORIO_ATUAL = os.path.dirname(os.path.abspath(__file__))
CAMINHO_LEXICO = os.path.join(DIRETORIO_ATUAL, "banco", "lexico.txt")

def carregar_dicionario(caminho_arquivo):
    if not os.path.exists(caminho_arquivo):
        print(f"Aviso: Arquivo '{caminho_arquivo}' não encontrado. Dicionário vazio.")
        return set()
    
    with open(caminho_arquivo, "r", encoding="utf-8") as f:
        palavras = {linha.strip().lower() for linha in f if linha.strip()}
    
    return palavras

DICIONARIO = carregar_dicionario(CAMINHO_LEXICO)

def agrupar_em_linhas(letras):

    if not letras:
        return []

    letras_ordenadas = sorted(
        letras,
        key=lambda letra:
            letra["y"]
            + letra["h"] / 2
    )

    linhas = []

    for letra in letras_ordenadas:

        centro_letra = (
            letra["y"]
            + letra["h"] / 2
        )

        melhor_linha = None
        menor_distancia = float("inf")

        for linha in linhas:

            centros = [
                item["y"]
                + item["h"] / 2
                for item in linha
            ]

            alturas = [
                item["h"]
                for item in linha
            ]

            centro_linha = median(
                centros
            )

            altura_referencia = median(
                alturas
            )

            tolerancia = max(
                8,
                altura_referencia * 0.6
            )

            distancia = abs(
                centro_letra
                - centro_linha  
            )

            if (
                distancia <= tolerancia
                and
                distancia < menor_distancia
            ):
                melhor_linha = linha
                menor_distancia = distancia

        if melhor_linha is None:
            linhas.append([letra])

        else:
            melhor_linha.append(letra)

    linhas.sort(
        key=lambda linha:
            median(
                item["y"] + item["h"] / 2
                for item in linha
            )
    )

    for linha in linhas:
        linha.sort(
            key=lambda letra: letra["x"]
        )

    return linhas

def distancia_levenshtein(s1: str, s2: str) -> int:
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distancia_anterior = list(range(len(s1) + 1))

    for i, c1 in enumerate(s2):
        distancia_atual = [i + 1]

        for j, c2 in enumerate(s1):
            insercao_ou_remocao = min(
                distancia_anterior[j + 1] + 1,
                distancia_atual[j] + 1
            )

            substituicao = distancia_anterior[j] + (0 if c1 == c2 else 1)

            distancia_atual.append(min(insercao_ou_remocao, substituicao))

        distancia_anterior = distancia_atual

        return distancia_anterior[-1]

def corrigir_palavra(palavra):

    if isinstance(palavra, list):
        previsoes = [letra["previsao"] for letra in palavra]

        tem_letra = any(c.isalpha() for c in previsoes)
        tem_numero = any(c.isdigit() for c in previsoes)

        if tem_letra and not tem_numero:
            texto_original = "".join(previsoes)

        elif tem_numero and not tem_letra:
            texto_original = "".join(previsoes)

        else:
            resultado = []
            for letra in palavra:
                caractere = letra["previsao"]

                if caractere.isdigit():
                    confianca_numero = letra["confianca"]
                    confianca_letra = letra["confianca_letra"]

                    if confianca_letra >= confianca_numero * 0.60:
                        resultado.append(letra["melhor_letra"])
                    else:
                        resultado.append(caractere)
                else:
                    resultado.append(caractere)

            texto_original = "".join(resultado)

    elif isinstance(palavra, str):
        texto_original = palavra

    else:
        return ""

    if not texto_original:
        return ""

    if any(c.isdigit() for c in texto_original):
        return texto_original

    palavra_limpa = texto_original.lower()

    if palavra_limpa in DICIONARIO:
        return texto_original

    candidatos = [
        p for p in DICIONARIO
        if abs(len(p) - len(palavra_limpa)) <= 2
    ]

    if not candidatos:
        return texto_original

    melhor_palavra = min(
        candidatos,
        key = lambda p: distancia_levenshtein(palavra_limpa, p)
    )

    limiar = max(2, len(palavra_limpa) // 3)
    if distancia_levenshtein(palavra_limpa, melhor_palavra) <= limiar:
        return melhor_palavra.upper() if texto_original.isupper else melhor_palavra

    return texto_original

def reconstruir_texto(
    linhas,
    espacos_por_linha
):

    resultado = []

    for numero_linha, linha in enumerate(
        linhas
    ):

        linha = sorted(
            linha,
            key=lambda letra: letra["x"]
        )

        espacos = espacos_por_linha[
            numero_linha
        ]

        palavra_atual = []

        for indice, letra in enumerate(
            linha
        ):

            palavra_atual.append(
                letra
            )

            if indice in espacos:

                palavra = corrigir_palavra(
                    palavra_atual
                )

                resultado.append(
                    palavra
                )

                resultado.append(" ")

                palavra_atual = []

        if palavra_atual:

            palavra = corrigir_palavra(
                palavra_atual
            )

            resultado.append(
                palavra
            )

        if (
            numero_linha
            < len(linhas) - 1
        ):
            resultado.append("\n")

    return "".join(resultado)