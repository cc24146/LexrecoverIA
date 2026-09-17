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