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
    
    if len(palavra_limpa) <= 3:
        return texto_original

    candidatos = [
        p for p in DICIONARIO
        if abs(len(p) - len(palavra_limpa)) <= 2
    ]

    if not candidatos:
        return texto_original

    distancias = [
        (
            candidato,
            distancia_levenshtein(
                palavra_limpa,
                candidato
            )
        )
        for candidato in candidatos
    ]

    menor_distancia = min(
        distancia
        for _, distancia in distancias
    )

    melhores_candidatos = [
        candidato
        for candidato, distancia in distancias
        if distancia == menor_distancia
    ]

    if len(melhores_candidatos) != 1:
        return texto_original

    melhor_palavra = melhores_candidatos[0]

    limiar = 1

    if menor_distancia <= limiar:
        if texto_original.isupper():
            return melhor_palavra.upper()

        return melhor_palavra

    return texto_original

def analisar_palavra(palavra):
    if not isinstance(palavra, str):
        raise TypeError("A palavra deve ser uma string.")

    resultado = {
        "original": palavra,
        "conhecida": False,
        "candidatos": [],
        "distancia": None,
        "motivo": "",
    }

    if not palavra:
        resultado["motivo"] = "entrada_vazia"
        return resultado

    if not DICIONARIO:
        resultado["motivo"] = "dicionario_vazio"
        return resultado

    palavra_normalizada = palavra.lower()

    if palavra_normalizada in DICIONARIO:
        resultado["conhecida"] = True
        resultado["distancia"] = 0
        resultado["motivo"] = "presente_no_lexico"
        return resultado

    if any(caractere.isdigit() for caractere in palavra):
        resultado["motivo"] = "contem_numero"
        return resultado

    if len(palavra_normalizada) <= 3:
        resultado["motivo"] = "palavra_curta"
        return resultado

    candidatos = []

    for candidato in DICIONARIO:
        if abs(len(candidato) - len(palavra_normalizada)) > 1:
            continue

        distancia = distancia_levenshtein(
            palavra_normalizada,
            candidato
        )

        if distancia == 1:
            candidatos.append(candidato)

    candidatos.sort()

    if palavra.isupper():
        candidatos = [
            candidato.upper()
            for candidato in candidatos
        ]
    elif palavra.istitle():
        candidatos = [
            candidato.capitalize()
            for candidato in candidatos
        ]

    resultado["candidatos"] = candidatos

    if not candidatos:
        resultado["motivo"] = "sem_candidato_a_uma_edicao"
    else:
        resultado["distancia"] = 1
        resultado["motivo"] = (
            "candidato_unico"
            if len(candidatos) == 1
            else "multiplos_candidatos"
        )

    return resultado