import re

from .corretor import (
    corrigir_palavra,
    analisar_palavra
)


def corrigir_texto(texto):
    texto_corrigido = re.sub(
        r"\w+(?:[-'’]\w+)*",
        lambda trecho: corrigir_palavra(
            trecho.group()
        ),
        texto
    )

    return texto_corrigido

def analisar_texto(texto):
    if not isinstance(texto, str):
        raise TypeError("O texto deve ser uma string.")

    palavras = []

    for trecho in re.finditer(
        r"\w+(?:[-‐‑–'’]\w+)*",
        texto
    ):
        palavra = trecho.group()

        if not palavra.isalpha():
            analise = {
                "original": palavra,
                "conhecida": None,
                "candidatos": [],
                "distancia": None,
                "motivo": "expressao_protegida",
            }

        else:
            analise = analisar_palavra(palavra)

        analise["inicio"] = trecho.start()
        analise["fim"] = trecho.end()

        palavras.append(analise)

    return {
        "texto_bruto": texto,
        "palavras": palavras,
    }