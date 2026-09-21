import re

from .corretor import corrigir_palavra


def corrigir_texto(texto):
    texto_corrigido = re.sub(
        r"\w+(?:[-'’]\w+)*",
        lambda trecho: corrigir_palavra(
            trecho.group()
        ),
        texto
    )

    return texto_corrigido