from statistics import median

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

def corrigir_palavra(caracteres):

    previsoes = [
        letra["previsao"]
        for letra in caracteres
    ]

    tem_letra = any(
        c.isalpha()
        for c in previsoes
    )

    tem_numero = any(
        c.isdigit()
        for c in previsoes
    )

    # Apenas letras
    if tem_letra and not tem_numero:
        return "".join(previsoes)

    # Apenas números
    if tem_numero and not tem_letra:
        return "".join(previsoes)

    resultado = []

    # Mistura letras + números
    for letra in caracteres:

        caractere = letra[
            "previsao"
        ]

        if caractere.isdigit():

            confianca_numero = letra[
                "confianca"
            ]

            confianca_letra = letra[
                "confianca_letra"
            ]

            if (
                confianca_letra
                >=
                confianca_numero * 0.60
            ):
                resultado.append(
                    letra["melhor_letra"]
                )

            else:
                resultado.append(
                    caractere
                )

        else:
            resultado.append(
                caractere
            )

    return "".join(resultado)

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