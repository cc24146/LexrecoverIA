import numpy as np
from statistics import median


def calcular_projecao_vertical(
    binary,
    linha,
    margem_y=2
):
    x_inicio = min(
        letra["x"]
        for letra in linha
    )

    x_fim = max(
        letra["x"] + letra["w"]
        for letra in linha
    )

    y_inicio = min(
        letra["y"]
        for letra in linha
    )

    y_fim = max(
        letra["y"] + letra["h"]
        for letra in linha
    )

    y_inicio = max(
        0,
        y_inicio - margem_y
    )

    y_fim = min(
        binary.shape[0],
        y_fim + margem_y
    )

    recorte = binary[
        y_inicio:y_fim,
        x_inicio:x_fim
    ]

    projecao = np.count_nonzero(
        recorte,
        axis=0
    )

    altura = recorte.shape[0]

    tolerancia = max(
        1,
        int(altura * 0.03)
    )

    return (
        projecao,
        x_inicio,
        tolerancia
    )


def encontrar_gaps(
    projecao,
    x_inicio,
    tolerancia
):
    gaps = []

    inicio_gap = None

    for indice, valor in enumerate(projecao):

        if valor <= tolerancia:

            if inicio_gap is None:
                inicio_gap = indice

        else:

            if inicio_gap is not None:

                fim_gap = indice - 1

                gaps.append({
                    "inicio": (
                        x_inicio + inicio_gap
                    ),
                    "fim": (
                        x_inicio + fim_gap
                    ),
                    "largura": (
                        fim_gap
                        - inicio_gap
                        + 1
                    )
                })

                inicio_gap = None

    if inicio_gap is not None:

        fim_gap = len(projecao) - 1

        gaps.append({
            "inicio": x_inicio + inicio_gap,
            "fim": x_inicio + fim_gap,
            "largura": (
                fim_gap
                - inicio_gap
                + 1
            )
        })

    return gaps


def medir_gaps_entre_caracteres(
    linha,
    gaps_projecao
):
    linha = sorted(
        linha,
        key=lambda letra: letra["x"]
    )

    medidas = []

    for i in range(len(linha) - 1):

        anterior = linha[i]
        atual = linha[i + 1]

        inicio_intervalo = (
            anterior["x"]
            + anterior["w"]
        )

        fim_intervalo = atual["x"]

        if fim_intervalo <= inicio_intervalo:
            medidas.append(0)
            continue

        maior_gap = 0

        for gap in gaps_projecao:

            inicio = max(
                inicio_intervalo,
                gap["inicio"]
            )

            fim = min(
                fim_intervalo - 1,
                gap["fim"]
            )

            if fim >= inicio:

                largura = (
                    fim - inicio + 1
                )

                maior_gap = max(
                    maior_gap,
                    largura
                )

        medidas.append(maior_gap)

    return medidas


def calcular_limiar_adaptativo(
    gaps,
    larguras
):
    gaps_validos = [
        gap
        for gap in gaps
        if gap > 0
    ]

    largura_referencia = median(
        larguras
    )

    fallback = max(
        3,
        largura_referencia * 0.8
    )

    if len(gaps_validos) < 3:
        return fallback

    centro_1 = float(
        min(gaps_validos)
    )

    centro_2 = float(
        max(gaps_validos)
    )

    if centro_1 == centro_2:
        return fallback

    # K-means com dois grupos
    for _ in range(20):

        grupo_1 = []
        grupo_2 = []

        for gap in gaps_validos:

            if (
                abs(gap - centro_1)
                <=
                abs(gap - centro_2)
            ):
                grupo_1.append(gap)

            else:
                grupo_2.append(gap)

        if not grupo_1 or not grupo_2:
            return fallback

        novo_1 = (
            sum(grupo_1)
            / len(grupo_1)
        )

        novo_2 = (
            sum(grupo_2)
            / len(grupo_2)
        )

        if (
            abs(novo_1 - centro_1) < 0.01
            and
            abs(novo_2 - centro_2) < 0.01
        ):
            break

        centro_1 = novo_1
        centro_2 = novo_2

    centro_letras = min(
        centro_1,
        centro_2
    )

    centro_palavras = max(
        centro_1,
        centro_2
    )

    separacao = (
        centro_palavras
        - centro_letras
    )

    separacao_minima = max(
        3,
        largura_referencia * 0.35
    )

    if separacao < separacao_minima:
        return fallback

    return (
        centro_letras
        + centro_palavras
    ) / 2


def detectar_espacos_linha(
    binary,
    linha
):
    linha = sorted(
        linha,
        key=lambda letra: letra["x"]
    )

    if len(linha) < 2:
        return set(), [], 0

    (
        projecao,
        x_inicio,
        tolerancia
    ) = calcular_projecao_vertical(
        binary,
        linha
    )

    gaps_projecao = encontrar_gaps(
        projecao,
        x_inicio,
        tolerancia
    )

    gaps = medir_gaps_entre_caracteres(
        linha,
        gaps_projecao
    )

    distancias = medir_distancias_horizontais(
        linha
    )

    larguras = [
        letra["w"]
        for letra in linha
    ]

    limiar_projecao = calcular_limiar_adaptativo(
        gaps,
        larguras
    )

    limiar_distancia = calcular_limiar_adaptativo(
        distancias,
        larguras
    )

    indices_espacos = set()

    for i in range(len(linha) - 1):

        espaco_por_projecao = (
            gaps[i] > limiar_projecao
        )

        espaco_por_distancia = (
            distancias[i] > limiar_distancia
            and
            distancias[i] > median(larguras) * 0.40
        )

        if (
            espaco_por_projecao
            or espaco_por_distancia
        ):
            indices_espacos.add(i)

    return (
        indices_espacos,
        gaps,
        limiar_projecao
    )

def medir_distancias_horizontais(linha):
    linha = sorted(
        linha,
        key=lambda letra: letra["x"]
    )

    distancias = []

    for i in range(len(linha) - 1):
        anterior = linha[i]
        atual = linha[i + 1]

        fim_anterior = (
            anterior["x"]
            + anterior["w"]
        )

        distancia = (
            atual["x"]
            - fim_anterior
        )

        distancias.append(
            max(0, distancia)
        )

    return distancias