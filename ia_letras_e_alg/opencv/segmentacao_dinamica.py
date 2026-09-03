import cv2
import math
import numpy as np
import torch


def separar_palavras(linha, espacos):
    linha = sorted(
        linha,
        key=lambda letra: letra["x"]
    )

    palavras = []
    palavra_atual = []

    for indice, letra in enumerate(linha):
        palavra_atual.append(letra)

        if indice in espacos:
            palavras.append(palavra_atual)
            palavra_atual = []

    if palavra_atual:
        palavras.append(palavra_atual)

    return palavras


def recortar_palavra(binary, componentes, margem=2):
    x_inicio = min(
        item["x"]
        for item in componentes
    )

    y_inicio = min(
        item["y"]
        for item in componentes
    )

    x_fim = max(
        item["x"] + item["w"]
        for item in componentes
    )

    y_fim = max(
        item["y"] + item["h"]
        for item in componentes
    )

    x_inicio = max(
        0,
        x_inicio - margem
    )

    y_inicio = max(
        0,
        y_inicio - margem
    )

    x_fim = min(
        binary.shape[1],
        x_fim + margem
    )

    y_fim = min(
        binary.shape[0],
        y_fim + margem
    )

    imagem = binary[
        y_inicio:y_fim,
        x_inicio:x_fim
    ]

    return (
        imagem,
        x_inicio,
        y_inicio
    )


def normalizar_segmento(imagem, max_size=20):
    pontos = cv2.findNonZero(imagem)

    if pontos is None:
        return None

    x, y, w, h = cv2.boundingRect(
        pontos
    )

    imagem = imagem[
        y:y + h,
        x:x + w
    ]

    if w > h:
        new_w = max_size
        new_h = int(
            h * max_size / w
        )

    else:
        new_h = max_size
        new_w = int(
            w * max_size / h
        )

    new_w = max(
        1,
        new_w
    )

    new_h = max(
        1,
        new_h
    )

    imagem = cv2.resize(
        imagem,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    pad_top = (
        28 - new_h
    ) // 2

    pad_bottom = (
        28 - new_h
        - pad_top
    )

    pad_left = (
        28 - new_w
    ) // 2

    pad_right = (
        28 - new_w
        - pad_left
    )

    imagem = cv2.copyMakeBorder(
        imagem,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=0
    )

    return imagem


def gerar_cortes_candidatos(imagem):
    altura, largura = imagem.shape

    if largura <= 1:
        return (
            [0, largura],
            np.zeros(
                largura,
                dtype=np.float32
            )
        )

    projecao = np.count_nonzero(
        imagem,
        axis=0
    ).astype(np.float32)

    kernel = (
        np.ones(
            5,
            dtype=np.float32
        )
        / 5
    )

    suavizada = np.convolve(
        projecao,
        kernel,
        mode="same"
    )

    valores_internos = suavizada[
        1:-1
    ]

    if len(valores_internos) > 0:
        limite_vale = np.percentile(
            valores_internos,
            40
        )
    else:
        limite_vale = 0

    cortes = {
        0,
        largura
    }

    for x in range(
        1,
        largura - 1
    ):
        if (
            suavizada[x]
            <= suavizada[x - 1]
            and
            suavizada[x]
            <= suavizada[x + 1]
            and
            suavizada[x]
            <= limite_vale
        ):
            cortes.add(x)

    passo = max(
        2,
        int(
            altura * 0.18
        )
    )

    for x in range(
        passo,
        largura,
        passo
    ):
        cortes.add(x)

    return (
        sorted(cortes),
        projecao
    )


def gerar_segmentos(
    imagem,
    cortes
):
    altura = imagem.shape[0]

    largura_minima = max(
        3,
        int(
            altura * 0.15
        )
    )

    largura_maxima = max(
        largura_minima + 1,
        int(
            altura * 1.25
        )
    )

    segmentos = []

    for i in range(
        len(cortes) - 1
    ):
        for j in range(
            i + 1,
            len(cortes)
        ):
            inicio = cortes[i]
            fim = cortes[j]

            largura = (
                fim - inicio
            )

            if largura < largura_minima:
                continue

            if largura > largura_maxima:
                break

            recorte = imagem[
                :,
                inicio:fim
            ]

            if (
                np.count_nonzero(
                    recorte
                )
                < 5
            ):
                continue

            normalizado = (
                normalizar_segmento(
                    recorte
                )
            )

            if normalizado is None:
                continue

            segmentos.append({
                "inicio_indice": i,
                "fim_indice": j,
                "inicio": inicio,
                "fim": fim,
                "largura": largura,
                "imagem": normalizado
            })

    return segmentos


def classificar_segmentos(
    segmentos,
    model,
    labels,
    device
):
    if not segmentos:
        return

    imagens = np.stack([
        segmento["imagem"]
        for segmento in segmentos
    ])

    tensor = torch.from_numpy(
        imagens
    ).float()

    tensor = (
        tensor / 255.0
    )

    tensor = (
        tensor
        .unsqueeze(1)
        .to(device)
    )

    with torch.no_grad():
        output = model(
            tensor
        )

        probabilidades = torch.softmax(
            output,
            dim=1
        )

        confiancas, indices = (
            probabilidades.max(
                dim=1
            )
        )

    for (
        segmento,
        indice,
        confianca
    ) in zip(
        segmentos,
        indices,
        confiancas
    ):
        segmento["caractere"] = (
            labels[
                indice.item()
            ]
        )

        segmento["confianca"] = (
            confianca.item()
        )


def melhor_sequencia(
    imagem,
    cortes,
    segmentos,
    projecao
):
    quantidade = len(
        cortes
    )

    altura = imagem.shape[0]

    dp = [
        float("-inf")
        for _ in range(quantidade)
    ]

    anterior = [
        None
        for _ in range(quantidade)
    ]

    escolha = [
        None
        for _ in range(quantidade)
    ]

    dp[0] = 0.0

    segmentos_por_inicio = {}

    for segmento in segmentos:
        indice = segmento[
            "inicio_indice"
        ]

        if indice not in segmentos_por_inicio:
            segmentos_por_inicio[
                indice
            ] = []

        segmentos_por_inicio[
            indice
        ].append(
            segmento
        )

    for i in range(
        quantidade
    ):
        if dp[i] == float("-inf"):
            continue

        candidatos = (
            segmentos_por_inicio.get(
                i,
                []
            )
        )

        for segmento in candidatos:
            j = segmento[
                "fim_indice"
            ]

            confianca = max(
                segmento["confianca"],
                1e-8
            )

            score = ( math.log(confianca) + 0.25)

            proporcao = (
                segmento["largura"]
                / max(
                    altura,
                    1
                )
            )

            if proporcao < 0.15:
                score -= (
                    0.15
                    - proporcao
                ) * 4

            if proporcao > 1.0:
                score -= (
                    proporcao
                    - 1.0
                ) * 2

            if j < quantidade - 1:
                x_corte = cortes[j]

                if (
                    0
                    <= x_corte
                    < len(projecao)
                ):
                    ocupacao = (
                        projecao[x_corte]
                        / max(
                            altura,
                            1
                        )
                    )

                    score -= (
                        ocupacao * 0.8
                    )

            novo_score = (
                dp[i]
                + score
            )

            if novo_score > dp[j]:
                dp[j] = novo_score
                anterior[j] = i
                escolha[j] = segmento

    ultimo = (
        quantidade - 1
    )

    if (
        dp[ultimo]
        == float("-inf")
    ):
        return (
            "",
            [],
            float("-inf")
        )

    caminho = []

    atual = ultimo

    while atual != 0:
        segmento = escolha[
            atual
        ]

        if segmento is None:
            return (
                "",
                [],
                float("-inf")
            )

        caminho.append(
            segmento
        )

        atual = anterior[
            atual
        ]

    caminho.reverse()

    texto = "".join(
        segmento["caractere"]
        for segmento in caminho
    )

    return (
        texto,
        caminho,
        dp[ultimo]
    )


def reconhecer_palavra_dinamica(
    binary,
    componentes,
    model,
    labels,
    device
):
    (
        imagem,
        x_inicio,
        y_inicio
    ) = recortar_palavra(
        binary,
        componentes
    )

    (
        cortes,
        projecao
    ) = gerar_cortes_candidatos(
        imagem
    )

    segmentos = gerar_segmentos(
        imagem,
        cortes
    )

    classificar_segmentos(
        segmentos,
        model,
        labels,
        device
    )

    (
        texto,
        caminho,
        score
    ) = melhor_sequencia(
        imagem,
        cortes,
        segmentos,
        projecao
    )

    return {
        "texto": texto,
        "score": score,
        "segmentos": caminho,
        "x": x_inicio,
        "y": y_inicio,
        "imagem": imagem,
        "cortes": cortes
    }