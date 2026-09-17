def distancia_levenshtein(
    referencia,
    previsao
):
    anterior = list(
        range(
            len(previsao) + 1
        )
    )

    for i, caractere_ref in enumerate(
        referencia,
        start=1
    ):
        atual = [i]

        for j, caractere_pred in enumerate(
            previsao,
            start=1
        ):
            custo = (
                0
                if caractere_ref == caractere_pred
                else 1
            )

            atual.append(
                min(
                    atual[j - 1] + 1,
                    anterior[j] + 1,
                    anterior[j - 1] + custo
                )
            )

        anterior = atual

    return anterior[-1]


def calcular_cer(
    referencias,
    previsoes
):
    erros = 0
    total_caracteres = 0

    for referencia, previsao in zip(
        referencias,
        previsoes
    ):
        erros += distancia_levenshtein(
            referencia,
            previsao
        )

        total_caracteres += len(
            referencia
        )

    if total_caracteres == 0:
        return 0.0

    return (
        erros
        / total_caracteres
    )