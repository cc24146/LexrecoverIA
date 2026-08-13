from processamento import detectar_espacos_por_projecao

def agrupar_em_linhas(letras):
    letras.sort(key=lambda l: l["y"])
    linhas = []
    linha_atual = []

    for l in letras:
        if not linha_atual:
            linha_atual.append(l)
        else:
            y_medio_linha = sum(item["y"] for item in linha_atual) / len(linha_atual)
            altura_referencia = l["imagem"].shape[0] 
            tolerancia = altura_referencia * 0.7 
            
            if abs(l["y"] - y_medio_linha) < tolerancia:
                linha_atual.append(l)
            else:
                linhas.append(linha_atual)
                linha_atual = [l]
    if linha_atual:
        linhas.append(linha_atual)
    return linhas

def reconstruir_texto(linhas, binary):
    letras_ordenadas_lista = []

    for i, linha in enumerate(linhas):

        linha.sort(key=lambda l: l["x"])

        espacos = detectar_espacos_por_projecao(
            binary,
            linha
        )

        for indice, l in enumerate(linha):

            if indice > 0:

                anterior = linha[indice - 1]

                fim_anterior = (
                    anterior["x"]
                    + anterior["w"]
                )

                inicio_atual = l["x"]

                tem_espaco = False

                for gap in espacos:

                    if (
                        gap["inicio"] >= fim_anterior
                        and
                        gap["fim"] <= inicio_atual
                    ):
                        tem_espaco = True
                        break

                if tem_espaco:
                    letras_ordenadas_lista.append(" ")

            letras_ordenadas_lista.append(
                l["previsao"]
            )

        if i < len(linhas) - 1:
            letras_ordenadas_lista.append("\n")

    return "".join(letras_ordenadas_lista)