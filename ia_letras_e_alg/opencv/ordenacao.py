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

def reconstruir_texto(linhas):
    letras_ordenadas_lista = []

    for i, linha in enumerate(linhas):
        linha.sort(key=lambda l: l["x"])
        
        for indice, l in enumerate(linha):
            if indice > 0:
                letra_anterior = linha[indice - 1]
                fim_anterior = letra_anterior["x"] + letra_anterior["w"]
                distancia_horizontal = l["x"] - fim_anterior
                largura_referencia = (letra_anterior["w"] + l["w"]) / 2
            
                if distancia_horizontal > largura_referencia * 0.8:
                    letras_ordenadas_lista.append(" ")
            
            letras_ordenadas_lista.append(l["previsao"])
        
        if i < len(linhas) - 1:
            letras_ordenadas_lista.append("\n")

    return "".join(letras_ordenadas_lista)