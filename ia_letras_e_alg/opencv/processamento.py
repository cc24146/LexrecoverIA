import cv2
import torch
import numpy as np


def calcular_projecao_vertical(binary, linha, margem_y=2): #margem 2 para segurança

    #descobre a area que cada linha ocupa 
    x_inicio = min(letra["x"] for letra in linha) #descobre começo da linha de acordo com o menor valor de x encontrado
    x_fim = max(
        letra["x"] + letra["w"]
        for letra in linha
    )#descobre o final da linha de acordo com o a soma do maior tamanho de x encontrado com a largura calculada da letra 


    #mesma logica do x
    y_inicio = min(letra["y"] for letra in linha)
    y_fim = max(
        letra["y"] + letra["h"]
        for letra in linha
    )

    #aplicação da margem vertical 
    y_inicio = max(0, y_inicio - margem_y)#max evita utilizar cordenada negativa(inexistente)
    y_fim = min(binary.shape[0], y_fim + margem_y)#mesma logica de cima (binary.shape é a altura total da imagem) 

    #recorta apenas essa linha
    recorte_linha = binary[
        y_inicio:y_fim,
        x_inicio:x_fim
    ]

    #cntagem de pixels brancos por coluna
    projecao = np.count_nonzero(
        recorte_linha,
        axis=0 #vertical
    )

    return projecao, x_inicio

def encontrar_gaps_verticais(projecao,x_inicio,tolerancia=0): #tolerancia define quando é considerado uma coluna vazia
    gaps = []

    inicio_gap = None #define se esta dentro de um espaço vazio ou não, caso diferente de none indica a posição inicial do gap

    for i, valor in enumerate(projecao):

        coluna_vazia = valor <= tolerancia # verificacao de coluna vazia

        if coluna_vazia: 
            if inicio_gap is None:
                inicio_gap = i # determina o começo do gap ao chegar em uma coluna vazia

        else:
            if inicio_gap is not None:
                fim_gap = i - 1 # identifica o indice de quando acaba o gap decrescendo no valor do indice de acesso atual 

                largura = fim_gap - inicio_gap + 1

                gaps.append({
                    "inicio": x_inicio + inicio_gap,
                    "fim": x_inicio + fim_gap,
                    "largura": largura
                })

                inicio_gap = None

    #tratamento necessario caso o vetor de projecao termine em um gap
    if inicio_gap is not None:
        fim_gap = len(projecao) - 1

        gaps.append({
            "inicio": x_inicio + inicio_gap,
            "fim": x_inicio + fim_gap,
            "largura": fim_gap - inicio_gap + 1
        })

    return gaps

def calcular_limite_espaco(gaps, linha):

    #pega as larguras, ignora os gaps de 1 pixel e coloca em ordem crescente
    larguras = sorted(
        gap["largura"]
        for gap in gaps
        if gap["largura"] >= 2
    )

    if len(larguras) < 2: # ??? nn entendi isso aqui direito, vou rever
        largura_mediana = np.median([
            letra["w"]
            for letra in linha
        ])

        return largura_mediana * 0.7

    #faz o calculo dos gaps, pegando a largura de i + 1 e subitraindo largura de i
    saltos = []

    for i in range(len(larguras) - 1):
        salto = larguras[i + 1] - larguras[i]

        saltos.append(salto)

    #identifica qual foi o maior salto feito
    indice_maior_salto = np.argmax(saltos)

    menor_gap_palavra = larguras[
        indice_maior_salto + 1
    ]

    maior_gap_letra = larguras[
        indice_maior_salto
    ]

    #limite para passar a ser espaço entre palavra ou letra
    limite = (
        maior_gap_letra
        + menor_gap_palavra
    ) / 2

    return limite

def detectar_espacos_por_projecao(binary, linha):
    #calculo de projecao
    projecao, x_inicio = calcular_projecao_vertical(
        binary,
        linha
    )

    #encontra gaps
    gaps = encontrar_gaps_verticais(
        projecao,
        x_inicio,
        tolerancia=0
    )
    
    #calcul de limite que separa gap de letras e de linhas
    limite = calcular_limite_espaco(
        gaps,
        linha
    )
    
    #guarda os gaps grandes
    espacos_palavras = [
        gap
        for gap in gaps
        if gap["largura"] > limite
    ]
    return espacos_palavras

def carregar_e_binarizar(caminho_imagem):
    image = cv2.imread(caminho_imagem)
    if image is None:
        return None, None, None
    
    image_boxes = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    
    return image, image_boxes, binary

def extrair_contornos(binary):
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def processar_letra(binary, contour, max_size=20):
    x, y, w, h = cv2.boundingRect(contour)
    letra = binary[y:y+h, x:x+w]

    if w > h:
        new_w = max_size
        new_h = int(h * (max_size / w))
    else:
        new_h = max_size
        new_w = int(w * (max_size / h))
        
    new_w = max(1, new_w)
    new_h = max(1, new_h)

    letra_redimensionada = cv2.resize(letra, (new_w, new_h), interpolation=cv2.INTER_AREA)

    pad_top = (28 - new_h) // 2
    pad_bottom = 28 - new_h - pad_top
    pad_left = (28 - new_w) // 2
    pad_right = 28 - new_w - pad_left

    letra_nova = cv2.copyMakeBorder(
        letra_redimensionada,
        pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT,
        value=0
    )
    return x, y, w, h, letra_nova