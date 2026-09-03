import cv2
import os
import sys
import torch


raiz_projeto = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

if raiz_projeto not in sys.path:
    sys.path.append(raiz_projeto)


from crnn.model import CRNN

from processamento import (
    carregar_e_binarizar,
    extrair_contornos,
    processar_letra
)

from ordenacao import (
    agrupar_em_linhas,
    reconstruir_texto,
    corrigir_palavra
)

from espacos import (
    detectar_espacos_linha
)

from segmentacao_dinamica import (
    separar_palavras,
    reconhecer_palavra_dinamica
)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


model = CRNN().to(device)

caminho_modelo = os.path.join(
    raiz_projeto,
    "crnn",
    "best_model.pth"
)

if not os.path.exists(caminho_modelo):
    print("Modelo não encontrado:")
    print(caminho_modelo)
    print("Treine o modelo antes de executar o OCR.")
    sys.exit()


model.load_state_dict(
    torch.load(
        caminho_modelo,
        map_location=device
    )
)

model.eval()


emnist_labels = (
    "0123456789"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abdefghnqrt"
)


imagem_caminho = os.path.join(
    raiz_projeto,
    "opencv",
    "imagens",
    "image.png"
)


(
    image,
    image_boxes,
    binary
) = carregar_e_binarizar(
    imagem_caminho
)


if image is None:
    print("Imagem não encontrada:")
    print(imagem_caminho)
    sys.exit()


contours = extrair_contornos(
    binary
)


print(
    f"Contornos encontrados: "
    f"{len(contours)}"
)


letras = []

debug_dir = os.path.join(
    raiz_projeto,
    "opencv",
    "debug_caracteres"
)

os.makedirs(
    debug_dir,
    exist_ok=True
)

contador_debug = 0


for contour in contours:

    area = cv2.contourArea(
        contour
    )

    if area < 30:
        continue


    (
        x,
        y,
        w,
        h,
        letra_nova
    ) = processar_letra(
        binary,
        contour
    )

    cv2.rectangle(
        image_boxes,
        (x, y),
        (x + w, y + h),
        (0, 0, 255),
        2
    )

    contador_debug += 1

    cv2.imwrite(
        os.path.join(
            debug_dir,
            f"{contador_debug:03d}.png"
        ),
        letra_nova
    )

    letra_normalizada = (
        letra_nova.astype("float32")
        / 255.0
    )


    letra_tensor = torch.from_numpy(
        letra_normalizada
    )

    letra_tensor = (
        letra_tensor
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )


    with torch.no_grad():

        output = model(
            letra_tensor
        )

        probabilidades = torch.softmax(
            output,
            dim=1
        )

        (
            confianca_geral,
            predicted
        ) = probabilidades.max(
            dim=1
        )

        predicted = (
            predicted.item()
        )

        confianca_geral = (
            confianca_geral.item()
        )


        probabilidades_letras = (
            probabilidades[:, 10:]
        )

        (
            confianca_letra,
            predicted_letra
        ) = probabilidades_letras.max(
            dim=1
        )

        predicted_letra = (
            predicted_letra.item()
            + 10
        )

        confianca_letra = (
            confianca_letra.item()
        )


    caractere = emnist_labels[
        predicted
    ]

    melhor_letra = emnist_labels[
        predicted_letra
    ]


    letras.append({
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "imagem": letra_nova,
        "previsao": caractere,
        "melhor_letra": melhor_letra,
        "confianca": confianca_geral,
        "confianca_letra": confianca_letra
    })


print(
    f"Caracteres utilizados: "
    f"{len(letras)}"
)


linhas = agrupar_em_linhas(
    letras
)


print(
    f"Linhas encontradas: "
    f"{len(linhas)}"
)


espacos_por_linha = []


for numero, linha in enumerate(
    linhas
):

    (
        espacos,
        gaps,
        limiar
    ) = detectar_espacos_linha(
        binary,
        linha
    )

    espacos_por_linha.append(
        espacos
    )


    print()
    print(
        f"--- Linha {numero + 1} ---"
    )

    print(
        "Caracteres:",
        "".join(
            letra["previsao"]
            for letra in linha
        )
    )

    print(
        "Gaps:",
        gaps
    )

    print(
        f"Limiar de espaço: "
        f"{limiar:.2f}px"
    )

    print(
        "Espaço depois dos índices:",
        sorted(espacos)
    )


    for indice in espacos:

        if indice >= len(linha) - 1:
            continue

        anterior = linha[
            indice
        ]

        proxima = linha[
            indice + 1
        ]

        fim_anterior = (
            anterior["x"]
            + anterior["w"]
        )

        inicio_proxima = (
            proxima["x"]
        )

        x_espaco = (
            fim_anterior
            + inicio_proxima
        ) // 2

        y_inicio = min(
            letra["y"]
            for letra in linha
        )

        y_fim = max(
            letra["y"]
            + letra["h"]
            for letra in linha
        )

        cv2.line(
            image_boxes,
            (x_espaco, y_inicio),
            (x_espaco, y_fim),
            (255, 0, 0),
            2
        )

texto_dinamico_linhas = []

for numero_linha, linha in enumerate(
    linhas
):
    espacos = espacos_por_linha[
        numero_linha
    ]

    palavras = separar_palavras(
        linha,
        espacos
    )

    palavras_reconhecidas = []

    for palavra in palavras:

        resultado = reconhecer_palavra_dinamica(
            binary,
            palavra,
            model,
            emnist_labels,
            device
        )

        texto_palavra = resultado[
            "texto"
        ]

        if not texto_palavra:
            texto_palavra = corrigir_palavra(
                palavra
            )

        palavras_reconhecidas.append(
            texto_palavra
        )

        altura_palavra = resultado[
            "imagem"
        ].shape[0]

        for segmento in resultado[
            "segmentos"
        ]:

            x_inicio_segmento = (
                resultado["x"]
                + segmento["inicio"]
            )

            x_fim_segmento = (
                resultado["x"]
                + segmento["fim"]
            )

            y_inicio_segmento = (
                resultado["y"]
            )

            y_fim_segmento = (
                resultado["y"]
                + altura_palavra
            )

            cv2.rectangle(
                image_boxes,
                (
                    x_inicio_segmento,
                    y_inicio_segmento
                ),
                (
                    x_fim_segmento,
                    y_fim_segmento
                ),
                (0, 255, 255),
                1
            )

    texto_dinamico_linhas.append(
        " ".join(
            palavras_reconhecidas
        )
    )


texto_dinamico = "\n".join(
    texto_dinamico_linhas
)

texto_completo = reconstruir_texto(
    linhas,
    espacos_por_linha
)


print()
print("==============================")
print("TEXTO DETECTADO")
print("==============================")
print()

print(
    texto_completo
)

print()
print("==============================")
print("SEGMENTACAO DINAMICA")
print("==============================")
print()

print(
    texto_dinamico
)


cv2.imshow(
    "OCR - Retangulos e Espacos",
    image_boxes
)

cv2.waitKey(0)
cv2.destroyAllWindows()



# import cv2
# import os
# import torch
# import sys
# import torchvision.transforms as transforms


# raiz_projeto = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# if raiz_projeto not in sys.path:
#     sys.path.append(raiz_projeto)

# from crnn.model import CRNN
# model = CRNN()

# model.load_state_dict(torch.load("best_model.pth"))

# model.eval()


# emnist_labels = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt"

# # ---------------------------------------------------
# # 1) Ler imagem
# # ---------------------------------------------------

# image = cv2.imread("opencv/imagens/dificil.png")

# if image is None:
#     print("Imagem não encontrada.")
#     exit()

# # Faz uma cópia apenas para desenhar os retângulos
# image_boxes = image.copy()

# # ---------------------------------------------------
# # 2) Converter para cinza
# # ---------------------------------------------------

# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # ---------------------------------------------------
# # 3) Binarização
# # ---------------------------------------------------

# _, binary = cv2.threshold(
#     gray,
#     127,
#     255,
#     cv2.THRESH_BINARY_INV
# )

# # ---------------------------------------------------
# # 4) Encontrar contornos
# # ---------------------------------------------------

# contours, hierarchy = cv2.findContours(
#     binary,
#     cv2.RETR_EXTERNAL,
#     cv2.CHAIN_APPROX_SIMPLE
# )

# print(f"Contornos encontrados: {len(contours)}")

# # ---------------------------------------------------
# # 5) Criar pasta para salvar caracteres
# # ---------------------------------------------------



# contador = 0

# # ---------------------------------------------------
# # 6) Percorrer cada contorno
# # ---------------------------------------------------

# letras = []

# for contour in contours:

#     area = cv2.contourArea(contour)

#     # Ignora pequenos ruídos
#     if area < 30:
#         continue

#     x, y, w, h = cv2.boundingRect(contour)

#     # Desenha um retângulo
#     cv2.rectangle(
#         image_boxes,
#         (x, y),
#         (x + w, y + h),
#         (0, 255, 0),
#         2
#     )

#     # Recorta o caractere
#     letra = binary[y:y+h, x:x+w]

#     falta_w = max(0, 28 - w)
#     falta_h = max(0, 28 - h)
 
#     max_size = 20
    
#     if w > h:
#         new_w = max_size
#         new_h = int(h * (max_size / w))
#     else:
#         new_h = max_size
#         new_w = int(w * (max_size / h))
        
#     new_w = max(1, new_w)
#     new_h = max(1, new_h)

#     letra_redimensionada = cv2.resize(letra, (new_w, new_h), interpolation=cv2.INTER_AREA)

#     pad_top = (28 - new_h) // 2
#     pad_bottom = 28 - new_h - pad_top
#     pad_left = (28 - new_w) // 2
#     pad_right = 28 - new_w - pad_left

#     letra_nova = cv2.copyMakeBorder(
#         letra_redimensionada,
#         pad_top, pad_bottom, pad_left, pad_right,
#         cv2.BORDER_CONSTANT,
#         value=0
#     )
#     contador += 1


#     letra_tensor = torch.from_numpy(letra_nova).float()
    
#     letra_tensor = letra_tensor / 255.0
    
#     letra_tensor = letra_tensor.unsqueeze(0).unsqueeze(0) 

#     with torch.no_grad():
#         output = model(letra_tensor)  
#         predicted = output.argmax(dim=1).item()

#     # print(f"Imagem {contador:03d}.png previsto como:", emnist_labels[predicted])
#     # nome = f"opencv/caracteres/{contador:03d}.png"

#     # cv2.imwrite(nome, letra_nova)

#     letras.append({
#         "x" : x,
#         "y" : y,
#         "w" : w,
#         "h" : h,
#         "imagem" : letra_nova,
#         "previsao" : emnist_labels[predicted]
#     })

# print(f"Caracteres salvos: {contador}")

# letras.sort(key=lambda l: l["y"])
# linhas = []
# linha_atual = []

# for l in letras:
#     if not linha_atual:
#         linha_atual.append(l)
#     else:
#         y_medio_linha = sum(item["y"] for item in linha_atual) / len(linha_atual)
#         altura_referencia = l["imagem"].shape[0] 
#         tolerancia = altura_referencia * 0.7 
        
#         if abs(l["y"] - y_medio_linha) < tolerancia:
#             linha_atual.append(l)
#         else:
#             linhas.append(linha_atual)
#             linha_atual = [l]
# if linha_atual:
#     linhas.append(linha_atual)

# letras_ordenadas_lista = []
# contador_ordenado = 0

# for i, linha in enumerate(linhas):

#     linha.sort(key=lambda l: l["x"])
#     largura_media = sum(letra["w"] for letra in linha) / len(linha)
    
#     for indice, l in enumerate(linha):
#         contador_ordenado += 1
        
#         if indice > 0:
#             letra_anterior = linha[indice - 1]

#             fim_anterior = letra_anterior["x"] + letra_anterior["w"]

#             distancia_horizontal = l["x"] - fim_anterior
#             largura_referencia = (letra_anterior["w"] + l["w"]) / 2
        
#             if distancia_horizontal > largura_referencia * 0.8:
#                 letras_ordenadas_lista.append(" ")

        
#         letras_ordenadas_lista.append(l["previsao"])
    
#     if i < len(linhas) - 1:
#         letras_ordenadas_lista.append("\n")

# texto_completo = "".join(letras_ordenadas_lista)
# print("Texto detectado completo:\n")
# print(texto_completo)




# # ---------------------------------------------------
# # 7) Mostrar resultado
# # ---------------------------------------------------

# cv2.imshow("Retangulos", image_boxes)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

