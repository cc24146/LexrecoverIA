import cv2
import os
import torch
import sys

# Ajuste do path para o modelo
raiz_projeto = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if raiz_projeto not in sys.path:
    sys.path.append(raiz_projeto)

from cnn.model import CNN
# Importando as funções que criamos nos outros arquivos
from processamento import carregar_e_binarizar, extrair_contornos, processar_letra
from ordenacao import agrupar_em_linhas, reconstruir_texto

# Configuração do Modelo
model = CNN()
model.load_state_dict(torch.load(os.path.join(raiz_projeto, "cnn/best_model.pth")))
model.eval()

emnist_labels = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt"

# 1) Processamento da Imagem
imagem_caminho = "opencv/imagens/dificil.png"
image, image_boxes, binary = carregar_e_binarizar(imagem_caminho)

if image is None:
    print("Imagem não encontrada.")
    exit()

contours = extrair_contornos(binary)
print(f"Contornos encontrados: {len(contours)}")

letras = []

# 2) Loop de Predição
for contour in contours:
    if cv2.contourArea(contour) < 30:
        continue

    x, y, w, h, letra_nova = processar_letra(binary, contour)

    # Desenha na imagem de exibição
    cv2.rectangle(image_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Preparação para o PyTorch
    letra_tensor = torch.from_numpy(letra_nova).float() / 255.0
    letra_tensor = letra_tensor.unsqueeze(0).unsqueeze(0) 

    with torch.no_grad():
        output = model(letra_tensor)  
        predicted = output.argmax(dim=1).item()

    letras.append({
        "x": x, "y": y, "w": w, "h": h,
        "imagem": letra_nova,
        "previsao": emnist_labels[predicted]
    })

# 3) Ordenação e montagem do texto
linhas = agrupar_em_linhas(letras)
texto_completo = reconstruir_texto(
    linhas,
    binary
)

print("Texto detectado completo:\n")
print(texto_completo)

# 4) Exibir resultado visual
cv2.imshow("Retangulos", image_boxes)
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

# from model import CNN
# model = CNN()

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

