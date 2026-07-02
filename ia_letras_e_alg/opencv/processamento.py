import cv2
import torch

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