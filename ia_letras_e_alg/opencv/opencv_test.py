import cv2
import os


# ---------------------------------------------------
# 1) Ler imagem
# ---------------------------------------------------

image = cv2.imread("opencv/imagens/pagina.png")

if image is None:
    print("Imagem não encontrada.")
    exit()

# Faz uma cópia apenas para desenhar os retângulos
image_boxes = image.copy()

# ---------------------------------------------------
# 2) Converter para cinza
# ---------------------------------------------------

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# ---------------------------------------------------
# 3) Binarização
# ---------------------------------------------------

_, binary = cv2.threshold(
    gray,
    127,
    255,
    cv2.THRESH_BINARY_INV
)

# ---------------------------------------------------
# 4) Encontrar contornos
# ---------------------------------------------------

contours, hierarchy = cv2.findContours(
    binary,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

print(f"Contornos encontrados: {len(contours)}")

# ---------------------------------------------------
# 5) Criar pasta para salvar caracteres
# ---------------------------------------------------

os.makedirs("opencv/caracteres", exist_ok=True)

contador = 0

# ---------------------------------------------------
# 6) Percorrer cada contorno
# ---------------------------------------------------

for contour in contours:

    area = cv2.contourArea(contour)

    # Ignora pequenos ruídos
    if area < 30:
        continue

    x, y, w, h = cv2.boundingRect(contour)

    # Desenha um retângulo
    cv2.rectangle(
        image_boxes,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2
    )

    # Recorta o caractere
    letra = binary[y:y+h, x:x+w]

    falta_w = max(0, 28 - w)
    falta_h = max(0, 28 - h)
 
    max_size = 20
    
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
    contador += 1


    nome = f"opencv/caracteres/{contador:03d}.png"

    cv2.imwrite(nome, letra_nova)

print(f"Caracteres salvos: {contador}")

def retornarContador():
    return contador
# ---------------------------------------------------
# 7) Mostrar resultado
# ---------------------------------------------------

cv2.imshow("Retangulos", image_boxes)

cv2.waitKey(0)
cv2.destroyAllWindows()

