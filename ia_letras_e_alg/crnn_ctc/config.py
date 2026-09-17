CARACTERES = (
    " "
    "0123456789"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "ÀÁÉÍÓÚ"
    "àáâãçèéêíóôõúü"
    ".,-!?;:'\"()%/ºª°"
)

BLANK_INDEX = 0

CHAR_TO_INDEX = {
    caractere: indice + 1
    for indice, caractere in enumerate(CARACTERES)
}

INDEX_TO_CHAR = {
    indice + 1: caractere
    for indice, caractere in enumerate(CARACTERES)
}

NUM_CLASSES = len(CARACTERES) + 1