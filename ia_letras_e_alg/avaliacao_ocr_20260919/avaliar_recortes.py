"""Diagnóstico dos recortes revisados. Execute manualmente; não treina."""
import hashlib
import json
import sys
import unicodedata
from datetime import datetime
from pathlib import Path

import cv2

PROJETO = Path(r"C:\Users\Admin\Documents\GitHub\LexrecoverIA\ia_letras_e_alg")
sys.path.insert(0, str(PROJETO))

from opencv.reconhecimento_ctc import ReconhecedorCTC, MODEL_PATH
from crnn_ctc.pos_processamento import corrigir_texto
from crnn_ctc.metrics import distancia_levenshtein

PASTA = Path(__file__).resolve().parent


def normalizar(texto):
    texto = unicodedata.normalize("NFC", texto).lower()
    texto = "".join(
        c if c.isalnum() or c.isspace() else " "
        for c in texto
    )
    return " ".join(texto.split())


def main():
    documento = json.loads(
        (PASTA / "manifesto_revisado.json").read_text(encoding="utf-8")
    )
    candidatos = [
        r for r in documento["registros"]
        if r["usar_avaliacao_por_linha"]
    ]
    if not candidatos:
        raise RuntimeError("Não há recortes selecionados.")

    modelo = ReconhecedorCTC()
    totais = {
        "caracteres": 0, "palavras": 0,
        "bruto_caracteres": 0, "corrigido_caracteres": 0,
        "bruto_palavras": 0, "corrigido_palavras": 0,
    }
    linhas = []

    for indice, registro in enumerate(candidatos, start=1):
        caminho = Path(registro["imagem_limpa_absoluta"])
        imagem = cv2.imread(str(caminho))
        if imagem is None:
            raise FileNotFoundError(caminho)

        referencia = normalizar(registro["texto_referencia"])
        if not referencia:
            raise ValueError(f"Referência vazia: {registro['reference_id']}")

        bruto = modelo.reconhecer(imagem)
        corrigido = corrigir_texto(bruto)
        totais["caracteres"] += len(referencia)
        totais["palavras"] += len(referencia.split())

        medidas = {}
        for nome, texto in (("bruto", bruto), ("corrigido", corrigido)):
            previsao = normalizar(texto)
            erros_c = distancia_levenshtein(referencia, previsao)
            erros_p = distancia_levenshtein(
                referencia.split(), previsao.split()
            )
            totais[nome + "_caracteres"] += erros_c
            totais[nome + "_palavras"] += erros_p
            medidas[nome] = {
                "edicoes_caracteres": erros_c,
                "edicoes_palavras": erros_p,
                "CER": erros_c / len(referencia),
                "WER": erros_p / len(referencia.split()),
            }

        linhas.append({
            "id": registro["reference_id"],
            "imagem": str(caminho),
            "sha256_recorte": hashlib.sha256(caminho.read_bytes()).hexdigest(),
            "referencia": registro["texto_referencia"],
            "bruto": bruto,
            "corrigido": corrigido,
            "medidas": medidas,
        })
        print(f"{indice}/{len(candidatos)} — {registro['reference_id']}")

    resumo = {}
    for nome in ("bruto", "corrigido"):
        resumo[nome] = {
            "CER": totais[nome + "_caracteres"] / totais["caracteres"],
            "WER": totais[nome + "_palavras"] / totais["palavras"],
        }
        print(
            f"{nome}: CER {resumo[nome]['CER']:.2%} | "
            f"WER {resumo[nome]['WER']:.2%}"
        )

    resultado = {
        "escopo": "diagnóstico de desenvolvimento por linha; referências provisórias",
        "modelo": str(MODEL_PATH),
        "sha256_modelo": hashlib.sha256(Path(MODEL_PATH).read_bytes()).hexdigest(),
        "normalizacao": "NFC, minúsculas, pontuação como espaço, espaços compactados",
        "numero_recortes": len(linhas),
        "totais": totais,
        "resumo": resumo,
        "linhas": linhas,
    }
    saida = PASTA / (
        "resultado_recortes_"
        + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        + ".json"
    )
    with saida.open("x", encoding="utf-8") as arquivo:
        json.dump(resultado, arquivo, ensure_ascii=False, indent=2)
    print(f"Resultado salvo em: {saida}")


if __name__ == "__main__":
    main()

