import hashlib
import json
from datetime import datetime
from pathlib import Path

import cv2

from opencv.main_ctc import detectar_linhas, PROJECT_ROOT


REFERENCIAS = Path(
    r"C:\Users\Admin\.codex\.chatgpt-projects"
    r"\g-p-6a7495fbdef08191a7b8f96807c971b2"
    r"\avaliacao_ocr_20260919\referencias.json"
)


def salvar_imagem(caminho, imagem):
    if not cv2.imwrite(str(caminho), imagem):
        raise OSError(f"Não foi possível salvar: {caminho}")


def main():
    dados = json.loads(
        REFERENCIAS.read_text(encoding="utf-8")
    )

    identificador = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    destino = (
        Path(PROJECT_ROOT)
        / "opencv"
        / "recortes_revisao"
        / identificador
    )

    destino.mkdir(parents=True, exist_ok=False)

    # Guarda as referências usadas nesta exportação.
    (destino / "referencias_usadas.json").write_text(
        json.dumps(dados, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    registros = []

    for amostra in dados["samples"]:
        origem = Path(amostra["image_path"])

        assinatura = hashlib.sha256(
            origem.read_bytes()
        ).hexdigest().upper()

        if assinatura != amostra["current_image_sha256"]:
            raise RuntimeError(
                f"A imagem mudou desde a revisão: {origem}"
            )

        imagem = cv2.imread(str(origem))

        if imagem is None:
            raise FileNotFoundError(origem)

        resultados = detectar_linhas(imagem)
        referencias = amostra["reference_lines"]

        pasta = destino / f"manuscrito_{amostra['id']:02d}"
        pasta.mkdir()

        mesma_quantidade = (
            len(resultados) == len(referencias)
        )

        for indice, resultado in enumerate(resultados, start=1):
            x, y, largura, altura = (
                int(valor)
                for valor in resultado["caixa"]
            )

            nome = f"linha_{indice:02d}"

            salvar_imagem(
                pasta / f"{nome}_limpa.png",
                resultado["imagem"]
            )

            salvar_imagem(
                pasta / f"{nome}_original.png",
                imagem[y:y + altura, x:x + largura]
            )

            # Mesmo com contagens iguais, a associação
            # precisa ser conferida visualmente.
            sugestao = None

            if mesma_quantidade:
                sugestao = referencias[indice - 1]["text"]

            registros.append({
                "manuscrito": amostra["id"],
                "recorte": indice,
                "origem": str(origem),
                "sha256_origem": assinatura,
                "caixa": [x, y, largura, altura],
                "imagem_limpa": (
                    f"{pasta.name}/{nome}_limpa.png"
                ),
                "imagem_original": (
                    f"{pasta.name}/{nome}_original.png"
                ),
                "texto_sugerido": sugestao,
                "aprovado": False,
                "quantidade_compativel": mesma_quantidade,
            })

        # Salva o progresso após cada imagem.
        (destino / "manifesto.json").write_text(
            json.dumps(
                registros,
                ensure_ascii=False,
                indent=2
            ),
            encoding="utf-8"
        )

        situacao = (
            "conferir conteúdo"
            if mesma_quantidade
            else "REVISAR QUANTIDADE"
        )

        print(
            f"Manuscrito {amostra['id']}: "
            f"{len(resultados)} recortes / "
            f"{len(referencias)} referências — {situacao}"
        )

    print(f"\nExportação concluída:\n{destino}")


if __name__ == "__main__":
    main()