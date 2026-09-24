import os
import sys

import cv2
import numpy as np


PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(
        0,
        PROJECT_ROOT
    )


from opencv.reconhecimento_ctc import ReconhecedorCTC
from crnn_ctc.pos_processamento import analisar_texto

def filtrar_pequenos_componentes(binaria):
    quantidade, rotulos, estatisticas, _ = (
        cv2.connectedComponentsWithStats(
            binaria,
            connectivity=8
        )
    )

    altura, largura = binaria.shape

    area_minima = max(
        3,
        round(altura * largura * 0.00002)
    )

    mascara_limpa = np.zeros_like(
        binaria
    )

    for indice in range(1, quantidade):
        area = estatisticas[
            indice,
            cv2.CC_STAT_AREA
        ]

        if area >= area_minima:
            mascara_limpa[
                rotulos == indice
            ] = 255

    return mascara_limpa

def extrair_componentes(binaria):
    quantidade, rotulos, estatisticas, _ = (
        cv2.connectedComponentsWithStats(
            binaria,
            connectivity=8
        )
    )

    componentes = []

    for indice in range(1, quantidade):
        x, y, largura, altura, area = (
            int(valor)
            for valor in estatisticas[indice]
        )

        componentes.append({
            "id": indice,
            "x": x,
            "y": y,
            "largura": largura,
            "altura": altura,
            "area": area,
        })

    return componentes, rotulos

def detectar_linhas(
    imagem
):

    cinza = cv2.cvtColor(
        imagem,
        cv2.COLOR_BGR2GRAY
    )

    binaria_adaptativa = cv2.adaptiveThreshold(
            cinza,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            41,
            10
    )

        # Referência mais restritiva de onde existe tinta.
    _, binaria_otsu = cv2.threshold(
        cinza,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    quantidade, rotulos_adaptativos, _, _ = (
        cv2.connectedComponentsWithStats(
            binaria_adaptativa,
            connectivity=8
        )
    )

    # Identifica componentes adaptativos que também
    # possuem algum pixel de tinta na máscara Otsu.
    ids_confirmados = np.unique(
        rotulos_adaptativos[binaria_otsu > 0]
    )

    manter = np.zeros(
        quantidade,
        dtype=bool
    )

    manter[ids_confirmados] = True

    # O fundo nunca deve ser selecionado.
    manter[0] = False

    # Preserva o componente adaptativo inteiro.
    binaria = (
        manter[rotulos_adaptativos].astype(np.uint8)
        * 255
    )

    cv2.imwrite(
        os.path.join(
            PROJECT_ROOT,
            "opencv",
            "debug_binaria_adaptativa.png"
        ),
        binaria_adaptativa
    )
    
    cv2.imwrite(
        os.path.join(
            PROJECT_ROOT,
            "opencv",
            "debug_binaria_antes.png"
        ),
        binaria
    )

    #binaria = filtrar_pequenos_componentes(
    #    binaria
    #)

    cv2.imwrite(
        os.path.join(
            PROJECT_ROOT,
            "opencv",
            "debug_binaria_depois.png"
        ),
        binaria
    )

    quantidade_tinta = np.count_nonzero(
        binaria_otsu,
        axis=1
    )
    
    minimo_tinta = max(
        3,
        round(binaria.shape[1] * 0.01)
    )

    possui_tinta = (
        quantidade_tinta >= minimo_tinta
    ).astype(np.uint8)
    
    caminho_perfil = os.path.join(
        PROJECT_ROOT,
        "opencv",
        "perfil_tinta.csv"
    )

    np.savetxt(
        caminho_perfil,
        np.column_stack((
            np.arange(len(quantidade_tinta)),
            quantidade_tinta,
            possui_tinta
        )),
        delimiter=",",
        header="y,pixels_tinta,ativo",
        comments="",
        fmt="%d"
    )

    altura = imagem.shape[0]

    gap_maximo = 1

    intervalos = []

    inicio = None
    ultimo = None

    for y, ativo in enumerate(
        possui_tinta
    ):

        if ativo:

            if inicio is None:

                inicio = y

            elif (
                ultimo is not None
                and y - ultimo - 1
                > gap_maximo
            ):

                intervalos.append(
                    (
                        inicio,
                        ultimo
                    )
                )

                inicio = y

            ultimo = y

    if inicio is not None:

        intervalos.append(
            (
                inicio,
                ultimo
            )
        )

    componentes, rotulos = extrair_componentes(binaria)

    if not componentes or not intervalos:
        return []

    # Estima a altura dos traços maiores.
    alturas = [
        componente["altura"]
        for componente in componentes
    ]

    altura_referencia = float(
        np.percentile(alturas, 75)
    )

    # Faixas muito finas não iniciam uma linha.
    # Seus componentes ainda podem ser associados a outra faixa.
    altura_minima = max(
        5,
        round(altura_referencia * 0.35)
    )

    faixas = [
        (inicio, fim)
        for inicio, fim in intervalos
        if fim - inicio + 1 >= altura_minima
    ]

    if not faixas:
        return []

    grupos = [[] for _ in faixas]

    for componente in componentes:
        topo = componente["y"]
        fim = topo + componente["altura"]

        # Mede quanto o componente ocupa de cada faixa.
        sobreposicoes = [
            max(
                0,
                min(fim, fim_faixa + 1)
                - max(topo, inicio_faixa)
            )
            for inicio_faixa, fim_faixa in faixas
        ]

        melhor = int(np.argmax(sobreposicoes))

        if sobreposicoes[melhor] == 0:
            # Pode ser um acento separado do corpo da linha.
            distancias = [
                max(
                    inicio_faixa - fim,
                    topo - (fim_faixa + 1),
                    0
                )
                for inicio_faixa, fim_faixa in faixas
            ]

            melhor = int(np.argmin(distancias))

            if distancias[melhor] > altura_referencia * 0.35:
                continue

        grupos[melhor].append(componente)

    linhas = []

    margem_x = 8
    margem_y = 4

    for grupo in grupos:
        if not grupo:
            continue

        x1 = max(
            0,
            min(c["x"] for c in grupo) - margem_x
        )

        y1 = max(
            0,
            min(c["y"] for c in grupo) - margem_y
        )

        x2 = min(
            imagem.shape[1],
            max(c["x"] + c["largura"] for c in grupo)
            + margem_x
        )

        y2 = min(
            imagem.shape[0],
            max(c["y"] + c["altura"] for c in grupo)
            + margem_y
        )

        largura = x2 - x1
        altura_linha = y2 - y1

        if largura < 15 or altura_linha < 5:
            continue

        ids = [
            componente["id"]
            for componente in grupo
        ]

        rotulos_recorte = rotulos[y1:y2, x1:x2]

        mascara_linha = np.isin(
            rotulos_recorte,
            ids
        )

        recorte_original = imagem[y1:y2, x1:x2]

        # Fundo branco com o mesmo tamanho e canais do recorte.
        recorte_limpo = np.full_like(
            recorte_original,
            255
        )

        # Copia somente a tinta dos componentes deste grupo.
        recorte_limpo[mascara_linha] = (
            recorte_original[mascara_linha]
        )

        linhas.append({
            "caixa": (x1, y1, largura, altura_linha),
            "imagem": recorte_limpo,
        })

    linhas.sort(
        key=lambda item: item["caixa"][1]
    )

    return linhas


def main():

    if len(sys.argv) > 1:

        caminho_imagem = (
            sys.argv[1]
        )

    else:

        caminho_imagem = os.path.join(
            PROJECT_ROOT,
            "opencv",
            "imagens",
            "dificil.png"
        )

    imagem = cv2.imread(
        caminho_imagem
    )

    if imagem is None:

        raise FileNotFoundError(
            f"Imagem não encontrada: "
            f"{caminho_imagem}"
        )

    reconhecedor = (
        ReconhecedorCTC()
    )

    caixas = detectar_linhas(
        imagem
    )

    imagem_debug = (
        imagem.copy()
    )

    pasta_debug = os.path.join(
        PROJECT_ROOT,
        "opencv",
        "debug_linhas_ctc"
    )

    os.makedirs(
        pasta_debug,
        exist_ok=True
    )

    textos = []

    print()
    print(
        "=============================="
    )

    print(
        "RECONHECIMENTO CTC"
    )

    print(
        "=============================="
    )

    for indice, resultado in enumerate(
        caixas,
        start=1
    ):

        x, y, w, h = resultado["caixa"]

        linha = resultado["imagem"]

        caminho_debug = os.path.join(
            pasta_debug,
            f"linha_{indice:02d}.png"
        )

        cv2.imwrite(
            caminho_debug,
            linha
        )

        texto = reconhecedor.reconhecer(
            linha,
            caminho_debug=os.path.join(
                pasta_debug,
                f"entrada_rede_{indice:02d}.png"
            )
        )

        textos.append(
            texto
        )

        print(
            f"Linha {indice}: "
            f"{texto}"
        )

        cv2.rectangle(
            imagem_debug,
            (x, y),
            (
                x + w,
                y + h
            ),
            (
                0,
                255,
                0
            ),
            2
        )

        cv2.putText(
            imagem_debug,
            str(indice),
            (
                x,
                max(
                    20,
                    y - 5
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (
                0,
                0,
                255
            ),
            2
        )

    texto_bruto = "\n".join(textos)

    print()
    print("==============================")
    print("TEXTO BRUTO (CTC)")
    print("==============================")
    print(texto_bruto)

    analise = analisar_texto(texto_bruto)

    print()
    print("==============================")
    print("SUGESTÕES PARA REVISÃO")
    print("==============================")

    encontrou_sugestao = False

    for palavra in analise["palavras"]:
        candidatos = palavra["candidatos"]

        if not candidatos:
            continue

        encontrou_sugestao = True

        limite_exibicao = 5

        opcoes = ", ".join(
            candidatos[:limite_exibicao]
        )

        restantes = max(
            0,
            len(candidatos) - limite_exibicao
        )

        if palavra.get("presente_no_corpus", False):
            situacao = (
                "registrada no corpus; "
                "não indica erro"
            )
        else:
            situacao = (
                "não encontrada nas listas consultadas"
            )

        print(
            f'{palavra["original"]} '
            f'[{situacao}]'
        )

        print(
            f"  Alternativas: {opcoes}"
        )
        if restantes:
            print(
                f"  Outras {restantes} alternativas "
                "não exibidas."
            )

    if not encontrou_sugestao:
        print("Nenhuma sugestão encontrada.")

    print("As sugestões não foram aplicadas ao texto bruto.")

    cv2.imshow(
        "Linhas detectadas - CTC",
        imagem_debug
    )

    cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()