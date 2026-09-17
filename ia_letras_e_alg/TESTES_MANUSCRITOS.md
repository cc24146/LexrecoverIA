# Avaliação inicial das fotografias de manuscritos

Data: 16/09/2026. Execução com `opencv/main_ctc.py`, modelo `crnn_ctc/final_model.pth` e CUDA. Código e modelo foram mantidos durante a rodada. As observações abaixo vêm das capturas e saídas compartilhadas manualmente; não foi calculado CER para estas imagens.

| Imagem | Linhas reais | Regiões detectadas | Observação |
|---|---:|---:|---|
| manuscrito 1.jpeg | 2 | 5 | Duas linhas isoladas e três regiões falsas. |
| manuscrito 2.jpeg | 5 | 3 | Últimas três linhas agrupadas. |
| manuscrito 3.jpeg | 5 | 2 | Últimas quatro linhas agrupadas. |
| manuscrito 4.jpeg | 5 | 4 | Segunda e terceira linhas agrupadas. |
| manuscrito 5.jpeg | 6 | 5 | Quarta e quinta linhas agrupadas. |
| manuscrito 6.jpeg | 5 | 4 | Região extra no topo e últimas três linhas agrupadas. |
| manuscrito 7.jpeg | 4 | 7 | Quatro regiões falsas e primeiras duas linhas agrupadas. |
| manuscrito 8.jpeg | 4 | 3 | Duas regiões falsas e todas as quatro linhas agrupadas. |
| manuscrito 9.jpeg | 4 | 6 | Quatro regiões falsas e últimas três linhas agrupadas. |
| manuscrito 10.jpeg | 4 | 7 | Quatro regiões falsas e primeiras duas linhas agrupadas. |
| manuscrito 11.jpeg | Pelo menos 5 visíveis | 7 | Três linhas agrupadas; regiões fora do texto; região superior cortada na captura, pendente de conferência na imagem original. |

## Conclusões observacionais

Nenhuma captura mostrou separação inteiramente correta. Os dois problemas recorrentes foram agrupamento de várias linhas em um recorte e marcas de fundo interpretadas como texto. Também houve erros em linhas isoladas, portanto a detecção não explica todos os erros.

As imagens mostram principalmente escrita em quadro branco com reflexos. Há conteúdo repetido entre algumas fotos; esta rodada não deve ser tratada como 11 textos independentes nem como avaliação representativa de todo manuscrito.

## Continuidade

1. Preservar esta referência e os pesos atuais.
2. Conferir as transcrições exatamente como aparecem nas imagens, mantendo acentos, maiúsculas e pontuação.
3. Comparar o reconhecimento de algumas linhas recortadas manualmente com os recortes automáticos.
4. Melhorar a detecção e filtragem de regiões, sem alterar os pesos.
5. Repetir as 11 imagens e registrar resultados em pastas separadas.
6. Investigar a preparação e o reconhecimento das linhas corretamente isoladas.

As funções anteriores de processamento e segmentação permanecem no projeto para avaliação de possível reutilização. Nenhuma melhoria no algoritmo foi aplicada durante esta rodada.
