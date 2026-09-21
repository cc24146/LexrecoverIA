# Revisão dos recortes exportados

Exportação: C:\Users\Admin\Documents\GitHub\LexrecoverIA\ia_letras_e_alg\opencv\recortes_revisao\20260919_214951_281022

Todos os 50 recortes limpos foram inspecionados visualmente. Os arquivos originais e o manifesto do projeto não foram alterados.

- 49 regiões contêm linhas de texto.
- 1 região extra: manuscrito 4, recorte 02, pequena marca sem linha legível.
- 2 linhas com pontuação perdida: manuscrito 9/01 e manuscrito 11/02, aspas de Twin Peaks ausentes ou incompletas.
- 47 candidatos à avaliação isolada do reconhecimento; ainda não aprovados para treinamento.
- O manuscrito 3 recortado está sem o texto vizinho. Usar a nova exportação para futuros resultados; não associar a saída antiga a estes recortes.
- Persistem pequenos ruídos em alguns recortes, especialmente 7/01 e 10/04. A primeira letra do manuscrito 4 está próxima ao limite da própria imagem de origem.

## Correspondência corrigida do manuscrito 4

| Recorte exportado | Referência |
|---|---|
| 01 | m04_l01 |
| 02 | Excluir da avaliação por linha: marca sem texto |
| 03 | m04_l02 |
| 04 | m04_l03 |
| 05 | m04_l04 |
| 06 | m04_l05 |

Não remover recorte 02 da avaliação ponta a ponta da segmentação: ele é um falso positivo real. Excluí-lo é apropriado apenas para diagnosticar o reconhecedor com linhas selecionadas.

## Próxima execução

O script avaliar_recortes.py lê manifesto_revisado.json, reconhece os 47 candidatos, registra previsões brutas/corrigidas e computa CER/WER normalizados.
Não segmenta novamente as fotografias, não altera pesos e não treina. Carrega o reconhecedor atual do projeto.
Salva um arquivo novo com data/hora nesta pasta de avaliação.
Os resultados não são diretamente comparáveis ao CER anterior por fotografia: o escopo, os recortes e o tratamento das fronteiras de linha são diferentes.

Todas as transcrições são propostas. As confirmações específicas do usuário foram preservadas, mas não se infere aprovação humana das demais linhas.

