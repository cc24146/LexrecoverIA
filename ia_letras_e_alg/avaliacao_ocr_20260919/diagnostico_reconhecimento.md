# Diagnóstico do reconhecimento por linha

Execução manual do usuário: resultado_recortes_20260919_215750_708565.json.
47 recortes selecionados, sem repetir a segmentação. Referências ainda provisórias, normalização sem caixa/pontuação; acentos e números preservados.

## Resultado

| Medida | Bruto | Corrigido |
|---|---:|---:|
| CER | 26,39% | 26,46% |
| WER | 59,15% | 57,02% |
| Edições por caracteres (1.315 caracteres) | 347 | 348 |
| Edições por palavras (235 palavras) | 139 | 134 |
| Linhas sem erro na comparação normalizada | 4 | 3 |

O corretor reduz o erro de caracteres em 6 linhas, aumenta em 6 e mantém a mesma contagem em 35. Mesma contagem não significa mesmo texto.

## CER bruto por manuscrito, apenas recortes selecionados

| Manuscrito | Recortes | CER |
|---|---:|---:|
| m01 | 2 | 41.46% |
| m02 | 5 | 16.28% |
| m03 | 5 | 26.74% |
| m04 | 5 | 33.09% |
| m05 | 6 | 21.64% |
| m06 | 5 | 29.24% |
| m07 | 4 | 30.00% |
| m08 | 4 | 17.24% |
| m09 | 3 | 30.71% |
| m10 | 4 | 16.22% |
| m11 | 4 | 29.80% |

9 e 11 não incluem a primeira linha de Twin Peaks por perda de aspas. 4 exclui uma detecção sem texto. O resultado da imagem 3 é da versão recortada, diferente da execução antiga.
Não comparar diretamente com a avaliação anterior por fotografia ou com o teste BRESSAY.

## Evidências

- Linhas normalizadas corretas no bruto: m02_l02, m03_l01, m08_l04, m10_l04.
- O corretor transforma suas ações passadas em suas nações passada, removendo um acerto completo.
- Inspeção textual de crnn_ctc/banco/lexico.txt: ação, nações e passada estão presentes; não foram encontradas entradas exatas ações e passadas.
- O corretor retorna palavras conhecidas sem modificação, mas para desconhecidas aceita candidato único a uma edição de distância, sem contexto.
- final_model.pth e best_model.pth idênticos por SHA-256. Não há ganho a esperar da simples troca de nomes.
- Segmentar melhor reduziu agrupamentos indevidos, mas não resolveu os erros nas linhas selecionadas.
- As entradas de 32 pixels inspecionadas preservam texto visualmente identificável. Isso não prova que a redução seja inofensiva nem que todo erro seja causado pelo treinamento.

## Decisão de trabalho

Manter pesos e processamento atuais como referência de desenvolvimento. Não treinar imediatamente com todos estes recortes e usar os mesmos para declarar melhora.
O material atual já orientou os ajustes e contém textos repetidos; 47 recortes são um diagnóstico, não evidência de generalização.
Manter a previsão bruta como referência e a corrigida como variante separada. Não adicionar somente palavras do teste ao léxico e apresentar o resultado como ganho independente.

## Próxima informação necessária

Existem outras imagens manuscritas que ainda não foram usadas nos ajustes? Identificar pasta, quantidade aproximada, autores (se conhecidos), tipo de suporte (quadro/papel) e disponibilidade de transcrições.
Com material novo, preparar divisão por autor/imagem e agrupar textos repetidos antes de qualquer treino complementar. Preservar teste não usado na escolha de parâmetros.
Se ainda não houver material novo, organizar a coleta e transcrição; não é necessário repetir a rodada das 11 imagens.

Nenhum treinamento ou alteração de código do projeto foi executado pelo assistente nesta análise.

