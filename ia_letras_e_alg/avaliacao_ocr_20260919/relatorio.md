# Avaliação de desenvolvimento — 19/09/2026

## Estado e limites

Nenhum arquivo do projeto foi modificado e nenhuma inferência ou treinamento foi executado nesta preparação. Foram lidas as imagens atuais, transcritas referências propostas e reaproveitadas as saídas relatadas pelo usuário.

O JSON contém 49 referências por linha, 11 imagens, as saídas bruta/corrigida com quebras convertidas em espaços e assinaturas SHA-256 dos arquivos atuais. As assinaturas não comprovam qual imagem foi usada em uma execução antiga.

Todos os campos de aprovação permanecem falsos. crop_path é nulo: não se deve tratar esta lista como pares imagem de linha/texto prontos para treinar.

## Proveniência pendente

O usuário confirmou que recortou o manuscrito 3 após o teste para remover texto vizinho. A saída antiga continua arquivada como histórico, mas está excluída das métricas da imagem atual. Não é necessário repetir OCR para exportar os recortes.

O usuário confirmou Ninar nos manuscritos 2/3 e Pinheiro sem s no 3. A referência foi atualizada. Demais textos e pontuação são propostas para revisão. Preservar 1999, milhares e Quincas Borbas como escritos, sem corrigir o conteúdo.

## Métricas

Normalização: Unicode NFC; minúsculas; cada caractere que não é letra, número ou espaço vira espaço; sequências de espaços são compactadas; espaços das extremidades removidos. Acentos e números preservados.
CER = distância de Levenshtein por caracteres / caracteres da referência.
WER = distância de Levenshtein por palavras separadas por espaço / palavras da referência.
Inserções, exclusões e substituições têm custo 1. Agregação pela soma das edições dividida pela soma dos comprimentos, e não média simples das porcentagens.
Pontuação não é avaliada diretamente, mas separa tokens.
Não interpretar 100-CER como porcentagem de frases corretas.

| Escopo | CER bruto | CER corrigido | WER bruto | WER corrigido |
|---|---:|---:|---:|---:|
| 11 imagens, cálculo preliminar anterior | 28.32% | 28.39% | 62.55% | 60.56% |
| 10 imagens, excluindo a 3 pendente | 28.23% | 28.30% | 61.86% | 59.75% |

Subtotal de 10 imagens: 1.339 caracteres, 236 palavras; 378/379 edições por caracteres e 146/141 por palavras (bruto/corrigido).
Esses valores continuam provisórios até revisão das referências. Não são diretamente comparáveis ao CER histórico do BRESSAY.
O conjunto foi usado para ajustar a segmentação, inclui conteúdo repetido e não constitui teste independente.

## Próxima etapa após revisão

1. Resolver a correspondência da imagem 3 com a execução; não misturar versões silenciosamente.
2. Aprovar transcrições por linha, preservando ortografia, acentos, números e pontuação visíveis.
3. Exportar recortes para pastas permanentes por imagem de origem e execução, sem reaproveitar nomes globais linha_01.png.
4. Conferir a correspondência visual entre cada recorte e sua referência. Não associar automaticamente apenas pela ordem: houve caixa extra na imagem 4, texto vizinho na 3 e pontuação possivelmente perdida nas 9/11.
5. Registrar o hash da imagem, versão do código, checkpoint e parâmetros na exportação.
6. Medir CER/WER por linha após aprovação dos pares.
7. Só então decidir por treinamento complementar.

## Separação dos dados

Manter estas imagens como desenvolvimento no momento. Não dividir aleatoriamente linhas da mesma foto entre treino e validação.
Os manuscritos 2/3 compartilham texto; 9/11 compartilham Twin Peaks; 10/11 têm conteúdo relacionado na mesma região de quadro. Manter conteúdos relacionados no mesmo grupo se houver uma divisão futura.
Identidades dos escritores não são conhecidas e não foram inferidas por aparência. Registrar quando informadas.
Um teste independente deve conter imagens/escritores reservados que não tenham orientado os ajustes. Se estas imagens forem usadas para treino, deixam de servir como avaliação independente desse treino.

## Modelo verificado

final_model.pth e best_model.pth têm conteúdo idêntico:
56BFBE9F0B7B47582C6A6988A83A88B67B9F440AE18307D99BC58D29147C6B75

model_last.pth é diferente. Não foi carregado nem avaliado.
O código atual salva best_model quando o CER de validação diminui; final_model não é atualizado automaticamente pelo trecho de treinamento consultado.
