# Retomada do trabalho — 20/09/2026

## Estado salvo

- Segmentação por faixas Otsu e componentes de máscara combinada, com exportação de recortes originais e limpos.
- Reconhecimento mantém texto bruto separado do corretor léxico.
- analisar_palavra foi adicionada e testada manualmente; analisar_texto ainda não foi integrada em pos_processamento.py.
- Próxima etapa: análise do texto com sugestões rastreáveis, revisão da cobertura do léxico e posterior correção contextual.
- Avaliação manual de 47 recortes: CER bruto 26,39%, corrigido 26,46%; WER bruto 59,15%, corrigido 57,02%. Referências provisórias e conjunto de desenvolvimento, sem teste independente.
- Dois recortes com aspas perdidas e uma detecção extra permanecem documentados.

## Arquivos locais

Esta pasta preserva uma cópia dos relatórios, referências, scripts e resultados preparados durante a análise. Os scripts e manifestos ainda contêm caminhos absolutos deste computador; ajustar os caminhos antes de usar em outra máquina.
O exportador atual aponta para a pasta original da avaliação fora do repositório, que foi preservada. Corrigir essa dependência é uma pendência de portabilidade.
Pesos, ambientes, dataset BRESSAY, diagnósticos e recortes exportados não entram neste commit. Precisam de cópia separada para uso em outro computador.
Os recortes referenciados pelos manifestos continuam no disco local em opencv/recortes_revisao e não foram apagados.
O commit não garante reprodução completa em outra máquina sem esses arquivos e a configuração dos caminhos.