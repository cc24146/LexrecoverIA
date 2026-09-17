# Reconhecimento de texto manuscrito

O projeto contém duas abordagens: classificação de caracteres com segmentação dinâmica e reconhecimento de linhas inteiras com CRNN + CTC. Os métodos anteriores permanecem disponíveis para comparação e evolução.

## Executar no Windows

Abra o PowerShell na pasta `ia_letras_e_alg` e utilize o ambiente instalado:

```powershell
.\venv\Scripts\python.exe -B .\opencv\main_ctc.py
```

Para escolher uma imagem:

```powershell
.\venv\Scripts\python.exe -B .\opencv\main_ctc.py ".\opencv\imagens\manuscrito 1.jpeg"
```

Pressione uma tecla com a janela da imagem em foco para encerrar. Os recortes são salvos em `opencv/debug_linhas_ctc/`; execuções posteriores reutilizam seus nomes.

## Requisitos locais

O reconhecimento depende de PyTorch, NumPy e OpenCV e do arquivo `crnn_ctc/final_model.pth`. O ambiente virtual, os conjuntos de dados e os pesos não são versionados. Uma cópia nova do repositório precisa de um ambiente configurado e dos pesos treinados para executar o reconhecimento.

Ambiente utilizado nos testes: Python 3.14, torch 2.14.0+cu126, numpy 2.5.3 e opencv-python 5.0.0.93; GPU GTX 970 de 4 GB. Essas são as versões observadas no ambiente local.

## Organização

- `crnn_ctc/`: arquitetura, alfabeto, decodificação, treinamento e avaliação por linhas.
- `crnn/`: classificador anterior de caracteres.
- `opencv/main_ctc.py` e `reconhecimento_ctc.py`: execução do reconhecimento por linhas.
- `opencv/main.py`, `processamento.py`, `ordenacao.py`, `espacos.py` e `segmentacao_dinamica.py`: métodos anteriores preservados.
- `opencv/imagens/`: imagens de avaliação, incluídas no repositório.
- `data/` e `dataset_bressay/`: dados locais de treinamento e avaliação.

## Resultados e próximo passo

Resultado registrado no teste final do BRESSAY: CER de 14,32%, perda de 0,5087 e 5.094 linhas avaliadas. O melhor CER de validação foi 15,41%. Esses números não representam o desempenho nas fotografias externas.

Os 11 testes manuais estão resumidos em [TESTES_MANUSCRITOS.md](TESTES_MANUSCRITOS.md). A próxima etapa é melhorar a detecção de linhas, mantendo os pesos fixos e comparando com os resultados iniciais. Depois serão investigados os erros restantes em linhas corretamente isoladas.
