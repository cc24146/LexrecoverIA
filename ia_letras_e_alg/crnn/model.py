# # define a rede neural
# import torch.nn as nn   # camadas da rede neural como convolucao e camada linear
# import torch.nn.functional as F # funções usadas

# class CNN(nn.Module):
#     def __init__(self):
#         super().__init__()
        
#         self.conv1 = nn.Conv2d(1, 32, 3)    # cria a primeira camada convulocional com 1 canal (imagem em preto e branco) com 32 filtros e cada filtro tem tamanho 3x3 -> padroes simples como bordas e tracos
#         self.conv2 = nn.Conv2d(32, 64, 3)   # cria a segunda camada convulocional. Recebe 32 mapas de caracteristicas da camada anterior e transforma isso em 64 novos mapas -> padroes mais complexos
        
#         self.fc1 = nn.Linear(64 * 5 * 5, 128)   # primeira camada conectada. Passagem da parte visual para a parte "decisoria" da rede. Tamanho dos dados depois das convolucoes e pooling. Pega tudo e reduz para 128 valores
#         self.fc2 = nn.Linear(128, 47)   # transforma os 128 valores para 47 saidas -> valores de 0 - 9 e letras de a - Z

#     def forward(self, x):   # define como os dados vao passar pela rede
#         x = F.relu(self.conv1(x))   # a imagem passa pela primeira convolucao, relu zera os valores negativos e mantém os positivos
#         x = F.max_pool2d(x, 2)  # reduz o tamanho da imagem de caracteristicas pela metade. Pega blocos 2x2 e guarda o maior valor de cada bloco
        
#         x = F.relu(self.conv2(x))   # segunda convolucao
#         x = F.max_pool2d(x, 2)
        
#         x = x.view(-1, 64 * 5 * 5)  # os dados viram uma lista grande de numeros para poder entrar na camada linear
        
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
        
#         return x

import torch
import torch.nn as nn
import torch.nn.functional as F

class CRNN(nn.Module):
    def __init__(self, num_classes=47, hidden_size=128):
        super().__init__()
        
        # 1. Extração visual (CNN)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        
        # 2. Recorrência temporal/seqüencial (RNN - LSTM)
        # Após 2 convs e 2 max_pools na imagem 28x28, a saída é (batch, 64, 5, 5)
        # Tratamos a largura (5) como a sequência de passos no tempo (seq_len = 5)
        # E multiplicamos os canais * altura (64 * 5 = 320) como o vetor de recursos por passo.
        self.lstm = nn.LSTM(
            input_size=64 * 5, 
            hidden_size=hidden_size, 
            num_layers=2, 
            batch_first=True, 
            bidirectional=True
        )
        
        # 3. Classificação
        # Bidirecional dobra o hidden_size (128 * 2 = 256)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # Passagem Convolucional
        x = F.relu(self.conv1(x))   # Shape: (N, 32, 26, 26)
        x = F.max_pool2d(x, 2)      # Shape: (N, 32, 13, 13)
        
        x = F.relu(self.conv2(x))   # Shape: (N, 64, 11, 11)
        x = F.max_pool2d(x, 2)      # Shape: (N, 64, 5, 5)
        
        # Reorganização para entrada na LSTM:
        # De: (Batch, Canais, Altura, Largura) -> Para: (Batch, Largura, Canais, Altura)
        x = x.permute(0, 3, 1, 2)
        
        batch_size, seq_len, channels, height = x.size()
        x = x.reshape(batch_size, seq_len, channels * height) # Shape: (N, 5, 320)
        
        # Passagem Recorrente (LSTM)
        out, _ = self.lstm(x)       # Shape: (N, 5, 256)
        
        # Agrupamento das sequências (média temporal)
        out = out.mean(dim=1)       # Shape: (N, 256)
        
        # Camada Final
        out = self.fc(out)          # Shape: (N, 47)
        return out