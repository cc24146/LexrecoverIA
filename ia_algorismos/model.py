# define a rede neural
import torch.nn as nn   # camadas da rede neural como convolucao e camada linear
import torch.nn.functional as F # funções usadas

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(1, 32, 3)    # cria a primeira camada convulocional com 1 canal (imagem em preto e branco) com 32 filtros e cada filtro tem tamanho 3x3 -> padroes simples como bordas e tracos
        self.conv2 = nn.Conv2d(32, 64, 3)   # cria a segunda camada convulocional. Recebe 32 mapas de caracteristicas da camada anterior e transforma isso em 64 novos mapas -> padroes mais complexos
        
        self.fc1 = nn.Linear(64 * 5 * 5, 128)   # primeira camada conectada. Passagem da parte visual para a parte "decisoria" da rede. Tamanho dos dados depois das convolucoes e pooling. Pega tudo e reduz para 128 valores
        self.fc2 = nn.Linear(128, 10)   # transforma os 128 valores para 10 saidas -> valores de 0 - 9

    def forward(self, x):   # define como os dados vao passar pela rede
        x = F.relu(self.conv1(x))   # a imagem passa pela primeira convolucao, relu zera os valores negativos e mantém os positivos
        x = F.max_pool2d(x, 2)  # reduz o tamanho da imagem de caracteristicas pela metade. Pega blocos 2x2 e guarda o maior valor de cada bloco
        
        x = F.relu(self.conv2(x))   # segunda convolucao
        x = F.max_pool2d(x, 2)
        
        x = x.view(-1, 64 * 5 * 5)  # os dados viram uma lista grande de numeros para poder entrar na camada linear
        
        x = F.relu(self.fc1(x)) #
        x = self.fc2(x)
        
        return x