#ensina o modelo
import torch
import torch.nn as nn

def train(model, train_loader): # modelo e dados de treino em lotes
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # ajusta para dar menos erros
    loss_fn = nn.CrossEntropyLoss() # compara o que o modelo previu com as resposta certa

    for epoch in range(5):  # repete o treino 5 vezes sobre todo o conjunto de treino
        total_loss = 0  # soma os errps de todos os lotes daquela epoca

        for images, labels in train_loader: # percorre o conjunto de treino em lotes
            optimizer.zero_grad()   # zera o gradiente para nao atrapalhar os calculos do lote atual

            outputs = model(images) # passa as imagens pela rede -> chama o forward do model.py
            loss = loss_fn(outputs, labels) # calcula o erro

            loss.backward() # verifica que parte da rede errou mais
            optimizer.step()    # muda os pesos usando os gradientes calculados

            total_loss += loss.item()   # soma os erros desse lote com o total

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")   # mostra o erro de cada epoca
    
def evaluate(model, test_loader):
    model.eval()  # modo avaliação
    
    correct = 0
    total = 0

    with torch.no_grad():  # não calcula gradiente
        for images, labels in test_loader:
            outputs = model(images)
            
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy:.2f}%")