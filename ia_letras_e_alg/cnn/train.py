import torch
import torch.nn as nn


def train(model, train_loader, test_loader):

    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    best_val_acc = 0
    for epoch in range(8):

        model.train()

        total_loss = 0
        correct = 0
        total = 0

        # ======================
        # TREINAMENTO
        # ======================

        for images, labels in train_loader:

            optimizer.zero_grad()

            outputs = model(images)

            loss = loss_fn(outputs, labels)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_accuracy = 100 * correct / total

        train_losses.append(total_loss)
        train_accs.append(train_accuracy)

        # ======================
        # VALIDAÇÃO
        # ======================

        val_loss, val_accuracy = evaluate(model, test_loader)

        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), "cnn/best_model.pth")
            print(f"Novo melhor modelo! Accuracy = {val_accuracy:.2f}%")

        val_losses.append(val_loss)
        val_accs.append(val_accuracy)

        print(
            f"Epoch {epoch+1:2d} | "
            f"Train Loss: {total_loss:.2f} | "
            f"Train Acc: {train_accuracy:.2f}% | "
            f"Val Loss: {val_loss:.2f} | "
            f"Val Acc: {val_accuracy:.2f}%"
        )

    return train_losses, train_accs, val_losses, val_accs


def evaluate(model, test_loader):

    model.eval()

    loss_fn = nn.CrossEntropyLoss()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            outputs = model(images)

            loss = loss_fn(outputs, labels)

            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    return total_loss, accuracy


# #ensina o modelo
# import torch
# import torch.nn as nn

# def train(model, train_loader): # modelo e dados de treino em lotes
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # ajusta para dar menos erros
#     loss_fn = nn.CrossEntropyLoss() # compara o que o modelo previu com as resposta certa

#     for epoch in range(5):  # repete o treino 5 vezes sobre todo o conjunto de treino
#         total_loss = 0  # soma os errps de todos os lotes daquela epoca

#         for images, labels in train_loader: # percorre o conjunto de treino em lotes
#             optimizer.zero_grad()   # zera o gradiente para nao atrapalhar os calculos do lote atual

#             outputs = model(images) # passa as imagens pela rede -> chama o forward do model.py
#             loss = loss_fn(outputs, labels) # calcula o erro

#             loss.backward() # verifica que parte da rede errou mais
#             optimizer.step()    # muda os pesos usando os gradientes calculados

#             total_loss += loss.item()   # soma os erros desse lote com o total

#         print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")   # mostra o erro de cada epoca
    
# def evaluate(model, test_loader):
#     model.eval()  # modo avaliação
    
#     correct = 0
#     total = 0

#     with torch.no_grad():  # não calcula gradiente
#         for images, labels in test_loader:
#             outputs = model(images)
            
#             _, predicted = torch.max(outputs, 1)
            
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()

#     accuracy = 100 * correct / total
#     print(f"Accuracy: {accuracy:.2f}%")