import torch
import torch.nn as nn

from .config import NUM_CLASSES


class CRNNCTC(nn.Module):

    def __init__(
        self,
        num_classes=NUM_CLASSES,
        hidden_size=256
    ):
        super().__init__()

        self.cnn = nn.Sequential(

            nn.Conv2d(
                1,
                64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),

            nn.Conv2d(
                128,
                256,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),

            nn.Conv2d(
                256,
                256,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=(2, 1),
                stride=(2, 1)
            ),

            nn.Conv2d(
                256,
                512,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(
                512,
                512,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=(2, 1),
                stride=(2, 1)
            ),

            nn.Conv2d(
                512,
                512,
                kernel_size=(2, 1)
            ),
            nn.ReLU()
        )

        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(
            hidden_size * 2,
            num_classes
        )

    def forward(self, x):

        x = self.cnn(x)

        x = x.squeeze(2)

        x = x.permute(
            0,
            2,
            1
        )

        x, _ = self.lstm(x)

        x = self.fc(x)

        return x

    def calcular_comprimentos_saida(
        self,
        larguras_entrada
    ):
        return larguras_entrada // 4
