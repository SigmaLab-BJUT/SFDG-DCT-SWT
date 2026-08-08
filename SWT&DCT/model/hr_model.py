import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self) -> None:
        super(Classifier, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            # nn.Conv2d(64, 64, 3, 1, 1),
            # nn.BatchNorm2d(64),
            # nn.ReLU(),
            # nn.MaxPool2d(2, 2, 0),

            nn.Flatten(),
            # nn.Linear(1024,512),
            # # # nn.BatchNorm1d(64),
            # nn.ReLU(),
            # # nn.Dropout(0.5),
            nn.Linear(1024, 64),
            nn.ReLU(),
            nn.Dropout(0.8),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.net(x)