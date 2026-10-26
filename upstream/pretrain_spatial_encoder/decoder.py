import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, input_features=128, hidden_features=128, num_classes=10):
        """
        input_features (Q): Encoderから来る特徴次元
        num_classes: SED用のクラス数
        """
        super(Decoder, self).__init__()

        self.fc_sed = nn.Sequential(
            nn.Linear(input_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, num_classes),
            nn.Sigmoid()
        )

        self.fc_doa = nn.Sequential(
            nn.Linear(input_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, num_classes),
            nn.Tanh()
        )

    def forward(self, x):
        """
        x: (batch, input_features)
        returns:
          - sed_output: (batch, num_classes)
          - doa_output: (batch, num_classes)
        """
        sed_output = self.fc_sed(x)
        doa_output = self.fc_doa(x)

        return sed_output, doa_output

