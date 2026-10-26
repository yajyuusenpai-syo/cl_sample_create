import torch
import torch.nn as nn

from feature_extractor import FeatureExtractor
from encoder import Encoder
from decoder import Decoder

class SELDModel(nn.Module):
    def __init__(self,
                 input_ch=2,
                 n_fft=1024,
                 hop_length=512,
                 num_classes=10):
        super(SELDModel, self).__init__()

        self.feature_extractor = FeatureExtractor(n_fft=n_fft, hop_length=hop_length)
        self.encoder = Encoder(input_channels=self.feature_extractor.input_ch*2)
        self.decoder = Decoder(input_features=self.encoder.output_features, num_classes=num_classes)
        
    def forward(self, x):
        """
        x: (batch, channels=2, time)
        returns:
          - sed_output: (batch, time, num_classes)
          - doa_output: (batch, time, num_classes*3)
        """
        features = self.feature_extractor(x)  # (batch, time, freq, ch)
        encoded = self.encoder(features)      # (batch, encoder_output_size)
        sed_output, doa_output = self.decoder(encoded)

        return sed_output, doa_output
