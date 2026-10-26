import torch
import torch.nn as nn

from feature_extractor import FeatureExtractor
from encoder import Encoder

class SELDModel(nn.Module):
    def __init__(self,
                 input_ch=2,
                 n_fft=1024,
                 hop_length=512,
                 num_classes=10):
        super(SELDModel, self).__init__()

        self.feature_extractor = FeatureExtractor(n_fft=n_fft, hop_length=hop_length)
        self.encoder = Encoder(input_channels=self.feature_extractor.input_ch*2)
        
    def forward(self, x):
        """
        x: (batch, channels=2, time)
        returns:
          - sed_output: (batch, time, num_classes)
          - doa_output: (batch, time, num_classes*3)
        """
        features = self.feature_extractor(x)  # (batch, time, freq, ch)
        encoded = self.encoder(features)      # (batch, encoder_output_size)

        return encoded

    def load_from_pretrained(self, path):
        params = {
          k:v for k,v in (torch.load(path, weights_only=True)["model_state_dict"]).items()
          if not k.startswith("decoder")
        }
        self.load_state_dict(params)

    def load_default_state_dict(self):
        ckpt_path = "pretrain_spatial_encoder/output/ckpt/model_epoch_49.pt"
        self.load_from_pretrained(ckpt_path)


