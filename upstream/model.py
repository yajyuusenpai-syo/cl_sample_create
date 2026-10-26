import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from htsat import create_htsat_model
from seldnet import SELDModel
from text_encoder import RobertaTextEncoder

class AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel_encoder = create_htsat_model()
        self.spatial_encoder = SELDModel()
        self.resampler = torchaudio.transforms.Resample(
            orig_freq = 16000,
            new_freq = 48000,
        )

        self.mel_feature_dim = 1024
        self.spatial_feature_dim = 256
    
    def get_output_dim(self):
        return self.mel_feature_dim + self.spatial_feature_dim

    def load_default_state_dict(self):
        self.mel_encoder.load_default_state_dict()
        self.spatial_encoder.load_default_state_dict()

    def forward(self, x):
        B = len(x)

        mel_encoded = self.mel_encoder({
            "waveform": self.resampler((x[:, 0, :] + x[:, 1, :]) / 2)
        })["embedding"]
        assert mel_encoded.shape == (B, self.mel_feature_dim), f"{mel_encoded.shape=}"

        spatial_encoded = self.spatial_encoder(x)
        assert spatial_encoded.shape == (B, self.spatial_feature_dim), f"{spatial_encoded.shape=}"

        return torch.cat(
            [mel_encoded, spatial_encoded],
            dim=1
        )

class CLAPEncoder(nn.Module):
    def __init__(
        self,
        joint_embed_shape: int = 512,
    ):
        super().__init__()

        self.audio_encoder = AudioEncoder()
        self.audio_projection = nn.Sequential(
            nn.Linear(self.audio_encoder.get_output_dim(), joint_embed_shape),
            nn.ReLU(),
            nn.Linear(joint_embed_shape, joint_embed_shape),
        )

        self.text_encoder = RobertaTextEncoder()
        self.text_projection = nn.Sequential(
            nn.Linear(512, joint_embed_shape),
            nn.ReLU(),
            nn.Linear(joint_embed_shape, joint_embed_shape),
        )

        self.logit_scale = nn.Parameter(torch.tensor(np.log(1 / (0.07))))

    def load_default_state_dict(self):
        self.audio_encoder.load_default_state_dict()
        self.text_encoder.load_default_state_dict()

    def load_pretrained(self, url=None):
        if url is None:
            url = "https://huggingface.co/sarulab-speech/SpatialCLAP/resolve/main/ckpt/l1proposed-spatial_contrastive-model_epoch_49.pt"
        ckpt = torch.hub.load_state_dict_from_url(url, map_location="cpu")["model_state_dict"]
        self.load_state_dict(ckpt)

    def embed_audio(self, x):
        encoded = self.audio_encoder(x)
        projected_encoded = self.audio_projection(encoded)
        return F.normalize(projected_encoded, dim=-1)
    
    def embed_text(self, x):
        encoded = self.text_encoder(x)
        projected_encoded = self.text_projection(encoded)
        return F.normalize(projected_encoded, dim=-1)

    def forward(self, audio=None, text=None):
        if audio is None:
            z_audio = None
        else:
            z_audio = self.embed_audio(audio)
        
        if text is None:
            z_text = None
        else:
            z_text = self.embed_text(text)

        return {
            "audio": z_audio,
            "text": z_text,
        }
