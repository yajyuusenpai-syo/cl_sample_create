import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class CLAPLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, clap_output, metas, logit_scale):
        """
        audio_features: Tensor of shape (N, D)
        text_features: Tensor of shape (N, D)
        """
        audio_features = clap_output["audio"]
        text_features = clap_output["text"]

        # τ is learnable → use exp(logit_scale)
        temperature = logit_scale.exp().clamp(max=np.log(100))

        logits_per_audio = audio_features @ text_features.T * temperature
        logits_per_text = text_features @ audio_features.T * temperature

        labels = torch.arange(audio_features.size(0), device=audio_features.device)

        loss = (
            F.cross_entropy(logits_per_audio, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2
        return loss
