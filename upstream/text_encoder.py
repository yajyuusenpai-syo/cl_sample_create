import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizer
from typing import List


class RobertaTextEncoder(nn.Module):
    def __init__(self, joint_embed_dim=512, mlp_act='relu'):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

        self.input_dim = 768  # fixed for roberta-base
        self.joint_embed_dim = joint_embed_dim

        if mlp_act == 'relu':
            act_layer = nn.ReLU()
        elif mlp_act == 'gelu':
            act_layer = nn.GELU()
        else:
            raise NotImplementedError(f"Unsupported activation: {mlp_act}")

        self.text_projection = nn.Sequential(
            nn.Linear(self.input_dim, joint_embed_dim),
            act_layer,
            nn.Linear(joint_embed_dim, joint_embed_dim)
        )

    def forward(self, texts: List[str]):
        """
        text: dictionary with keys "input_ids" and "attention_mask"
        Returns: normalized embedding of shape [batch_size, joint_embed_dim]
        """
        tokenized = self.tokenizer(
            texts,
            padding=True,
            return_tensors="pt"
        )
        text = {
            key: value.to(next(self.parameters()).device)
            for key, value in tokenized.items()
        }
        
        x = self.roberta(
            input_ids=text["input_ids"],
            attention_mask=text["attention_mask"]
        )["pooler_output"]
        x = self.text_projection(x)
        x = nn.functional.normalize(x, dim=-1)
        return x

    def load_default_state_dict(self):
        pass
