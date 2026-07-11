"""
RoBERTa + MLP Baseline
简单的文本编码 + MLP分类头，不使用多维特征、hetero attention等
"""

import torch
import torch.nn as nn
from transformers import AutoModel


class RoBERTaMLP(nn.Module):
    """
    RoBERTa编码器 + 池化 + MLP分类头
    """

    def __init__(
        self,
        encoder_name: str = "roberta-large",
        pooling: str = "mean",
        cls_hidden: int = 256,
        num_classes: int = 2,
        dropout: float = 0.25,
    ):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.hidden_dim = self.encoder.config.hidden_size
        self.pooling = pooling

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, cls_hidden),
            nn.GELU(),
            nn.LayerNorm(cls_hidden),
            nn.Dropout(dropout),
            nn.Linear(cls_hidden, num_classes)
        )

    def _pooling(self, outputs, attention_mask=None):
        hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden_dim]

        if self.pooling == "cls":
            return hidden_states[:, 0, :]
        elif self.pooling == "max":
            return torch.max(hidden_states, dim=1)[0]
        else:  # mean
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                return sum_embeddings / sum_mask
            return torch.mean(hidden_states, dim=1)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self._pooling(outputs, attention_mask)
        logits = self.classifier(pooled)
        return logits

    def get_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self):
        return sum(p.numel() for p in self.parameters())
