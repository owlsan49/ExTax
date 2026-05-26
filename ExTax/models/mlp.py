"""
模型模块 - 支持可扩展性
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any


class BaseModel(nn.Module):
    """模型基类"""

    name: str = "base"

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__()
        self.config = config or {}

    def forward(self, x):
        raise NotImplementedError

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())


class MLPClassifier(nn.Module):
    """多层感知机分类器"""

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dims: List[int] = None,
        output_dim: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 32]

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class PersuasionMLP(BaseModel):
    """说服力检测MLP模型"""

    name = "PersuasionMLP"

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dims: List[int] = None,
        num_classes: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        self.mlp = MLPClassifier(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=num_classes,
            dropout=dropout
        )

    def forward(self, x):
        return self.mlp(x)


# 模型注册表
MODEL_REGISTRY: Dict[str, type] = {
    'mlp': MLPClassifier,
    'persuasion_mlp': PersuasionMLP,
}


def register_model(name: str, model_class: type):
    """注册新模型"""
    MODEL_REGISTRY[name] = model_class


def build_model(model_name: str, **kwargs) -> nn.Module:
    """构建模型"""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")

    return MODEL_REGISTRY[model_name](**kwargs)