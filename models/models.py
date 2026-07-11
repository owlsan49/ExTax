"""
PersuRoleEmoNet 模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
from transformers import AutoModel


class HeteroMultiHeadAttention(nn.Module):
    """异质多头注意力：说服头(6维)、角色头(6维)、情感头(5维)"""

    def __init__(
        self,
        dropout: float = 0.1
    ):
        super().__init__()

        self.head_dims = [6, 6, 5]  # 说服维、角色维、情感维
        self.num_heads = len(self.head_dims)

        # 输入投影到各自维度
        self.q_proj = nn.ModuleList([
            nn.Linear(dim, dim) for dim in self.head_dims
        ])
        self.k_proj = nn.ModuleList([
            nn.Linear(dim, dim) for dim in self.head_dims
        ])
        self.v_proj = nn.ModuleList([
            nn.Linear(dim, dim) for dim in self.head_dims
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(17)

    def forward(
        self,
        query: torch.Tensor,  # [batch_size, q_len, dim]
        key: torch.Tensor,  # [batch_size, k_len, dim]
        value: torch.Tensor,  # [batch_size, k_len, dim]
        attention_mask: Optional[torch.Tensor] = None  # [batch_size, q_len] or [batch_size, q_len, k_len]
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            query: 查询向量
            key: 键向量
            value: 值向量
            attention_mask: 注意力掩码

        Returns:
            输出向量 [batch_size, q_len, sum(head_dims)]
        """
        head_outputs = []
        focus_range = [slice(0, 6), slice(6, 12), slice(12, 17)]
        for i in range(self.num_heads):
            q = self.q_proj[i](query[:, :, focus_range[i]])
            k = self.k_proj[i](key[:, :, focus_range[i]])
            v = self.v_proj[i](value[:, :, focus_range[i]])

            # 计算注意力分数
            scores = torch.einsum('bqd,bkd->bqk', q, k) / (self.head_dims[i] ** 0.5)

            # 应用注意力掩码
            if attention_mask is not None:
                if attention_mask.dim() == 2:
                    mask = attention_mask.unsqueeze(2)  # [batch_size, q_len, 1]
                else:
                    mask = attention_mask  # [batch_size, q_len, k_len]
                scores = scores.masked_fill(mask == 0, float('-inf'))

            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)

            # 加权求和
            output = torch.einsum('bqk,bkd->bqd', attn_weights, v)
            head_outputs.append(output)

        # 拼接多头输出
        output = torch.cat(head_outputs, dim=-1)  # [batch_size, q_len, 6+6+5]
        output += query
        output = self.layer_norm(output)

        return output



class PersuRoleEmoNet(nn.Module):
    """
    多任务学习模型

    使用预训练的Transformer作为文本编码器，包含:
    - Encoder: 预训练Transformer (如roberta-large)
    - multi_feat_mlp: 多维特征提取头 (输出17维: 6+6+5)
    - cls_head: 分类头 (fake/real)

    两阶段训练:
    - 'pretrain': 第一阶段，训练multi_feat_mlp，返回multi_feat输出
    - 'cls': 第二阶段，输入multi_feat输出，输出分类结果
    """

    def __init__(
        self,
        encoder_name: str = "roberta-large",
        pooling: str = "mean",
        persu_hidden: int = 256,
        num_classes: int = 2,
        dropout: float = 0.25,
        max_length: int = 512,
        num_learnable_queries: int = 0,
        num_hetero_attention: int = 3,
        use_hetero_attention: bool = True,
        use_sum_weight: bool = True,
    ):
        super().__init__()

        # 预训练文本编码器
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.hidden_dim = self.encoder.config.hidden_size
        self.pooling = pooling
        self.num_learnable_queries = num_learnable_queries
        self.use_hetero_attention = use_hetero_attention
        self.use_sum_weight = use_sum_weight

        # 可学习 query
        if num_learnable_queries > 0:
            self.learnable_queries = nn.Parameter(torch.randn(num_learnable_queries, 17) * 0.02)
            if use_sum_weight:
                # 初始时 position 0 权重接近 1，其他接近 0
                init_weight = torch.zeros(1 + num_learnable_queries)
                init_weight[0] = 10.0
                self.sum_weight = nn.Parameter(init_weight)

        # 非线性维度变换 MLP: hidden_dim -> 17
        self.dim_transform = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 17)
        )

        # HeteroAttention Block
        if use_hetero_attention:
            self.hetero_attention_block = nn.ModuleList([
                HeteroMultiHeadAttention(dropout=dropout)
                for _ in range(num_hetero_attention)
            ])

        # 说服维特征mlp 
        self.persu_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, persu_hidden),
            nn.GELU(),
            nn.LayerNorm(persu_hidden),
            nn.Dropout(dropout),
            nn.Linear(persu_hidden, 6)
        )
        
        # 角色特征mlp 
        self.role_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, persu_hidden),
            nn.GELU(),
            nn.LayerNorm(persu_hidden),
            nn.Dropout(dropout),
            nn.Linear(persu_hidden, 6)
        )
        
        # 情感特征mlp 
        self.emo_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, persu_hidden),
            nn.GELU(),
            nn.LayerNorm(persu_hidden),
            nn.Dropout(dropout),
            nn.Linear(persu_hidden, 5)
        )


        # 分类头 - fake/real
        self.cls_head = nn.Sequential(
            nn.Linear(6+6+5, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
        self.max_len = max_length

        # 可学习 gated pooling
        if pooling == "gated":
            self.gated_pooling = nn.Parameter(torch.ones(max_length))
        

    def _pooling(self, outputs, attention_mask=None):
        """池化操作"""
        hidden_states = outputs.last_hidden_state  # [batch_size, seq_len, hidden_dim]
        batch_size, seq_len, _ = hidden_states.shape

        if self.pooling == "cls":
            return hidden_states[:, 0, :]
        elif self.pooling == "max":
            return torch.max(hidden_states, dim=1)[0]
        elif self.pooling == "gated":
            # gated pooling: 根据实际长度截断 gated 并 softmax，然后加权求和
            if not hasattr(self, 'gated_pooling'):
                # fallback to mean if gated_pooling not created
                if attention_mask is not None:
                    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                    sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
                    sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                    return sum_embeddings / sum_mask
                return torch.mean(hidden_states, dim=1)

            if attention_mask is None:
                lengths = torch.full((batch_size,), seq_len, dtype=torch.long, device=hidden_states.device)
            else:
                lengths = attention_mask.sum(dim=1).long()  # [batch_size]

            # 取每个样本对应的 gated 参数并 softmax
            gated_expanded = self.gated_pooling[:seq_len].unsqueeze(0).expand(batch_size, -1)
            valid_mask = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0).expand(batch_size, -1)
            length_mask = valid_mask < lengths.unsqueeze(1)  # [batch_size, seq_len]

            gated_masked = gated_expanded.masked_fill(~length_mask, float('-inf'))
            weights = torch.softmax(gated_masked, dim=1)  # [batch_size, seq_len]

            # 加权求和
            pooled = torch.einsum('bn,bnh->bh', weights, hidden_states)

            return pooled
        else:  # "mean"
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                return sum_embeddings / sum_mask
            return torch.mean(hidden_states, dim=1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        stage: str = 'cls',
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            input_ids: 输入token IDs [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            stage: 当前阶段

        Returns:
            包含不同输出的字典
        """
        batch_size, seq_len = input_ids.shape
        
        # 编码文本
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = self._pooling(outputs, attention_mask)

        results = {'embeddings': embeddings}

        # 第一阶段：多维特征对齐 (输出17维: 6+6+5)
        persu_feat = self.persu_mlp(embeddings)
        role_feat = self.role_mlp(embeddings)
        emo_feat = self.emo_mlp(embeddings)
        multi_feat = torch.cat([persu_feat, role_feat, emo_feat], dim=-1)
        results['multi_feat'] = multi_feat

        if stage == 'cls':
            if self.use_hetero_attention and hasattr(self, 'hetero_attention_block'):
                # 第二阶段：经过 HeteroAttention Block
                hidden_states = outputs.last_hidden_state
                transformed = self.dim_transform(hidden_states)  # [batch_size, seq_len, 17]

                # 变换 attention_mask 为 3D: [batch_size, 1, seq_len]
                attn_mask = attention_mask.unsqueeze(1) if attention_mask is not None else None

                # query 由 multi_feat 和 learnable_queries 拼接
                query_input = multi_feat.unsqueeze(1)
                if self.num_learnable_queries > 0:
                    expanded_learnable_queries = self.learnable_queries.unsqueeze(0).expand(batch_size, -1, -1)
                    query_input = torch.cat([query_input, expanded_learnable_queries], dim=1)  # [batch_size, 1+num_learnable_queries, 17]

                # 依次通过 HeteroMultiHeadAttention
                hetero_output = query_input
                for attn in self.hetero_attention_block:
                    hetero_output = attn(
                        query=hetero_output,
                        key=transformed,
                        value=transformed,
                        attention_mask=attn_mask
                    )

                # 与 sum_weight 的 softmax 做内积，或 mean pooling
                if self.num_learnable_queries > 0 and self.use_sum_weight and hasattr(self, 'sum_weight'):
                    weight = F.softmax(self.sum_weight, dim=0)  # [1 + num_learnable_queries]
                    cls_input = torch.einsum('n,bnd->bd', weight, hetero_output)  # [batch_size, 17]
                else:
                    cls_input = torch.mean(hetero_output, dim=1)  # [batch_size, 17]
            else:
                cls_input = multi_feat  # [batch_size, 17]

            results['cls_logits'] = self.cls_head(cls_input)

        return results

    def set_trainable(self, stage: str):
        """根据阶段设置哪些参数可训练"""
        for param in self.parameters():
            param.requires_grad = False

        if stage == 'align':
            # 第一阶段：训练multi_feat_mlp
            for layer in [self.persu_mlp, self.role_mlp, self.emo_mlp]:
                for p in layer.parameters():
                    p.requires_grad = True
            parts = ['persu_mlp', 'role_mlp', 'emo_mlp']
            if hasattr(self, 'gated_pooling'):
                self.gated_pooling.requires_grad = True
                parts.append('gated_pooling')
            print(f"Stage: {stage} | Trainable: {', '.join(parts)}")

        elif stage == 'cls':
            # 第二阶段：训练 cls 相关参数
            for p in self.cls_head.parameters():
                p.requires_grad = True
            parts = ['cls_head']
            for p in self.dim_transform.parameters():
                p.requires_grad = True
            parts.append('dim_transform')
            if hasattr(self, 'hetero_attention_block'):
                for attn in self.hetero_attention_block:
                    for p in attn.parameters():
                        p.requires_grad = True
                parts.append('hetero_attention_block')
            if self.num_learnable_queries > 0:
                self.learnable_queries.requires_grad = True
                parts.append('learnable_queries')
                if hasattr(self, 'sum_weight'):
                    self.sum_weight.requires_grad = True
                    parts.append('sum_weight')
            print(f"Stage: {stage} | Trainable: {', '.join(parts)}")

        else:
            raise ValueError(f"Unknown stage: {stage}")

    def get_trainable_params(self):
        """获取可训练参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self):
        """获取总参数数量"""
        return sum(p.numel() for p in self.parameters())