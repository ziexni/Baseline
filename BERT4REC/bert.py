import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class BERTEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int,
                 max_len: int, dropout_rate: float = 0.1):
        super().__init__()
        self.token_embeddings      = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.positional_embeddings = nn.Embedding(max_len, embed_size)
        self.layer_norm            = nn.LayerNorm(embed_size, eps=1e-12)
        self.dropout               = nn.Dropout(p=dropout_rate)

    def forward(self, seq: torch.Tensor,
                segment_label: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_length = seq.size()
        position_ids = torch.arange(seq_length, device=seq.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_length)
        embeddings   = (self.token_embeddings(seq)
                        + self.positional_embeddings(position_ids))
        return self.dropout(self.layer_norm(embeddings))


class MultiHeadedAttention(nn.Module):
    def __init__(self, head_num, hidden_dim, dropout_rate_attn=0.1):
        super().__init__()
        assert hidden_dim % head_num == 0
        self.hidden_dim    = hidden_dim
        self.head_num      = head_num
        self.head_dim      = hidden_dim // head_num
        self.query_linear  = nn.Linear(hidden_dim, hidden_dim)
        self.key_linear    = nn.Linear(hidden_dim, hidden_dim)
        self.value_linear  = nn.Linear(hidden_dim, hidden_dim)
        self.scale         = math.sqrt(self.head_dim)
        self.dropout       = nn.Dropout(p=dropout_rate_attn)
        self.output_linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, q, k, v, mask=None):
        B = q.size(0)
        query = self.query_linear(q).view(B,-1,self.head_num,self.head_dim).permute(0,2,1,3)
        key   = self.key_linear(k).view(B,-1,self.head_num,self.head_dim).permute(0,2,1,3)
        value = self.value_linear(v).view(B,-1,self.head_num,self.head_dim).permute(0,2,1,3)

        scores = torch.matmul(query, key.transpose(-1,-2)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attention, value)
        out = out.permute(0,2,1,3).contiguous().view(B,-1,self.hidden_dim)
        return self.output_linear(out)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, hidden_dim: int, ff_dim: int, dropout_rate: float = 0.1):
        super().__init__()
        self.fc1        = nn.Linear(hidden_dim, ff_dim)
        self.fc2        = nn.Linear(ff_dim, hidden_dim)
        self.dropout    = nn.Dropout(p=dropout_rate)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.fc2(self.dropout(self.activation(self.fc1(x))))


class TransformerEncoder(nn.Module):
    """
    ✅ 원본 레포와 동일한 Pre-LN 방식
    x = x + Attention(LayerNorm(x))
    x = x + FFN(LayerNorm(x))
    """
    def __init__(self, hidden_dim, head_num, ff_dim,
                 dropout_rate=0.1, dropout_rate_attn=0.1):
        super().__init__()
        self.attention  = MultiHeadedAttention(head_num, hidden_dim, dropout_rate_attn)
        self.ffn        = PositionwiseFeedForward(hidden_dim, ff_dim, dropout_rate)
        self.attn_norm  = nn.LayerNorm(hidden_dim, eps=1e-12)   # Pre-LN: attn 전
        self.ffn_norm   = nn.LayerNorm(hidden_dim, eps=1e-12)   # Pre-LN: ffn 전
        self.dropout    = nn.Dropout(p=dropout_rate)

    def forward(self, x, mask):
        # ✅ Pre-LN attention
        x = x + self.dropout(self.attention(self.attn_norm(x),
                                            self.attn_norm(x),
                                            self.attn_norm(x), mask))
        # ✅ Pre-LN FFN
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        return x


class BERT(nn.Module):
    def __init__(self, vocab_size=30522, max_len=512, hidden_dim=768,
                 encoder_num=12, head_num=12, dropout_rate=0.1,
                 dropout_rate_attn=0.1, initializer_range=0.02):
        super().__init__()
        self.ff_dim = hidden_dim * 4

        self.embedding    = BERTEmbeddings(vocab_size, hidden_dim, max_len, dropout_rate)
        self.transformers = nn.ModuleList([
            TransformerEncoder(hidden_dim, head_num, self.ff_dim,
                               dropout_rate, dropout_rate_attn)
            for _ in range(encoder_num)
        ])

        self.initializer_range = initializer_range
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(0.0, self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, seq, segment_info=None):
        mask = (seq > 0).unsqueeze(1).unsqueeze(1)   # (B, 1, 1, T)
        x    = self.embedding(seq, segment_info)
        for transformer in self.transformers:
            x = transformer(x, mask)
        return x   # ✅ output head 제거 — weight tying이 직접 처리
