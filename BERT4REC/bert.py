import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple # code 가독성

class BERTEmbeddings(nn.Module):
    """
    BERT Embeddings:
        Token Embeddings : Token 정보
        Position Embeddings : 위치 정보
        Segment Embeddings : 여러 문장 입력 시 문장 구분에 사용
    """
    def __init__(self, vocab_size: int, embed_size: int, max_len: int, dropout_rate: float = 0.3):
        super(BERTEmbeddings, self).__init__()

        # item_id -> embedding 
        self.token_embeddings = nn.Embedding(vocab_size, embed_size, padding_idx=0)

        # 위치 정보 embedding
        self.positional_embeddings = nn.Embedding(max_len, embed_size)

        # normalization + regularization
        self.layer_norm = nn.LayerNorm(embed_size, eps=1e-6)
        self.dropout = nn.Dropout(p=dropout_rate)
    
    def forward(self, seq: torch.Tensor, segment_label: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_length = seq.size()

        # position index 생성: (B, T)
        position_ids = torch.arange(seq_length, device=seq.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_length)

        # embedding 계산
        token_embeddings = self.token_embeddings(seq)
        positional_embeddings = self.positional_embeddings(position_ids)

        # token + position
        embeddings = token_embeddings + positional_embeddings

        # layer norm + dropout
        return self.dropout(self.layer_norm(embeddings))
    
class MultiHeadedAttention(nn.Module):
    """
    Multi-Head Self Attention
    """

    def __init__(self, head_num, hidden_dim, dropout_rate_attn=0.1):
        super(MultiHeadedAttention, self).__init__()

        assert hidden_dim % head_num == 0

        self.hidden_dim = hidden_dim
        self.head_num = head_num
        self.head_dim = hidden_dim // head_num

        # Q, K, V projection
        self.query_linear = nn.Linear(hidden_dim, hidden_dim)
        self.key_linear = nn.Linear(hidden_dim, hidden_dim)
        self.value_linear = nn.Linear(hidden_dim, hidden_dim)

        self.scale = math.sqrt(self.head_dim)
        self.dropout = nn.Dropout(p=dropout_rate_attn)

        # output projection
        self.output_linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, q, k, v, mask=None):
        B = q.size(0)

        # Q, V, K 셍성
        query = self.query_linear(q)
        key   = self.key_linear(k)
        value = self.value_linear(v)

        # (B, T, D) → (B, H, T, D/H)
        query = query.view(B, -1, self.head_num, self.head_dim).permute(0, 2, 1, 3)
        key   = key.view(B, -1, self.head_num, self.head_dim).permute(0, 2, 1, 3)
        value = value.view(B, -1, self.head_num, self.head_dim).permute(0, 2, 1, 3)

        # attention score 계산
        scores = torch.matmul(query, key.transpose(-1, -2)) / self.scale

        # padding mask 적용
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # attention 확률
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        # weighted sum
        out = torch.matmul(attention, value)

        # 다시 (B, T, D)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, -1, self.hidden_dim)

        return self.output_linear(out), attention
    
class SublayerConnection(nn.Module):
    def __init__(self, hidden_dim: int, dropout_rate: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.dropout    = nn.Dropout(p=dropout_rate)

    def forward(self, x, sublayer_out):
        # ✅ 표준 Post-LN: LayerNorm(x + dropout(sublayer_out))
        return self.layer_norm(x + self.dropout(sublayer_out))
    
class PositionwiseFeedForward(nn.Module):
    """
    FFN: (D → 4D → D)
    """

    def __init__(self, hidden_dim: int, ff_dim: int, dropout_rate: float = 0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, hidden_dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.fc2(self.dropout(self.activation(self.fc1(x))))
    
class TransformerEncoder(nn.Module):
    """
    Transformer Block = Attention + FFN
    """

    def __init__(self, hidden_dim, head_num, ff_dim, dropout_rate=0.1, dropout_rate_attn=0.1):
        super(TransformerEncoder, self).__init__()

        self.attention = MultiHeadedAttention(head_num, hidden_dim, dropout_rate_attn)

        self.input_sublayer = SublayerConnection(hidden_dim, dropout_rate)
        self.feed_forward   = PositionwiseFeedForward(hidden_dim, ff_dim, dropout_rate)
        self.output_sublayer = SublayerConnection(hidden_dim, dropout_rate)

        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x, mask):
        # self-attention
        attn_out, _ = self.attention(x, x, x, mask)

        # residual 1
        x = self.input_sublayer(x, attn_out)

        # FFN + residual
        x = self.output_sublayer(x, self.feed_forward(x))

        return self.dropout(x)
    
class BERT(nn.Module):
    """
    BERT4Rec backbone (Transformer encoder stack)
    """

    def __init__(
        self,
        vocab_size=30522,
        max_len=512,
        hidden_dim=768,
        encoder_num=12,
        head_num=12,
        dropout_rate=0.1,
        dropout_rate_attn=0.1,
        initializer_range=0.02
    ):
        super(BERT, self).__init__()

        self.ff_dim = hidden_dim * 4

        # embedding layer
        self.embedding = BERTEmbeddings(vocab_size, hidden_dim, max_len)

        # transformer stack
        self.transformers = nn.ModuleList([
            TransformerEncoder(hidden_dim, head_num, self.ff_dim,
                               dropout_rate, dropout_rate_attn)
            for _ in range(encoder_num)
        ])

        # weight initialization
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
        """
        입력:
          seq (B, T)

        출력:
          (B, T, hidden_dim)
        """

        # padding mask 생성 (0 = PAD)
        mask = (seq > 0).unsqueeze(1).unsqueeze(1)

        # embedding
        x = self.embedding(seq, segment_info)

        # transformer encoder stack
        for transformer in self.transformers:
            x = transformer(x, mask)

        return x
