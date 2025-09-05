import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

from attention import MultiHeadAttention
try:
    from utils import pad_msk, fut_mask, comb
except Exception:
    pad_msk = fut_mask = comb = None


from decoder import PositionwiseFFN
#will be the same whether it is encoder or decoder
class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        pos_encoding: str = "rope",
        rope_base: float = 10000.0,
        rpb_num_buckets: int = 32,
        rpb_max_distance: int = 128,
        rpb_bidirectional: bool = True,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            pos_encoding=pos_encoding,
            rope_base=rope_base,
            rpb_num_buckets=rpb_num_buckets,
            rpb_max_distance=rpb_max_distance,
            rpb_bidirectional=rpb_bidirectional,
        )
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=norm_eps)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, dict]:
        h = self.norm1(src)
        h, attn_w = self.self_attn(h, h, h, attn_mask=None, key_padding_mask=src_key_padding_mask, use_rope_q=True, use_rope_k=True)
        src = src + self.drop1(h)

        h = self.norm2(src)
        h = self.ffn(h)
        src = src + self.drop2(h)

        return src, {"self_attn": attn_w}


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        pad_idx: int,
        dropout: float = 0.1,
        pos_encoding: str = "rope",
        tie_embeddings: bool = False,
        rope_base: float = 10000.0,
        rpb_num_buckets: int = 32,
        rpb_max_distance: int = 128,
        rpb_bidirectional: bool = True,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_idx = pad_idx

        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.drop = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
                pos_encoding=pos_encoding,
                rope_base=rope_base,
                rpb_num_buckets=rpb_num_buckets,
                rpb_max_distance=rpb_max_distance,
                rpb_bidirectional=rpb_bidirectional,
                norm_eps=norm_eps,
            )
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model, eps=norm_eps)

    def forward(self, src_tokens: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, List[dict]]:
        x = self.embed(src_tokens) * (self.d_model ** 0.5)
        x = self.drop(x)

        attn_caches = []
        for layer in self.layers:
            x, cache = layer(x, src_key_padding_mask)
            attn_caches.append(cache["self_attn"])

        x = self.final_norm(x)
        return x, attn_caches

#main explanations needed written in decoder.py
if __name__ == "__main__":
    torch.manual_seed(0)
    B, S = 2, 7
    V, D, H, L = 5000, 128, 8, 2
    pad_idx = 0
    src = torch.tensor([[4,5,6,0,0,0,0],[7,8,9,10,11,0,0]], dtype=torch.long)
    src_pad = (src == pad_idx).unsqueeze(1).unsqueeze(2)  # [B,1,1,S]

    enc = TransformerEncoder(V, D, L, H, d_ff=4*D, pad_idx=pad_idx, pos_encoding="rope")
    memory, caches = enc(src, src_pad)
    print("memory:", memory.shape, "self_attn[0]:", caches[0].shape)
