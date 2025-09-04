import math
from typing import Optional,Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from utils import pad_msk, fut_mask, comb 
except Exception:
    pad_msk = fut_mask = comb = None

def rotate(x: torch.tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    #makes a last dimension of two with first one having odd indexed stuff and second having even indexed stuff then interleaves
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def rope_cos_sin(seq_len: int, dim: int, base: float, device, dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    # RoPE needs even dimension (because it rotates pairs: even/odd)
    if dim % 2 != 0:
        raise ValueError(f"RoPE requires even head_dim; got {dim}")
    half = dim // 2
    # build inverse frequencies (like in sinusoidal positions but scaled by base)
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    # positions 0..L-1 as a column vector [L, 1]
    t = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(1)
    freqs = t * inv_freq.unsqueeze(0)
    emb = torch.cat([freqs, freqs], dim=-1).to(dtype)
    return emb.cos(), emb.sin()

#essentially above apply RoPe formula 1/base^... for hidden_dim/2 then rotate embedding i.e per position frequency associated with it then spread out and cos and sin

def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    while cos.dim() < x.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    return (x * cos) + (rotate(x) * sin)
#break 1

class RelativePositionBias(nn.Module):
    def __init__(self, num_heads: int, num_buckets: int = 32, max_distance: int = 128, bidirectional: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.bidirectional = bidirectional
        total_buckets = num_buckets * 2 if bidirectional else num_buckets
        self.emb = nn.Embedding(total_buckets, num_heads)

    def _relative_position_bucket(self, rel_pos: torch.Tensor) -> torch.Tensor:
        n = self.num_buckets
        if self.bidirectional:
            sign = (rel_pos > 0).to(torch.long)
            rel = rel_pos.abs()
            half = n
            max_exact = half // 2
            is_small = rel < max_exact
            large = max_exact + (
                (torch.log(rel.float() / max_exact + 1e-6) / math.log(self.max_distance / max_exact))
                * (half - max_exact)
            ).to(torch.long)
            large = large.clamp(max=half - 1)
            buckets = torch.where(is_small, rel, large)
            buckets = buckets + sign * half
            total = 2 * n
            return buckets.clamp(min=0, max=total - 1)
        else:
            rel = (-rel_pos).clamp(min=0)
            max_exact = n // 2
            is_small = rel < max_exact
            large = max_exact + (
                (torch.log(rel.float() / max_exact + 1e-6) / math.log(self.max_distance / max_exact))
                * (n - max_exact)
            ).to(torch.long)
            large = large.clamp(max=n - 1)
            buckets = torch.where(is_small, rel, large)
            return buckets.clamp(min=0, max=n - 1)

    def forward(self, q_len: int, k_len: int, device=None, dtype=None) -> torch.Tensor:
        q_pos = torch.arange(q_len, device=device)
        k_pos = torch.arange(k_len, device=device)
        rel = k_pos[None, :] - q_pos[:, None]
        buckets = self._relative_position_bucket(rel)
        bias = self.emb(buckets).permute(2, 0, 1)
        return bias.unsqueeze(0)

class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        pos_encoding: str = "rope",
        rope_base: float = 10000.0,
        rpb_num_buckets: int = 32,
        rpb_max_distance: int = 128,
        rpb_bidirectional: bool = True,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.pos_encoding = pos_encoding.lower()
        self.rope_base = rope_base
        if self.pos_encoding == "rpb":
            self.rpb = RelativePositionBias(
                num_heads=num_heads,
                num_buckets=rpb_num_buckets,
                max_distance=rpb_max_distance,
                bidirectional=rpb_bidirectional,
            )
        else:
            self.rpb = None

    def _shape(self, x: torch.Tensor, B: int) -> torch.Tensor:
        return x.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        use_rope_q: bool = True,
        use_rope_k: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, QL, _ = query.shape
        _, KL, _ = key.shape

        Q = self._shape(self.W_q(query), B)
        K = self._shape(self.W_k(key),   B)
        V = self._shape(self.W_v(value), B)

        if self.pos_encoding == "rope":
            cos_q, sin_q = rope_cos_sin(QL, self.head_dim, self.rope_base, Q.device, Q.dtype)
            cos_k, sin_k = rope_cos_sin(KL, self.head_dim, self.rope_base, K.device, K.dtype)
            if use_rope_q:
                Q = apply_rope(Q, cos_q, sin_q)
            if use_rope_k:
                K = apply_rope(K, cos_k, sin_k)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if self.pos_encoding == "rpb":
            bias = self.rpb(QL, KL, device=scores.device, dtype=scores.dtype)
            scores = scores + bias

        mask = comb(attn_mask, key_padding_mask) if comb is not None else (attn_mask if key_padding_mask is None else key_padding_mask if attn_mask is None else (attn_mask | key_padding_mask))
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)

        out = out.transpose(1, 2).contiguous().view(B, QL, self.d_model)
        out = self.W_o(out)
        return out, attn

if __name__ == "__main__":
    torch.manual_seed(0)
    B, Q, K, D, H = 2, 5, 7, 64, 8
    xq = torch.randn(B, Q, D)
    xk = torch.randn(B, K, D)
    xv = torch.randn(B, K, D)

    if fut_mask is not None:
        tri = torch.tril(torch.ones((Q, K), dtype=torch.bool))
        fut = (~tri).unsqueeze(0).unsqueeze(0)
    else:
        tri = torch.tril(torch.ones((Q, K), dtype=torch.bool))
        fut = (~tri).unsqueeze(0).unsqueeze(0)

    if pad_msk is not None:
        fake_keys = torch.zeros(B, K, dtype=torch.long)
        fake_keys[:, -2:] = 0
        kpad = pad_msk(fake_keys, pad_idx=0)
    else:
        kpad = torch.zeros(B, 1, 1, K, dtype=torch.bool)
        kpad[:, :, :, -2:] = True

    mha = MultiHeadAttention(D, H, pos_encoding="rope")
    y, a = mha(xq, xk, xv, attn_mask=fut, key_padding_mask=kpad)
    print("out:", y.shape, "attn:", a.shape)