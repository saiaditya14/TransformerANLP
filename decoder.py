from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import MultiHeadAttention

try:
    from utils import pad_msk, fut_mask
except Exception:
    pad_msk = fut_mask = None

class PositionwiseFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, activation: str = "gelu"):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation.lower()
        if self.activation not in ("gelu", "relu", "silu"):
            raise ValueError("activation must be 'gelu', 'relu', or 'silu'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        if self.activation == "gelu":
            x = F.gelu(x)
        elif self.activation == "relu":
            x = F.relu(x)
        else:
            x = F.silu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        self_pos_encoding: str = "rope",
        cross_pos_encoding: str = "none",
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
            pos_encoding=self_pos_encoding,
            rope_base=rope_base,
            rpb_num_buckets=rpb_num_buckets,
            rpb_max_distance=rpb_max_distance,
            rpb_bidirectional=rpb_bidirectional,
        )
        self.cross_attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            pos_encoding=cross_pos_encoding,
            rope_base=rope_base,
            rpb_num_buckets=rpb_num_buckets,
            rpb_max_distance=rpb_max_distance,
            rpb_bidirectional=rpb_bidirectional,
        )
        #calls from before
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=norm_eps)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.drop3 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor],
        self_key_padding_mask: Optional[torch.Tensor],
        cross_key_padding_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, dict]:
        h = self.norm1(x)
        h, self_attn_weights = self.self_attn(
            h, h, h,
            attn_mask=self_attn_mask,
            key_padding_mask=self_key_padding_mask,
            use_rope_q=True, use_rope_k=True,
        )
        #self attention for tokens produced so far
        x = x + self.drop1(h)
        #residual connections to not lose info
        h = self.norm2(x)
        h, cross_attn_weights = self.cross_attn(
            h, memory, memory, #memory encoder outputs
            attn_mask=None,
            key_padding_mask=cross_key_padding_mask,
            use_rope_q=True, use_rope_k=True,
        )
        #contextualised vectors attneding on encoder output tokens
        x = x + self.drop2(h)

        h = self.norm3(x)
        h = self.ffn(h)
        x = x + self.drop3(h)

        cache = {"self_attn": self_attn_weights, "cross_attn": cross_attn_weights}
        return x, cache

class TransformerDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        pad_idx: int,
        dropout: float = 0.1,
        self_pos_encoding: str = "rope",
        cross_pos_encoding: str = "none", #cos what I cant calculate relatives across encoder out and decoder in order change in translation
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
        #the learned ebeddings for input tokens
        self.drop = nn.Dropout(dropout)
        #dropout heheheha
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
                self_pos_encoding=self_pos_encoding,
                cross_pos_encoding=cross_pos_encoding,
                rope_base=rope_base,
                rpb_num_buckets=rpb_num_buckets,
                rpb_max_distance=rpb_max_distance,
                rpb_bidirectional=rpb_bidirectional,
                norm_eps=norm_eps,
            )
            for _ in range(num_layers)
            #jsut setting up layers
        ])
        self.final_norm = nn.LayerNorm(d_model, eps=norm_eps)

        self.generator = nn.Linear(d_model, vocab_size, bias=False)
        #scale to probabilities across vocabulary matrix
        if tie_embeddings:
            #possible to tie embeddings i.e vector * embmat for encoding and decoding = hidden * (embmat)^T essentially reuse transpose
            if self.embed.weight.shape != self.generator.weight.shape:
                raise ValueError("Cannot tie embeddings: shape mismatch.")
            self.generator.weight = self.embed.weight

    def forward(
        self,
        tgt_tokens: torch.Tensor,
        memory: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor],
        self_key_padding_mask: Optional[torch.Tensor],
        cross_key_padding_mask: Optional[torch.Tensor]
    ):
        x = self.embed(tgt_tokens) * (self.d_model ** 0.5)
        x = self.drop(x)
        #apply dropout for regularisation
        caches = {"self_attn": [], "cross_attn": []}
        for layer in self.layers:
            x, cache = layer(
                x,
                memory,
                self_attn_mask=self_attn_mask,
                self_key_padding_mask=self_key_padding_mask,
                cross_key_padding_mask=cross_key_padding_mask,
            )
            caches["self_attn"].append(cache["self_attn"])
            caches["cross_attn"].append(cache["cross_attn"])

        x = self.final_norm(x)
        logits = self.generator(x)
        #generate output tokens
        return logits, caches

if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, S = 2, 5, 7
    V, D, H, L = 5000, 128, 8, 2
    pad_idx = 0

    tgt = torch.tensor([[4, 5, 6, 0, 0],
                        [7, 8, 0, 0, 0]], dtype=torch.long)
    memory = torch.randn(B, S, D)

    if fut_mask is not None:
        fut = fut_mask(T)
    else:
        tri = torch.tril(torch.ones((T, T), dtype=torch.bool))
        fut = (~tri).unsqueeze(0).unsqueeze(0)

    dec_pad = (tgt == pad_idx).unsqueeze(1).unsqueeze(2) if pad_msk is None else pad_msk(tgt, pad_idx)
    enc_pad = torch.zeros(B, S, dtype=torch.bool)
    enc_pad[:, -2:] = True
    enc_pad = enc_pad.unsqueeze(1).unsqueeze(2)

    dec = TransformerDecoder(V, D, L, H, d_ff=4*D, pad_idx=pad_idx, self_pos_encoding="rope")
    logits, caches = dec(tgt, memory, fut, dec_pad, enc_pad)
    print("logits:", logits.shape, "self_attn[0]:", caches["self_attn"][0].shape)
