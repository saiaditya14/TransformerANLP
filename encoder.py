import torch
import torch.nn as nn
import torch.nn.functional as F

class dotprodatt(nn.Module):
    #Attention(Q,K,V) = softmax(QK^T/sqrt(d==hidden/heads) + mask)
    def __init__(self,dropout: float):
        super().__init__()
        #Python OOP when I create this class I make it clear it is inheriting from nn.Module and I am now just calling its constructor
        self.dropout = nn.Dropout(dropout)
        #atahcing dropout layer
    def forward(self,Q,K,V,mask=None):
        d = Q.size(-1)
        #last dimension
        scores = torch.matmul(Q,K.transpose(-2,-1))/d**0.5
        if mask is not None:
            scores = scores.masked_fill(mask,-1e9)
            #everywhere mask is true block
        attn = F.softmax(scores,dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn,V)
        return out,attn
if __name__ == "__main__":
    torch.manual_seed(0)

    batch_size = 2
    heads = 1
    seq_len = 4
    d_k = 8  # hidden size per head

    # [batch, heads, seq_len, d_k]
    Q = torch.randn(batch_size, heads, seq_len, d_k)
    K = torch.randn(batch_size, heads, seq_len, d_k)
    V = torch.randn(batch_size, heads, seq_len, d_k)

    # build a simple pad mask: last two tokens in each sequence are PAD
    pad_mask = torch.tensor([[0, 0, 1, 1],
                             [0, 1, 1, 1]], dtype=torch.bool)  # [batch, seq_len]
    pad_mask = pad_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]

    attn_layer = dotprodatt(dropout=0.1)
    out, attn = attn_layer(Q, K, V, mask=pad_mask)

    print("Output shape:", out.shape)   # [batch, heads, seq_len, d_k]
    print("Attention shape:", attn.shape)  # [batch, heads, seq_len, seq_len]
    print("Attention weights (batch 0):")
    print(attn[0, 0])