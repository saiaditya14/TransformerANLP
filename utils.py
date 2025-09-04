import torch
#Convention: True means blocked
def pad_msk(seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    #inputs, datatypes and output also keyword args for this one
    #this one returns tensor from shape (batch,seq) to (batch,1,1,seq), projection across all heads
    if seq.dim() != 2:
        raise ValueError(f"wtf")
    return (seq == pad_idx).unsqueeze(1).unsqueeze(2)

def fut_mask(dim: int) -> torch.Tensor:
    #this one returns uppertraiangular projected for each sentence (batch) and each head
    if dim <= 0:
        raise ValueError(f"wtf")
    mask = torch.zeros((dim,dim), dtype=torch.bool)
    for i in range(dim):
        for j in range(i+1,dim):
            mask[i][j] = True
    return mask.unsqueeze(0).unsqueeze(0)

def comb(m1: torch.Tensor | None, m2: torch.Tensor | None) -> torch.Tensor:
    if m1 is None: return m2
    elif m2 is None: return m1
    return m1|m2


if __name__ == "__main__":
    seq = torch.tensor([[5, 6, 7, 0, 0],
                        [9, 8, 0, 0, 0]])
    pad_block = pad_msk(seq, pad_idx=0)          
    fut_block = fut_mask(5)    
    both_block = comb(pad_block, fut_block)

    print("pad_block shape:", pad_block.shape)   
    print("fut_block shape:", fut_block.shape)   
    print("both_block shape:", both_block.shape) 