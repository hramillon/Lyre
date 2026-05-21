import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, ff_dim, n_kv_heads, dropout=0.1, attn_class=None):
        super().__init__()
        # on prépare les layers
        #attention
        self.ln1 = RMSNorm(embed_dim)
        self.attn = attn_class(embed_dim, n_heads, n_kv_heads, dropout)
        # MLP
        self.ln2, self.ffn = RMSNorm(embed_dim),SwiGLU(embed_dim, ff_dim, dropout)
    def forward(self, x,cos, sin):
        # Calcul de l'attention sur l'entrée normalisée
        x = x + self.attn(self.ln1(x), cos, sin)
        # Calcul du FFN sur l'entrée normalisée
        return x + self.ffn(self.ln2(x))


# ---------------
# SwiGLU
# ---------------

class SwiGLU(nn.Module):
    def __init__(self, embed_dim, ff_dim, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(embed_dim, ff_dim, bias=False)
        self.w2 = nn.Linear(embed_dim, ff_dim, bias=False)
        self.w3 = nn.Linear(ff_dim, embed_dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.w3(F.silu(self.w1(x)) * self.w2(x)))

# ---------------
# RMSNorm
# ---------------
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight

# ---------------
# RoPE
# ---------------
#head = 64
def precompute_rope(head_dim, max_len, base=10000, device=None):
    # une fréquence theta par paire de dimensions
    theta = 1.0 / (base ** (
        torch.arange(0, head_dim, 2, device=device).float() / head_dim
    ))
    # toutes les positions
    positions = torch.arange(max_len, device=device).float()
    # matrice (max_len, head_dim//2) angle de chaque position/dimension
    freqs = torch.outer(positions, theta)
    return freqs.cos(), freqs.sin()

def apply_rope(x, cos, sin):
    T = x.shape[2]
    cos, sin = cos[:T], sin[:T]
    
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    
    # Calcul direct
    out_even = x1 * cos - x2 * sin
    out_odd = x1 * sin + x2 * cos
    
    # reconstruit sans allouer / problème de RAM
    out = torch.empty_like(x)
    out[..., ::2] = out_even
    out[..., 1::2] = out_odd
    
    return out

    return x_rot.flatten(-2) 