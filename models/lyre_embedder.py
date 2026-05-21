import os
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

from archi import RMSNorm, SwiGLU, precompute_rope, apply_rope, TransformerBlock

"""
Le RAG sera une partie très importante de l'architecture pour pallier les lacunes de la LLM en culture générale pour ceci
 - on entraine un modèle de 50-100M de param avec un trnsfo encodeur de type BERT
 - on entraine sur mmarco french des paires avec questions -> réponses de cette façon on pourra regarder sur internet quelles sont le sparagraphes qui correspondent le plus 
   à notre question. La LLM récupère et reformule (ça le fine tunning s'en occupe de lui apprendre à faire ça)
- On garde les implémentaions Lyre2
"""

# LA MAJORITE DE CE ".py" EST EN GROS UN COPIER COLLER DE "lyre.py" LE LIRE POUR COMPRENDRE

# FIX SYSTEME
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


# =============================================================================
# HYPERPARAMÈTRES
# =============================================================================
#Pour Rag

MAX_LEN          = 512
VOCAB_SIZE       = 32768
#Default GPT2 has 768 for 50k but mistral 1024 for 30k.
EMBEDDING_DIM    = 768
# 16 layers 16 heads
N_BLOCKS         = 12
N_HEADS          = 12
FEED_FORWARD_DIM = 2048
DROPOUT          = 0.1
# 2-3Go
EPOCHS           = 1
# pour le grouped Query Attention
N_KV_HEADS = 4
MODEL_SAVE_DIR   = "/tmp/checkpoint/"
# will try to  increase it 
BATCH_SIZE = 64

SAVE_EVERY       = 500 * ACCUM_STEPS 
ACCUM_STEPS = 1
RESUME_PATH      = os.path.join(MODEL_SAVE_DIR, "latest_best2.pt")

CHECKPOINT_PATH      = os.path.join(MODEL_SAVE_DIR, "checks.pt")

# =============================================================================
# INIT DDP (NCCL)
# =============================================================================

torch.cuda.set_device(local_rank)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(MODEL_SAVE_DIR):
    os.makedirs(MODEL_SAVE_DIR)

# =============================================================================
# DATASET
# =============================================================================

tokenizer = AutoTokenizer.from_pretrained("token/tokenizer_lyre.json")

class MMarcoDataset(Dataset):
    def __init__(self, max_len=512):
        self.data = load_dataset('unicamp-dl/mmarco', 'french')['train']
        self.max_len = max_len

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        query   = tokenizer(item['query'],    max_length=self.max_len, truncation=True, padding='max_length', return_tensors='pt')
        pos     = tokenizer(item['positive'], max_length=self.max_len, truncation=True, padding='max_length', return_tensors='pt')
        neg     = tokenizer(item['negative'], max_length=self.max_len, truncation=True, padding='max_length', return_tensors='pt')
        return query, pos, neg

def make_dataloader(dataset, batch_size):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

# =============================================================================
# MODÈLE
# =============================================================================

# rien à changer en comparaison à Lyre si vous voulez comprendre lire commentaires de lyre.py à la class
class BidirectionalAttention(nn.Module):
    def __init__(self, embed_dim, n_heads, n_kv_heads=4, dropout=0.1):
        super().__init__()
        self.n_kv_heads = n_kv_heads
        self.n_heads, self.head_dim = n_heads, embed_dim // n_heads
        self.n_rep      = n_heads // n_kv_heads
        self.q_proj  = nn.Linear(embed_dim, embed_dim, bias=False)
        self.kv_proj = nn.Linear(embed_dim, 2 * n_kv_heads * self.head_dim, bias=False)
        self.proj    = nn.Linear(embed_dim, embed_dim)
        self.drop    = nn.Dropout(dropout)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(x).view(B, T, 2, self.n_kv_heads, self.head_dim)
        k, v = kv.unbind(2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.drop.p if self.training else 0.0,
            is_causal=False  # <-- seul changement
        )
        return self.proj(out.transpose(1, 2).contiguous().view(B, T, C))

TransformerBlock(..., attn_class=BidirectionalAttention)

# plus de head qui sert à la génération
class LyreEmbedder(nn.Module):
    def __init__(self, vocab_size, max_len, embed_dim, n_heads, ff_dim, n_blocks, n_kv_heads, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, n_heads, ff_dim, n_kv_heads, dropout) for _ in range(n_blocks)])
        self.ln_f = RMSNorm(embed_dim)
        cos, sin = precompute_rope(embed_dim // n_heads, max_len)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx):
        x = self.drop(self.token_emb(idx))
        for block in self.blocks:
            x = block(x, self.cos, self.sin)
        x = self.ln_f(x)
        # mean pooling — représentation globale de la séquence
        return x.mean(dim=1)

# =============================================================================
# SETUP & REPRISE
# =============================================================================

model = LyreEmbedder(
    VOCAB_SIZE, MAX_LEN, EMBEDDING_DIM, N_HEADS,
    FEED_FORWARD_DIM, N_BLOCKS, N_KV_HEADS, DROPOUT,
).to(device)

total = sum(p.numel() for p in model.parameters())
print(f"[ARCH] Paramètres: {total/1e6:.1f}M")

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
loader = make_dataloader(dataset, BATCH_SIZE)
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lambda s: (
        s / 1000 if s < 1000
        else 0.5 * (1.0 + math.cos(
            math.pi * (s - 1000) / ((len(loader) * EPOCHS) - 1000)
        ))
    )
)

start_step = 0
best_loss = float('inf')

if os.path.exists(RESUME_PATH):
    ckpt = torch.load(RESUME_PATH, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    start_step = ckpt['step'] + 1
    best_loss  = ckpt['loss']
    print(f"[INFO] Resume au step {start_step} (Loss: {best_loss:.4f})")

# =============================================================================
# TRAIN
# =============================================================================

# On regarde la similarité dans l'espace
def triplet_loss(q, p, n, margin=0.2):
    q = F.normalize(q, dim=-1)
    p = F.normalize(p, dim=-1)
    n = F.normalize(n, dim=-1)
    
    sim_pos = (q * p).sum(dim=-1)  # similarité query/positif
    sim_neg = (q * n).sum(dim=-1)  # similarité query/négatif
    
    return F.relu(sim_neg - sim_pos + margin).mean()

total_steps = len(loader) * EPOCHS
t_start = time.time()

for epoch in range(EPOCHS):
    model.train()
    for step_in_epoch, (query, pos, neg) in enumerate(loader):
        global_step = epoch * len(loader) + step_in_epoch
        if global_step < start_step:
            continue

        query_ids = query['input_ids'].squeeze(1).to(device, non_blocking=True)
        pos_ids   = pos['input_ids'].squeeze(1).to(device, non_blocking=True)
        neg_ids   = neg['input_ids'].squeeze(1).to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            q_emb = model(query_ids)
            p_emb = model(pos_ids)
            n_emb = model(neg_ids)
            loss = triplet_loss(q_emb, p_emb, n_emb)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        current_loss_val = loss.item()
        elapsed = time.time() - t_start
        steps_done = global_step - start_step + 1
        eta = (total_steps - global_step) * (elapsed / steps_done)

        print(
            f"Step {global_step}/{total_steps} | "
            f"Loss: {current_loss_val:.4f} | "
            f"Passé: {time.strftime('%H:%M:%S', time.gmtime(elapsed))} | "
            f"ETA: {time.strftime('%H:%M:%S', time.gmtime(eta))}"
        )

        if current_loss_val < best_loss:
            best_loss = current_loss_val
            torch.save({
                'step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': best_loss,
            }, RESUME_PATH)
            print(f" >> [SAVE] Nouveau best à {best_loss:.4f} au step {global_step}")

        if (step_in_epoch + 1) % SAVE_EVERY == 0:
            torch.save({
                'step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': current_loss_val,
            }, CHECKPOINT_PATH)
            print(f" >> [SAVE] Checkpoint {current_loss_val:.4f} at {global_step}")