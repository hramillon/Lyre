import os
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.backends.cuda import sdp_kernel

# FIX SYSTEME
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# =============================================================================
# HYPERPARAMÈTRES
# =============================================================================
#Pour Rag
MAX_LEN          = 1024
#vocab donnée par lyre_token.py
VOCAB_SIZE       = 32768
#Default GPT2 has 768 for 50k but mistral 1024 for 30k.
EMBEDDING_DIM    = 1024
# 16 layers 16 heads
N_BLOCKS         = 16
N_HEADS          = 16
# 4 * 1024 by convention
FEED_FORWARD_DIM = 4096
DROPOUT          = 0.1
# 14-20 Go is sufficient I don't want the model to overfit. However cuz there is a lot of data we can maybe try a second epoch to slightly increase the results without ouverfit
EPOCHS           = 1
"""
à absolument faire sur les trois machines 
mkdir -p /tmp/checkpoint

cp ./checkpoint/latest_best.pt /tmp/checkpoint/latest_best.pt
cp corpus_encoded.bin /tmp/corpus_encoded.bin

à absolument faire à la fin de la session d'entrianement

cp /tmp/checkpoint/latest_best.pt checkpoint/latest_best.pt

torchrun --nproc_per_node=1 --nnodes=3 --node_rank=0 --rdzv_id=lyre --rdzv_backend=c10d --rdzv_endpoint=10.0.104.4:29505 models/lyre.py

"""
BIN_FILE         = "/tmp/corpus_encoded.bin"
MODEL_SAVE_DIR   = "/tmp/checkpoint"
# memory issues 20go for 8 by gpus is too small
BATCH_SIZE_PER_GPU = 8
# by increasing BATCH_SIZE_PER_GPU we should decrease it to prevent the models to diverge
ACCUM_STEPS      = 128 

SAVE_EVERY       = 10 * ACCUM_STEPS 
RESUME_PATH      = os.path.join(MODEL_SAVE_DIR, "latest_best.pt")

# =============================================================================
# INIT DDP (NCCL)
# =============================================================================

## !! power SGD à mettre en oeuvre pour le second modèle de Lyre, ethernet empêche de baisser le ACCUM_STEPS   !!
## mettre un schéma du modèle avant le lancement afin de maitriser le nombre de param du modèle.
local_rank = int(os.environ["LOCAL_RANK"])
dist.init_process_group(backend="nccl")
rank, world_size = dist.get_rank(), dist.get_world_size()
is_chief = rank == 0

torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")

if is_chief:
    print(f"[INFO] {world_size} GPU(s), Ada 4000. Accumulation: {ACCUM_STEPS}")
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)

# =============================================================================
# DATASET
# =============================================================================
class CorpusDataset(Dataset):
    def __init__(self, bin_file: str, max_len: int, vocab_size: int):
        self.raw_data = np.fromfile(bin_file, dtype="int32")
        # n = nb à entrainer
        self.n = (len(self.raw_data) - 1) // max_len
        self.max_len, self.vocab_size = max_len, vocab_size
    def __len__(self): return self.n
    def __getitem__(self, idx):
        #idée simple code un peu compliqué on récupère le bon chunk à l'ID donnée
        offset = idx * self.max_len
        # on récupère le raw data on avance au chunk de max len * idx
        chunk = np.clip(self.raw_data[offset : offset + self.max_len + 1].copy(), 0, self.vocab_size - 1)
        #on rècupère le chunk correspondant 
        return torch.from_numpy(chunk[:-1].astype(np.int64)), torch.from_numpy(chunk[1:].astype(np.int64))

def make_dataloader(dataset, batch_size, world_size, rank):
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0, pin_memory=True), sampler

#14Go, 1024 , 30074
dataset = CorpusDataset(BIN_FILE, MAX_LEN, VOCAB_SIZE)
dist.barrier(device_ids=[local_rank])

# =============================================================================
# MODÈLE
# =============================================================================
# casual masque le futur (génération de texte)
class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout=0.1):
        super().__init__()
        #head dim  = 1024/16 =  64
        self.n_heads, self.head_dim = n_heads, embed_dim // n_heads
        # au lieu de faire trois lignes pour Query, Key, Value :  on fait entrer un tenseur de taille embed dim on sort trois embed dim
        # le proj concatène les résultats des différentes couches d'attention
        # dropout basique, régularisation simple en 0.1
        self.qkv, self.proj, self.drop = nn.Linear(embed_dim, 3 * embed_dim, bias=False), nn.Linear(embed_dim, embed_dim), nn.Dropout(dropout)
    def forward(self, x):
        # tenseur d'entrée x , divisée par son B(atch size) T(ime) =1024 C(hannel 1024)
        B, T, C = x.shape
        # créer des vues différentes pour Query, Key, View
        q, k, v = self.qkv(x).split(C, dim=2)
        # view divise c en deux puis transpose B ,n_heads ,T, head_dim
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        # calcul en parallèle on multiplie la matrice q par la transposée k / on applique le  masque triangulaire inférieure
        ## Softmax ( Q.k^{T}/sqrt(d_k) + M) * V ou M est le masque causal et d_k = head dim = embed_dim//n_head  = 64
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=self.drop.p if self.training else 0.0, is_causal=True)
        return self.proj(out.transpose(1, 2).contiguous().view(B, T, C))

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, ff_dim, dropout=0.1):
        super().__init__()
        # on prépare les layers
        #attention
        self.ln1, self.attn = nn.LayerNorm(embed_dim, eps=1e-5), CausalSelfAttention(embed_dim, n_heads, dropout)
        # MLP
        self.ln2, self.ffn = nn.LayerNorm(embed_dim, eps=1e-5), nn.Sequential(nn.Linear(embed_dim, ff_dim), nn.GELU(), nn.Linear(ff_dim, embed_dim), nn.Dropout(dropout))
    def forward(self, x):
        # Calcul de l'attention sur l'entrée normalisée
        x = x + self.attn(self.ln1(x))
        # Calcul du FFN sur l'entrée normalisée
        return x + self.ffn(self.ln2(x))

class Lyre(nn.Module):
    def __init__(self, vocab_size, max_len, embed_dim, n_heads, ff_dim, n_blocks, dropout=0.1):
        super().__init__()
        # token_emb : 30074 * 1024 pos_emb : 1024 * 1024
        self.token_emb, self.pos_emb = nn.Embedding(vocab_size, embed_dim), nn.Embedding(max_len, embed_dim)
        self.drop = nn.Dropout(dropout)
        # block transformer 16 layers
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, n_heads, ff_dim, dropout) for _ in range(n_blocks)])
        # normalisation  + MLP
        self.ln_f, self.head = nn.LayerNorm(embed_dim, eps=1e-5), nn.Linear(embed_dim, vocab_size, bias=False)
        self.head.weight = self.token_emb.weight
        self._init_weights()
    def _init_weights(self):
        # papier original de GPT2 on initalise selon une loi normale centrée réduite ajustée N(0,0.02²)
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Embedding): nn.init.normal_(m.weight, std=0.02)
    def forward(self, idx):
        # entrée  = Batch * Time
        B, T = idx.shape
        # dropout sur la somme de mes embeddings donc sémantique du mot + position
        x = self.drop(self.token_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device)))
        for block in self.blocks: x = block(x)
        return self.head(self.ln_f(x))

# =============================================================================
# SETUP & REPRISE
# =============================================================================
# créer norte modèle
model = Lyre(VOCAB_SIZE, MAX_LEN, EMBEDDING_DIM, N_HEADS, FEED_FORWARD_DIM, N_BLOCKS, DROPOUT).to(device)
# distributed data parallel pour entrainer sur plusieurs PC gradient dans des paquets de 25Mo
model = DDP(model, device_ids=[local_rank], bucket_cap_mb=25)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
# assure aucun chevauchement entre ordinateurs
loader, sampler = make_dataloader(dataset, BATCH_SIZE_PER_GPU, world_size, rank)
# scheduler avec courbe cosinus 
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: (s/1000 if s < 1000 else 0.5 * (1.0 + math.cos(math.pi * (s-1000)/( (len(loader)*EPOCHS)-1000)))))
loss_fn = nn.CrossEntropyLoss()

start_step = 0
best_loss = float('inf')

# reprise de l'netrainement
if os.path.exists(RESUME_PATH):
    map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
    checkpoint = torch.load(RESUME_PATH, map_location=map_location)
    model.module.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_step = checkpoint['step'] + 1 
    best_loss = checkpoint['loss']
    if is_chief: print(f"[INFO] Resume au step {start_step} (Loss: {best_loss:.4f})")

# =============================================================================
# TRAIN
# =============================================================================
#info
total_steps = len(loader) * EPOCHS
t_start = time.time()
#1
for epoch in range(EPOCHS):
    sampler.set_epoch(epoch)
    model.train()
    
    for step_in_epoch, (x, y) in enumerate(loader):
        # Reprise exacte : on skip les batchs déjà vus
        if step_in_epoch < (start_step % len(loader)):
            continue

        is_sync_step = (step_in_epoch + 1) % ACCUM_STEPS == 0
        context = torch.enable_grad() if is_sync_step else model.no_sync()

        with context:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                with sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True):
                    loss = loss_fn(model(x.to(device, non_blocking=True)).view(-1, VOCAB_SIZE), y.to(device, non_blocking=True).view(-1))
                    loss = loss / ACCUM_STEPS
            loss.backward()
        
        if is_sync_step:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            if is_chief:
                current_total_step = epoch * len(loader) + step_in_epoch
                current_loss_val = loss.item() * ACCUM_STEPS
                
                elapsed = time.time() - t_start
                # Speed basée sur les steps réellement effectués cette session
                steps_done_session = (step_in_epoch - (start_step % len(loader))) + 1
                speed = elapsed / steps_done_session
                eta = (total_steps - current_total_step) * speed
                
                print(f"Step {current_total_step}/{total_steps} | Loss: {current_loss_val:.4f} | Passé: {time.strftime('%H:%M:%S', time.gmtime(elapsed))} | ETA: {time.strftime('%H:%M:%S', time.gmtime(eta))}")

                # SAUVEGARDE CONDITIONNELLE : Intervalle + Meilleure Loss à améliorer pas assezd e sauvegarde en fin d'entrainement
                if (step_in_epoch + 1) % SAVE_EVERY == 0 and current_loss_val < best_loss:
                    best_loss = current_loss_val
                    torch.save({
                        'step': step_in_epoch,
                        'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'loss': best_loss,
                    }, RESUME_PATH)
                    print(f" >> [SAVE] Nouveau best à {best_loss:.4f} au step {step_in_epoch}")

                optimizer.zero_grad(set_to_none=True)

dist.destroy_process_group()