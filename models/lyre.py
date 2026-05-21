import os
import math
import time
import sys, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.utils.checkpoint import checkpoint
from torch.distributed.algorithms.ddp_comm_hooks import powerSGD_hook as powerSGD

from archi import RMSNorm, SwiGLU, precompute_rope, apply_rope, TransformerBlock

print("DEBUT DU SCRIPT 1", flush=True)
sys.stdout.flush()

"""
L'objectif principale pour la seconde version de Lyre et quelle ait une meilleure perplexité pour éviter avoir des phrases plus cohérentes. d'après mes recerches rajouter des parametres n'est clairement pas suffisant( il faudrait en rajouter beaucoup ) soit dit en passant on peut augmenter la quantité de donnée sur laquelle est entrainée l'IA on passe donc de 14 à 20Go de données on peut aussi penser à soit faire une autre epoch soit entrainée sur en plus du opensubttles + une partie de CulturaX que je n'ai pas telechargé en français.
le second objectif est d'être plus souple sur la window size ce que le mécanisme RoPE permet ce qui permettre d'améliorer l'efficatcité du RAG (réussi dans la mesure ou on a RoPE)
Enfin il faut améliorer l'efficactité du transfert entre les machines (POwerSGD) et faire baisser la RAM pour baisser ACCUM_STEPS et augmenter BatchSize. (réussi mois de 1s on tente de baisser le ACCUM_STEPS à . !!)

Listes des améliorations pour la seconde version de Lyre :
    - augmentation data 14Go -> 19.2
    - RoPE pour remplacer pos_emb d'autant plus important pour du RAG qui demande une window size assez élevée pendant à l'inférence (fait)
    - RMSNorm remplacer à partir de GPT3 il me semble (fait)
    - SwiGLU (fait)
    - Gradient checkpointing (à faire pour baisser la VRAM)
    - PowerSGD (fait)
    -> nouveau problème de RAM pas envie de faire baisser le batch size solution
        - implémenter GQA pour partager les têtes d'attention

Il est clait que le principal avantage de cette IA est d'être en capacité de faire des phrases correctes cependant elle ne pourra pas comprendre des questions compliquées. Son utilité reposera donc principalement sur
    - Un bon fine tuning pour lui faire faire excatement ce que l'on veut 
    - Un très bon système de RAG qui réussit à récupérer les bonnes infortmations sur Internet/la discussion/info internes Ce RAG devra donc reposer sur une couche embedding assez puissante.
"""
# FIX SYSTEME
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

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
FEED_FORWARD_DIM = 2752
DROPOUT          = 0.1
# 14-20 Go is sufficient I don't want the model to overfit. However cuz there is a lot of data we can maybe try a second epoch to slightly increase the results
EPOCHS           = 1
# pour le grouped Query Attention
N_KV_HEADS = 4
"""
dans version 2 ça devient important le seul moment ou les GPU ne travail pas c'est durant la période de warmup du sgd et du save donc on essaie de faire ça en local pour le rendre indtant au lieu d'attendre 20s en passant par le réseau

à absolument faire sur les trois machines 
mkdir -p /tmp/checkpoint

cp ./checkpoint/latest_best.pt /tmp/checkpoint/latest_best.pt
cp corpus_encoded.bin /tmp/corpus_encoded.bin

à absolument faire à la fin de la session d'entrianement

cp /tmp/checkpoint/latest_best.pt checkpoint/latest_best.pt

torchrun --nproc_per_node=1 --nnodes=3 --node_rank=0 --master_addr=10.0.104.4 --master_port=29505 models/lyre.py


"""
BIN_FILE         = "ressources/corpus_encoded.bin"
MODEL_SAVE_DIR   = "checkpoint/"
# memory issues ,20go for 8 by gpus is too small
BATCH_SIZE_PER_GPU = 16
# by increasing BATCH_SIZE_PER_GPU we should decrease it to prevent the models to diverge
ACCUM_STEPS      = 32 

SAVE_EVERY       = 100 * ACCUM_STEPS 
RESUME_PATH      = os.path.join(MODEL_SAVE_DIR, "latest_best2.pt")

CHECKPOINT_PATH      = os.path.join(MODEL_SAVE_DIR, "checks.pt")

# =============================================================================
# INIT DDP (NCCL)
# =============================================================================

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

#19Go, 1024 , 32768
dataset = CorpusDataset(BIN_FILE, MAX_LEN, VOCAB_SIZE)
dist.barrier(device_ids=[local_rank])

print("DEBUT DU SCRIPT 1", flush=True)
sys.stdout.flush()

# =============================================================================
# MODÈLE
# =============================================================================

# casual masque les mots futurs (génération de texte)
class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, n_heads,n_kv_heads=4, dropout=0.1):
        super().__init__()
        self.n_kv_heads = n_kv_heads
        #head dim  = 1024/16 =  64
        self.n_heads, self.head_dim = n_heads, embed_dim // n_heads
        self.n_rep      = n_heads // n_kv_heads
        # (anvienne version) au lieu de faire trois lignes pour Query, Key, Value :  on fait entrer un tenseur de taille embed dim on sort trois embed dim
        # (nouvelle QVA) q d'un coté kv de l'autre 
        # le proj concatène les résultats des différentes couches d'attention
        # dropout basique, régularisation simple en 0.1
        self.q_proj  = nn.Linear(embed_dim, embed_dim, bias=False)
        self.kv_proj = nn.Linear(embed_dim, 2 * n_kv_heads * self.head_dim, bias=False)
        self.proj    = nn.Linear(embed_dim, embed_dim)
        self.drop    = nn.Dropout(dropout)
    def forward(self, x, cos, sin):
        # tenseur d'entrée x , divisée par son B(atch size) T(ime) =1024 C(hannel 1024)
        B, T, C = x.shape
        # créer des vues différentes pour Query, Key, View
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(x).view(B, T, 2, self.n_kv_heads, self.head_dim)
        k, v = kv.unbind(2)
        # view divise c en deux puis transpose B ,n_heads ,T, head_dim
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        #RoPE
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)

        # calcul en parallèle on multiplie la matrice q par la transposée k / on applique le  masque triangulaire inférieure
        ## Softmax ( Q.k^{T}/sqrt(d_k) + M) * V ou M est le masque causal et d_k = head dim = embed_dim//n_head  = 64
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.drop.p if self.training else 0.0,
            is_causal=True
        )
        return self.proj(out.transpose(1, 2).contiguous().view(B, T, C))

class Lyre(nn.Module):
    def __init__(self, vocab_size, max_len, embed_dim, n_heads, ff_dim, n_blocks, n_kv_heads, dropout=0.1, use_gradient_checkpointing=True):
        super().__init__()
        # token_emb : 32768 * 1024 pos_emb : remplacé par RoPE
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.drop = nn.Dropout(dropout)
        # block transformer 16 layers
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, n_heads, ff_dim, n_kv_heads, dropout,CausalSelfAttention) for _ in range(n_blocks)])
        # normalisation  + MLP
        self.ln_f, self.head = RMSNorm(embed_dim), nn.Linear(embed_dim, vocab_size, bias=False)
        self.head.weight = self.token_emb.weight
        #def cos et sin
        cos, sin = precompute_rope(embed_dim // n_heads, max_len)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)
        #gain VRAM
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self._init_weights()
    def _init_weights(self):
        # papier original de GPT2 on initalise selon une loi normale centrée réduite ajustée N(0,0.02²)
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Embedding): nn.init.normal_(m.weight, std=0.02)
    def forward(self, idx):
        # entrée  = Batch * Time
        B, T = idx.shape
        # dropout sur la somme de mes embeddings donc sémantique du mot + position
        x = self.drop(self.token_emb(idx))
        for block in self.blocks:
            if self.use_gradient_checkpointing and self.training:
                x = checkpoint(block, x, self.cos, self.sin, use_reentrant=False)
            else:
                x = block(x, self.cos, self.sin)

        return self.head(self.ln_f(x))

# =============================================================================
# SETUP & REPRISE
# =============================================================================
print("DEBUT DU SCRIPT", flush=True)
sys.stdout.flush()

# créer norte modèle
model = Lyre(
    VOCAB_SIZE, MAX_LEN, EMBEDDING_DIM, N_HEADS,
    FEED_FORWARD_DIM, N_BLOCKS,N_KV_HEADS, DROPOUT,
    use_gradient_checkpointing=True
).to(device)

if is_chief:
    total = sum(p.numel() for p in model.parameters())
    print(f"[ARCH] Paramètres: {total/1e6:.1f}M")
    print(f"[ARCH] Layers: {N_BLOCKS} | Heads: {N_HEADS} | Embed: {EMBEDDING_DIM} | FF: {FEED_FORWARD_DIM}")

# distributed data parallel pour entrainer sur plusieurs PC gradient dans des paquets de 50Mo pour que PowerSgd soit efficace
#on fait un *2 donc on risque de mieux converger...
model = DDP(model, device_ids=[local_rank], bucket_cap_mb=50)

powersgd_state = powerSGD.PowerSGDState(
    process_group=dist.group.WORLD,
    matrix_approximation_rank=4,
    warm_start=True,
    use_error_feedback=False,
    start_powerSGD_iter=10,
    min_compression_rate=2.0,
)

powersgd_state.error_dict = {}
model.register_comm_hook(
    state=powersgd_state,
    hook=powerSGD.powerSGD_hook,
)


optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
# assure aucun chevauchement entre ordinateurs
loader, sampler = make_dataloader(dataset, BATCH_SIZE_PER_GPU, world_size, rank)
# scheduler avec courbe cosinus 
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lambda s: (
        s / 1000 if s < 1000
        else 0.5 * (1.0 + math.cos(
            math.pi * (s - 1000) / ((len(loader) * EPOCHS) - 1000)
        ))
    )
)

loss_fn = nn.CrossEntropyLoss()

start_step = 0
best_loss = float('inf')

# reprise de l'entrainement
if os.path.exists(RESUME_PATH):
    map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
    ckpt = torch.load(RESUME_PATH, map_location=map_location)
    
    model.module.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    start_step = ckpt['step'] + 1
    best_loss  = ckpt['loss']

    # PowerSGD
    if 'powersgd_iter' in ckpt:
        powersgd_state.iter = ckpt['powersgd_iter']
        powersgd_state.error_dict = {}

    if is_chief:
        print(f"[INFO] Resume au step {start_step} (Loss: {best_loss:.4f})")

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
        global_step = epoch * len(loader) + step_in_epoch
        if global_step < start_step:
            continue

        is_sync_step = (step_in_epoch + 1) % ACCUM_STEPS == 0
        context = torch.enable_grad() if is_sync_step else model.no_sync()

        with context:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):                    
                    logits = model(x.to(device, non_blocking=True))
                    loss = loss_fn(logits.view(-1, VOCAB_SIZE), y.to(device, non_blocking=True).view(-1))
                    loss = loss / ACCUM_STEPS
            loss.backward()
        
        if is_sync_step:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            
            if is_chief:
                current_total_step = epoch * len(loader) + step_in_epoch
                current_loss_val = loss.item() * ACCUM_STEPS
                
                elapsed = time.time() - t_start
                steps_done_session = current_total_step - start_step + 1  # steps réels cette session
                speed = elapsed / steps_done_session if steps_done_session > 0 else 0
                remaining_steps = total_steps - current_total_step
                eta = remaining_steps * speed

                print(
                    f"Step {current_total_step}/{total_steps} | "
                    f"Loss: {current_loss_val:.4f} | "
                    f"Passé: {time.strftime('%H:%M:%S', time.gmtime(elapsed))} | "
                    f"ETA: {time.strftime('%H:%M:%S', time.gmtime(eta))}"
                )

                # SAUVEGARDE CONDITIONNELLE : Intervalle + Meilleure Loss à améliorer pas assez de sauvegarde en fin d'entrainement
                if current_loss_val < best_loss:
                    best_loss = current_loss_val
                    torch.save({
                        'step': step_in_epoch,
                        'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'loss': best_loss,
                        'powersgd_iter': powersgd_state.iter,
                    }, RESUME_PATH)
                    print(f" >> [SAVE] Nouveau best à {best_loss:.4f} au step {step_in_epoch}")


                if (step_in_epoch + 1) % SAVE_EVERY== 0 :
                        current_loss = current_loss_val
                        torch.save({
                            'step': step_in_epoch,
                            'model_state_dict': model.module.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'loss': current_loss,
                            'powersgd_iter': powersgd_state.iter,
                        }, CHECKPOINT_PATH)
                        print(f" >> [SAVE] Checkpoint {current_loss:.4f} at {step_in_epoch}")
dist.destroy_process_group()