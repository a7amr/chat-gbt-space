import os, torch

# allocator: avoids fragmentation on T4
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# enable SDPA kernels (Flash/efficient) for attention
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.enable_flash_sdp(True)         # PyTorch >=2.1
torch.backends.cuda.enable_mem_efficient_sdp(True)  # fallback if no flash
torch.backends.cuda.enable_math_sdp(False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device:", device)


# === training/data config ===
TOKEN_MODE   = "bpe"      # 'bpe' (tiktoken) or 'char'
EXTRA_GLOBS  = []         # e.g. ["/kaggle/input/space3/*.jsonl"] if you want to target a specific dataset

# Fit-on-T4 defaults (OOM-safe)
block_size   = 128        # context length
batch_size   = 16         # tokens per step per GPU
grad_accum_steps = 3      # effective batch = 32 * 2 = 64

print("Config -> block_size:", block_size, "| batch_size:", batch_size, "| grad_accum_steps:", grad_accum_steps)


# ==== UNIVERSAL DATA PIPELINE (Kaggle auto-discovery) ====
import os, glob, json, random, math, torch

FALLBACK_TXT = "space_corpus_final.txt"  # if you uploaded a single txt

def iter_jsonl(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                obj = json.loads(line)
                t = obj.get("text") or ""
                if t: 
                    yield t
            except:
                continue

def find_jsonl_files():
    patterns = [
        "/kaggle/input/*/*.jsonl",
        "/kaggle/input/*/*/*.jsonl",
        "/kaggle/working/*.jsonl",
        "space_sync_*.jsonl",
    ] + list(EXTRA_GLOBS)
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    return sorted(set(files))

raw_text = None
jsonl_files = find_jsonl_files()

if jsonl_files:
    print("Found JSONL shards:", len(jsonl_files))
    total_lines = 0
    texts = []
    for fn in jsonl_files:
        cnt = 0
        for t in iter_jsonl(fn):
            texts.append(t); cnt += 1
        total_lines += cnt
        print(f"  + {os.path.basename(fn)} | {cnt:,} lines")
    raw_text = "\n\n".join(texts); del texts
    print("Total lines:", f"{total_lines:,}")
elif os.path.exists(FALLBACK_TXT):
    print("Loading TXT:", FALLBACK_TXT)
    raw_text = open(FALLBACK_TXT, "r", encoding="utf-8", errors="ignore").read()
else:
    raise FileNotFoundError(
        "No corpus found. I looked for *.jsonl in /kaggle/input/** and for "
        f"{FALLBACK_TXT} in working dir. Add your dataset on the right (Input) "
        "or set EXTRA_GLOBS to match your filenames."
    )

print("Raw chars:", len(raw_text))

# ---- tokenizer: BPE (tiktoken) or char ----
if TOKEN_MODE == "bpe":
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    def encode(s): return enc.encode_ordinary(s)
    def decode(ids): return enc.decode(ids)
    vocab_size = enc.n_vocab
    print("Tokenizer: BPE (tiktoken) | vocab_size:", vocab_size)
else:
    the_chars = sorted(list(set(raw_text)))
    stoi = {ch:i for i,ch in enumerate(the_chars)}
    itos = {i:ch for i,ch in enumerate(the_chars)}
    def encode(s): return [stoi[c] for c in s if c in stoi]
    def decode(ids): return "".join(itos[i] for i in ids)
    vocab_size = len(the_chars)
    print("Tokenizer: CHAR | vocab_size:", vocab_size)

# ---- tokenize to tensor ----
ids = encode(raw_text); del raw_text
data = torch.tensor(ids, dtype=torch.long)
print("Total tokens:", len(data))

# ---- split ----
n = int(0.98 * len(data))
train_data = data[:n]
val_data   = data[n:]
print(f"Split -> train: {len(train_data):,} | val: {len(val_data):,}")

# ---- batching ----
def get_batch(split: str):
    src = train_data if split == 'train' else val_data
    ix = torch.randint(0, len(src) - block_size - 1, (batch_size,))
    x = torch.stack([src[i:i+block_size] for i in ix])
    y = torch.stack([src[i+1:i+1+block_size] for i in ix])
    return x.to(device), y.to(device)

print("Data pipeline ready. Example batch shapes:", tuple(t.shape for t in get_batch('train')))



import math, time, os, torch, torch.nn as nn, torch.nn.functional as F
from contextlib import nullcontext

# ---- model size tuned for T4 ----
n_layer = 5
n_head  = 5
n_embd  = 320
dropout = 0.15
use_ckpt = True  # activation checkpointing to save memory

# ---- optim/training ----
learning_rate = 3e-4
weight_decay  = 0.1
max_iters     = 8000
eval_interval = 500
eval_iters    = 200
grad_clip     = 1.0

SAVE_EVERY = 1000
CKPT_DIR   = "ckpts"
os.makedirs(CKPT_DIR, exist_ok=True)

# ---- blocks ----
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))
    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

def _build_rope_cache(head_dim, seq_len, device, base=10_000):
    idx = torch.arange(0, head_dim, 2, device=device).float()
    inv_freq = 1.0 / (base ** (idx / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.einsum('t,f->tf', t, inv_freq)
    cos = torch.cos(freqs)[None, :, None, :]
    sin = torch.sin(freqs)[None, :, None, :]
    return cos, sin

def _apply_rope(q, k, cos, sin):
    q1, q2 = q[..., ::2], q[..., 1::2]
    k1, k2 = k[..., ::2], k[..., 1::2]
    qc = q1 * cos - q2 * sin; qs = q1 * sin + q2 * cos
    kc = k1 * cos - k2 * sin; ks = k1 * sin + k2 * cos
    q_rot = torch.stack([qc, qs], dim=-1).flatten(-2)
    k_rot = torch.stack([kc, ks], dim=-1).flatten(-2)
    return q_rot, k_rot

class MHA_RoPE(nn.Module):
    def __init__(self, n_embd, n_head, dropout, block_size):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head  = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3*n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).view(1,1,block_size,block_size), persistent=False)
        self.rope_cos = self.rope_sin = None
        self.block_size = block_size
    def _maybe_rope(self, T, dev):
        if (self.rope_cos is None) or (self.rope_cos.size(1) < T):
            self.rope_cos, self.rope_sin = _build_rope_cache(self.head_dim, max(T, self.block_size), dev)
    def forward(self, x):
        B,T,C = x.shape
        q,k,v = self.qkv(x).split(C, dim=-1)
        q = q.view(B,T,self.n_head,self.head_dim)
        k = k.view(B,T,self.n_head,self.head_dim)
        v = v.view(B,T,self.n_head,self.head_dim)

        self._maybe_rope(T, x.device)
        cos, sin = self.rope_cos[:, :T], self.rope_sin[:, :T]
        q,k = _apply_rope(q,k,cos,sin)

        q = q.transpose(1,2); k = k.transpose(1,2); v = v.transpose(1,2)
        att = (q @ k.transpose(-2,-1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:,:,:T,:T]==0, float('-inf'))
        att = self.attn_drop(F.softmax(att, dim=-1))
        y = att @ v
        y = y.transpose(1,2).contiguous().view(B,T,C)
        return self.proj_drop(self.proj(y))

class SwiGLU(nn.Module):
    def __init__(self, d_model, mult=4):
        super().__init__()
        inner = int(mult*d_model)
        self.w1 = nn.Linear(d_model, inner, bias=False)
        self.w2 = nn.Linear(d_model, inner, bias=False)
        self.w3 = nn.Linear(inner, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        return self.drop(self.w3(F.silu(self.w1(x)) * self.w2(x)))

class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout, block_size):
        super().__init__()
        self.n1 = RMSNorm(n_embd)
        self.attn = MHA_RoPE(n_embd, n_head, dropout, block_size)
        self.n2 = RMSNorm(n_embd)
        self.mlp = SwiGLU(n_embd, mult=4)
    def forward(self, x):  # single-tensor signature (for checkpointing)
        x = x + self.attn(self.n1(x))
        x = x + self.mlp(self.n2(x))
        return x

class GPTMiniRoPE(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, dropout, block_size):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(n_embd,n_head,dropout,block_size) for _ in range(n_layer)])
        self.nf = RMSNorm(n_embd)
        self.lm = nn.Linear(n_embd, vocab_size, bias=False)
        self.apply(self._init)
    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)
    def forward(self, idx, targets=None):
        from torch.utils.checkpoint import checkpoint
        x = self.drop(self.tok(idx))
        for blk in self.blocks:
            if self.training and use_ckpt:
                x = checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        logits = self.lm(self.nf(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
    @torch.no_grad()
    def generate(self, idx, max_new_tokens=220, temperature=0.4, top_k=60):
        self.eval()
        for _ in range(max_new_tokens):
            logits,_ = self(idx[:, -block_size:])
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k:
                v,_ = torch.topk(logits, k=min(top_k, logits.size(-1)))
                logits = torch.where(logits < v[:,[-1]], torch.full_like(logits, -float('inf')), logits)
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx

m = GPTMiniRoPE(vocab_size, n_embd, n_head, n_layer, dropout, block_size).to(device)
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate, weight_decay=weight_decay)

use_amp = (device == 'cuda')
amp_ctx = (lambda: torch.amp.autocast(device_type='cuda')) if use_amp else (lambda: nullcontext())
scaler  = torch.amp.GradScaler('cuda') if use_amp else None

@torch.no_grad()
def estimate_loss():
    m.eval()
    vals = []
    for _ in range(eval_iters):
        xb,yb = get_batch('val')
        with amp_ctx():
            _, loss = m(xb,yb)
        vals.append(loss.item())
    m.train()
    return sum(vals)/len(vals)

print("Model params:", sum(p.numel() for p in m.parameters())/1e6, "M")
print("Config OK:", block_size, batch_size, grad_accum_steps, n_layer, n_head, n_embd)

import time
t0 = time.time()
best_val = float('inf')

RESUME = None  # e.g., "ckpts/space_rope_step3000.pt"
if RESUME and os.path.exists(RESUME):
    m.load_state_dict(torch.load(RESUME, map_location=device))
    print("Resumed:", RESUME)

for it in range(1, max_iters+1):
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0

    for _ in range(grad_accum_steps):
        xb, yb = get_batch('train')
        if use_amp:
            with amp_ctx():
                _, loss = m(xb,yb)
                loss = loss / grad_accum_steps
            scaler.scale(loss).backward()
        else:
            _, loss = m(xb,yb)
            loss = loss / grad_accum_steps
            loss.backward()
        total_loss += loss.item()

    # clip & step
    if use_amp:
        torch.nn.utils.clip_grad_norm_(m.parameters(), grad_clip)
        scaler.step(optimizer); scaler.update()
    else:
        torch.nn.utils.clip_grad_norm_(m.parameters(), grad_clip)
        optimizer.step()

    if it % 50 == 0:
        print(f"step {it:5d} | train {total_loss:.4f}")

    if it % eval_interval == 0:
        val = estimate_loss()
        print(f"step {it:5d} | train {total_loss:.4f} | val {val:.4f} | elapsed {int((time.time()-t0)/60)}m")
        if val < best_val:
            best_val = val
            path = f"{CKPT_DIR}/space_rope_best.pt"
            torch.save(m.state_dict(), path)
            print("  ↳ saved best:", path)
        if device == 'cuda':
            torch.cuda.empty_cache()

    if it % SAVE_EVERY == 0:
        path = f"{CKPT_DIR}/space_rope_step{it}.pt"
        torch.save(m.state_dict(), path)
        print("  ↳ checkpoint:", path)

torch.save(m.state_dict(), "space_rope_final.pt")
print("done. total minutes:", int((time.time()-t0)/60))


# Train
t0 = time.time()
best_val = float('inf')

# resume? set RESUME = path or None
RESUME = None  # e.g., "ckpts/space_rope_step3000.pt"
if RESUME and os.path.exists(RESUME):
    m.load_state_dict(torch.load(RESUME, map_location=device))
    print("Resumed:", RESUME)

for it in range(1, max_iters+1):
    xb,yb = get_batch('train')

    optimizer.zero_grad(set_to_none=True)
    if use_amp:
        with amp_ctx(): _, loss = m(xb,yb)
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), grad_clip)
        scaler.step(optimizer); scaler.update()
    else:
        _, loss = m(xb,yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), grad_clip)
        optimizer.step()

    if it % eval_interval == 0:
        val = estimate_loss()
        print(f"step {it:5d} | train {loss.item():.4f} | val {val:.4f} | elapsed {int((time.time()-t0)/60)}m")
        if val < best_val:
            best_val = val
            path = f"{CKPT_DIR}/space_rope_best.pt"
            torch.save(m.state_dict(), path)
            print("  ↳ saved best:", path)

    if it % SAVE_EVERY == 0:
        path = f"{CKPT_DIR}/space_rope_step{it}.pt"
        torch.save(m.state_dict(), path)
        print("  ↳ checkpoint:", path)

torch.save(m.state_dict(), "space_rope_final.pt")
print("done. total minutes:", int((time.time()-t0)/60))





@torch.no_grad()
def ask_greedy(q):
    primer = "You are SpaceSystemsGPT.\nRules: Answer in 2–4 complete sentences. No lists, no links.\n"
    prompt = f"{primer}Q: {q}\nA:"
    seed = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    out = m.generate(seed, max_new_tokens=220, temperature=0.25, top_k=50)[0].tolist()
    txt = decode(out)
    a = txt.find("A:")
    if a != -1: txt = txt[a+2:]
    return " ".join(txt.split()).strip()

questions = [
    "What is orbital inclination?",
    "Explain a Hohmann transfer in simple terms.",
    "What is delta-v and why is it important?",
    "How does a gravity assist work?",
    "Name common Earth orbits and typical uses.",
]
for i,q in enumerate(questions,1):
    print(f"Q{i}: {q}\nA: {ask_greedy(q)}\n")
