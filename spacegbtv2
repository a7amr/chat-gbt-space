# --- Cell 0: environment & device ---
import os, json, glob, math, random, time, string
import numpy as np
import torch, torch.nn as nn
from torch.nn import functional as F
from contextlib import nullcontext

torch.manual_seed(256)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device:", device, "| torch:", torch.__version__)



# --- Cell 1: config ---

# Where to read JSONL: Kaggle dataset path by default
DATASET_DIR = "/kaggle/input/space2"   # change if local, e.g. "./data/space"
MERGED_JSONL = None                   # optional single merged file; set a path or leave None

# Tokenizer mode: 'char' (letters) or 'bpe' (tiktoken)
TOKEN_MODE = 'char'   # <- switch to 'bpe' to try the second method

# Model + training
block_size    = 256
batch_size    = 64
n_layer       = 8
n_head        = 8
n_embd        = 512
dropout       = 0.2

learning_rate = 3e-4
max_iters     = 6000
eval_interval = 500
eval_iters    = 300
grad_clip     = 1.0

# Checkpoints
CKPT_DIR   = "ckpts"
SAVE_EVERY = 1000
os.makedirs(CKPT_DIR, exist_ok=True)

use_amp = (device == 'cuda')  # mixed precision
print("config ready")



# --- Cell 2: load corpus into a single text string ---

def iter_jsonl(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                obj = json.loads(line)
                t = obj.get("text") or ""
                if t: yield t
            except:
                pass

texts = []
if MERGED_JSONL and os.path.exists(MERGED_JSONL):
    print("Loading merged file:", MERGED_JSONL)
    texts = list(iter_jsonl(MERGED_JSONL))
else:
    assert os.path.isdir(DATASET_DIR), f"Missing DATASET_DIR: {DATASET_DIR}"
    files = sorted(glob.glob(os.path.join(DATASET_DIR, "*.jsonl")))
    print("Found JSONL files:", len(files))
    if not files:
        print("DATASET_DIR contents:", os.listdir(DATASET_DIR))
        raise FileNotFoundError("No *.jsonl found in DATASET_DIR.")
    for fn in files:
        texts.extend(iter_jsonl(fn))

text = "\n\n".join(texts)
print("Total characters:", len(text), "| docs:", len(texts))
del texts


# --- Cell 3: tokenizer + tensors ---
if TOKEN_MODE == 'char':
    the_chars  = sorted(list(set(text)))
    stoi = {ch:i for i,ch in enumerate(the_chars)}
    itos = {i:ch for i,ch in enumerate(the_chars)}
    encode = lambda s: [stoi[c] for c in s if c in stoi]
    decode = lambda ids: ''.join(itos[i] for i in ids)
    vocab_size = len(the_chars)
else:
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    encode = lambda s: enc.encode_ordinary(s)
    decode = lambda ids: enc.decode(ids)
    vocab_size = enc.n_vocab

data = torch.tensor(encode(text), dtype=torch.long)
print("TOKEN_MODE:", TOKEN_MODE, "| vocab_size:", vocab_size, "| data_len:", len(data))

# splits
n = int(0.98 * len(data))
train_data = data[:n]
val_data   = data[n:]

def get_batch(split):
    src = train_data if split == "train" else val_data
    ix = torch.randint(len(src) - block_size, (batch_size,))
    x = torch.stack([src[i:i+block_size] for i in ix])
    y = torch.stack([src[i+1:i+1+block_size] for i in ix])
    return x.to(device), y.to(device)



# --- Cell 4: GPT model ---

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x); q = self.query(x); v = self.value(x)
        dk = k.size(-1)
        wei = (q @ k.transpose(-2, -1)) * (dk ** -0.5)
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_head)])
        self.proj  = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout)
        )
    def forward(self, x): return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa   = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1  = nn.LayerNorm(n_embd)
        self.ln2  = nn.LayerNorm(n_embd)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.pos_emb_table         = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f   = nn.LayerNorm(n_embd)
        self.lm_head= nn.Linear(n_embd, vocab_size)
    def forward(self, idx, targets=None):
        B,T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.pos_emb_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            B,T,V = logits.shape
            loss = F.cross_entropy(logits.view(B*T, V), targets.view(B*T))
        return logits, loss
    @torch.no_grad()
    def generate(self, idx, max_new_tokens=200, temperature=0.7, top_k=None, rep_penalty=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]

            # repetition penalty
            if rep_penalty != 1.0:
                for b in range(idx.shape[0]):
                    for tkn in idx[b].tolist():
                        logits[b, tkn] /= rep_penalty

            # temperature + top-k
            logits = logits / max(temperature, 1e-8)
            if top_k is not None:
                v, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
                thresh = v[:, [-1]]
                logits = torch.where(logits < thresh, torch.full_like(logits, -float('inf')), logits)

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
        return idx



# --- Cell 5: init model, AMP, loss eval ---
m = GPTModel().to(device)
optimizer = torch.optim.Adam(m.parameters(), lr=learning_rate)

amp_ctx = (lambda: torch.amp.autocast(device_type='cuda')) if use_amp else (lambda: nullcontext())
scaler  = torch.amp.GradScaler(device='cuda') if use_amp else None

@torch.no_grad()
def estimate_loss():
    out = {}
    m.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters, device=device)
        for _ in range(eval_iters):
            xb, yb = get_batch(split)
            with amp_ctx():
                _, loss = m(xb, yb)
            losses[_] = loss.item()
        out[split] = losses.mean().item()
    m.train()
    return out




# === Init model, AMP, eval helper, and training loop (fixed) ===
import time
import math
import torch
from contextlib import nullcontext

# assumes: GPTModel, get_batch, learning_rate, max_iters, eval_interval,
# eval_iters, grad_clip, device, CKPT_DIR, SAVE_EVERY are already defined

# 1) Model + optimizer
m = GPTModel().to(device)
optimizer = torch.optim.Adam(m.parameters(), lr=learning_rate)

# 2) AMP setup (new torch.amp API)
use_amp = (device == 'cuda')
amp_ctx = (lambda: torch.amp.autocast(device_type='cuda')) if use_amp else (lambda: nullcontext())
scaler  = torch.amp.GradScaler(device='cuda') if use_amp else None

# 3) Evaluation helper (fixed indexing; no "_" reuse)
@torch.no_grad()
def estimate_loss():
    out = {}
    m.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)  # CPU tensor is fine for averaging
        for i in range(eval_iters):
            xb, yb = get_batch(split)
            with amp_ctx():
                logits, loss = m(xb, yb)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    m.train()
    return out

# 4) (Optional) resume
start_step = 0
# To resume, uncomment and set the step:
# ckpt_path = f"{CKPT_DIR}/spacegpt_step3000.pt"
# m.load_state_dict(torch.load(ckpt_path, map_location=device))
# start_step = 3000

# 5) Train loop
t0 = time.time()
m.train()
for it in range(start_step, max_iters):
    # periodic eval
    if it % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {it}: train/val ~ {losses['train']:.4f} / {losses['val']:.4f}")

    xb, yb = get_batch('train')
    optimizer.zero_grad(set_to_none=True)

    if use_amp:
        with amp_ctx():
            logits, loss = m(xb, yb)
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        logits, loss = m(xb, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), grad_clip)
        optimizer.step()

    # checkpoint
    if (it + 1) % SAVE_EVERY == 0:
        ck = f"{CKPT_DIR}/spacegpt_step{it+1}.pt"
        torch.save(m.state_dict(), ck)
        print("Saved:", ck)

# final save
torch.save(m.state_dict(), "spacegpt_final.pt")
print("Saved final -> spacegpt_final.pt | minutes:", int((time.time() - t0) / 60))





# === Demo Q&A: same five questions (clean sampler, works for char or bpe) ===
import string
import torch
import torch.nn.functional as F

# Build a light ASCII whitelist mask for CHAR mode to avoid weird Unicode
if 'TOKEN_MODE' in globals() and TOKEN_MODE == 'char':
    safe_chars = set(string.ascii_letters + string.digits + " .,:;!?'-()/\n")
    # itos must exist when TOKEN_MODE='char'
    safe_ids   = {i for i,ch in itos.items() if ch in safe_chars}
    char_mask  = torch.zeros(vocab_size, device=device)
    if len(safe_ids) < vocab_size:
        bad = [i for i in range(vocab_size) if i not in safe_ids]
        if bad:
            char_mask[bad] = -1e9
else:
    # BPE mode (or if TOKEN_MODE not set): no mask needed
    char_mask = torch.zeros(vocab_size, device=device)

@torch.no_grad()
def generate_clean_ids(seed_ids, max_new_tokens=240, temperature=0.45, top_k=60, rep_penalty=1.15):
    """Conservative sampling with optional repetition penalty and char whitelist."""
    m.eval()
    out = seed_ids
    for _ in range(max_new_tokens):
        idx_cond = out[:, -block_size:]
        logits, _ = m(idx_cond)
        logits = logits[:, -1, :] + char_mask

        # repetition penalty (soft)
        if rep_penalty != 1.0:
            for b in range(out.size(0)):
                for tkn in out[b].tolist():
                    logits[b, tkn] /= rep_penalty

        # temperature + top-k
        logits = logits / max(temperature, 1e-8)
        if top_k is not None:
            v, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
            thresh = v[:, [-1]]
            logits = torch.where(logits < thresh, torch.full_like(logits, -float('inf')), logits)

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        out = torch.cat([out, next_id], dim=1)

        # early stop for CHAR mode on blank line
        if 'TOKEN_MODE' in globals() and TOKEN_MODE == 'char':
            nl = stoi.get('\n', -1)
            if out.size(1) >= 2 and out[0, -2:].tolist() == [nl, nl]:
                break
    return out

def demo_answer(question, greedy=False):
    primer = "You are SpaceSystemsGPT.\nRules: Answer in 2–4 complete sentences. No lists, no links.\n"
    prompt = f"{primer}Q: {question}\nA:"
    T  = 0.25 if greedy else 0.45
    K  = 40   if greedy else 60
    RP = 1.20 if greedy else 1.15
    seed = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    ids  = generate_clean_ids(seed, max_new_tokens=240, temperature=T, top_k=K, rep_penalty=RP)[0].tolist()
    txt  = decode(ids)
    # Cut at first blank line to avoid drift
    return txt.split("\n\n")[0].strip()

# ---- The same five questions ----
questions = [
    "What is orbital inclination?",
    "Explain a Hohmann transfer in simple terms.",
    "What is delta-v and why is it important?",
    "How does a gravity assist work?",
    "Name common Earth orbits and typical uses."
]

for i, q in enumerate(questions, 1):
    print(f"Q{i}: {q}\n{demo_answer(q)}\n")
