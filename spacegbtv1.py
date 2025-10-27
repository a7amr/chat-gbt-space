import os, glob, json

DATASET_DIR = "/kaggle/input/space"   # <— your dataset name
assert os.path.isdir(DATASET_DIR), f"Not found: {DATASET_DIR}"

files = sorted(glob.glob(os.path.join(DATASET_DIR, "*.jsonl")))
print("Found JSONL files:", len(files))
for f in files[:10]:
    print(" •", os.path.basename(f))


    # Turn JSONL shards into one big text string for char training

def iter_jsonl(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                obj = json.loads(line)
                t = obj.get("text") or ""
                if t:
                    yield t
            except:
                pass

texts = []
for fn in files:
    texts.extend(iter_jsonl(fn))

text = "\n\n".join(texts)
print("Total characters:", len(text), "| docs:", len(texts))
del texts


MERGED_JSONL = "/kaggle/working/space_corpus_merged.jsonl"

with open(MERGED_JSONL, "w", encoding="utf-8") as out:
    for fn in files:
        with open(fn, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                out.write(line)

print("Wrote:", MERGED_JSONL)



MERGED_JSONL = "/kaggle/working/space_corpus_merged.jsonl"



# Step 1: imports & device
import os, json, glob, math, random, time
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(256)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device:", device)



# Step 2: config

# DATA: point to your merged file OR shards. We’ll auto-pick what exists.
MERGED_JSONL = "/kaggle/working/space_corpus_merged.jsonl"   # optional final corpus
SHARD_GLOB   = "/kaggle/working/space_corpus_merged.jsonl"        # shards from the crawler
DATA_MODE    = "letters"                    # char-level as in class

# GPT training hyperparams (start here; you can scale up later)
block_size    = 256      # sequence length (char tokens)
batch_size    = 64       # try 128 if GPU fits
max_iters     = 6000     # training steps
eval_interval = 500
learning_rate = 3e-4
eval_iters    = 300
n_embd        = 512
n_head        = 8
n_layer       = 8
dropout       = 0.2

# Checkpointing
CKPT_DIR      = "ckpts"
os.makedirs(CKPT_DIR, exist_ok=True)
SAVE_EVERY    = 1000     # save every N steps

use_amp       = True     # mixed-precision if cuda
print("config ok")



# Load corpus from the Kaggle dataset: /kaggle/input/space/*.jsonl
import os, glob, json

DATASET_DIR = "/kaggle/input/space"

def iter_jsonl(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                obj = json.loads(line)
                t = obj.get("text") or ""
                if t:
                    yield t
            except:
                pass

files = sorted(glob.glob(os.path.join(DATASET_DIR, "*.jsonl")))
print("Found JSONL files:", len(files))
if not files:
    # Helpful message if nothing is found
    print("Dataset dir contents:", os.listdir(DATASET_DIR))
    raise FileNotFoundError(f"No *.jsonl files found in {DATASET_DIR}")

texts = []
for fn in files:
    texts.extend(iter_jsonl(fn))

text = "\n\n".join(texts)
print("Total characters:", len(text), "| docs:", len(texts))
del texts



# === LOAD CORPUS FROM DATASET ===
# Replaces any previous "MERGED_JSONL/SHARD_GLOB" logic

import os, glob, json

DATASET_DIR = "/kaggle/input/space"   # <- your dataset name shown on the right panel

def iter_jsonl(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                obj = json.loads(line)
                t = obj.get("text") or ""
                if t:
                    yield t
            except:
                pass

# list all .jsonl files in the dataset (handles spaces/parentheses in names)
files = sorted(glob.glob(os.path.join(DATASET_DIR, "*.jsonl")))
print("Found JSONL files:", len(files))
for f in files[:10]:
    print(" •", os.path.basename(f))

if not files:
    # helpful debug
    print("Dataset contents:", os.listdir(DATASET_DIR) if os.path.isdir(DATASET_DIR) else "dataset dir missing")
    raise FileNotFoundError(f"No *.jsonl files found under {DATASET_DIR}")

# build one big text blob (letters approach)
texts = []
for fn in files:
    texts.extend(iter_jsonl(fn))

text = "\n\n".join(texts)
print("Total characters:", len(text), "| docs:", len(texts))
del texts


# Step 4: char vocab + encode/decode + split

the_chars  = sorted(list(set(text)))
vocab_size = len(the_chars)
print("vocab_size:", vocab_size)
print("sample chars:", "".join(the_chars[:120]))

stoi = { ch:i for i, ch in enumerate(the_chars) }
itos = { i:ch for i, ch in enumerate(the_chars) }

encode = lambda s: [stoi[c] for c in s if c in stoi]
decode = lambda ids: ''.join([itos[i] for i in ids])

data = torch.tensor(encode(text), dtype=torch.long)
print("encoded tensor:", data.shape)

n = int(0.98 * len(data))  # big train split, small val
train_data = data[:n]
val_data   = data[n:]

def get_batch(split):
    src = train_data if split == "train" else val_data
    ix = torch.randint(len(src) - block_size, (batch_size,))
    x = torch.stack([src[i:i+block_size] for i in ix])
    y = torch.stack([src[i+1:i+1+block_size] for i in ix])
    return x.to(device), y.to(device)


# Step 5: GPT (char) model

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x); q = self.query(x)
        dk = k.size(-1)
        wei = q @ k.transpose(-2, -1) * (dk ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
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
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)                     # (B,T,C)
        pos_emb = self.pos_emb_table(torch.arange(T, device=idx.device))  # (T,C)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)                                      # (B,T,V)
        loss = None
        if targets is not None:
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B*T, V), targets.view(B*T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=200, temperature=0.8, top_k=None, rep_penalty=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(1e-8, temperature)

            # repetition penalty
            if rep_penalty != 1.0:
                for b in range(idx.shape[0]):
                    for tkn in idx[b].tolist():
                        logits[b, tkn] /= rep_penalty

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = torch.where(logits < v[:, [-1]], torch.full_like(logits, -float('Inf')), logits)

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
        return idx



from contextlib import nullcontext

use_amp = (device == 'cuda')  # only meaningful on GPU

# Autocast context
autocast_ctx = (lambda: torch.amp.autocast(device_type='cuda')) if use_amp else nullcontext

# Grad scaler
scaler = torch.amp.GradScaler(device='cuda') if use_amp else None




# Step 6: init model + eval helper

m = GPTModel().to(device)
optimizer = torch.optim.Adam(m.parameters(), lr=learning_rate)
from contextlib import nullcontext
use_amp = (device == 'cuda')
autocast_ctx = (lambda: torch.amp.autocast(device_type='cuda')) if use_amp else nullcontext
scaler = torch.amp.GradScaler(device='cuda') if use_amp else None


@torch.no_grad()
def estimate_loss():
    out = {}
    m.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split)
            with torch.amp.autocast(device_type='cuda') if use_amp else nullcontext():

                _, loss = m(xb, yb)
            losses[k] = loss.item()
    m.train()
    out['train'] = losses.mean().item()
    out['val']   = losses.mean().item()  # same-size buckets, simple view
    return out




# Step 7: train (checkpoint every SAVE_EVERY steps)

start_step = 0
# To resume, uncomment one of these (adjust filename) then run this cell:
# m.load_state_dict(torch.load("ckpts/spacegpt_step3000.pt", map_location=device)); start_step = 3000
# print("Resumed from step", start_step)

m.train()
t0 = time.time()
for it in range(start_step, max_iters):
    if it % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {it}: train/val loss ~ {losses['train']:.4f}")

    xb, yb = get_batch('train')
    optimizer.zero_grad(set_to_none=True)
    if use_amp:
        with autocast_ctx():
            _, loss = m(xb, yb)
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        _, loss = m(xb, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        optimizer.step()


    if (it+1) % SAVE_EVERY == 0:
        ck = f"{CKPT_DIR}/spacegpt_step{it+1}.pt"
        torch.save(m.state_dict(), ck)
        print(f"Saved checkpoint -> {ck}")

# final save
torch.save(m.state_dict(), "spacegpt_final.pt")
print("Saved final -> spacegpt_final.pt | elapsed min:", int((time.time()-t0)/60))




# Step 8: sampling helpers + 5 demo questions

def sample(prompt, T=0.7, K=60, RP=1.10, N=220):
    seed = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    out_ids = m.generate(seed, max_new_tokens=N, temperature=T, top_k=K, rep_penalty=RP)[0].tolist()
    txt = decode(out_ids)
    # cut drift after two blank lines if any
    return txt.split("\n\n")[0].strip()

def ask_clean(q):
    prompt = f"You are SpaceSystemsGPT.\nQ: {q}\nA:"
    return sample(prompt, T=0.6, K=50, RP=1.08, N=280)

# 5 ready-to-run demo questions (each prints one answer)
print("Q1:", "What is orbital inclination?")
print(ask_clean("What is orbital inclination?")); print()

print("Q2:", "Explain a Hohmann transfer in simple terms.")
print(ask_clean("Explain a Hohmann transfer in simple terms.")); print()

print("Q3:", "What is delta-v and why is it important?")
print(ask_clean("What is delta-v and why is it important?")); print()

print("Q4:", "How does a gravity assist work?")
print(ask_clean("How does a gravity assist work?")); print()

print("Q5:", "Name common Earth orbits and typical uses.")
print(ask_clean("Name common Earth orbits and typical uses.")); print()
