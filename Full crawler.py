!pip -q install aiohttp aiofiles beautifulsoup4 trafilatura lxml ftfy tqdm nest_asyncio

# ===== Config =====
OUT_BASENAME   = "space_clean"              # output prefix -> will create space_clean_000.jsonl, _001.jsonl, ...
SHARD_BYTES    = 100 * 1024 * 1024          # rotate every ~100MB per file
TARGET_BYTES   = 350 * 1024 * 1024          # stop when total written >= target
USER_AGENT     = "SpaceSystemsGPT/1.0 (+edu; contact: student@example.edu)"
TIMEOUT        = 25
CONCURRENCY    = 12                          # polite but parallel
POLITE_DELAY   = (0.15, 0.6)                 # jitter per request
MIN_CHARS      = 900                         # drop very short pages
MAX_PAGES      = 200000                      # safety cap
ASCII_RATIO_MIN = 0.35                       # filter non-English-ish

# Allowlist: EDIT to your topic
ALLOWED_DOMAINS = {
    "wikipedia.org", "nasa.gov", "esa.int", "noaa.gov",
    "jpl.nasa.gov", "caltech.edu", "space.com", "nssdc.gsfc.nasa.gov",
    "gsfc.nasa.gov", "science.nasa.gov", "earthdata.nasa.gov",
    "solarsystem.nasa.gov", "mars.nasa.gov", "webb.nasa.gov"
}

# Hard blacklist (domains): social/media/tracking
BLACKLIST_DOMAINS = {
    "twitter.com", "x.com", "facebook.com", "instagram.com", "tiktok.com",
    "youtube.com", "youtu.be", "reddit.com", "linkedin.com", "medium.com",
    "pinterest.com", "snapchat.com", "threads.net", "discord.com"
}

# File extensions to skip
DISALLOWED_EXTS = {
    ".pdf",".jpg",".jpeg",".png",".gif",".svg",".webp",".webm",".mp4",".mp3",
    ".zip",".gz",".tar",".ppt",".pptx",".doc",".docx",".xls",".xlsx",".apk"
}

# Seeds (start pages). Add more deep links if you want faster expansion.
SEEDS = [
    "https://science.nasa.gov/solar-system/",
    "https://science.nasa.gov/mission/",
    "https://solarsystem.nasa.gov/basics/",
    "https://www.esa.int/Science_Exploration",
    "https://en.wikipedia.org/wiki/Orbital_inclination",
    "https://en.wikipedia.org/wiki/Orbital_elements",
    "https://en.wikipedia.org/wiki/Delta-v",
    "https://en.wikipedia.org/wiki/Hohmann_transfer_orbit",
    "https://en.wikipedia.org/wiki/Low_Earth_orbit",
    "https://nssdc.gsfc.nasa.gov/planetary/planetfact.html",
    "https://jpl.nasa.gov/missions"
]

# ===== Helpers =====
import os, re, json, random, hashlib, asyncio, nest_asyncio, aiohttp, aiofiles
from urllib.parse import urljoin, urldefrag, urlparse
from bs4 import BeautifulSoup
import trafilatura, ftfy
from tqdm import tqdm

nest_asyncio.apply()

SOCIAL_RE = re.compile(r"(instagram|youtube|twitter|x\.com|facebook|tiktok|subscribe|follow|@[\w_]+)", re.I)
WHITESPACE_RE = re.compile(r"[ \t]+")

def normalize_text(txt: str) -> str:
    txt = ftfy.fix_text(txt).replace("\ufeff"," ")
    txt = WHITESPACE_RE.sub(" ", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt.strip()

def is_englishish(txt: str) -> bool:
    letters = sum(ch.isalpha() for ch in txt)
    return (letters / max(1, len(txt))) >= ASCII_RATIO_MIN

def remove_social_lines(txt: str) -> str:
    lines = []
    for ln in txt.splitlines():
        s = ln.strip()
        if not s: 
            lines.append("")
            continue
        if SOCIAL_RE.search(s): 
            continue
        if s.startswith(("•","-","@","#")) and SOCIAL_RE.search(s):
            continue
        lines.append(s)
    out = "\n".join(lines)
    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    return out

def hash_text(txt: str) -> str:
    return hashlib.md5(txt[:5000].encode("utf-8","ignore")).hexdigest()

def looks_html(ct: str) -> bool:
    return ct and ("text/html" in ct or "application/xhtml+xml" in ct)

def allowed_url(url: str) -> bool:
    u = urlparse(url)
    host = u.netloc.lower()
    if any(b in host for b in BLACKLIST_DOMAINS):
        return False
    if not any(a in host for a in ALLOWED_DOMAINS):
        return False
    path = u.path.lower()
    if any(path.endswith(ext) for ext in DISALLOWED_EXTS):
        return False
    return True

def extract_links(base: str, html: str) -> set:
    soup = BeautifulSoup(html, "lxml")
    out = set()
    for a in soup.select("a[href]"):
        href = a.get("href","")
        href = urljoin(base, href)
        href, _ = urldefrag(href)
        if allowed_url(href):
            out.add(href)
    return out

def html_to_text(html: str) -> str:
    try:
        txt = trafilatura.extract(html) or ""
    except Exception:
        txt = ""
    if not txt:
        soup = BeautifulSoup(html, "lxml")
        for s in soup(["script","style","noscript","header","footer","nav","aside","form","iframe"]):
            s.decompose()
        txt = soup.get_text(" ")
    txt = normalize_text(txt)
    txt = remove_social_lines(txt)
    return txt




# === Space/Science corpus crawler (resume-safe) ===
# Paste this cell and run. It writes shards: space_clean_000.jsonl, _001.jsonl, ...
# You can stop/restart anytime; it will resume without losing progress.

import os, re, json, random, asyncio, hashlib, nest_asyncio
from collections import deque, defaultdict
from urllib.parse import urljoin, urldefrag, urlparse

nest_asyncio.apply()

# --- config ---
OUT_BASENAME     = "space_clean"
SHARD_BYTES      = 100 * 1024 * 1024           # rotate every ~100 MB
TARGET_BYTES     = 350 * 1024 * 1024           # stop around ~350 MB total
MAX_PAGES        = 500000                      # hard cap
TIMEOUT          = 20
CONCURRENCY      = 18                          # polite but parallel (adjust 12–20)
POLITE_DELAY     = (0.08, 0.30)                # polite jitter per request
MIN_CHARS        = 600                         # drop very short pages
ASCII_RATIO_MIN  = 0.32                        # filter non-Englishish
USER_AGENT       = "SpaceSystemsGPT/1.0 (+edu)"

# Allowlist: high-quality technical/science
ALLOWED_DOMAINS = {
    "wikipedia.org",
    "nasa.gov", "science.nasa.gov", "solarsystem.nasa.gov",
    "jpl.nasa.gov", "gsfc.nasa.gov", "earthdata.nasa.gov",
    "mars.nasa.gov", "ames.nasa.gov", "arc.nasa.gov",
    "esa.int",
    "nssdc.gsfc.nasa.gov",
    "lpi.usra.edu", "planetary.org",
}

# Hard blacklist (social, UGC, tracking)
BLACKLIST_DOMAINS = {
    "twitter.com","x.com","facebook.com","instagram.com","tiktok.com",
    "youtube.com","youtu.be","reddit.com","linkedin.com","medium.com",
    "pinterest.com","threads.net","discord.com","snapchat.com"
}

# Skip binary/doc attachments
DISALLOWED_EXTS = {
    ".pdf",".jpg",".jpeg",".png",".gif",".svg",".webp",".bmp",".tif",".tiff",
    ".mp4",".webm",".mov",".avi",".mp3",".wav",".zip",".gz",".tar",".tgz",
    ".ppt",".pptx",".doc",".docx",".xls",".xlsx",".apk"
}

# Seeds (start points). Add more if you want.
SEEDS = [
  # Wikipedia — orbital mechanics & core topics
  "https://en.wikipedia.org/wiki/Orbital_mechanics",
  "https://en.wikipedia.org/wiki/Hohmann_transfer_orbit",
  "https://en.wikipedia.org/wiki/Bi-elliptic_transfer",
  "https://en.wikipedia.org/wiki/Low_Earth_orbit",
  "https://en.wikipedia.org/wiki/Delta-v",
  "https://en.wikipedia.org/wiki/Tsiolkovsky_rocket_equation",
  "https://en.wikipedia.org/wiki/Orbital_elements",
  "https://en.wikipedia.org/wiki/Inclination_(orbit)",
  "https://en.wikipedia.org/wiki/Geostationary_orbit",
  "https://en.wikipedia.org/wiki/Sun-synchronous_orbit",
  "https://en.wikipedia.org/wiki/Gravity_assist",
  "https://en.wikipedia.org/wiki/Lagrange_point",
  "https://en.wikipedia.org/wiki/Atmospheric_entry",
  "https://en.wikipedia.org/wiki/Reaction_control_system",
  "https://en.wikipedia.org/wiki/Attitude_control",
  # NASA / ESA hubs
  "https://science.nasa.gov/solar-system/",
  "https://science.nasa.gov/planetary-science/",
  "https://science.nasa.gov/astrophysics/",
  "https://science.nasa.gov/heliophysics/",
  "https://solarsystem.nasa.gov/basics/",
  "https://solarsystem.nasa.gov/planets/overview/",
  "https://solarsystem.nasa.gov/missions/",
  "https://www.jpl.nasa.gov/missions",
  "https://nssdc.gsfc.nasa.gov/planetary/factsheet/",
  "https://www.esa.int/Science_Exploration/Space_Science",
  "https://www.esa.int/Applications/Observing_the_Earth",
]

EXTRA_SEEDS = [
  "https://science.nasa.gov/earth/",
  "https://mars.nasa.gov/mars2020/",
  "https://mars.nasa.gov/msl/",
  "https://earthdata.nasa.gov/learn/backgrounders",
]

# --- deps ---
import aiohttp
from bs4 import BeautifulSoup
try:
    import trafilatura
    _HAS_TRAFILATURA = True
except Exception:
    _HAS_TRAFILATURA = False

from tqdm import tqdm
import ftfy

# --- cleaners / helpers ---
SOCIAL_RE = re.compile(r"(instagram|youtube|twitter|x\.com|facebook|tiktok|subscribe|follow|@[\w_]+)", re.I)
WHITESPACE_RE = re.compile(r"[ \t]+")

def normalize_text(txt: str) -> str:
    txt = ftfy.fix_text(txt).replace("\ufeff"," ")
    txt = WHITESPACE_RE.sub(" ", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt.strip()

def remove_social_lines(txt: str) -> str:
    keep = []
    for ln in txt.splitlines():
        s = ln.strip()
        if not s:
            keep.append(""); continue
        if SOCIAL_RE.search(s):
            continue
        if s.startswith(("-", "•", "@")) and SOCIAL_RE.search(s):
            continue
        keep.append(s)
    out = "\n".join(keep)
    return re.sub(r"\n{3,}", "\n\n", out).strip()

BAD_TEXT_PATTERNS = re.compile(
    r"(cookies|cookie policy|accept cookies|enable javascript|javascript required|"
    r"page not found|404|access denied|sign in|subscribe|privacy policy|"
    r"enable cookies|strictly necessary|data protection|content unavailable)",
    re.I
)

def looks_like_boilerplate(txt: str) -> bool:
    if BAD_TEXT_PATTERNS.search(txt[:2000]):  # obvious boilerplate up front
        return True
    # very few sentences → nav or stub
    if txt.count(".") < 5 and txt.count("\n") < 15 and len(txt) < 1200:
        return True
    # extremely low lexical variety on short pages
    words = re.findall(r"[A-Za-z]{3,}", txt.lower())
    if len(txt) < 2000 and len(set(words)) < 80:
        return True
    return False

def looks_html(ct: str) -> bool:
    return ct and ("text/html" in ct or "application/xhtml+xml" in ct)

def html_to_text(html: str) -> str:
    # prefer trafilatura if available
    txt = ""
    if _HAS_TRAFILATURA:
        try:
            txt = trafilatura.extract(html) or ""
        except Exception:
            txt = ""
    if not txt:
        soup = BeautifulSoup(html, "lxml")
        for s in soup(["script","style","noscript","header","footer","nav","aside","form","iframe"]):
            s.decompose()
        txt = soup.get_text(" ")
    txt = normalize_text(txt)
    txt = remove_social_lines(txt)
    if looks_like_boilerplate(txt):
        return ""
    return txt

def extract_links(base: str, html: str) -> set:
    soup = BeautifulSoup(html, "lxml")
    out = set()
    for a in soup.select("a[href]"):
        href = a.get("href","")
        href = urljoin(base, href)
        href, _ = urldefrag(href)
        out.add(href)
    return out

def allowed_url(url: str) -> bool:
    u = urlparse(url)
    host = u.netloc.lower()
    path = u.path.lower()

    if any(b in host for b in BLACKLIST_DOMAINS):
        return False
    if not any(a in host for a in ALLOWED_DOMAINS):
        return False
    if any(path.endswith(ext) for ext in DISALLOWED_EXTS):
        return False

    # host-specific path rules to stay in content-rich areas
    if "wikipedia.org" in host:
        if not path.startswith("/wiki/"): return False
        # exclude File:, Help:, Talk:, Template:, etc.
        tail = path.split("/wiki/")[-1]
        if ":" in tail: return False

    if "science.nasa.gov" in host:
        ok = path.startswith("/solar-system/") or path.startswith("/mission/") or \
             path.startswith("/planetary-science/") or path.startswith("/astrophysics/") or \
             path.startswith("/heliophysics/") or path.startswith("/earth/")
        if not ok: return False

    if "solarsystem.nasa.gov" in host:
        ok = ("/basics" in path) or ("/planets" in path) or ("/moons" in path) or \
             ("/missions" in path) or ("/resources" in path)
        if not ok: return False

    if "jpl.nasa.gov" in host and not path.startswith("/missions"):
        return False

    if "esa.int" in host and not ("/Science_Exploration" in path or "/Applications/Observing_the_Earth" in path):
        return False

    return True

def is_englishish(txt: str) -> bool:
    letters = sum(ch.isalpha() for ch in txt)
    return (letters / max(1, len(txt))) >= ASCII_RATIO_MIN

def hash_text(txt: str) -> str:
    return hashlib.md5(txt[:5000].encode("utf-8","ignore")).hexdigest()

# --- resume: load existing shards to seed seen_hashes, written_total ---
seen_urls, seen_hashes = set(), set()

def preload_seen_from_existing():
    total_bytes = 0
    for fn in sorted(os.listdir(".")):
        if fn.startswith(OUT_BASENAME) and fn.endswith(".jsonl"):
            total_bytes += os.path.getsize(fn)
            with open(fn,"r",encoding="utf-8",errors="ignore") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        t = obj.get("text") or ""
                        if t:
                            seen_hashes.add(hash_text(t))
                    except:
                        pass
    return total_bytes

written_total = preload_seen_from_existing()
print(f"[resume] preloaded hashes={len(seen_hashes)} | bytes already on disk={written_total:,}")

# --- shard writer ---
class ShardWriter:
    def __init__(self, base, shard_bytes):
        self.base = base; self.shard_bytes = shard_bytes
        self.idx = 0; self.file = None; self.written_in_shard = 0
        self._open_new_if_needed()
    def _open_new_if_needed(self):
        while True:
            name = f"{self.base}_{self.idx:03d}.jsonl"
            if not os.path.exists(name) or os.path.getsize(name) < self.shard_bytes:
                self.file = open(name, "a", encoding="utf-8")
                self.written_in_shard = os.path.getsize(name)
                self.name = name
                break
            self.idx += 1
    def write(self, obj: dict):
        s = json.dumps(obj, ensure_ascii=False) + "\n"
        b = s.encode("utf-8")
        if self.written_in_shard + len(b) > self.shard_bytes:
            self.file.close(); self.idx += 1; self._open_new_if_needed()
        self.file.write(s); self.file.flush()
        self.written_in_shard += len(b)
        return len(b)
    def close(self):
        if self.file: self.file.close()

writer = ShardWriter(OUT_BASENAME, SHARD_BYTES)

# --- async fetch infra ---
import aiohttp
from tqdm import tqdm

sem = asyncio.Semaphore(CONCURRENCY)
timeout = aiohttp.ClientTimeout(total=None, connect=8, sock_read=TIMEOUT)
conn = aiohttp.TCPConnector(limit=CONCURRENCY, limit_per_host=max(6, CONCURRENCY//2), ttl_dns_cache=300)
headers = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Encoding": "gzip, deflate",
    "Accept-Language": "en,en-US;q=0.9",
}

async def fetch(session: aiohttp.ClientSession, url: str):
    async with sem:
        await asyncio.sleep(random.uniform(*POLITE_DELAY))
        try:
            async with session.get(url, timeout=TIMEOUT, allow_redirects=True) as r:
                if r.status != 200:
                    return None, None, r.status
                ct = r.headers.get("content-type","")
                if not looks_html(ct):
                    return None, None, r.status
                return await r.text(), ct, r.status
        except asyncio.TimeoutError:
            return None, None, "timeout"
        except Exception:
            return None, None, "error"

async def crawl_to_size():
    global written_total, MIN_CHARS, ASCII_RATIO_MIN

    q = deque(SEEDS)
    frontier_seen = set(SEEDS)
    per_host_visits = defaultdict(int)
    HOST_SOFT_LIMIT = 450

    reasons = defaultdict(int)
    saved = fetched = 0
    last_saved_iter = 0
    iters = 0

    pbar = tqdm(total=TARGET_BYTES, unit="B", unit_scale=True, desc="bytes written", initial=written_total)

    async with aiohttp.ClientSession(headers=headers, timeout=timeout, connector=conn) as session:
        while q and written_total < TARGET_BYTES and iters < MAX_PAGES:
            iters += 1
            url = q.popleft()

            if url in seen_urls or not allowed_url(url):
                reasons["not_allowed_or_seen"] += 1
                continue
            seen_urls.add(url)

            host = urlparse(url).netloc.lower()
            if per_host_visits[host] > HOST_SOFT_LIMIT:
                q.append(url)
                continue
            per_host_visits[host] += 1

            html, _, status = await fetch(session, url)
            if html is None:
                reasons[str(status)] += 1
                continue

            fetched += 1
            txt = html_to_text(html)

            if len(txt) < MIN_CHARS:
                reasons["too_short"] += 1
            elif not is_englishish(txt):
                reasons["non_englishish"] += 1
            else:
                h = hash_text(txt)
                if h in seen_hashes:
                    reasons["dup_hash"] += 1
                else:
                    seen_hashes.add(h)
                    b = writer.write({"url": url, "text": txt})
                    written_total += b
                    saved += 1
                    last_saved_iter = iters
                    pbar.update(b)

            # expand frontier
            for u in extract_links(url, html):
                if u not in frontier_seen and allowed_url(u):
                    frontier_seen.add(u)
                    q.append(u)

            if (saved % 25 == 0 and saved > 0) or (iters % 500 == 0):
                print(f"[progress] saved={saved} fetched={fetched} bytes={written_total/1e6:.1f} MB "
                      f"frontier={len(q)} reasons={dict(list(reasons.items())[:8])}")

            # auto-reseed if frontier shrinks
            if len(q) < 200:
                pushed = 0
                for s in EXTRA_SEEDS:
                    if s not in frontier_seen and allowed_url(s):
                        frontier_seen.add(s); q.append(s); pushed += 1
                if pushed:
                    print(f"[auto] frontier low → injected {pushed} seeds (frontier={len(q)})")

            # auto-relax if no saves for a while
            if iters - last_saved_iter > 1500:
                old_min, old_ratio = MIN_CHARS, ASCII_RATIO_MIN
                MIN_CHARS = max(500, MIN_CHARS - 100)
                ASCII_RATIO_MIN = max(0.28, ASCII_RATIO_MIN - 0.02)
                last_saved_iter = iters
                print(f"[auto] relaxed filters: MIN_CHARS {old_min}→{MIN_CHARS}, "
                      f"ASCII_RATIO_MIN {old_ratio:.2f}→{ASCII_RATIO_MIN:.2f}")

            if written_total >= TARGET_BYTES:
                break

    pbar.close()
    writer.close()
    print(f"Saved pages: {saved} | Total bytes: {written_total:,}")
    print("Reasons summary:", dict(reasons))

# run
await crawl_to_size()


# Merge shards -> single corpus file (dedup again for safety)
import os, glob, json, hashlib

FINAL_JSONL = "space_corpus_350mb.jsonl"
def h(t): return hashlib.md5(t[:5000].encode("utf-8","ignore")).hexdigest()

seen=set(); kept=0
with open(FINAL_JSONL,"w",encoding="utf-8") as out:
    for fn in sorted(glob.glob(f"{OUT_BASENAME}_*.jsonl")):
        for line in open(fn,"r",encoding="utf-8",errors="ignore"):
            try:
                t = json.loads(line).get("text","")
            except:
                t = ""
            if len(t) < 500: 
                continue
            hh = h(t)
            if hh in seen: 
                continue
            seen.add(hh); kept += 1
            out.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")

print("Merged ->", FINAL_JSONL, "| docs:", kept, "| size MB:", os.path.getsize(FINAL_JSONL)/1024/1024)

