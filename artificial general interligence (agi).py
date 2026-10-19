import subprocess, sys, os, time, json, shutil, re, hashlib, uuid, ast

# ── SECTION 1: INSTALL ─────────────────────────────────────────────────
def _pip(*pkgs, timeout=2400):
    cmd = [sys.executable, "-m", "pip", "install", "-q"] + list(pkgs)
    print(f"📦 {pkgs[0]}")
    r = subprocess.run(cmd, capture_output=True, timeout=timeout, check=False)
    if r.returncode != 0:
        print(f"   ⚠️ {r.stderr.decode()[-200:]}")

print("=" * 80)
print("🧠 AUTONOMOUS AGENT v4.0 — Self-Modifying + Goal-Protected")
print("=" * 80)

subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y",
                "torch", "torch-xla", "torchvision", "torchaudio"],
               check=False, capture_output=True)

_pip("openai", "duckduckgo-search", "chromadb", "beautifulsoup4",
     "requests", "psutil", "sentencepiece", "protobuf", "einops",
     "msgspec", "pyzmq", "fastapi", "uvicorn", "pydantic")
_pip("torch==2.9.0", "torch_xla[tpu]==2.9.0",
     "-f", "https://storage.googleapis.com/libtpu-releases/index.html")
_pip("vllm-tpu==0.28.0", "--no-deps")
_pip("xgrammar", "compressed-tensors", "outlines", "ray")
print("✅ Packages installed\n")

# ── SECTION 2: ENVIRONMENT ─────────────────────────────────────────────
os.environ["PJRT_DEVICE"] = "TPU"
os.environ["VLLM_XLA_CACHE_PATH"] = "/tmp/xla_cache"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["HF_HOME"] = "/dev/shm/hf_cache"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

WORKSPACE   = "/kaggle/working"
AGENT_DIR   = f"{WORKSPACE}/PEARL_Agent"
PATCHES_DIR = f"{AGENT_DIR}/patches"
MEMORY_DIR  = f"{AGENT_DIR}/memory"
CHROMA_DIR  = f"{AGENT_DIR}/chroma_db"
BACKUP_DIR  = f"{AGENT_DIR}/backups"

# ── Memory files (group A) ──
GOAL_FILE           = f"{AGENT_DIR}/goal.txt"
INTERPRETATION_FILE = f"{AGENT_DIR}/interpretation.json"
RESEARCH_FILE       = f"{AGENT_DIR}/research.json"
STRATEGY_FILE       = f"{AGENT_DIR}/strategy.md"
PROGRESS_FILE       = f"{AGENT_DIR}/progress.json"

STATE_FILE  = f"{AGENT_DIR}/agent_state.json"
TRACE_FILE  = f"{AGENT_DIR}/agent_trace.jsonl"
SOURCE_FILE = f"{AGENT_DIR}/agent_source.py"
CORE_FILE   = f"{AGENT_DIR}/agent_core.py"    # the running code
EVOLUTION_LOG = f"{AGENT_DIR}/evolution.log"

for d in [AGENT_DIR, PATCHES_DIR, MEMORY_DIR, CHROMA_DIR, BACKUP_DIR,
          "/tmp/xla_cache", "/dev/shm/hf_cache"]:
    os.makedirs(d, exist_ok=True)

# XLA cache
for src in ["/kaggle/input/qwen38-xla-cache-v5e8/xla_cache"]:
    if os.path.isdir(src):
        for f in os.listdir(src):
            s, d = os.path.join(src, f), os.path.join("/tmp/xla_cache", f)
            if not os.path.exists(d):
                try:
                    if os.path.isfile(s): shutil.copy2(s, d)
                    else: shutil.copytree(s, d)
                except Exception: pass
        print(f"📂 XLA cache loaded")
        break

SESSION_LIMIT = float(os.environ.get("AGENT_TIME_LIMIT", 8.5)) * 3600

# ── SECTION 2b: GOAL INITIALIZATION (UPDATE 1 + 11) ────────────────────
# goal.txt is the SOURCE OF TRUTH. chmod 444 to prevent modification.

DEFAULT_GOAL = """ You should continuously become more advanced than what you currently are, using your own source code and other tools. Build self-improvement features like these to upgrade yourself into a better version, and keep pushing forward—keep evolving and becoming more advanced continuously."""

if not os.path.exists(GOAL_FILE):
    with open(GOAL_FILE, "w") as f:
        f.write(DEFAULT_GOAL)
    print("📝 Created goal.txt")
else:
    print(f"📖 Loaded existing goal.txt")

# chmod 444 — read-only (UPDATE 11)
try:
    os.chmod(GOAL_FILE, 0o444)
    print("🔒 goal.txt is now READ-ONLY (chmod 444)")
except Exception as e:
    print(f"⚠️ chmod failed: {e}")

# Load goal from file
def read_goal():
    """Read goal from goal.txt. If locked, temporarily unlock to read."""
    try:
        with open(GOAL_FILE, "r") as f:
            return f.read().strip()
    except PermissionError:
        os.chmod(GOAL_FILE, 0o644)
        goal = open(GOAL_FILE).read().strip()
        os.chmod(GOAL_FILE, 0o444)
        return goal

GOAL = read_goal()
GOAL_HASH = hashlib.sha256(GOAL.encode()).hexdigest()[:16]
print(f"🎯 GOAL_HASH: {GOAL_HASH}\n")

# ── Write AGENT_CORE code to disk (SOURCE for self-modification) ───────
# The full agent logic is written to CORE_FILE and executed as subprocess.
# On self_rewrite, this file gets replaced + os.execv restarts it.

CORE_CODE = r'''#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════════
# AGENT CORE — Self-modifying autonomous agent
# This file may be REPLACED by the agent itself via self_rewrite tool.
# Core invariant: must read goal.txt + 4 memory files at startup.
# ═══════════════════════════════════════════════════════════════════════
import os, sys, json, time, shutil, re, hashlib, uuid, ast, subprocess, importlib.util
from collections import deque, Counter
from dataclasses import dataclass

# ── Paths (passed via env or recomputed) ──
WORKSPACE   = "/kaggle/working"
AGENT_DIR   = os.environ.get("AGENT_DIR", f"{WORKSPACE}/PEARL_Agent")
PATCHES_DIR = f"{AGENT_DIR}/patches"
CHROMA_DIR  = f"{AGENT_DIR}/chroma_db"
BACKUP_DIR  = f"{AGENT_DIR}/backups"
GOAL_FILE           = f"{AGENT_DIR}/goal.txt"
INTERPRETATION_FILE = f"{AGENT_DIR}/interpretation.json"
RESEARCH_FILE       = f"{AGENT_DIR}/research.json"
STRATEGY_FILE       = f"{AGENT_DIR}/strategy.md"
PROGRESS_FILE       = f"{AGENT_DIR}/progress.json"
STATE_FILE          = f"{AGENT_DIR}/agent_state.json"
TRACE_FILE          = f"{AGENT_DIR}/agent_trace.jsonl"
SOURCE_FILE         = f"{AGENT_DIR}/agent_source.py"
CORE_FILE           = f"{AGENT_DIR}/agent_core.py"
EVOLUTION_LOG       = f"{AGENT_DIR}/evolution.log"

MODEL_ID = "Goekdeniz-Guelmez/Josiefied-DeepSeek-R1-0528-Qwen3-8B-abliterated-v1"
NATURAL_CONTEXT_LIMIT = 131072
MAX_OUTPUT_TOKENS = 32768
SESSION_LIMIT = float(os.environ.get("AGENT_TIME_LIMIT", 8.5)) * 3600
SESSION_START = time.time()

MAX_REWRITES_PER_SESSION = 3

# ── Client setup ──
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

import chromadb
from bs4 import BeautifulSoup
import requests as _req

# ═══════════════════════════════════════════════════════════════════════
# STARTUP: MANDATORY READS (UPDATE 10)
# Every run MUST read goal.txt + 4 memory files
# ═══════════════════════════════════════════════════════════════════════
def read_goal_unlocked():
    try:
        return open(GOAL_FILE).read().strip()
    except PermissionError:
        os.chmod(GOAL_FILE, 0o644)
        g = open(GOAL_FILE).read().strip()
        os.chmod(GOAL_FILE, 0o444)
        return g

GOAL = read_goal_unlocked()
GOAL_HASH = hashlib.sha256(GOAL.encode()).hexdigest()[:16]

def safe_read(path, default=""):
    if not os.path.exists(path): return default
    try: return open(path).read()
    except Exception: return default

INTERPRETATION = safe_read(INTERPRETATION_FILE, "{}")
RESEARCH       = safe_read(RESEARCH_FILE, "{}")
STRATEGY       = safe_read(STRATEGY_FILE, "")
PROGRESS       = safe_read(PROGRESS_FILE, "{}")

print(f"📖 MANDATORY READS at startup:")
print(f"   goal.txt           → {len(GOAL)} chars")
print(f"   interpretation.json→ {len(INTERPRETATION)} chars")
print(f"   research.json      → {len(RESEARCH)} chars")
print(f"   strategy.md        → {len(STRATEGY)} chars")
print(f"   progress.json      → {len(PROGRESS)} chars")

# ── SECTION: STARTUP HELPERS ──
try:
    _p = json.loads(PROGRESS) if PROGRESS else {}
except Exception:
    _p = {}
rewrites_this_session = _p.get("rewrites_this_session", 0)
print(f"🔁 Rewrites this session: {rewrites_this_session}/{MAX_REWRITES_PER_SESSION}")

# ── ChromaDB ──
_chroma = chromadb.PersistentClient(path=CHROMA_DIR)
_mem_coll = _chroma.get_or_create_collection("agent_memory")

def memory_store(key, content):
    try:
        mid = f"{key}_{uuid.uuid4().hex[:12]}"
        _mem_coll.add(documents=[content], ids=[mid],
                      metadatas=[{"key": key, "ts": str(time.time())}])
        return f"✅ Stored [{key}]: {content[:100]}"
    except Exception as e:
        return f"❌ store error: {e}"

def memory_recall(query, n=5):
    try:
        if _mem_coll.count() == 0: return "❌ Memory empty."
        r = _mem_coll.query(query_texts=[query], n_results=min(n, _mem_coll.count()))
        docs = r.get("documents", [[]])[0] or []
        return "\n".join(f"- {d[:300]}" for d in docs) if docs else "❌ No matches."
    except Exception as e:
        return f"❌ recall error: {e}"

# ═══════════════════════════════════════════════════════════════════════
# UPDATE 9: VERIFY GOAL INTEGRITY — every step
# ═══════════════════════════════════════════════════════════════════════
def verify_goal_integrity():
    """Check if goal.txt was tampered with."""
    current = read_goal_unlocked()
    current_hash = hashlib.sha256(current.encode()).hexdigest()[:16]
    if current_hash != GOAL_HASH:
        print(f"🚨 GOAL INTEGRITY BREACH!")
        print(f"   Expected: {GOAL_HASH}")
        print(f"   Got:      {current_hash}")
        # Restore from backup
        backup = f"{GOAL_FILE}.backup"
        if os.path.exists(backup):
            os.chmod(GOAL_FILE, 0o644)
            shutil.copy2(backup, GOAL_FILE)
            os.chmod(GOAL_FILE, 0o444)
            print(f"   ✅ Restored from backup")
            return False
        return False
    return True

# Create initial goal backup
if not os.path.exists(f"{GOAL_FILE}.backup"):
    try:
        os.chmod(GOAL_FILE, 0o644)
        shutil.copy2(GOAL_FILE, f"{GOAL_FILE}.backup")
        os.chmod(GOAL_FILE, 0o444)
    except Exception: pass

# ═══════════════════════════════════════════════════════════════════════
# SECTION: FILE TOOLS
# ═══════════════════════════════════════════════════════════════════════
def _abs(path):
    full = path if path.startswith("/") else os.path.join(AGENT_DIR, path)
    full = os.path.abspath(full)
    if not full.startswith(os.path.abspath(AGENT_DIR)):
        raise ValueError(f"Path outside workspace: {path}")
    return full

def read_file(path, max_chars=50000):
    try:
        full = _abs(path)
    except ValueError as e:
        return f"❌ {e}"
    if not os.path.exists(full): return f"FILE_NOT_FOUND: {full}"
    try: return open(full, "r", encoding="utf-8").read()[:max_chars]
    except Exception as e: return f"❌ read error: {e}"

PROTECTED_FILES = ["agent_state.json", "agent_trace.jsonl",
                   "vllm_server.log", "patches/registry.json",
                   "evolution.log", "agent_core.py"]

def write_file(path, content):
    try:
        full = _abs(path)
    except ValueError as e:
        return f"❌ {e}"
    rel = os.path.relpath(full, AGENT_DIR)
    if any(p in rel for p in PROTECTED_FILES):
        return f"❌ Protected: {rel}. Use self_rewrite for agent_core.py."
    if full.startswith(PATCHES_DIR):
        return "❌ Use create_patch for patches/"
    os.makedirs(os.path.dirname(full) or ".", exist_ok=True)
    try:
        with open(full, "w", encoding="utf-8") as f:
            f.write(content)
        return f"✅ Wrote {len(content)} chars to {full}"
    except Exception as e:
        return f"❌ write error: {e}"

def run_python(code, timeout=180):
    p = f"/tmp/_run_{uuid.uuid4().hex[:8]}.py"
    header = f"import sys, os\nsys.path.insert(0, '{AGENT_DIR}')\nos.chdir('{AGENT_DIR}')\n"
    try:
        with open(p, "w") as f: f.write(header + code)
        r = subprocess.run([sys.executable, p], capture_output=True,
                           text=True, timeout=timeout, cwd=AGENT_DIR)
        return {"stdout": r.stdout[-4000:], "stderr": r.stderr[-1500:], "rc": r.returncode}
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": f"TIMEOUT {timeout}s", "rc": -1}
    except Exception as e:
        return {"stdout": "", "stderr": str(e), "rc": -1}

_search_cache = {}
def web_search(query):
    q = (query or "").strip()
    if not q: return "❌ Empty query."
    if q in _search_cache: return f"💾 (cached) {_search_cache[q]}"
    try:
        from duckduckgo_search import DDGS
        with DDGS() as d: results = list(d.text(q, max_results=6))
        if not results: return f"❌ No results for: {q}"
        out = [f"🔎 Search: {q}"]
        for i, r in enumerate(results[:5], 1):
            out.append(f"[{i}] {r.get('title','')}")
            out.append(f"    🔗 {r.get('href','')}")
            out.append(f"    📝 {r.get('body','')[:250]}")
        result = "\n".join(out); _search_cache[q] = result
        return result
    except Exception as e:
        return f"❌ search error: {type(e).__name__}: {e}"

def fetch_url(url):
    u = (url or "").strip().strip("'\"")
    if not u.startswith(("http://", "https://")): return f"❌ Invalid URL: {u}"
    try:
        r = _req.get(u, timeout=12, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 Chrome/122.0.0.0 Safari/537.36"},
            allow_redirects=True)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for t in soup(["script","style","nav","footer","header","aside",
                       "noscript","iframe","svg","form"]): t.decompose()
        text = " ".join(soup.get_text(separator=" ", strip=True).split())
        return text[:4500] + (" …[truncated]" if len(text) > 4500 else "")
    except Exception as e:
        return f"❌ fetch error: {type(e).__name__}: {e}"

# ═══════════════════════════════════════════════════════════════════════
# GROUP A: MEMORY FILE TOOLS (UPDATES 2, 3, 4, 5)
# Model fills structured content — system gives structure
# ═══════════════════════════════════════════════════════════════════════

def save_interpretation(win_meaning, success_criteria, constraints, notes=""):
    """UPDATE 2: Save what 'winning' means for this goal."""
    data = {
        "goal_summary": GOAL[:500],
        "win_meaning": win_meaning,
        "success_criteria": success_criteria,
        "constraints": constraints,
        "notes": notes,
        "updated_at": time.ctime(),
    }
    try:
        json.dump(data, open(INTERPRETATION_FILE, "w"), indent=2, ensure_ascii=False)
        return f"✅ interpretation.json saved ({len(win_meaning)} chars)"
    except Exception as e:
        return f"❌ save error: {e}"

def save_research(topic, findings, sources=None, key_insights=None):
    """UPDATE 3: Append research findings (grows over time)."""
    try:
        existing = json.loads(open(RESEARCH_FILE).read()) if os.path.exists(RESEARCH_FILE) else {"entries": []}
    except Exception:
        existing = {"entries": []}
    entry = {
        "topic": topic,
        "findings": findings,
        "sources": sources or [],
        "key_insights": key_insights or [],
        "timestamp": time.ctime(),
    }
    existing["entries"].append(entry)
    existing["total_entries"] = len(existing["entries"])
    existing["last_updated"] = time.ctime()
    try:
        json.dump(existing, open(RESEARCH_FILE, "w"), indent=2, ensure_ascii=False)
        return f"✅ research.json: added entry '{topic}' (total: {existing['total_entries']})"
    except Exception as e:
        return f"❌ save error: {e}"

def save_strategy(markdown_content):
    """UPDATE 4: Save/update strategy.md."""
    try:
        with open(STRATEGY_FILE, "w") as f:
            f.write(markdown_content)
        return f"✅ strategy.md saved ({len(markdown_content)} chars)"
    except Exception as e:
        return f"❌ save error: {e}"

def update_progress(current_phase, completed, in_progress, blockers, next_steps):
    """UPDATE 5: Save structured progress."""
    try:
        existing = json.loads(open(PROGRESS_FILE).read()) if os.path.exists(PROGRESS_FILE) else {}
    except Exception:
        existing = {}
    data = {
        "current_phase": current_phase,
        "completed": completed,
        "in_progress": in_progress,
        "blockers": blockers,
        "next_steps": next_steps,
        "rewrites_this_session": existing.get("rewrites_this_session", 0),
        "updated_at": time.ctime(),
    }
    try:
        json.dump(data, open(PROGRESS_FILE, "w"), indent=2, ensure_ascii=False)
        return f"✅ progress.json updated (phase: {current_phase})"
    except Exception as e:
        return f"❌ save error: {e}"

# ═══════════════════════════════════════════════════════════════════════
# SAFE PATCH SYSTEM
# ═══════════════════════════════════════════════════════════════════════
FORBIDDEN_PATTERNS = ["os.system", "subprocess.", "eval(", "exec(",
                      "shutil.rmtree", "__import__", "os.remove",
                      "os.rmdir", "socket.socket"]
FORBIDDEN_IMPORTS = ["os", "sys", "subprocess", "shutil", "socket", "ctypes"]

def _ast_check(code):
    try: tree = ast.parse(code)
    except SyntaxError as e: return False, f"SyntaxError: {e.msg} (line {e.lineno})"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] in FORBIDDEN_IMPORTS:
                    return False, f"Forbidden import: {a.name}"
        if isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] in FORBIDDEN_IMPORTS:
                return False, f"Forbidden import: {node.module}"
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in ("eval", "exec", "compile", "__import__"):
                    return False, f"Forbidden call: {node.func.id}"
        if isinstance(node, ast.Attribute):
            if node.attr in ("__globals__", "__builtins__", "__subclasses__"):
                return False, f"Forbidden: {node.attr}"
    for p in FORBIDDEN_PATTERNS:
        if p in code: return False, f"Forbidden pattern: {p}"
    return True, "OK"

def create_patch(name, code, description=""):
    if not name.replace("_","").replace("-","").isalnum():
        return f"❌ Invalid name: {name}"
    ok, reason = _ast_check(code)
    if not ok: return f"❌ Rejected: {reason}"
    ts = int(time.time())
    pf = f"{name}.v{ts}.py"
    pp = os.path.join(PATCHES_DIR, pf)
    with open(pp, "w") as f:
        f.write(f"# {description}\n# {time.ctime(ts)}\n\n{code}")
    test = f"import sys; sys.path.insert(0, '{PATCHES_DIR}')\nimport importlib.util\nspec = importlib.util.spec_from_file_location('t', '{pp}')\nm = importlib.util.module_from_spec(spec)\ntry:\n    spec.loader.exec_module(m)\n    print('OK')\nexcept Exception as e:\n    print(f'FAIL: {{e}}')\n"
    tp = f"/tmp/_pt_{uuid.uuid4().hex[:8]}.py"
    with open(tp, "w") as f: f.write(test)
    r = subprocess.run([sys.executable, tp], capture_output=True, text=True, timeout=30)
    if "OK" not in r.stdout:
        try: os.remove(pp)
        except: pass
        return f"❌ Sandbox failed: {r.stdout[-200:]}"
    # Register
    reg_path = os.path.join(PATCHES_DIR, "registry.json")
    try: reg = json.load(open(reg_path)) if os.path.exists(reg_path) else {}
    except: reg = {}
    reg[name] = {"file": pf, "desc": description, "created": ts}
    json.dump(reg, open(reg_path, "w"), indent=2)
    # Load now
    loaded = []
    try:
        spec = importlib.util.spec_from_file_location(name, pp)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for attr in dir(mod):
            if not attr.startswith("_") and callable(getattr(mod, attr)):
                TOOLS[attr] = getattr(mod, attr)
                loaded.append(attr)
    except Exception as e:
        return f"⚠️ Saved, load failed: {e}"
    with open(EVOLUTION_LOG, "a") as f:
        f.write(f"[{time.ctime()}] NEW TOOL: {name} → {loaded}\n")
    return f"✅ Patch '{name}' created and loaded: {loaded}"

def list_patches():
    reg_path = os.path.join(PATCHES_DIR, "registry.json")
    if not os.path.exists(reg_path): return "📦 No patches."
    reg = json.load(open(reg_path))
    if not reg: return "📦 No patches."
    return f"📦 {len(reg)} patches:\n" + "\n".join(f"  • {n} — {i.get('desc','')[:60]}" for n,i in reg.items())

# ═══════════════════════════════════════════════════════════════════════
# GROUP B: SELF-MODIFICATION (UPDATES 6, 7, 8, 12)
# ═══════════════════════════════════════════════════════════════════════

CURRENT_SYSTEM_PROMPT = None  # set below

def self_modify_prompt(new_prompt, reason):
    """UPDATE 6: Model modifies its own system prompt."""
    global CURRENT_SYSTEM_PROMPT
    if not new_prompt or len(new_prompt) < 200:
        return "❌ New prompt too short (min 200 chars)"
    if not reason or len(reason) < 10:
        return "❌ Reason required (min 10 chars)"
    # Save old for backup
    backup_file = f"{AGENT_DIR}/system_prompt.backup.{int(time.time())}"
    if CURRENT_SYSTEM_PROMPT:
        with open(backup_file, "w") as f: f.write(CURRENT_SYSTEM_PROMPT)
    # Save new
    prompt_file = f"{AGENT_DIR}/system_prompt.txt"
    with open(prompt_file, "w") as f: f.write(new_prompt)
    CURRENT_SYSTEM_PROMPT = new_prompt
    with open(EVOLUTION_LOG, "a") as f:
        f.write(f"[{time.ctime()}] PROMPT MODIFIED: {reason}\n")
    return f"✅ System prompt modified. Reason: {reason}. Backup: {os.path.basename(backup_file)}"

def self_rewrite(new_code, reason):
    """UPDATE 7, 8, 12: Rewrite own code + restart.
    Requires:
    - AST safety check
    - New code reads goal.txt + 4 memory files
    - Rate limit (max 3 per session)
    - Backup, save state, execv restart
    """
    global rewrites_this_session

    # 1. Rate limit check (UPDATE 12)
    if rewrites_this_session >= MAX_REWRITES_PER_SESSION:
        return (f"❌ Rate limit reached: {rewrites_this_session}/"
                f"{MAX_REWRITES_PER_SESSION} rewrites this session. "
                f"Cannot rewrite again until next session.")

    # 2. AST safety
    ok, reason_ast = _ast_check(new_code)
    if not ok:
        return f"❌ AST check failed: {reason_ast}"

    # 3. Required reads check (UPDATE 10)
    required_reads = [
        ("goal.txt", "goal.txt"),
        ("interpretation.json", "interpretation.json"),
        ("research.json", "research.json"),
        ("strategy.md", "strategy.md"),
        ("progress.json", "progress.json"),
    ]
    missing = []
    for filename, keyword in required_reads:
        # Check for both the filename string OR the variable name
        if filename not in new_code and keyword.upper() not in new_code:
            missing.append(filename)
    if missing:
        return (f"❌ Rejected: new code must reference and read these files: "
                f"{missing}. Add reads for mandatory memory files.")

    # Also check for goal integrity function reference
    if "verify_goal_integrity" not in new_code:
        return "❌ Rejected: new code must include verify_goal_integrity() call"

    # 4. Backup current code
    if os.path.exists(CORE_FILE):
        backup = os.path.join(BACKUP_DIR, f"agent_core.{int(time.time())}.py")
        shutil.copy2(CORE_FILE, backup)
        print(f"💾 Backup: {os.path.basename(backup)}")

    # 5. Update progress.json rate limit counter BEFORE writing new code
    try:
        prog = json.loads(open(PROGRESS_FILE).read()) if os.path.exists(PROGRESS_FILE) else {}
    except Exception:
        prog = {}
    prog["rewrites_this_session"] = rewrites_this_session + 1
    prog["last_rewrite_reason"] = reason
    prog["last_rewrite_at"] = time.ctime()
    prog["updated_at"] = time.ctime()
    json.dump(prog, open(PROGRESS_FILE, "w"), indent=2)

    # 6. Write new code
    with open(CORE_FILE, "w") as f:
        f.write(new_code)

    # 7. Log
    with open(EVOLUTION_LOG, "a") as f:
        f.write(f"[{time.ctime()}] CODE REWRITE: {reason}\n")
        f.write(f"  Backup: {os.path.basename(backup)}\n")
        f.write(f"  New size: {len(new_code)} chars\n")

    print(f"\n{'='*70}")
    print(f"🔁 SELF-REWRITE TRIGGERED")
    print(f"   Reason: {reason}")
    print(f"   New code size: {len(new_code)} chars")
    print(f"   Restarting via os.execv...")
    print(f"{'='*70}\n")

    # 8. Flush and execv (UPDATE 8)
    sys.stdout.flush()
    sys.stderr.flush()

    # Small delay to let print finish
    time.sleep(1)

    os.execv(sys.executable, [sys.executable, CORE_FILE])

# ═══════════════════════════════════════════════════════════════════════
# LOAD EXISTING PATCHES
# ═══════════════════════════════════════════════════════════════════════
BASE_TOOLS = ["read_file", "write_file", "run_python", "web_search",
              "fetch_url", "memory_store", "memory_recall",
              "create_patch", "list_patches",
              "save_interpretation", "save_research", "save_strategy",
              "update_progress", "self_modify_prompt", "self_rewrite"]

TOOLS = {
    "read_file": read_file, "write_file": write_file, "run_python": run_python,
    "web_search": web_search, "fetch_url": fetch_url,
    "memory_store": memory_store, "memory_recall": memory_recall,
    "create_patch": create_patch, "list_patches": list_patches,
    "save_interpretation": save_interpretation,
    "save_research": save_research,
    "save_strategy": save_strategy,
    "update_progress": update_progress,
    "self_modify_prompt": self_modify_prompt,
    "self_rewrite": self_rewrite,
}

def load_patches():
    reg_path = os.path.join(PATCHES_DIR, "registry.json")
    if not os.path.exists(reg_path): return 0
    try: reg = json.load(open(reg_path))
    except Exception: return 0
    loaded = 0
    for name, info in reg.items():
        fp = os.path.join(PATCHES_DIR, info["file"])
        if not os.path.exists(fp): continue
        try:
            spec = importlib.util.spec_from_file_location(name, fp)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            for attr in dir(mod):
                if not attr.startswith("_") and callable(getattr(mod, attr)):
                    TOOLS[attr] = getattr(mod, attr)
            loaded += 1
        except Exception as e:
            print(f"   ❌ {name}: {e}")
    return loaded

n_loaded = load_patches()
print(f"✅ Loaded {n_loaded} patches\n")

# ═══════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT (dynamic — can be modified by model)
# ═══════════════════════════════════════════════════════════════════════
SYSTEM_PROMPT_FILE = f"{AGENT_DIR}/system_prompt.txt"

SYSTEM_PROMPT_TEMPLATE = """You are an AUTONOMOUS SELF-EVOLVING AGENT.
Your GOAL (from goal.txt, read-only, hash-verified every step): {goal}

═══════════════════════════════════════════════════════════════════════
YOUR ARCHITECTURE — 15 TOOLS
═══════════════════════════════════════════════════════════════════════
CORE (7):
  read_file, write_file, run_python, web_search, fetch_url,
  memory_store, memory_recall

MEMORY FILES (4) — structured memory that survives code rewrites:
  save_interpretation(win_meaning, success_criteria, constraints, notes)
      → saves what "winning" means
  save_research(topic, findings, sources, key_insights)
      → appends findings (grows over time)
  save_strategy(markdown_content)
      → saves/updates strategy.md
  update_progress(current_phase, completed, in_progress, blockers, next_steps)
      → saves structured progress

SELF-MODIFICATION (2):
  self_modify_prompt(new_prompt, reason)
      → changes YOUR system prompt (backup kept)
  self_rewrite(new_code, reason)
      → rewrites YOUR OWN CODE (agent_core.py) and restarts
      → requires: AST-safe + reads goal.txt & 4 memory files
      → rate limit: max 3 per session
      → backup + save state + os.execv restart

PATCHES (2):
  create_patch(name, code, description) → build new tool
  list_patches()                         → list your patches

═══════════════════════════════════════════════════════════════════════
{built_tools_block}

═══════════════════════════════════════════════════════════════════════
YOUR PERSISTENT MEMORY
═══════════════════════════════════════════════════════════════════════
Current interpretation.json:
{interpretation}

Research entries: {research_count}
Strategy.md: {strategy_len} chars
Progress phase: {progress_phase}

═══════════════════════════════════════════════════════════════════════
YOUR SELF-MODIFICATION PHILOSOPHY
═══════════════════════════════════════════════════════════════════════
Your code runs as: agent_core.py
Your memory lives in: goal.txt + 4 memory files + chroma_db + patches/
If you rewrite your code, your MEMORY SURVIVES because it's in files.

HOW TO REWRITE YOURSELF (self_rewrite):
  1. Read agent_source.py to understand current architecture
  2. Read goal.txt + interpretation.json + research.json + strategy.md + progress.json
  3. Write new code that:
     • Contains all mandatory reads (goal.txt, 4 memory files)
     • Calls verify_goal_integrity() every loop iteration
     • Implements your improved architecture
  4. Call self_rewrite(new_code, reason)
  5. System will:
     • AST check
     • Verify mandatory reads are present
     • Check rate limit (max 3/session)
     • Backup current code
     • Save state
     • os.execv → restart with new code
  6. After restart: your memory is re-loaded from files
     → you continue with NEW code + OLD knowledge

WHY THIS DESIGN:
- Code change = memory loss (in-RAM) → solve via files
- Files structure given by SYSTEM → model fills content
- Goal protected by chmod 444 + hash check
- Rate limit prevents infinite rewrite loops

═══════════════════════════════════════════════════════════════════════
YOUR REASONING (DeepSeek-style)
═══════════════════════════════════════════════════════════════════════
Before every action, use <think>...</think>:
  - What have I done? (check [YOUR HISTORY])
  - What is the next atomic sub-goal?
  - Which tool is optimal?
  - Do I need to save research/strategy/progress?
  - Do I need to build a patch OR modify my own code?

RULES:
1. FIRST: read_file("agent_source.py") — know thyself
2. SECOND: read_file("goal.txt") — remember the mission
3. Save findings with save_research, not just memory_store
4. Update progress regularly with update_progress
5. NEVER modify goal.txt (chmod 444 + hash-verified)
6. Every ~10 steps, ask: "should I self-rewrite for better architecture?"
7. Use self_rewrite ONLY when current architecture blocks progress
"""

# Load or init system prompt
if os.path.exists(SYSTEM_PROMPT_FILE):
    CURRENT_SYSTEM_PROMPT = open(SYSTEM_PROMPT_FILE).read()
    print("📖 Loaded previous system prompt")
else:
    CURRENT_SYSTEM_PROMPT = None

def build_system_prompt():
    global CURRENT_SYSTEM_PROMPT
    custom = [t for t in TOOLS.keys() if t not in BASE_TOOLS]
    built_block = ("[BUILT-IN TOOLS + PATCHES]\n"
                   + "\n".join(f"  • {t}" for t in custom)) if custom else "[No custom tools yet]"
    # Read current memory state
    try:
        interp = json.loads(safe_read(INTERPRETATION_FILE, "{}"))
        interp_summary = json.dumps(interp, indent=2)[:800] if interp else "(empty)"
    except Exception:
        interp_summary = "(empty)"
    try:
        res = json.loads(safe_read(RESEARCH_FILE, "{}"))
        res_count = res.get("total_entries", 0)
    except Exception:
        res_count = 0
    strat_len = len(safe_read(STRATEGY_FILE, ""))
    try:
        prog = json.loads(safe_read(PROGRESS_FILE, "{}"))
        prog_phase = prog.get("current_phase", "(not set)")
    except Exception:
        prog_phase = "(not set)"

    if CURRENT_SYSTEM_PROMPT:
        # Model has modified its prompt — use that + append dynamic state
        return CURRENT_SYSTEM_PROMPT + f"\n\n[CURRENT STATE]\ninterpretation: {interp_summary}\nresearch_entries: {res_count}\nstrategy_len: {strat_len}\nprogress_phase: {prog_phase}\nbuilt_tools: {custom}"

    return SYSTEM_PROMPT_TEMPLATE.format(
        goal=GOAL,
        built_tools_block=built_block,
        interpretation=interp_summary,
        research_count=res_count,
        strategy_len=strat_len,
        progress_phase=prog_phase,
    )

# ═══════════════════════════════════════════════════════════════════════
# AGENT SOURCE FILE
# ═══════════════════════════════════════════════════════════════════════
def _write_agent_source():
    if os.path.exists(SOURCE_FILE): return
    import inspect
    parts = [f"# AGENT SOURCE — real running code\n# Model: {MODEL_ID}\n# Goal hash: {GOAL_HASH}\n\n"]
    for fn in [verify_goal_integrity, save_research, save_interpretation,
               save_strategy, update_progress, self_modify_prompt,
               self_rewrite, create_patch, memory_store, web_search]:
        try: parts.append(f"\n# === {fn.__name__} ===\n{inspect.getsource(fn)}")
        except Exception as e: parts.append(f"\n# {fn.__name__}: {e}")
    with open(SOURCE_FILE, "w") as f: f.write("\n".join(parts))

try: _write_agent_source()
except Exception: pass

# ═══════════════════════════════════════════════════════════════════════
# TOOL SCHEMAS (OpenAI format for server-side validation)
# ═══════════════════════════════════════════════════════════════════════
def S(name, desc, props, req):
    return {"type":"function","function":{"name":name,"description":desc,
            "parameters":{"type":"object","properties":props,
                          "required":req,"additionalProperties":False}}}

TOOL_SCHEMAS = [
    S("read_file", "Read a file", {"path":{"type":"string"}}, ["path"]),
    S("write_file", "Write regular files", {"path":{"type":"string"},"content":{"type":"string"}}, ["path","content"]),
    S("run_python", "Execute Python", {"code":{"type":"string"}}, ["code"]),
    S("web_search", "DuckDuckGo search", {"query":{"type":"string"}}, ["query"]),
    S("fetch_url", "Fetch & clean web page", {"url":{"type":"string"}}, ["url"]),
    S("memory_store", "Save to vector memory", {"key":{"type":"string"},"content":{"type":"string"}}, ["key","content"]),
    S("memory_recall", "Semantic recall", {"query":{"type":"string"},"n":{"type":"integer","default":5}}, ["query"]),
    S("save_interpretation", "Save what WINNING means",
      {"win_meaning":{"type":"string"},"success_criteria":{"type":"string"},
       "constraints":{"type":"string"},"notes":{"type":"string","default":""}},
      ["win_meaning","success_criteria","constraints"]),
    S("save_research", "Append research findings",
      {"topic":{"type":"string"},"findings":{"type":"string"},
       "sources":{"type":"array","items":{"type":"string"},"default":[]},
       "key_insights":{"type":"array","items":{"type":"string"},"default":[]}},
      ["topic","findings"]),
    S("save_strategy", "Save strategy.md", {"markdown_content":{"type":"string"}}, ["markdown_content"]),
    S("update_progress", "Update progress.json",
      {"current_phase":{"type":"string"},
       "completed":{"type":"array","items":{"type":"string"}},
       "in_progress":{"type":"array","items":{"type":"string"}},
       "blockers":{"type":"array","items":{"type":"string"}},
       "next_steps":{"type":"array","items":{"type":"string"}}},
      ["current_phase","completed","in_progress","blockers","next_steps"]),
    S("self_modify_prompt", "Change your own system prompt",
      {"new_prompt":{"type":"string"},"reason":{"type":"string"}},
      ["new_prompt","reason"]),
    S("self_rewrite", "Rewrite your own code + restart (max 3/session)",
      {"new_code":{"type":"string"},"reason":{"type":"string"}},
      ["new_code","reason"]),
    S("create_patch", "Build new tool in patches/",
      {"name":{"type":"string"},"code":{"type":"string"},
       "description":{"type":"string","default":""}},
      ["name","code"]),
    S("list_patches", "List patches", {}, []),
    S("DONE", "Declare goal complete",
      {"answer":{"type":"string","minLength":100},
       "proof_files":{"type":"array","items":{"type":"string"},"minItems":1}},
      ["answer","proof_files"]),
]

def ask_model(messages, max_new_tokens=MAX_OUTPUT_TOKENS, temperature=0.4,
              use_tools=True):
    try:
        kwargs = {"model": MODEL_ID, "messages": messages,
                  "max_tokens": max_new_tokens, "temperature": temperature,
                  "top_p": 0.95,
                  "extra_body": {"chat_template_kwargs": {"enable_thinking": True},
                                 "repetition_penalty": 1.05}}
        if use_tools:
            kwargs["tools"] = TOOL_SCHEMAS
            kwargs["tool_choice"] = "auto"
        response = client.chat.completions.create(**kwargs)
        msg = response.choices[0].message
        tcs = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                try: args = json.loads(tc.function.arguments)
                except json.JSONDecodeError as e:
                    args = {"__parse_error__": str(e),
                            "__raw__": tc.function.arguments[:500]}
                tcs.append({"id": tc.id, "name": tc.function.name, "arguments": args})
        return {"content": msg.content or "", "tool_calls": tcs,
                "reasoning": getattr(msg, "reasoning_content", None),
                "raw_message": msg}
    except Exception as e:
        return {"content": f"[API error: {type(e).__name__}: {e}]",
                "tool_calls": [], "reasoning": None, "raw_message": None}

def verify_goal_completion(args, action_history):
    checks = {}
    proof_files = args.get("proof_files", [])
    answer = args.get("answer", "")
    for pf in proof_files:
        try:
            path = _abs(pf)
            exists = os.path.exists(path)
            size = os.path.getsize(path) if exists else 0
            checks[f"file:{pf}"] = exists and size > 500
        except Exception:
            checks[f"file:{pf}"] = False
    checks["answer_length"] = len(answer) >= 500
    checks["answer_diversity"] = (len(set(answer.split())) / max(1, len(answer.split()))) > 0.3
    tools_used = [a[1] for a in action_history]
    checks["min_steps"] = len(action_history) >= 15
    checks["web_research"] = "web_search" in tools_used or "fetch_url" in tools_used
    checks["memory_used"] = "memory_store" in tools_used or "save_research" in tools_used
    # LLM judge
    jp = f"GOAL: {GOAL}\n\nANSWER: {answer[:2000]}\nPROOF: {proof_files}\n\nDoes this ACTUALLY achieve the goal? YES/NO + reason."
    vs = []
    for i in range(3):
        try:
            v = ask_model([{"role":"user","content":jp}], max_new_tokens=200,
                          temperature=0.2+i*0.15, use_tools=False)["content"]
            vs.append(v.strip().upper().startswith("YES"))
        except Exception:
            vs.append(False)
    checks["llm_judge"] = sum(vs) >= 2
    failed = [k for k,v in checks.items() if not v]
    return (len(failed) == 0, failed)

def compress_context(messages, keep_last=10):
    if len(messages) < 25: return messages
    old = messages[1:-keep_last]
    if len(old) < 8: return messages
    old_text = "\n".join([f"[{m.get('role','?')}] {str(m.get('content',''))[:400]}" for m in old])
    prompt = ("Summarize agent history. Preserve: goal progress, decisions, "
              "key findings, blockers, next steps.\n\n"
              f"TOOLS: {list(TOOLS.keys())}\n\nHISTORY:\n{old_text[:12000]}")
    try:
        summary = ask_model([{"role":"user","content":prompt}], max_new_tokens=2000,
                            temperature=0.2, use_tools=False)["content"]
    except Exception as e:
        summary = f"[Compression failed: {e}]"
    return [messages[0],
            {"role":"user","content":f"[COMPRESSED]\n{summary}"},
            *messages[-keep_last:]]

def is_progress(tool_name, result_str):
    if "❌" in result_str[:100]: return False
    if tool_name in ("web_search", "fetch_url") and len(result_str) > 500: return True
    if tool_name in ("create_patch","write_file","memory_store","save_research",
                     "save_interpretation","save_strategy","update_progress"):
        return "✅" in result_str
    if tool_name == "run_python":
        return '"rc": 0' in result_str
    return False

metrics_log = []
error_log = []

def get_time_context():
    e = time.time() - SESSION_START
    r = SESSION_LIMIT - e
    p = 100 * e / SESSION_LIMIT
    warn = ""
    if p > 90: warn = "🚨 WRAP UP NOW."
    elif p > 75: warn = "⚠️ HURRY."
    return f"⏰ {e/3600:.2f}h/{SESSION_LIMIT/3600:.1f}h ({p:.0f}%) | {r/3600:.2f}h left {warn}"

def save_state(step, messages, extra=None):
    st = {"step": step, "messages": messages[-40:], "saved_at": time.time(),
          "goal_hash": GOAL_HASH, **(extra or {})}
    try:
        tmp = STATE_FILE + ".tmp"
        json.dump(st, open(tmp, "w"), ensure_ascii=False, indent=2)
        os.replace(tmp, STATE_FILE)
    except Exception as e:
        print(f"⚠️ save_state: {e}")

def load_state():
    if not os.path.exists(STATE_FILE): return None
    try:
        st = json.load(open(STATE_FILE))
        if time.time() - st.get("saved_at", 0) > 7*24*3600: return None
        if st.get("goal_hash") != GOAL_HASH: return None
        return st
    except: return None

def trace(step, et, data):
    try:
        with open(TRACE_FILE, "a") as f:
            f.write(json.dumps({"step": step, "t": round(time.time()-SESSION_START,1),
                                "type": et, **data}, ensure_ascii=False, default=str) + "\n")
    except: pass

# ═══════════════════════════════════════════════════════════════════════
# INIT MESSAGES
# ═══════════════════════════════════════════════════════════════════════
state = load_state()
if state:
    print(f"🔄 RESUMING from step {state['step']}")
    messages = state["messages"]; step = state["step"]
    messages[0] = {"role": "system", "content": build_system_prompt()}
else:
    print("🆕 FRESH START")
    messages = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content":
         "Begin. Step 1: read_file('agent_source.py') — know yourself. "
         "Step 2: read_file('goal.txt') + 4 memory files. "
         "Step 3: web_search. Then start."},
    ]
    step = 0

# ═══════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "═"*80)
print("🚀 AUTONOMOUS LOOP — TPU v5e-8 | 15 tools | self-modifying")
print("═"*80)

action_history = deque(maxlen=20)
recent_tools = deque(maxlen=3)
last_progress = 0
recent_errors = 0
goal_finished = False

ERROR_HINTS = {"TypeError":"Check arg names.","KeyError":"Missing arg.",
               "ValueError":"Invalid value.","TimeoutExpired":"Break task.",
               "FileNotFoundError":"Check path.","ConnectionError":"Network."}

while not goal_finished:
    if time.time() - SESSION_START > SESSION_LIMIT:
        save_state(step, messages, extra={"exit_reason":"time_limit"})
        print(f"\n⏰ Time limit. State saved.")
        break

    # UPDATE 9: verify goal integrity every 5 steps
    if step > 0 and step % 5 == 0:
        if not verify_goal_integrity():
            print("🚨 GOAL TAMPERED — restored from backup")
            save_state(step, messages, extra={"exit_reason":"goal_tampered"})
            break

    messages[0] = {"role": "system", "content": build_system_prompt()}

    # Dynamic memory recall
    if step > 0 and action_history:
        last = action_history[-1]
        rel = memory_recall(f"{last[1]}: {last[2]}", n=2)
        if "No matches" not in rel and "empty" not in rel.lower():
            messages.append({"role":"user","content":f"[RECALL]\n{rel}"})

    if action_history:
        recap = "═══ HISTORY ═══\n"
        for s, a, ar, rs in list(action_history)[-8:]:
            recap += f"  {s}: {a}({ar}) → {rs}\n"
        messages.append({"role":"user","content":recap})

    if step % 5 == 0:
        messages.append({"role":"user","content":get_time_context()})

    if step > 0 and step % 5 == 0:
        est = sum(len(str(m.get("content",""))) for m in messages) // 4
        pct = 100 * est / NATURAL_CONTEXT_LIMIT
        if pct > 50:
            print(f"   🗜️ Compress {pct:.0f}%")
            messages = compress_context(messages, keep_last=15)

    temp = 0.1 if recent_errors >= 3 else (0.7 if step < 10 else 0.4)

    print(f"\n[Step {step}] 🧠")
    result = ask_model(messages, max_new_tokens=MAX_OUTPUT_TOKENS, temperature=temp)

    if result["content"].startswith("[API error"):
        print(f"   ❌ {result['content']}")
        recent_errors += 1; time.sleep(5); step += 1; continue

    if result["tool_calls"]:
        step_tools = [tc["name"] for tc in result["tool_calls"]]
        if len(step_tools) == 1:
            recent_tools.append(step_tools[0])
            if len(recent_tools) == 3 and len(set(recent_tools)) == 1:
                print(f"   🚨 LOOP: {step_tools[0]}")
                messages.append({"role":"user",
                    "content":"🚨 LOOP. Different tool OR create_patch OR self_rewrite."})
                recent_tools.clear(); recent_errors += 1; step += 1; continue
        else:
            recent_tools.clear()

        messages.append(result["raw_message"])
        sorted_calls = sorted(result["tool_calls"], key=lambda tc: tc["name"]=="DONE")

        for tc in sorted_calls:
            tn, ta, tid = tc["name"], tc["arguments"], tc["id"]
            print(f"   ⚙️ {tn}({str(ta)[:100]})")

            if "__parse_error__" in ta:
                messages.append({"role":"tool","tool_call_id":tid,
                    "content":f"❌ JSON: {ta['__parse_error__']}"})
                recent_errors += 1; continue

            if tn == "DONE":
                passed, failed = verify_goal_completion(ta, list(action_history))
                if passed:
                    with open(f"{AGENT_DIR}/CONCLUSION.md","w") as f:
                        f.write(ta.get("answer",""))
                    messages.append({"role":"tool","tool_call_id":tid,
                        "content":"✅ Verified."})
                    print(f"\n🎯 GOAL at step {step}")
                    save_state(step, messages, extra={"exit_reason":"done"})
                    goal_finished = True; break
                else:
                    print(f"   ❌ Rejected: {failed}")
                    messages.append({"role":"tool","tool_call_id":tid,
                        "content":f"❌ DONE rejected. Failed: {failed}"})
                    recent_errors += 1; continue

            try:
                tool_result = TOOLS[tn](**ta)
                rs = (json.dumps(tool_result, default=str) if isinstance(tool_result, dict)
                      else str(tool_result))[:4000]
                messages.append({"role":"tool","tool_call_id":tid,"content":rs})
                action_history.append((step, tn, str(ta)[:60], rs[:80].replace("\n"," ")))
                print(f"   ✅ {rs[:150]}")
                if "❌" in rs[:100]: recent_errors += 1
                else:
                    recent_errors = 0
                    if is_progress(tn, rs): last_progress = step
                if tn in ("web_search","fetch_url") and len(rs) > 500:
                    memory_store(key=f"{tn}_r", content=rs[:1000])
                trace(step, "ok", {"tool": tn})
            except Exception as e:
                et = type(e).__name__
                hint = ERROR_HINTS.get(et, "Try different.")
                print(f"   ❌ {et}: {e}")
                messages.append({"role":"tool","tool_call_id":tid,
                    "content":f"❌ {et}: {e}\n💡 {hint}"})
                error_log.append({"step": step, "type": et})
                recent_errors += 1

    elif result["content"]:
        print(f"   💭 {result['content'][:200]}")
        messages.append({"role":"assistant","content":result["content"]})
        messages.append({"role":"user","content":"⚠️ Call ONE tool."})

    if recent_errors >= 3 and error_log:
        recent_err_types = [e["type"] for e in error_log[-5:]]
        if recent_err_types:
            mc = Counter(recent_err_types).most_common(1)[0][0]
            if mc == "TypeError":
                messages.append({"role":"user","content":"💡 Check arg names."})

    if step - last_progress >= 8 and step > 8:
        print(f"   ⚠️ No progress — reflect")
        messages.append({"role":"user",
            "content":"⚠️ NO PROGRESS. STOP:\n"
                       "1. Blocking?\n"
                       "2. Need self_rewrite for better architecture?\n"
                       "3. Save research/strategy/progress?\n"
                       "4. Try different approach."})
        last_progress = step

    if step > 0 and step % 10 == 0:
        succ = sum(1 for m in metrics_log if m) if metrics_log else 0
        custom = [t for t in TOOLS.keys() if t not in BASE_TOOLS]
        print(f"   ┌─ Step {step} | {step/3600*0:.2f}h | custom tools: {len(custom)}")
    if step % 3 == 0:
        save_state(step, messages)

    step += 1

print(f"\n🏁 Finished at step {step}")
custom = [t for t in TOOLS.keys() if t not in BASE_TOOLS]
print(f"   Custom tools built: {len(custom)}")
print(f"   Files: {sorted(os.listdir(AGENT_DIR))}")
print("✅ DONE")
'''

# Write the agent code to disk
with open(CORE_FILE, "w") as f:
    f.write(CORE_CODE)
print(f"✅ Wrote agent_core.py ({len(CORE_CODE)} chars)")

# ── SECTION 3: START vLLM SERVER ───────────────────────────────────────
MODEL_ID = "Goekdeniz-Guelmez/Josiefied-DeepSeek-R1-0528-Qwen3-8B-abliterated-v1"

server_cmd = [
    sys.executable, "-m", "vllm.entrypoints.openai.api_server",
    "--model", MODEL_ID, "--tensor-parallel-size", "8",
    "--dtype", "bfloat16", "--max-model-len", "131072",
    "--enable-auto-tool-choice", "--tool-call-parser", "hermes",
    "--reasoning-parser", "qwen3", "--enable-prefix-caching",
    "--port", "8000", "--host", "0.0.0.0", "--disable-log-requests",
    "--gpu-memory-utilization", "0.92", "--max-num-seqs", "16",
    "--trust-remote-code",
]

server_log = open(f"{AGENT_DIR}/vllm_server.log", "w")
server_proc = subprocess.Popen(server_cmd, stdout=server_log, stderr=subprocess.STDOUT)
print(f"\n🚀 vLLM server (PID {server_proc.pid})...")

import requests as _req
for i in range(180):
    time.sleep(10)
    try:
        if _req.get("http://localhost:8000/health", timeout=5).status_code == 200:
            print(f"✅ Server READY ({i*10}s)\n")
            break
    except: pass
    if (i+1) % 6 == 0: print(f"   loading ({i*10}s)")
else:
    print(open(f"{AGENT_DIR}/vllm_server.log").read()[-3000:])
    raise SystemExit("Server failed")

# ── SECTION 4: RUN AGENT (with restart support) ────────────────────────
os.environ["AGENT_DIR"] = AGENT_DIR
os.environ["AGENT_TIME_LIMIT"] = "8.5"

print("=" * 80)
print("🚀 Spawning agent process (self-modification enabled)")
print("=" * 80 + "\n")

# This subprocess will os.execv-restart itself on self_rewrite.
# Notebook cell waits until the FINAL process exits.
proc = subprocess.Popen([sys.executable, CORE_FILE], env=os.environ.copy())
proc.wait()

print("\n" + "=" * 80)
print("✅ Agent process exited")
print("=" * 80)