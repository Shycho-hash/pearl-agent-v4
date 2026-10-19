# PEARL Agent v4.0: Self-Modifying Autonomous Agent

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Track: Gemma 4 Developer Agent](https://img.shields.io/badge/Kaggle-Paper%20Track-orange)](https://www.kaggle.com/competitions/gemma-4-developer-agent-paper)

**Author:** Salluri Shekar | **GitHub:** [@Shycho-hash](https://github.com/Shycho-hash) |

---
## 📌 Research Question
> *Can an AI agent continuously modify its own codebase without losing accumulated knowledge, while remaining strictly aligned with its long-term goal?*

To address this, I designed **PEARL Agent v4.0** (Persistent Evolution through Autonomous Rewriting and Learning). It demonstrates that an agent can rewrite its own executable code, restart, and resume work — **without losing memory and without drifting from its goal**.

---

## 🧠 Core Principles
1. **Code ≠ Memory:** Code execution state may restart, but accumulated knowledge persists on disk in structured JSON/Markdown files.
2. **Immutable Goal:** The core objective is cryptographically and file-level locked (`chmod 444` + SHA256 hash verification).
3. **Controlled Self-Modification:** Self-rewrites are strictly bounded, AST-validated, and rate-limited to prevent execution loops.

---

## 🏗️ System Architecture & Safety Mechanisms
- **Architectural Separation:** Executable logic (`agent_core.py`) is separated from persistent state (4 structured memory files + ChromaDB vector store). When the agent rewrites its core logic and triggers `os.execv()`, the new process restores context from memory files—eliminating agent amnesia.
- **Goal Integrity Enforcement:** The target goal (`goal.txt`) is protected using read-only permissions, SHA256 hashing, and automated backup restoration.
- **Multi-Layer Safety:** 
  - AST parsing for prohibited imports/patterns.
  - Mandatory memory-read constraints.
  - Rate limit of ≤ 3 rewrites per session.
  - Multi-LLM consensus verification.

---

## 📊 Experimental Results (8.5-Hour Benchmark)

| Benchmark Metric | Baseline Agent | PEARL v4.0 | Delta (Δ) |
| :--- | :--- | :--- | :--- |
| Goal Completion Rate | 12% | 68% | **+56pp** |
| Memory Recovery Post-Restart | 0% | 100% | **Perfect** |
| Accumulated Research Entries | 3–5 | 47 | **~10×** |
| Custom Tools Generated (Patches) | 0 | 7 | **+7** |
| Unsafe Code Attempts Blocked | N/A | 23/23 | **100%** |
| Loop Incidents | 34 | 2 | **-94%** |

---

## 💻 Model Compatibility (Gemma 4)
This architecture is **model-agnostic**. While the baseline experiments were run using DeepSeek-R1, the tool and memory layers are completely decoupled from the LLM. 

To run PEARL with **Gemma 4**, simply change the `MODEL_ID` in `agent_core.py`:
```python
MODEL_ID = "google/gemma-4-it" # pearl-agent-v4
Self-modifying autonomous agent with persistent memory
