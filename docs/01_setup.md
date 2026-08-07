# Setup

Four environments. Only the first is required before Colab.

---

## 1. Local CPU — the gate

No GPU, no network, no HF token. **This must be green before anything touches Colab.**

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[cpu,dev]"

make cpu-all        # Windows: .\tasks.ps1 cpu-all
```

`cpu-all` = ruff + black --check + mypy + unit tests + contract tests.

What it proves, and why each part is worth the seconds it costs:

| suite | proves |
|---|---|
| `tests/unit/test_invariants.py` | **the golden test.** A laundered node passes both SBU invariants while containing the forgotten fact. If this is red, the paper's central argument is wrong — and you find out on a laptop in week one. |
| `tests/unit/test_derivation_closure.py` | closure is transitive; a diamond DAG does not double-decrement |
| `tests/contract/test_condition_c3_stub.py` | C3 runs end to end before a GPU exists, and tells you whether C3 is even measurable |
| `tests/unit/test_containment.py` | the three matching modes agree on positives; the false-positive floor on retain strings is 0 |
| `tests/integration/test_ou_bridge_parse.py` | catches open-unlearning schema drift with no GPU |

Python 3.10–3.13. See ADR-0002 for why the range is wider than open-unlearning's 3.11.

### Optional: the tiny-model integration test

Needs network (~1 MB download). Proves the real loader, chat template, and padding logic
before Colab:

```bash
make test-integration
```

---

## 2. Colab T4 — eval only

Runtime → Change runtime type → **T4 GPU**.

Open `notebooks/colab_phase0_repro.ipynb` and run it top to bottom. It handles secrets,
the private clone, the submodule, and the eval.

**Three T4 facts, encoded rather than remembered:**

| fact | consequence | where it is handled |
|---|---|---|
| T4 is Turing (SM 7.5), no bf16 | eval runs in fp16 | `hardware.detect()` → `recommended_dtype` |
| FlashAttention-2 needs SM80+ | `attn_implementation=sdpa`; **skip the `flash-attn==2.6.3` install** the upstream README prescribes | `hardware.detect()` → `recommended_attn`; `scripts/01_bootstrap_openunlearning.sh` |
| GA/NPO in fp16 NaNs silently | **no training here, ever** | `hardware.assert_training_allowed` refuses |

### Secrets

Colab sidebar → 🔑 Secrets. Both need *Notebook access* enabled.

| name | value |
|---|---|
| `HF_TOKEN` | HF token from an account that has **accepted the Llama 3.2 community licence** |
| `GH_PAT` | GitHub **fine-grained** PAT, scoped to `Re-derivation-Leakage` only, `Contents: Read and write` |

Accept the Llama 3.2 licence **now** if you have not:
<https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct>. It is a hard blocker and you
do not want to hit it at hour six.

### Expect one runtime restart

Colab preinstalls `torch`/`transformers` versions that open-unlearning will move. The
notebook says where to restart and what to re-run.

---

## 3. RTX 3090 (Ampere, SM 8.6, 24 GB) — training permitted

```bash
pip install -r requirements-gpu-ampere.txt
pip install --no-build-isolation flash-attn==2.6.3    # ~20 min to build
```

bf16 makes gradient-ascent-family objectives numerically sound, which is the only reason
training lives here.

**Caveat for any training run:** 24 GB single-GPU cannot reproduce upstream's 2× L40s
ZeRO-3 effective-batch-32 setup, and upstream warns their numbers shift when the
distributed setup changes. A training run here produces a **new number, not a
reproduction** — record it as such in `04_decisions.md`.

---

## 4. H100 (Hopper, SM 9.0, 80 GB)

Same as the 3090, with `configs/env/h100.yaml`. 80 GB holds upstream's effective batch
without ZeRO-3 sharding across two cards, making this the only device where a training
reproduction is arguably comparable. Still not identical — say so in the write-up.

---

## 5. The submodule

```bash
make submodule                             # git submodule update --init --recursive
bash scripts/01_bootstrap_openunlearning.sh
```

The bootstrap script detects compute capability and skips `flash-attn` below SM80
automatically. It also runs `setup_data.py --eval_logs`, which downloads TOFU/MUSE data
**and** the published eval logs — including the retain-model logs required to compute
`forget_quality` at all.

Record the SHA it prints in `docs/02_repro_targets.md`.

---

## 6. Environment variables

Copy `.env.example` to `.env` (gitignored). On Colab these come from Colab Secrets, never
from a file.

```
HF_TOKEN=hf_...
HF_HOME=/content/hf          # keep the cache OFF Drive: Drive I/O is slow, repos are GBs
TOKENIZERS_PARALLELISM=false
```

---

## 7. Pre-commit

```bash
pre-commit install
```

`nbstripout` is the load-bearing hook: a Colab notebook that has actually run will
otherwise commit an HF token or a GitHub PAT inside a traceback or a cell output.
`gitleaks` is the second line of defence, and CI greps for credential patterns as a third.
