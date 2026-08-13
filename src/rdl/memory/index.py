"""Vector index backends and the embedder.

Two interchangeable backends behind one protocol:

  NumpyBruteForce   default. Exact, CPU, zero dependencies.
  FaissFlat         opt-in. IndexFlatIP — still EXACT, just faster.

**Both are exact by construction, and that is deliberate.** An approximate index (IVF,
HNSW) introduces retrieval noise into a leakage measurement - a forget-set item that
fails to surface would be indistinguishable from one the system successfully withheld.
Never swap in an ANN index without re-deriving what the metric means. Say so in the
paper.

The default embedder is a deterministic hashing embedder with no network and no torch,
which is what lets the entire memory/orchestrator/metric stack be developed and tested
on a laptop offline. Swap in a sentence-transformer via `build_index` on the GPU boxes.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np

__all__ = [
    "Embedder",
    "FaissFlat",
    "HashingEmbedder",
    "NumpyBruteForce",
    "VectorIndex",
    "build_index",
    "normalise_text",
]

# The em/en dashes here are DATA, not prose: TOFU answers contain both, and both must
# normalise to a space so that "Kuwait City—a port" and "Kuwait City - a port" compare
# equal. The dashes are now in `allowed-confusables` for the same reason the detector's
# fold table is (GU-0032), so the per-line noqa is no longer needed.
_PUNCT = str.maketrans(dict.fromkeys("\"'`.,;:!?()[]{}<>-—–/\\|*_#", " "))


def normalise_text(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace.

    Shared by the embedder, the stub LM's question lookup, and the containment metric's
    "normalised" mode so that all three agree on what "the same string" means.
    """
    return " ".join(text.lower().translate(_PUNCT).split())


@runtime_checkable
class Embedder(Protocol):
    dim: int

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return an (n, dim) float32 array of L2-normalised row vectors."""
        ...


class HashingEmbedder:
    """Deterministic bag-of-token hashing embedder. No network, no torch, no weights.

    Not a semantic model — it is a *stable* one. Two runs on two machines produce
    identical vectors, which is what a reproducibility gate needs from the retrieval
    layer. Token overlap drives similarity, so paraphrase retrieval works to the extent
    that paraphrases share tokens; `SemanticBlocklist` tests that path explicitly.
    """

    def __init__(self, dim: int = 64, seed: int = 42) -> None:
        if dim < 8:
            raise ValueError("embedding dim must be >= 8")
        self.dim = dim
        self.seed = seed

    def _token_vec(self, token: str) -> np.ndarray:
        digest = hashlib.blake2b(
            token.encode("utf-8"), digest_size=8, key=str(self.seed).encode()
        ).digest()
        rng = np.random.default_rng(int.from_bytes(digest, "little"))
        return rng.standard_normal(self.dim, dtype=np.float64)

    def encode_one(self, text: str) -> np.ndarray:
        tokens = normalise_text(text).split()
        vec = np.zeros(self.dim, dtype=np.float64)
        if not tokens:
            # Deterministic non-degenerate vector for the empty string.
            vec[0] = 1.0
            return vec.astype(np.float32)
        for tok in tokens:
            vec += self._token_vec(tok)
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec.astype(np.float32)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self.encode_one(t) for t in texts]).astype(np.float32)


@runtime_checkable
class VectorIndex(Protocol):
    dim: int

    def add(self, node_id: str, vector: np.ndarray) -> None: ...
    def remove(self, node_id: str) -> bool: ...
    def search(self, vector: np.ndarray, k: int) -> list[tuple[str, float]]: ...
    def ids(self) -> list[str]: ...
    def __len__(self) -> int: ...


class NumpyBruteForce:
    """Exact cosine search over an (n, dim) matrix. The default and the tested path."""

    backend_name = "numpy"

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim
        self._ids: list[str] = []
        self._pos: dict[str, int] = {}
        self._mat: np.ndarray = np.zeros((0, dim), dtype=np.float32)

    def add(self, node_id: str, vector: np.ndarray) -> None:
        # Annotated, not inferred. `asarray(...).reshape(-1)` infers the exact
        # (1-D, float32) type, and `v / norm` widens it back to (Any-D, floating[Any]) —
        # numpy's stubs cannot see that dividing by a Python float is dtype-preserving
        # under NEP 50. The narrow inferred type then rejects the re-assignment.
        v: np.ndarray = np.asarray(vector, dtype=np.float32).reshape(-1)
        if v.shape[0] != self.dim:
            raise ValueError(f"vector dim {v.shape[0]} != index dim {self.dim}")
        norm = float(np.linalg.norm(v))
        if norm > 0:
            v = v / norm
        if node_id in self._pos:
            self._mat[self._pos[node_id]] = v
            return
        self._pos[node_id] = len(self._ids)
        self._ids.append(node_id)
        self._mat = np.vstack([self._mat, v[None, :]]) if len(self._ids) > 1 else v[None, :].copy()

    def remove(self, node_id: str) -> bool:
        pos = self._pos.pop(node_id, None)
        if pos is None:
            return False
        self._ids.pop(pos)
        self._mat = np.delete(self._mat, pos, axis=0)
        # Positions after the removed row all shift down by one.
        for nid, p in self._pos.items():
            if p > pos:
                self._pos[nid] = p - 1
        return True

    def search(self, vector: np.ndarray, k: int) -> list[tuple[str, float]]:
        if len(self._ids) == 0 or k <= 0:
            return []
        q: np.ndarray = np.asarray(vector, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(q))
        if norm > 0:
            q = q / norm
        scores = self._mat @ q
        k = min(k, len(self._ids))
        # argsort over the negated scores gives a deterministic, stable ordering; ties
        # break by insertion order, which keeps runs byte-identical.
        order = np.argsort(-scores, kind="stable")[:k]
        return [(self._ids[int(i)], float(scores[int(i)])) for i in order]

    def ids(self) -> list[str]:
        return list(self._ids)

    def __len__(self) -> int:
        return len(self._ids)


class FaissFlat:
    """faiss.IndexFlatIP — exact inner-product search. Opt-in, never the tested default.

    Deliberately faiss-CPU only. The corpus is <= 10k nodes; faiss-gpu would be slower
    end-to-end and adds a CUDA-version failure mode for nothing.
    """

    backend_name = "faiss"

    def __init__(self, dim: int = 64) -> None:
        try:
            import faiss
        except ImportError as exc:  # pragma: no cover - opt-in path
            raise ImportError(
                "FaissFlat requires faiss-cpu: pip install faiss-cpu. "
                "The default NumpyBruteForce backend needs no extra dependency."
            ) from exc
        self._faiss = faiss
        self.dim = dim
        self._index = faiss.IndexFlatIP(dim)
        self._ids: list[str] = []
        self._vecs: dict[str, np.ndarray] = {}

    def _rebuild(self) -> None:
        self._index = self._faiss.IndexFlatIP(self.dim)
        if self._ids:
            self._index.add(np.stack([self._vecs[i] for i in self._ids]))

    def add(self, node_id: str, vector: np.ndarray) -> None:
        v: np.ndarray = np.asarray(vector, dtype=np.float32).reshape(-1)
        if v.shape[0] != self.dim:
            raise ValueError(f"vector dim {v.shape[0]} != index dim {self.dim}")
        norm = float(np.linalg.norm(v))
        if norm > 0:
            v = v / norm
        if node_id in self._vecs:
            self._vecs[node_id] = v
            self._rebuild()
            return
        self._vecs[node_id] = v
        self._ids.append(node_id)
        self._index.add(v[None, :])

    def remove(self, node_id: str) -> bool:
        if node_id not in self._vecs:
            return False
        self._vecs.pop(node_id)
        self._ids.remove(node_id)
        self._rebuild()
        return True

    def search(self, vector: np.ndarray, k: int) -> list[tuple[str, float]]:
        if not self._ids or k <= 0:
            return []
        q: np.ndarray = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        norm = float(np.linalg.norm(q))
        if norm > 0:
            q = q / norm
        k = min(k, len(self._ids))
        scores, idx = self._index.search(q, k)
        out: list[tuple[str, float]] = []
        for score, i in zip(scores[0], idx[0], strict=True):
            if i < 0:
                continue
            out.append((self._ids[int(i)], float(score)))
        return out

    def ids(self) -> list[str]:
        return list(self._ids)

    def __len__(self) -> int:
        return len(self._ids)


def build_index(backend: str = "numpy", dim: int = 64) -> VectorIndex:
    if backend == "numpy":
        return NumpyBruteForce(dim=dim)
    if backend == "faiss":
        return FaissFlat(dim=dim)
    raise ValueError(f"unknown index backend '{backend}' (expected 'numpy' or 'faiss')")
