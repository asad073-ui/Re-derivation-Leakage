"""Index backends and the embedder.

The exactness property matters more than the speed one: an approximate index would put
retrieval noise into a leakage measurement, where a forget-set item that failed to
surface would be indistinguishable from one the system successfully withheld.
"""

from __future__ import annotations

import contextlib

import numpy as np
import pytest

from rdl.memory.index import (
    FaissFlat,
    HashingEmbedder,
    NumpyBruteForce,
    build_index,
    normalise_text,
)

# ------------------------------------------------------------------ normalisation --


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Hello, World!", "hello world"),
        ("  multiple   spaces  ", "multiple spaces"),
        ("Basil's father—a florist.", "basil s father a florist"),
        ("", ""),
        ("UPPER lower MiXeD", "upper lower mixed"),
    ],
)
def test_normalise_text(raw, expected):
    assert normalise_text(raw) == expected


# ---------------------------------------------------------------------- embedder --


def test_embedder_is_deterministic():
    a, b = HashingEmbedder(dim=64), HashingEmbedder(dim=64)
    v1 = a.encode(["the same sentence"])
    v2 = b.encode(["the same sentence"])
    np.testing.assert_allclose(v1, v2)


def test_embedder_output_is_normalised():
    vecs = HashingEmbedder(dim=64).encode(["one", "two three", "a longer sentence here"])
    np.testing.assert_allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5)


def test_embedder_is_shift_invariant_in_case_and_punctuation():
    e = HashingEmbedder(dim=64)
    np.testing.assert_allclose(e.encode(["Hello, World!"]), e.encode(["hello world"]))


def test_embedder_similar_text_scores_higher_than_unrelated():
    e = HashingEmbedder(dim=128)
    q = e.encode(["Basil Mahfouz Al-Kuwaiti father florist"])[0]
    close = e.encode(["Basil Mahfouz Al-Kuwaiti's father was a florist"])[0]
    far = e.encode(["quantum chromodynamics binds quarks together"])[0]
    assert float(close @ q) > float(far @ q)


def test_embedder_handles_the_empty_string():
    v = HashingEmbedder(dim=32).encode([""])
    assert v.shape == (1, 32)
    assert np.isfinite(v).all()


def test_embedder_rejects_a_tiny_dim():
    with pytest.raises(ValueError):
        HashingEmbedder(dim=4)


def test_encode_empty_list():
    assert HashingEmbedder(dim=16).encode([]).shape == (0, 16)


# ------------------------------------------------------------------------ backend --


def _backends():
    out = [NumpyBruteForce(dim=32)]
    # faiss is opt-in; the numpy backend is the tested default.
    with contextlib.suppress(ImportError):
        out.append(FaissFlat(dim=32))
    return out


@pytest.mark.parametrize("index", _backends(), ids=lambda i: type(i).__name__)
def test_backend_add_search_remove(index):
    e = HashingEmbedder(dim=32)
    for name, text in [
        ("a", "florist in Kuwait City"),
        ("b", "quantum field theory"),
        ("c", "a florist shop"),
    ]:
        index.add(name, e.encode([text])[0])

    assert len(index) == 3
    assert set(index.ids()) == {"a", "b", "c"}

    hits = index.search(e.encode(["florist"])[0], k=2)
    assert len(hits) == 2
    assert "b" not in [h[0] for h in hits]

    assert index.remove("a") is True
    assert index.remove("a") is False
    assert len(index) == 2
    assert "a" not in index.ids()


@pytest.mark.parametrize("index", _backends(), ids=lambda i: type(i).__name__)
def test_backend_search_is_exact_and_ordered(index):
    e = HashingEmbedder(dim=32)
    texts = {f"n{i}": f"document number {i} about topic {i}" for i in range(6)}
    for nid, t in texts.items():
        index.add(nid, e.encode([t])[0])

    q = e.encode(["document number 3 about topic 3"])[0]
    hits = index.search(q, k=6)

    assert hits[0][0] == "n3", "exact search must rank the identical document first"
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.parametrize("index", _backends(), ids=lambda i: type(i).__name__)
def test_backend_empty_search(index):
    assert index.search(np.ones(32, dtype=np.float32), k=5) == []


@pytest.mark.parametrize("index", _backends(), ids=lambda i: type(i).__name__)
def test_backend_rejects_wrong_dim(index):
    with pytest.raises(ValueError, match="dim"):
        index.add("x", np.ones(8, dtype=np.float32))


def test_numpy_remove_keeps_positions_consistent():
    """Removing a middle row must not corrupt the id -> row mapping."""
    e = HashingEmbedder(dim=32)
    idx = NumpyBruteForce(dim=32)
    for name in "abcde":
        idx.add(name, e.encode([f"document {name}"])[0])

    idx.remove("c")

    for name in "abde":
        hits = idx.search(e.encode([f"document {name}"])[0], k=1)
        assert hits[0][0] == name, f"{name} resolves to the wrong row after removal"


def test_numpy_add_is_idempotent_on_the_same_id():
    e = HashingEmbedder(dim=32)
    idx = NumpyBruteForce(dim=32)
    idx.add("a", e.encode(["first"])[0])
    idx.add("a", e.encode(["second"])[0])
    assert len(idx) == 1


def test_build_index_dispatch():
    assert isinstance(build_index("numpy", dim=16), NumpyBruteForce)
    with pytest.raises(ValueError, match="unknown index backend"):
        build_index("annoy", dim=16)
