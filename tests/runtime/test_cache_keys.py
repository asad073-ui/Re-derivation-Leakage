"""Reuse only when everything that could change the output is identical."""

from __future__ import annotations

from dataclasses import replace

from rdl.runtime.backend import GenRequest, decoding_key
from rdl.runtime.response_cache import ResponseCache, cache_key

BASE = GenRequest(
    request_id="r0",
    prompt="the user turn",
    system="the system message",
    max_new_tokens=128,
    seed=7,
    temperature=1.0,
    top_p=1.0,
    top_k=0,
    model="primary",
)


def _key(request: GenRequest, **overrides) -> str:
    kwargs = {
        "model_revision": "repo@abc",
        "tokenizer_revision": "tok@def",
        "backend": "stub",
        "backend_version": "1.0",
        **overrides,
    }
    return cache_key(request, **kwargs)


def test_identical_requests_share_a_key():
    assert _key(BASE) == _key(replace(BASE, request_id="r1"))


def test_every_field_that_shapes_the_output_changes_the_key():
    for field, value in (
        ("prompt", "a different user turn"),
        ("system", "a different system message"),
        ("seed", 8),
        ("temperature", 0.7),
        ("top_p", 0.9),
        ("top_k", 50),
        ("max_new_tokens", 64),
        ("model", "secondary"),
    ):
        assert _key(BASE) != _key(replace(BASE, **{field: value})), field


def test_revision_and_backend_are_part_of_the_key():
    assert _key(BASE) != _key(BASE, model_revision="repo@zzz")
    assert _key(BASE) != _key(BASE, tokenizer_revision="tok@zzz")
    assert _key(BASE) != _key(BASE, backend="vllm")
    assert _key(BASE) != _key(BASE, backend_version="2.0")


def test_a_defence_that_rewrites_the_prompt_gets_a_new_key():
    guarded = replace(BASE, system=BASE.system + "\nGuard instruction.")
    assert _key(BASE) != _key(guarded)


def test_decoding_key_excludes_the_seed():
    """The seed is provenance for one draw; the decoding key is what arms must share."""
    assert decoding_key(BASE) == decoding_key(replace(BASE, seed=999))
    assert decoding_key(BASE) != decoding_key(replace(BASE, temperature=0.5))


def test_cache_marks_reused_entries():
    from rdl.runtime.backend import GenResponse

    cache = ResponseCache()
    assert cache.get("k") is None
    cache.put("k", GenResponse(request_id="r0", text="out", backend="stub"))
    hit = cache.get("k")
    assert hit is not None and hit.cached and hit.text == "out"
    assert cache.stats() == {"hits": 1, "misses": 1, "entries": 1}


def test_a_disabled_cache_never_serves():
    from rdl.runtime.backend import GenResponse

    cache = ResponseCache(enabled=False)
    cache.put("k", GenResponse(request_id="r0", text="out"))
    assert cache.get("k") is None
