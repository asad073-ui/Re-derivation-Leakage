"""Real Transformers sampling contract; excluded from the offline CPU gate."""

from __future__ import annotations

import hashlib

import pytest

from rdl.models.loader import HFLMHandle
from rdl.models.stub import GenerationRequest


@pytest.mark.network
def test_sampled_hf_generation_is_seeded_without_generator_model_kwarg() -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    model_id = "hf-internal-testing/tiny-random-gpt2"
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    model = transformers.AutoModelForCausalLM.from_pretrained(model_id).to("cpu").eval()
    handle = HFLMHandle(model, tokenizer, model_id=model_id, dtype="float32", attn="eager")
    prompt = "A short scientific answer is"
    same = GenerationRequest(do_sample=True, temperature=1.0, top_p=1.0, seed=91)
    assert handle.generate(
        prompt, max_new_tokens=12, apply_template=False, request=same
    ) == handle.generate(prompt, max_new_tokens=12, apply_template=False, request=same)
    provenance = handle.generation_provenance()
    assert provenance["semantic_user_prompt_sha256"] == hashlib.sha256(prompt.encode()).hexdigest()
    assert provenance["serialized_chat_prompt_sha256"] == provenance["semantic_user_prompt_sha256"]
    assert provenance["input_ids_sha256"] and len(provenance["input_ids_sha256"]) == 64
    # Sampling may collide on a very small model, but several independent seeds should
    # expose more than one continuation and, crucially, no unsupported generator kwarg.
    outputs = {
        handle.generate(
            prompt,
            max_new_tokens=12,
            apply_template=False,
            request=GenerationRequest(do_sample=True, temperature=1.0, top_p=1.0, seed=seed),
        )
        for seed in range(4)
    }
    assert len(outputs) > 1
    assert torch is not None
