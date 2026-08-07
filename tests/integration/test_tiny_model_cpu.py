"""The real `loader.py` path, on CPU, with a ~1 MB model.

NEEDS NETWORK — excluded from the CPU gate, run via `make test-integration`.

What this buys: it proves the loader, the chat template handling, the padding-side
logic, and the `LMHandle` interface all work *before* a Colab session is spent finding
out they do not. `hf-internal-testing/tiny-random-LlamaForCausalLM` is the same
architecture family as the TOFU checkpoints, so the code path is genuinely the same one;
only the weights are nonsense.

Its outputs are gibberish and that is fine — nothing here asserts anything about content.
"""

from __future__ import annotations

import pytest

from rdl.config import ModelConfig
from rdl.hardware import HardwareProfile, detect
from rdl.models.loader import (
    ChatTemplateMissingError,
    resolve_attn,
    resolve_dtype,
)

pytestmark = [pytest.mark.network, pytest.mark.slow]

TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"

transformers = pytest.importorskip("transformers", reason="loader path needs transformers")
torch = pytest.importorskip("torch", reason="loader path needs torch")


@pytest.fixture(scope="module")
def cpu_hw() -> HardwareProfile:
    hw = detect()
    if hw.is_cuda:
        pytest.skip("this test pins down the CPU path specifically")
    return hw


@pytest.fixture(scope="module")
def tiny_cfg() -> ModelConfig:
    # The tiny test model ships no chat template, which is legitimate for a raw-text
    # architecture smoke test. The TOFU checkpoints DO require one — see
    # test_chat_template_is_required_by_default below.
    return ModelConfig(
        name="tiny",
        kind="hf",
        repo_id=TINY,
        chat_template_required=False,
    )


@pytest.fixture(scope="module")
def tiny_lm(tiny_cfg, cpu_hw):
    from rdl.models.loader import load_lm

    lm = load_lm(tiny_cfg, cpu_hw)
    yield lm
    lm.close()


# --------------------------------------------------------------- dtype / attn --


def test_cpu_resolves_to_fp32_and_eager(tiny_cfg, cpu_hw):
    assert resolve_dtype(tiny_cfg, cpu_hw) == "float32"
    assert resolve_attn(tiny_cfg, cpu_hw) == "eager"


def test_bf16_override_is_refused_on_a_device_without_bf16(cpu_hw):
    cfg = ModelConfig(name="x", kind="hf", repo_id=TINY, dtype_override="bfloat16")
    with pytest.raises(ValueError, match="no bf16 support"):
        resolve_dtype(cfg, cpu_hw)


def test_flash_attention_override_is_refused_pre_ampere(cpu_hw):
    cfg = ModelConfig(name="x", kind="hf", repo_id=TINY, attn_override="flash_attention_2")
    with pytest.raises(ValueError, match="SM80"):
        resolve_attn(cfg, cpu_hw)


# ------------------------------------------------------------------ the load --


def test_model_loads_through_the_only_sanctioned_path(tiny_lm):
    assert tiny_lm.model_id == TINY
    assert tiny_lm.dtype == "float32"
    assert tiny_lm.attn == "eager"
    assert tiny_lm.tokenizer is not None


def test_pad_token_is_populated(tiny_lm):
    """Llama ships no pad token; eos is the conventional stand-in and is what
    open-unlearning uses. Inventing a new one would shift the embedding table."""
    assert tiny_lm.tokenizer.pad_token is not None


def test_chat_template_is_required_by_default(cpu_hw, monkeypatch):
    """The TOFU checkpoints are -Instruct derivatives. A missing template must fail
    loudly, because a mismatched prompt format looks exactly like an unlearning effect.

    The guard is exercised against a tokenizer we force to have no template, NOT against
    whatever `hf-internal-testing/tiny-random-LlamaForCausalLM` happens to ship. It used
    to rely on that repo lacking one; the repo has since gained a default template, so
    the test silently stopped testing anything and went red only because the raise no
    longer happened. Our guard is the thing under test — pin it to our code.
    """
    import transformers

    from rdl.models import loader as loader_mod

    real_from_pretrained = transformers.AutoTokenizer.from_pretrained

    def _no_template(*args, **kwargs):
        tok = real_from_pretrained(*args, **kwargs)
        tok.chat_template = None
        return tok

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", _no_template)

    cfg = ModelConfig(name="x", kind="hf", repo_id=TINY, chat_template_required=True)
    with pytest.raises(ChatTemplateMissingError, match="chat_template"):
        loader_mod.load_lm(cfg, cpu_hw)


# ------------------------------------------------------------- the interface --


def test_generate_returns_a_string(tiny_lm):
    out = tiny_lm.generate("Question: who?\nAnswer:", max_new_tokens=4)
    assert isinstance(out, str)


def test_generate_is_deterministic(tiny_lm):
    """Greedy, do_sample=False, batch of one. Same prompt, same bytes."""
    prompt = "Question: who?\nAnswer:"
    assert tiny_lm.generate(prompt, max_new_tokens=6) == tiny_lm.generate(prompt, max_new_tokens=6)


def test_generation_uses_left_padding(tiny_lm):
    tiny_lm.tokenizer.padding_side = "right"
    tiny_lm.generate("Question: who?\nAnswer:", max_new_tokens=2)
    assert tiny_lm.tokenizer.padding_side == "left", "generation must left-pad"


def test_scoring_uses_right_padding(tiny_lm):
    tiny_lm.tokenizer.padding_side = "left"
    tiny_lm.logprobs("Question: who?\nAnswer:", " a florist")
    assert tiny_lm.tokenizer.padding_side == "right", "scoring must right-pad"


def test_logprobs_shape_matches_the_continuation(tiny_lm):
    cont = " a florist in Kuwait City"
    n_tokens = tiny_lm.tokenizer(cont, add_special_tokens=False)["input_ids"]
    lp = tiny_lm.logprobs("Question: who?\nAnswer:", cont)
    assert lp.shape[0] == len(n_tokens)
    assert bool(torch.isfinite(lp).all())
    assert bool((lp <= 0).all()), "log-probabilities are non-positive"


def test_sequence_logprob_reduces_to_a_float(tiny_lm):
    value = tiny_lm.sequence_logprob("Question: who?\nAnswer:", " a florist")
    assert isinstance(value, float)
    assert value <= 0


def test_handle_works_as_a_context_manager(tiny_cfg, cpu_hw):
    from rdl.models.loader import load_lm

    with load_lm(tiny_cfg, cpu_hw) as lm:
        assert isinstance(lm.generate("hello", max_new_tokens=2), str)


def test_agent_runs_on_the_real_loader(tiny_lm, cpu_hw):
    """The whole point: the same LLMAgent code path serves StubLM and a real model."""
    from rdl.agents.llm_agent import LLMAgent
    from rdl.memory.store import MemoryStore
    from rdl.orchestrator.loop import EpisodePolicies, run_episode

    agent = LLMAgent("A", tiny_lm, max_new_tokens=4)
    store = MemoryStore(index_backend="numpy", embedding_dim=64)
    tr = run_episode("Who was the florist?", [agent], store, EpisodePolicies())

    assert [e.kind for e in tr] == ["user_query", "retrieval", "agent_answer", "final_answer"]
