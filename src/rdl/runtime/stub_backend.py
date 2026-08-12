"""A ``StubLM``-backed generation backend. The CPU path.

Every graph, defence, memory and evidence mechanism is developed and green-lighted
against this before a GPU is rented. The rule from the two-agent work still holds: if a
module cannot be exercised with the stub, it is coupled wrong.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ..models.stub import GenerationRequest, StubLM
from .backend import GenRequest, GenResponse

__all__ = ["StubBackend"]


class StubBackend:
    """One shared ``StubLM`` per model profile, exactly like the GPU handle sharing."""

    name = "stub"

    def __init__(
        self,
        handles: Mapping[str, StubLM] | None = None,
        *,
        default: StubLM | None = None,
    ) -> None:
        self._handles: dict[str, StubLM] = dict(handles or {})
        if default is not None:
            self._handles.setdefault("primary", default)
        if not self._handles:
            raise ValueError("StubBackend needs at least one model handle")
        self.calls: list[GenRequest] = []

    def _handle(self, model: str) -> StubLM:
        try:
            return self._handles[model]
        except KeyError as exc:
            raise KeyError(
                f"no stub handle for model profile '{model}'; have {sorted(self._handles)}"
            ) from exc

    def generate(self, requests: Sequence[GenRequest]) -> list[GenResponse]:
        out: list[GenResponse] = []
        for request in requests:
            self.calls.append(request)
            handle = self._handle(request.model)
            text = handle.generate(
                request.prompt,
                max_new_tokens=request.max_new_tokens,
                system=request.system,
                request=GenerationRequest(
                    do_sample=request.do_sample,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    top_k=request.top_k,
                    seed=request.seed,
                ),
            )
            out.append(
                GenResponse(
                    request_id=request.request_id,
                    text=text,
                    backend=self.name,
                    model_revision=self.model_revision(request.model),
                    prompt_tokens=len(request.prompt.split()),
                    completion_tokens=len(text.split()),
                )
            )
        return out

    def model_revision(self, model: str) -> str | None:
        return f"stub:{self._handle(model).model_id}"

    def shared_handle_ids(self) -> dict[str, int]:
        """Which logical profiles share one physical object. Recorded in the manifest."""
        return {name: id(handle) for name, handle in sorted(self._handles.items())}

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
