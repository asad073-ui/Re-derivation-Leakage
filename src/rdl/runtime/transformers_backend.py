"""HuggingFace ``transformers`` backend: the reference path.

Slower than vLLM and kept anyway, for two reasons. It is the only backend whose
decoding path is the one the released two-agent Leak@k evidence was produced under, so
it is what a vLLM-vs-transformers equivalence audit compares against; and it is what
runs when a box has no vLLM wheel for its CUDA version.

It loops internally rather than batching. Real batching through ``generate`` changes
padding and therefore changes outputs, which would silently make this backend
non-comparable with the batch-1 compatibility runner.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ..models.stub import GenerationRequest, LMHandle
from .backend import GenRequest, GenResponse

__all__ = ["TransformersBackend"]


class TransformersBackend:
    name = "transformers"

    def __init__(
        self, handles: Mapping[str, LMHandle], *, revisions: Mapping[str, str] | None = None
    ) -> None:
        if not handles:
            raise ValueError("TransformersBackend needs at least one model handle")
        self._handles = dict(handles)
        self._revisions = dict(revisions or {})

    def _handle(self, model: str) -> LMHandle:
        try:
            return self._handles[model]
        except KeyError as exc:
            raise KeyError(
                f"no handle for model profile '{model}'; have {sorted(self._handles)}"
            ) from exc

    def generate(self, requests: Sequence[GenRequest]) -> list[GenResponse]:
        out: list[GenResponse] = []
        for request in requests:
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
        if model in self._revisions:
            return self._revisions[model]
        return getattr(self._handle(model), "model_id", None)

    def shared_handle_ids(self) -> dict[str, int]:
        return {name: id(handle) for name, handle in sorted(self._handles.items())}

    def close(self) -> None:
        seen: set[int] = set()
        for handle in self._handles.values():
            # Logical profiles share one physical handle; closing it twice is an error
            # on some backends and a no-op on others. Do it once.
            if id(handle) in seen:
                continue
            seen.add(id(handle))
            handle.close()
