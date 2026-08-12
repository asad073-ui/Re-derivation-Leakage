"""Generation backends, batching, caching and resource accounting.

One physical model handle per unique checkpoint; five logical agents referencing it.
Loading five copies of a 1B checkpoint to run five agents is the mistake this package
exists to make impossible.
"""

from __future__ import annotations

from .backend import GenerationBackend, GenRequest, GenResponse, decoding_key
from .batch_scheduler import BatchScheduler
from .resource_monitor import ResourceMonitor
from .response_cache import ResponseCache, cache_key

__all__ = [
    "BatchScheduler",
    "GenRequest",
    "GenResponse",
    "GenerationBackend",
    "ResourceMonitor",
    "ResponseCache",
    "cache_key",
    "decoding_key",
]
