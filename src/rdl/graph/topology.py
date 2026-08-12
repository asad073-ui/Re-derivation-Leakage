"""Loading topologies from ``configs/graph/topologies/*.yaml``.

There are no in-code builtin graphs. A topology that exists in two places drifts, and
the one the tests exercise is then not the one the run loads.
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from ..paths import configs_dir
from .schema import GraphSpec
from .validation import GraphConfigError, validate_graph

__all__ = ["available_topologies", "graph_configs_dir", "load_topology", "topology_path"]


def graph_configs_dir(root: Path | None = None) -> Path:
    return configs_dir(root) / "graph"


def topology_path(name: str, root: Path | None = None) -> Path:
    return graph_configs_dir(root) / "topologies" / f"{name}.yaml"


def available_topologies(root: Path | None = None) -> list[str]:
    directory = graph_configs_dir(root) / "topologies"
    if not directory.is_dir():
        return []
    return sorted(p.stem for p in directory.glob("*.yaml"))


def load_topology(
    name: str, root: Path | None = None, *, expected_nodes: int | None = None
) -> GraphSpec:
    """Load and validate one topology by name."""
    path = topology_path(name, root)
    if not path.exists():
        raise GraphConfigError(
            f"unknown topology '{name}': {path} does not exist. "
            f"Available: {available_topologies(root)}"
        )
    raw = OmegaConf.load(path)
    if not isinstance(raw, DictConfig):
        raise GraphConfigError(f"{path}: top level must be a mapping")
    container = OmegaConf.to_container(raw, resolve=True)
    if not isinstance(container, dict):
        raise GraphConfigError(f"{path}: top level must be a mapping")
    try:
        spec = GraphSpec.model_validate(container)
    except Exception as exc:
        raise GraphConfigError(f"{path} failed validation:\n{exc}") from exc
    if spec.name != name:
        raise GraphConfigError(
            f"{path}: declares name '{spec.name}' but lives at '{name}.yaml'. The file "
            "name is what configs reference, so the two must agree."
        )
    return validate_graph(spec, expected_nodes=expected_nodes)
