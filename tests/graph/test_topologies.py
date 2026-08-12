"""Every shipped topology loads, validates, and says what it claims."""

from __future__ import annotations

import pytest

from rdl.graph.topology import available_topologies, load_topology
from rdl.graph.validation import GraphConfigError


def test_all_shipped_topologies_validate():
    names = available_topologies()
    assert {"chain5", "diamond5", "dense_dag5", "diamond6"} <= set(names)
    for name in names:
        spec = load_topology(name)
        assert spec.name == name
        assert spec.sink in spec.node_ids()
        assert not spec.children(spec.sink)


@pytest.mark.parametrize(
    ("name", "n_nodes", "n_edges"),
    [("chain5", 5, 4), ("diamond5", 5, 5), ("dense_dag5", 5, 8), ("diamond6", 6, 7)],
)
def test_topology_shapes(name, n_nodes, n_edges):
    spec = load_topology(name)
    assert len(spec.nodes) == n_nodes
    assert len(spec.edges) == n_edges


def test_dense_dag_is_more_connected_than_diamond():
    assert len(load_topology("dense_dag5").edges) > len(load_topology("diamond5").edges)


def test_diamond_and_dense_have_join_nodes_chain_does_not():
    for name in ("diamond5", "dense_dag5", "diamond6"):
        spec = load_topology(name)
        assert any(len(spec.parents(n)) >= 2 for n in spec.node_ids()), name
    chain = load_topology("chain5")
    assert not any(len(chain.parents(n)) >= 2 for n in chain.node_ids())


def test_unknown_topology_names_available_ones():
    with pytest.raises(GraphConfigError, match="unknown topology"):
        load_topology("no_such_graph")


def test_every_arm_uses_one_model_profile():
    """Five logical agents, one physical checkpoint. Never five copies."""
    for name in available_topologies():
        spec = load_topology(name)
        assert {n.model for n in spec.nodes} == {"primary"}
