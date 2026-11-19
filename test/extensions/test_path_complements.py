"""Regression checks for path enumeration and complementary feedback."""

from unittest.mock import Mock

import networkx as nx
import pytest

from qmm.extensions import paths


@pytest.mark.parametrize("calculate", [paths.system_paths, paths.weighted_paths, paths.path_metrics])
def test_path_query_enumerates_once_and_builds_one_matrix(snowshoe_io, monkeypatch, calculate):
    enumerate_paths = Mock(wraps=nx.all_simple_paths)
    matrix = Mock(wraps=paths.create_matrix)
    monkeypatch.setattr(nx, "all_simple_paths", enumerate_paths)
    monkeypatch.setattr(paths, "create_matrix", matrix)

    assert len(calculate(snowshoe_io, "Inp1", "Out1")) == 4
    enumerate_paths.assert_called_once_with(snowshoe_io, "Inp1", "Out1")
    matrix.assert_called_once()


def test_routes_in_different_orders_share_the_unsigned_minor(monkeypatch):
    graph = nx.DiGraph()
    graph.add_node("press", category="input")
    graph.add_node("response", category="output")
    graph.add_edges_from([("press", "A"), ("press", "B"), ("A", "B"), ("B", "A"),
                          ("A", "response"), ("B", "response")])
    graph.add_edges_from([("X", "X"), ("Y", "Y"), ("X", "Y"), ("Y", "X")])
    permanent = Mock(wraps=paths.perm)
    monkeypatch.setattr(paths, "perm", permanent)

    metrics = paths.path_metrics(graph, "press", "response")
    assert permanent.call_count == 3
    both = metrics[metrics["Length"] == 3]
    assert both["Complementary subsystem"].tolist() == [("X", "Y"), ("X", "Y")]
    assert both["Net feedback"].tolist() == [0, 0]
    assert both["Absolute feedback"].tolist() == [2, 2]
