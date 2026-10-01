"""Compatibility between NetworkX digraphs and digraph-builder JSON models.

builder_edge_priors.json is export_digraph output that digraph-builder's own
normalizeModel and toLegacyModel (src/lib/model.js) loaded and saved again, so it
carries the keys the builder adds (shape, x, y, roundness, loopAngle, smooth).
builder_snowshoe.json is the builder's bundled snowshoe example with images removed.
"""

import json
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from qmm import export_digraph, import_digraph, load_digraph, get_simulations

DATA = Path(__file__).parent.parent / "data"
MODELS = ["snowshoe", "snowshoe_rp", "snowshoe_io", "chain", "mesocosm"]


def model_view(G, edge_drop=(), node_drop=()):
    """Signs, categories and attributes in a form that ignores import defaults."""
    def clean(data, drop):
        return {k: v for k, v in data.items()
                if k not in drop and v is not None and not (k == "dashes" and v is False)}
    nodes = {str(n): clean(d, set(node_drop)) for n, d in G.nodes(data=True)}
    edges = {(str(u), str(v)): clean(d, set(edge_drop) | {"arrows", "id"}) for u, v, d in G.edges(data=True)}
    return nodes, edges


def with_priors():
    G = nx.DiGraph(load_digraph("snowshoe_io"))
    G.graph["meta"] = {"title": "Priors", "authors": "QMM"}
    G.edges["R", "C"]["range"] = [0.67, 1]
    G.edges["C", "R"]["dist"] = "weak"
    G.edges["P", "C"]["dist"] = {"beta": [2, 5]}
    G.add_edge("R", "P", sign=1, dashes=True, inclusion=0.25, title="Uncertain")
    G.add_edge("P", "R", sign=-1, dashes=True, inclusion=0.25)
    return G


@pytest.mark.parametrize("name", MODELS)
def test_bundled_models_round_trip(name):
    G = load_digraph(name)
    assert model_view(import_digraph(export_digraph(G))) == model_view(G)


def test_priors_round_trip_through_a_file(tmp_path):
    G = with_priors()
    path = tmp_path / "model.json"
    model = export_digraph(G, path)
    assert json.loads(path.read_text()) == model
    H = import_digraph(str(path))
    assert H.graph == {"meta": {"title": "Priors", "authors": "QMM"}}
    assert model_view(H) == model_view(G)
    assert H.edges["P", "C"]["dist"] == {"beta": [2, 5]}
    assert H.edges["R", "P"]["inclusion"] == 0.25


def test_export_writes_sign_and_arrow_and_leaves_out_import_defaults():
    model = export_digraph(import_digraph(export_digraph(with_priors())))
    for edge in model["edges"]:
        assert edge["arrows"]["to"]["type"] == ("triangle" if edge["sign"] == 1 else "circle")
        assert edge.get("title", "") is not None
        assert edge.get("dashes", True) is True
    assert all("category" not in node and "title" not in node for node in model["nodes"])
    assert len({edge["id"] for edge in model["edges"]}) == len(model["edges"])


def test_export_follows_a_changed_sign_over_a_stale_arrow():
    H = nx.DiGraph(import_digraph({"nodes": [{"id": "A"}, {"id": "B"}], "edges": [
        {"from": "A", "to": "A", "sign": -1}, {"from": "B", "to": "B", "sign": -1},
        {"from": "A", "to": "B", "arrows": {"to": {"type": "triangle"}}, "id": "e1"},
        {"from": "B", "to": "A", "sign": -1}]}))
    H.edges["A", "B"]["sign"] = -1
    edge = next(e for e in export_digraph(H)["edges"] if e["from"] == "A" and e["to"] == "B")
    assert edge["id"] == "e1"
    assert (edge["sign"], edge["arrows"]["to"]["type"]) == (-1, "circle")


def test_export_converts_numpy_and_tuples_to_json():
    G = nx.DiGraph(load_digraph("snowshoe"))
    G.edges["R", "C"]["range"] = (np.float64(0.2), np.float64(0.4))
    G.graph["meta"] = {"version": np.int64(3)}
    model = export_digraph(G)
    json.dumps(model)
    assert next(e for e in model["edges"] if e["from"] == "R" and e["to"] == "C")["range"] == [0.2, 0.4]


def test_export_rejects_invalid_priors():
    G = nx.DiGraph(load_digraph("snowshoe"))
    G.edges["R", "C"]["inclusion"] = 0.5
    with pytest.raises(ValueError, match="Inclusion needs a dashed edge"):
        export_digraph(G)


def test_builder_saved_export_imports_with_its_priors():
    H = import_digraph(str(DATA / "builder_edge_priors.json"))
    G = nx.DiGraph(load_digraph("snowshoe_io"))
    assert sorted(H.edges(data="sign")) == sorted(list(G.edges(data="sign")) + [("P", "R", -1), ("R", "P", 1)])
    assert {n: d["category"] for n, d in H.nodes(data=True)} == dict(G.nodes(data="category"))
    priors = {(u, v): {k: d[k] for k in ("dist", "range", "inclusion", "stronger_than") if k in d}
              for u, v, d in H.edges(data=True)}
    assert {e: p for e, p in priors.items() if p} == {
        ("R", "C"): {"range": [0.67, 1], "stronger_than": [["C", "R"], ["Inp1", "C"]]},
        ("C", "R"): {"dist": "weak"},
        ("P", "C"): {"dist": {"beta": [2, 5]}},
        ("Inp1", "R"): {"range": [0.5, 0.5]},
        ("R", "P"): {"inclusion": 0.25},
        ("P", "R"): {"inclusion": 0.25},
    }
    sims = get_simulations(H, n_sim=200, perturb=("Inp1", 1), return_samples=True)
    assert np.all(sims["samples"]["b_R,Inp1"] == 0.5)
    assert np.all(sims["samples"]["a_C,R"] > sims["samples"]["a_R,C"])
    assert np.all(sims["samples"]["a_C,R"] > sims["samples"]["b_C,Inp1"])


def test_builder_saved_export_survives_another_round_trip():
    H = import_digraph(str(DATA / "builder_edge_priors.json"))
    again = import_digraph(export_digraph(H))
    assert model_view(again) == model_view(H)
    assert again.graph == H.graph


def test_builder_example_round_trips_without_losing_builder_keys():
    raw = json.loads((DATA / "builder_snowshoe.json").read_text())
    H = import_digraph(raw)
    assert H.edges["V", "P"]["dashes"] is True
    model = export_digraph(H)
    assert model["meta"] == raw["meta"]
    by_pair = {(e["from"], e["to"]): e for e in raw["edges"]}
    for edge in model["edges"]:
        before = by_pair[(edge["from"], edge["to"])]
        assert {k: v for k, v in edge.items() if k != "sign"} == {k: v for k, v in before.items() if k != "sign"}
    nodes = {n["id"]: n for n in raw["nodes"]}
    assert all(node == nodes[node["id"]] for node in model["nodes"])
