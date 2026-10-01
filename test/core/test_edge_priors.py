"""Per-edge priors set as edge attributes: dist, range and inclusion."""

import networkx as nx
import numpy as np
import pytest

from qmm import (
    compare_model_alternatives,
    get_dashed_alternatives,
    get_simulations,
    import_digraph,
    load_digraph,
    numerical_simulations,
    simulation_stability,
)
from qmm.core.helper import _prior_sampler, _random_sampler


def snowshoe_rp():
    return nx.DiGraph(load_digraph("snowshoe_rp"))


def with_dashed(inclusion=None, reverse_inclusion=None):
    G = nx.DiGraph(load_digraph("snowshoe"))
    G.add_edge("R", "P", sign=1, dashes=True)
    if inclusion is not None:
        G.edges["R", "P"]["inclusion"] = inclusion
    if reverse_inclusion is not None:
        G.add_edge("P", "R", sign=-1, dashes=True, inclusion=reverse_inclusion)
    return G


def test_no_priors_keeps_the_default_draws():
    G = snowshoe_rp()
    plain = get_simulations(G, n_sim=300, return_samples=True)
    for edge in G.edges:
        G.edges[edge]["dist"] = None
        G.edges[edge]["range"] = None
    same = get_simulations(G, n_sim=300, return_samples=True)
    assert all(np.array_equal(plain["samples"][k], same["samples"][k]) for k in plain["samples"])
    assert _prior_sampler(G, list(G.edges), "uniform") is None


def test_sampler_redraws_only_edges_with_priors():
    G = snowshoe_rp()
    G.edges["R", "C"]["dist"] = {"beta": [2, 8]}
    G.edges["C", "R"]["range"] = [0.4, 0.6]
    edges = list(G.edges)
    draw = _prior_sampler(G, edges, "uniform")
    values = np.array([draw(np.random.RandomState(s)) for s in range(4000)])
    plain = np.array([_random_sampler("uniform", len(edges), np.random.RandomState(s)) for s in range(4000)])
    beta, scaled = edges.index(("R", "C")), edges.index(("C", "R"))
    untouched = [i for i in range(len(edges)) if i not in (beta, scaled)]
    assert np.array_equal(values[:, untouched], plain[:, untouched])
    assert abs(values[:, beta].mean() - 0.2) < 0.01
    assert values[:, scaled].min() >= 0.4 and values[:, scaled].max() <= 0.6
    assert np.allclose(values[:, scaled], 0.4 + 0.2 * plain[:, scaled])


def test_get_simulations_respects_ranges_and_dists():
    G = snowshoe_rp()
    G.edges["R", "C"]["range"] = [0.67, 1]
    G.edges["P", "P"]["range"] = [0.3, 0.3]
    G.edges["C", "R"]["dist"] = "strong"
    sims = get_simulations(G, n_sim=500, return_samples=True)
    s = sims["samples"]
    assert s["a_C,R"].min() >= 0.67 and s["a_C,R"].max() <= 1
    assert np.all(s["a_P,P"] == 0.3)
    assert s["a_R,C"].mean() > 0.6


def fixed(G, weights):
    for edge, w in weights.items():
        G.edges[edge]["range"] = [w, w]
    return G


def test_numerical_simulations_with_fixed_weights_match_the_inverse():
    G = fixed(snowshoe_rp(), {("R", "R"): 0.5, ("R", "C"): 0.9, ("R", "P"): 0.1, ("C", "R"): 0.8,
                              ("C", "P"): 0.2, ("P", "C"): 0.7, ("P", "P"): 0.6})
    nodes = list(G)
    A = np.zeros((3, 3))
    for (u, v), w in {e: G.edges[e]["range"][0] * G.edges[e]["sign"] for e in G.edges}.items():
        A[nodes.index(v), nodes.index(u)] = w
    expected = np.sign(np.linalg.inv(-A))
    result = np.array(numerical_simulations(G, n_sim=50).tolist(), dtype=float)
    assert np.array_equal(result, expected)


def test_simulation_stability_uses_edge_ranges():
    G = fixed(snowshoe_rp(), {e: 0.5 for e in snowshoe_rp().edges})
    assert simulation_stability(G, n_sim=20)["Result"][0] == "100.00%"
    G.edges["R", "P"]["range"] = [1, 1]
    G.edges["R", "R"]["range"] = [0.05, 0.05]
    G.edges["P", "P"]["range"] = [0.05, 0.05]
    stable = np.all(np.real(np.linalg.eigvals(np.array([[-0.05, -0.5, 0], [0.5, 0, -0.5], [1, 0.5, -0.05]]))) < 0)
    assert simulation_stability(G, n_sim=20)["Result"][0] == ("100.00%" if stable else "0.00%")


@pytest.mark.parametrize("inclusion, expected", [(0, 0), (1, 1)])
def test_inclusion_zero_and_one_fix_the_structure(inclusion, expected):
    sims = get_simulations(with_dashed(inclusion), n_sim=300)
    assert set(sims["structures"]) == {expected}


def test_inclusion_sets_how_often_the_edge_is_kept():
    low = np.mean(get_simulations(with_dashed(0.2), n_sim=3000)["structures"])
    high = np.mean(get_simulations(with_dashed(0.8), n_sim=3000)["structures"])
    assert low < 0.3 and high > 0.7


def test_unset_inclusion_keeps_the_shared_random_probability():
    plain = get_simulations(with_dashed(), n_sim=300)
    assert 0.3 < np.mean(plain["structures"]) < 0.7


def test_reciprocal_pair_needs_one_inclusion():
    with pytest.raises(ValueError, match="same inclusion"):
        get_simulations(with_dashed(0.3, reverse_inclusion=0.6), n_sim=10)
    sims = get_simulations(with_dashed(0.3, reverse_inclusion=0.6), n_sim=10, pair_reciprocal=False)
    assert len(sims["interactions"]) == 2


def test_enumerate_rejects_inclusion():
    with pytest.raises(ValueError, match="uncertain_interactions='sample'"):
        get_simulations(with_dashed(0.5), n_sim=10, uncertain_interactions="enumerate")


def test_alternatives_drop_inclusion_with_dashes():
    G = with_dashed(0.5)
    assert all("inclusion" not in d for H in get_dashed_alternatives(G) for _, _, d in H.edges(data=True))
    table = compare_model_alternatives(G, perturb="R:+", observe="P:+", n_sim=200)
    assert len(table) == 2


def model(**edge):
    return {"nodes": [{"id": "A"}, {"id": "B"}],
            "edges": [{"from": "A", "to": "A", "sign": -1}, {"from": "B", "to": "B", "sign": -1},
                      {"from": "B", "to": "A", "sign": -1}, {"from": "A", "to": "B", "sign": 1, **edge}]}


@pytest.mark.parametrize("edge, message", [
    ({"dist": "huge"}, "Invalid dist"),
    ({"dist": {"beta": [0, 1]}}, "Invalid dist"),
    ({"dist": {"gamma": [1, 1]}}, "Invalid dist"),
    ({"range": [0.8, 0.2]}, "Invalid range"),
    ({"range": [-0.1, 0.2]}, "Invalid range"),
    ({"range": 0.5}, "Invalid range"),
    ({"inclusion": 0.5}, "Inclusion needs a dashed edge"),
    ({"dashes": True, "inclusion": 1.5}, "Invalid inclusion"),
    ({"dashes": True, "inclusion": True}, "Invalid inclusion"),
])
def test_import_rejects_invalid_priors(edge, message):
    with pytest.raises(ValueError, match=message):
        import_digraph(model(**edge))


def test_import_keeps_valid_priors():
    G = import_digraph(model(dashes=True, inclusion=0.4, dist={"beta": [2, 2]}, range=[0.2, 0.9]))
    assert {k: G.edges["A", "B"][k] for k in ("dist", "range", "inclusion")} == {
        "dist": {"beta": [2, 2]}, "range": [0.2, 0.9], "inclusion": 0.4}


def ordered():
    G = snowshoe_rp()
    G.edges["R", "C"]["stronger_than"] = [["C", "R"]]
    G.edges["C", "R"]["stronger_than"] = [["P", "P"]]
    return G


def test_orderings_hold_in_every_draw():
    s = get_simulations(ordered(), n_sim=2000, return_samples=True)["samples"]
    assert np.all(s["a_C,R"] > s["a_R,C"]) and np.all(s["a_R,C"] > s["a_P,P"])
    assert abs(s["a_C,R"].mean() - 0.75) < 0.03 and abs(s["a_P,P"].mean() - 0.25) < 0.03


def test_orderings_respect_each_edges_own_prior():
    G = ordered()
    G.edges["C", "R"]["range"] = [0.4, 0.6]
    G.edges["P", "P"]["dist"] = "weak"
    draw = _prior_sampler(G, list(G.edges), "uniform")
    edges = list(G.edges)
    values = np.array([draw(np.random.RandomState(s)) for s in range(2000)])
    a, b, c = (edges.index(e) for e in [("R", "C"), ("C", "R"), ("P", "P")])
    assert np.all(values[:, a] > values[:, b]) and np.all(values[:, b] > values[:, c])
    assert values[:, b].min() >= 0.4 and values[:, b].max() <= 0.6


def test_orderings_apply_to_numerical_simulations_and_stability():
    G = ordered()
    for f in (lambda H: numerical_simulations(H, n_sim=300), lambda H: simulation_stability(H, n_sim=300)):
        assert not f(G).equals(f(snowshoe_rp()))


def test_impossible_ordering_fails_plainly():
    G = ordered()
    G.edges["R", "C"]["range"] = [0.1, 0.2]
    G.edges["C", "R"]["range"] = [0.5, 0.6]
    with pytest.raises(RuntimeError, match="stronger_than"):
        get_simulations(G, n_sim=10)


def test_ordering_on_a_dropped_dashed_edge_is_dropped_with_it():
    G = with_dashed(0.5)
    G.edges["R", "C"]["stronger_than"] = [["R", "P"]]
    variants = get_dashed_alternatives(G)
    assert [H.edges["R", "C"]["stronger_than"] for H in variants] == [[], [["R", "P"]]]
    sims = get_simulations(G, n_sim=300, return_samples=True)
    s = sims["samples"]
    kept = np.array(sims["structures"]) == 1
    assert np.all(s["a_C,R"][kept] > s["a_P,R"][kept])


@pytest.mark.parametrize("weaker, message", [
    ([["A", "C"]], "Unknown edge"),
    ([["A", "B"]], "stronger than itself"),
    ("B,A", "Invalid stronger_than"),
    ([["B"]], "Invalid stronger_than"),
])
def test_import_rejects_invalid_orderings(weaker, message):
    with pytest.raises(ValueError, match=message):
        import_digraph(model(stronger_than=weaker))


def test_import_rejects_cyclic_orderings():
    raw = model(stronger_than=[["B", "A"]])
    raw["edges"][2]["stronger_than"] = [["A", "B"]]
    with pytest.raises(ValueError, match="Cyclic"):
        import_digraph(raw)


def test_orderings_match_numeric_json_ids():
    raw = {"nodes": [{"id": 1}, {"id": 2}],
           "edges": [{"from": 1, "to": 1, "sign": -1}, {"from": 2, "to": 2, "sign": -1},
                     {"from": 2, "to": 1, "sign": -1}, {"from": 1, "to": 2, "sign": 1, "stronger_than": [[2, 1]]}]}
    s = get_simulations(import_digraph(raw), n_sim=200, return_samples=True)["samples"]
    assert np.all(s["a_2,1"] > s["a_1,2"])
