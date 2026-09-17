"""Tests for qmm.extensions.paths module using snowshoe_io fixture."""

from test.feedback import cycle_expansion
import numpy as np
import pytest
import pandas as pd
import sympy as sp
import networkx as nx

from qmm.core.helper import get_nodes
from qmm.core.structure import create_matrix
from qmm.core.stability import system_feedback
from qmm.extensions.effects import cumulative_effects, get_simulations, iter_simulations
from qmm.extensions.paths import (
    get_cycles,
    cycles_table,
    get_paths,
    paths_table,
    complementary_feedback,
    system_paths,
    weighted_paths,
    path_metrics,
    pathway_effects,
    _simulate_pathway_effects,
)


# =============================================================================
# get_cycles
# =============================================================================

def test_get_cycles_form_symbolic_snowshoe_io(snowshoe_io):
    result = get_cycles(snowshoe_io)
    a_RR = sp.Symbol("a_R,R")
    a_PP = sp.Symbol("a_P,P")
    a_CR = sp.Symbol("a_C,R")
    a_RC = sp.Symbol("a_R,C")
    a_CP = sp.Symbol("a_C,P")
    a_PC = sp.Symbol("a_P,C")
    expected_cycles = [("P",), ("R",), ("C", "P"), ("C", "R")]
    expected_products = {-a_RR, -a_PP, -a_CR * a_RC, -a_CP * a_PC}
    assert list(result["Cycle"]) == expected_cycles
    assert set(result["Product"]) == expected_products


# =============================================================================
# cycles_table
# =============================================================================

def test_cycles_table_symbolic_loops_snowshoe_io(snowshoe_io):
    result = list(cycles_table(snowshoe_io)["Cycle"])
    expected = [
        "P $\\multimap$ P",
        "R $\\multimap$ R",
        "C $\\rightarrow$ P $\\multimap$ C",
        "C $\\multimap$ R $\\rightarrow$ C",
    ]
    assert result == expected
    assert list(cycles_table(snowshoe_io)["Sign"]) == ["\u2212", "\u2212", "\u2212", "\u2212"]


def test_cycles_table_labels_snowshoe_io(snowshoe_io):
    G = nx.DiGraph(snowshoe_io)
    nx.set_node_attributes(G, {"R": "Resource", "C": "Consumer", "P": "Predator"}, "label")
    result = list(cycles_table(G, labels=True)["Cycle"])
    expected = [
        "Predator $\\multimap$ Predator",
        "Resource $\\multimap$ Resource",
        "Consumer $\\rightarrow$ Predator $\\multimap$ Consumer",
        "Consumer $\\multimap$ Resource $\\rightarrow$ Consumer",
    ]
    assert result == expected
    assert list(cycles_table(G)["Cycle"]) == list(cycles_table(snowshoe_io)["Cycle"])


def test_get_cycles_ignores_output_cycles_snowshoe_io(snowshoe_io):
    G = nx.DiGraph(snowshoe_io)
    G.add_edge("Out1", "Out2", sign=1)
    G.add_edge("Out2", "Out1", sign=1)
    result = list(get_cycles(G)["Cycle"])
    expected = [("P",), ("R",), ("C", "P"), ("C", "R")]
    assert result == expected


# =============================================================================
# get_paths
# =============================================================================

def test_get_paths_form_symbolic_inp1_out1_snowshoe_io(snowshoe_io):
    result = get_paths(snowshoe_io, "Inp1", "Out1", form="symbolic")
    a_CR = sp.Symbol("a_C,R")
    a_PC = sp.Symbol("a_P,C")
    b_RI = sp.Symbol("b_R,Inp1")
    b_CI = sp.Symbol("b_C,Inp1")
    c_OC = sp.Symbol("c_Out1,C")
    c_OP = sp.Symbol("c_Out1,P")
    expected_paths = [
        ("Inp1", "R", "C", "P", "Out1"),
        ("Inp1", "R", "C", "Out1"),
        ("Inp1", "C", "P", "Out1"),
        ("Inp1", "C", "Out1"),
    ]
    expected = sp.Matrix([
        [a_CR * a_PC * b_RI * c_OP],
        [      -a_CR * b_RI * c_OC],
        [      -a_PC * b_CI * c_OP],
        [              b_CI * c_OC]])
    assert list(result["Path"]) == expected_paths
    assert sp.Matrix(result["Product"].tolist()) == expected


def test_get_paths_form_signed_inp1_out1_snowshoe_io(snowshoe_io):
    result = get_paths(snowshoe_io, "Inp1", "Out1", form="signed")
    expected = sp.Matrix([
        [ 1],
        [-1],
        [-1],
        [ 1]])
    assert list(result["Path"]) == [
        ("Inp1", "R", "C", "P", "Out1"),
        ("Inp1", "R", "C", "Out1"),
        ("Inp1", "C", "P", "Out1"),
        ("Inp1", "C", "Out1"),
    ]
    assert sp.Matrix(result["Product"].tolist()) == expected


def test_get_paths_form_binary_inp1_out1_snowshoe_io(snowshoe_io):
    result = get_paths(snowshoe_io, "Inp1", "Out1", form="binary")
    expected = sp.Matrix([
        [1],
        [1],
        [1],
        [1]])
    assert sp.Matrix(result["Product"].tolist()) == expected


@pytest.mark.parametrize("source", ["R", "C", "P"])
def test_get_paths_no_path_to_new_state_snowshoe_io_na_source(snowshoe_io_na, source):
    result = get_paths(snowshoe_io_na, source, "N", form="signed")
    expected = ((), sp.Integer(0))
    assert (result["Path"].iloc[0], result["Product"].iloc[0]) == expected


def test_get_paths_direct_input_output_symbolic_snowshoe_io_with_direct_edge(snowshoe_io_with_direct_edge):
    result = get_paths(snowshoe_io_with_direct_edge, "Inp1", "Out1", form="symbolic")
    d_Inp1_Out1 = sp.Symbol('d_Out1,Inp1')
    expected = any(d_Inp1_Out1 in product.free_symbols for product in result["Product"])
    assert expected


def test_system_paths_rejects_direct_io_edge(snowshoe_io_with_direct_edge):
    with pytest.raises(ValueError, match="Direct input to output edge"):
        system_paths(snowshoe_io_with_direct_edge, "Inp1", "Out1")


def test_weighted_paths_rejects_direct_io_edge(snowshoe_io_with_direct_edge):
    with pytest.raises(ValueError, match="Direct input to output edge"):
        weighted_paths(snowshoe_io_with_direct_edge, "Inp1", "Out1")


def test_path_metrics_rejects_direct_io_edge(snowshoe_io_with_direct_edge):
    with pytest.raises(ValueError, match="Direct input to output edge"):
        path_metrics(snowshoe_io_with_direct_edge, "Inp1", "Out1")


def test_get_paths_symbolic_output_to_output_edge_output_to_output_graph(output_to_output_graph):
    result = len(get_paths(output_to_output_graph, 'A', 'Out2', form='symbolic'))
    expected = 1
    assert result == expected


def test_get_paths_invalid_source_snowshoe_io(snowshoe_io):
    with pytest.raises(ValueError):
        get_paths(snowshoe_io, "Invalid", "Out1")


def test_get_paths_invalid_target_snowshoe_io(snowshoe_io):
    with pytest.raises(ValueError, match="Invalid target node"):
        get_paths(snowshoe_io, "Inp1", "Invalid")


def test_get_paths_rejects_output_source_snowshoe_io(snowshoe_io):
    with pytest.raises(ValueError, match="Invalid source node"):
        get_paths(snowshoe_io, "Out1", "R")


def test_get_paths_rejects_input_target_snowshoe_io(snowshoe_io):
    with pytest.raises(ValueError, match="Invalid target node"):
        get_paths(snowshoe_io, "R", "Inp1")


def test_get_paths_source_eq_target_snowshoe_io(snowshoe_io):
    result = get_paths(snowshoe_io, "R", "R")
    expected = (("R",), sp.Integer(1))
    assert (result["Path"].iloc[0], result["Product"].iloc[0]) == expected


# =============================================================================
# paths_table
# =============================================================================

def test_paths_table_inp1_out1_snowshoe_io(snowshoe_io):
    result = list(paths_table(snowshoe_io, "Inp1", "Out1")["Path"])
    expected = [
        "Inp1 $\\rightarrow$ R $\\rightarrow$ C $\\rightarrow$ P $\\rightarrow$ Out1",
        "Inp1 $\\rightarrow$ R $\\rightarrow$ C $\\multimap$ Out1",
        "Inp1 $\\multimap$ C $\\rightarrow$ P $\\rightarrow$ Out1",
        "Inp1 $\\multimap$ C $\\multimap$ Out1",
    ]
    assert result == expected


def test_paths_table_labels_fall_back_to_ids_snowshoe_io(snowshoe_io):
    G = nx.DiGraph(snowshoe_io)
    nx.set_node_attributes(G, {"Inp1": "Input", "Out1": "Output", "C": "Consumer"}, "label")
    result = list(paths_table(G, "Inp1", "Out1", labels=True)["Path"])
    assert result[-1] == "Input $\\multimap$ Consumer $\\multimap$ Output"
    assert result[0] == "Input $\\rightarrow$ R $\\rightarrow$ Consumer $\\rightarrow$ P $\\rightarrow$ Output"


def test_paths_table_no_path_available_snowshoe_io_na(snowshoe_io_na):
    result = paths_table(snowshoe_io_na, "R", "N")
    expected = None
    assert result == expected


def test_paths_table_invalid_source_snowshoe_io(snowshoe_io):
    with pytest.raises(ValueError, match="Invalid source node"):
        paths_table(snowshoe_io, "Invalid", "Out1")


def test_paths_table_self_response_snowshoe_io(snowshoe_io):
    result = paths_table(snowshoe_io, "R", "R")
    expected = (0, "R", "+")
    assert (result.loc[0, "Length"], result.loc[0, "Path"], result.loc[0, "Sign"]) == expected


# =============================================================================
# complementary_feedback
# =============================================================================

def test_complementary_feedback_form_symbolic_inp1_out1_snowshoe_io(snowshoe_io):
    result = complementary_feedback(snowshoe_io, "Inp1", "Out1", form="symbolic")
    a_PP = sp.Symbol("a_P,P")
    a_RR = sp.Symbol("a_R,R")
    expected = sp.Matrix([
        [          -1],
        [       -a_PP],
        [       -a_RR],
        [-a_PP * a_RR]])
    assert list(result["Path"]) == [
        ("Inp1", "R", "C", "P", "Out1"),
        ("Inp1", "R", "C", "Out1"),
        ("Inp1", "C", "P", "Out1"),
        ("Inp1", "C", "Out1"),
    ]
    assert sp.Matrix(result["Feedback"].tolist()) == expected


def test_complementary_feedback_invalid_source_snowshoe_io(snowshoe_io):
    with pytest.raises(ValueError):
        complementary_feedback(snowshoe_io, "Invalid", "Out1")


def test_complementary_feedback_source_eq_target_snowshoe_io(snowshoe_io):
    result = complementary_feedback(snowshoe_io, "R", "R", form="symbolic")
    expected = system_feedback(snowshoe_io.subgraph(['C', 'P']).copy(), level=2)
    assert list(result["Path"]) == [("R",)]
    assert sp.Matrix(result["Feedback"].tolist()) == expected


def test_complementary_feedback_invalid_form_feedback_test_graph(feedback_test_graph):
    with pytest.raises(ValueError):
        complementary_feedback(feedback_test_graph, 'A', 'B', form='invalid')


def test_complementary_feedback_invalid_form_no_path(snowshoe_io):
    with pytest.raises(ValueError, match="Invalid form"):
        complementary_feedback(snowshoe_io, 'Inp1', 'Out1', form='invalid')


# =============================================================================
# system_paths
# =============================================================================

def test_system_paths_form_symbolic_inp1_out1_snowshoe_io(snowshoe_io):
    result = system_paths(snowshoe_io, "Inp1", "Out1", form="symbolic")
    a_CR = sp.Symbol("a_C,R")
    a_PC = sp.Symbol("a_P,C")
    a_PP = sp.Symbol("a_P,P")
    a_RR = sp.Symbol("a_R,R")
    b_RI = sp.Symbol("b_R,Inp1")
    b_CI = sp.Symbol("b_C,Inp1")
    c_OC = sp.Symbol("c_Out1,C")
    c_OP = sp.Symbol("c_Out1,P")
    expected = sp.Matrix([
        [ a_CR * a_PC * b_RI * c_OP],
        [-a_CR * a_PP * b_RI * c_OC],
        [-a_PC * a_RR * b_CI * c_OP],
        [ a_PP * a_RR * b_CI * c_OC]])
    assert list(result["Path"]) == [
        ("Inp1", "R", "C", "P", "Out1"),
        ("Inp1", "R", "C", "Out1"),
        ("Inp1", "C", "P", "Out1"),
        ("Inp1", "C", "Out1"),
    ]
    assert sp.Matrix(result["Effect"].tolist()) == expected


def test_system_paths_form_signed_inp1_out1_snowshoe_io(snowshoe_io):
    result = system_paths(snowshoe_io, "Inp1", "Out1", form="signed")
    expected = sp.Matrix([
        [ 1],
        [-1],
        [-1],
        [ 1]])
    assert sp.Matrix(result["Effect"].tolist()) == expected


def test_system_paths_form_binary_inp1_out1_snowshoe_io(snowshoe_io):
    result = system_paths(snowshoe_io, "Inp1", "Out1", form="binary")
    expected = sp.Matrix([
        [1],
        [1],
        [1],
        [1]])
    assert sp.Matrix(result["Effect"].tolist()) == expected


def test_system_paths_source_eq_target_snowshoe_io(snowshoe_io):
    states = get_nodes(snowshoe_io, "state")
    cum = cumulative_effects(snowshoe_io, form="symbolic")
    for i, node in enumerate(states):
        result = system_paths(snowshoe_io, node, node, form="symbolic")
        assert list(result["Path"]) == [(node,)]
        assert sp.simplify(sp.expand(sum(result["Effect"])) - sp.expand(cum[i, i])) == 0


# =============================================================================
# weighted_paths
# =============================================================================

def test_weighted_paths_signed_inp1_out1_snowshoe_io(snowshoe_io):
    result = weighted_paths(snowshoe_io, "Inp1", "Out1")
    expected = sp.Matrix([
        [ 1],
        [-1],
        [-1],
        [ 1]])
    assert list(result["Path"]) == [
        ("Inp1", "R", "C", "P", "Out1"),
        ("Inp1", "R", "C", "Out1"),
        ("Inp1", "C", "P", "Out1"),
        ("Inp1", "C", "Out1"),
    ]
    assert sp.Matrix(result["Weight"].tolist()) == expected


def test_weighted_paths_nan_feedback_snowshoe_io(snowshoe_io):
    result = weighted_paths(snowshoe_io, "Inp1", "Out1")
    expected = (isinstance(result, pd.DataFrame), len(result))
    assert expected == (True, 4)


def test_weighted_paths_nan_feedback_replacement_nan_feedback_graph(nan_feedback_graph):
    result = weighted_paths(nan_feedback_graph, 'A', 'B')
    assert list(result["Path"]) == [("A", "B")]
    assert result["Weight"].iloc[0] == 0


def test_weighted_paths_invalid_source_snowshoe_io(snowshoe_io):
    with pytest.raises(ValueError):
        weighted_paths(snowshoe_io, "Invalid", "Out1")


def test_weighted_paths_source_eq_target_snowshoe_io(snowshoe_io):
    result = weighted_paths(snowshoe_io, "R", "R")
    expected = (("R",), 1)
    assert (result["Path"].iloc[0], result["Weight"].iloc[0]) == expected


# =============================================================================
# path_metrics
# =============================================================================

def test_path_metrics_inp1_out1_snowshoe_io(snowshoe_io):
    result = path_metrics(snowshoe_io, "Inp1", "Out1")
    expected_paths = [
        ("Inp1", "R", "C", "P", "Out1"),
        ("Inp1", "R", "C", "Out1"),
        ("Inp1", "C", "P", "Out1"),
        ("Inp1", "C", "Out1"),
    ]
    expected_complements = [(), ("P",), ("R",), ("R", "P")]
    expected_values = [
        [-1, 1, 0, 1, -1,  1,  1],
        [-1, 1, 0, 1, -1, -1, -1],
        [-1, 1, 0, 1, -1, -1, -1],
        [-1, 1, 0, 1, -1,  1,  1]]
    assert list(result["Path"]) == expected_paths
    assert list(result["Complementary subsystem"]) == expected_complements
    assert list(result["Sign"]) == ["+", "\u2212", "\u2212", "+"]
    assert result.iloc[:, 4:].values.tolist() == expected_values


@pytest.mark.parametrize("source, target", [("A1", "A1"), ("A2", "H2"), ("P", "A1")])
def test_path_metrics_match_weighted_and_system_paths_mesocosm(mesocosm, source, target):
    result = path_metrics(mesocosm, source, target)[["Path", "Weighted path", "System path"]].values.tolist()
    weights = weighted_paths(mesocosm, source, target)
    effects = system_paths(mesocosm, source, target, form="signed")
    expected = [list(row) for row in zip(weights["Path"], weights["Weight"], effects["Effect"])]
    assert result == expected


def test_path_metrics_zero_complementary_feedback_nan_feedback_graph(nan_feedback_graph):
    result = path_metrics(nan_feedback_graph, "A", "B").iloc[0, 4:].tolist()
    expected = [0, 0, 0, 0, 0, 0, 0]
    assert result == expected


def test_path_metrics_no_path_available_snowshoe_io_na(snowshoe_io_na):
    result = path_metrics(snowshoe_io_na, "R", "N")
    expected = pd.DataFrame()
    assert result.equals(expected)


def test_path_metrics_invalid_source_snowshoe_io(snowshoe_io):
    with pytest.raises(ValueError, match="Invalid source node"):
        path_metrics(snowshoe_io, "Invalid", "Out1")


def test_path_metrics_self_response_snowshoe_io(snowshoe_io):
    result = path_metrics(snowshoe_io, "R", "R")
    expected = (0, ("R",), ("C", "P"))
    assert (result.loc[0, "Length"], result.loc[0, "Path"], result.loc[0, "Complementary subsystem"]) == expected


# =============================================================================
# system_paths vs cumulative_effects comparison tests
# =============================================================================

@pytest.mark.parametrize("form", ["symbolic", "signed", "binary"])
def test_system_paths_matches_cumulative_effects(snowshoe_io_na, form):
    states, inputs, outputs = get_nodes(snowshoe_io_na, "state"), get_nodes(snowshoe_io_na, "input"), get_nodes(snowshoe_io_na, "output")
    rows, cols = states + outputs, states + inputs
    cum = cumulative_effects(snowshoe_io_na, form=form)
    for i, tgt in enumerate(rows):
        for j, src in enumerate(cols):
            result = sp.expand(sum(system_paths(snowshoe_io_na, src, tgt, form=form)["Effect"]))
            assert sp.simplify(result - sp.expand(cum[i, j])) == 0


# =============================================================================
# pathway_effects
# =============================================================================

def test_pathway_effects_terms_sum_mesocosm(mesocosm):
    responders = get_nodes(mesocosm, "state") + get_nodes(mesocosm, "output")
    for target in ["P", "A1", "C2"]:
        paths, terms, sims = _simulate_pathway_effects(mesocosm, "P", target, 200, "uniform", 3, "sample", True)
        result = (terms.shape, terms.sum(axis=1))
        expected = ((200, len(paths)), np.array([effect[responders.index(target)] for effect in sims["effects"]]))
        assert result[0] == expected[0]
        assert np.allclose(result[1], expected[1])


def test_pathway_effects_terms_sum_inp1_r_snowshoe_io(snowshoe_io):
    responders = get_nodes(snowshoe_io, "state") + get_nodes(snowshoe_io, "output")
    paths, terms, sims = _simulate_pathway_effects(snowshoe_io, "Inp1", "R", 150, "uniform", 5, "sample", True)
    result = terms.sum(axis=1)
    expected = np.array([effect[responders.index("R")] for effect in sims["effects"]])
    assert len(paths) > 0
    assert np.allclose(result, expected)


def test_pathway_effects_terms_sum_inp1_out1_snowshoe_io(snowshoe_io):
    responders = get_nodes(snowshoe_io, "state") + get_nodes(snowshoe_io, "output")
    paths, terms, sims = _simulate_pathway_effects(snowshoe_io, "Inp1", "Out1", 150, "uniform", 5, "sample", True)
    result = (len(paths), terms.sum(axis=1))
    expected = (4, np.array([effect[responders.index("Out1")] for effect in sims["effects"]]))
    assert result[0] == expected[0]
    assert np.allclose(result[1], expected[1])


def test_pathway_effects_table_inp1_out1_snowshoe_io(snowshoe_io):
    result = pathway_effects(snowshoe_io, "Inp1", "Out1", n_sim=300, seed=1)
    expected_cols = ["Length", "Path", "Sign", "Present", "Positive", "Negative", "Zero", "Contribution"]
    assert list(result.columns) == expected_cols
    assert len(result) == 4
    assert result["Contribution"].sum() == pytest.approx(1.0)
    assert result["Contribution"].is_monotonic_decreasing
    assert (result["Positive"] + result["Negative"] + result["Zero"]).tolist() == pytest.approx([1.0] * 4)
    assert set(result["Sign"]) == {"+", "\u2212"}
    for _, row in result.iterrows():
        dominant = row["Positive"] if row["Sign"] == "+" else row["Negative"]
        assert dominant == 1.0


def test_pathway_effects_self_response_snowshoe(snowshoe):
    result = pathway_effects(snowshoe, "R", "R", n_sim=100)
    expected = (1, 0, ("R",), 1.0, 1.0)
    assert (
        len(result),
        result.loc[0, "Length"],
        result.loc[0, "Path"],
        result.loc[0, "Contribution"],
        result.loc[0, "Positive"],
    ) == expected


def test_pathway_effects_uncertain_edges_snowshoe_dashed(snowshoe_dashed):
    result = pathway_effects(snowshoe_dashed, "R", "P", n_sim=300, seed=2, uncertain_interactions="sample")
    direct = result[result["Length"] == 1].iloc[0]
    assert direct["Zero"] > 0.2
    assert direct["Positive"] + direct["Zero"] == pytest.approx(1.0)
    assert result["Contribution"].sum() == pytest.approx(1.0)


def test_pathway_effects_invalid_target_snowshoe(snowshoe):
    with pytest.raises(ValueError, match="Invalid target node"):
        pathway_effects(snowshoe, "R", "X", n_sim=10)


def test_pathway_effects_rejects_output_source_snowshoe_io(snowshoe_io):
    with pytest.raises(ValueError, match="Invalid source node"):
        pathway_effects(snowshoe_io, "Out1", "R", n_sim=10)


def test_pathway_effects_rejects_input_target_snowshoe_io(snowshoe_io):
    with pytest.raises(ValueError, match="Invalid target node"):
        pathway_effects(snowshoe_io, "Inp1", "Inp1", n_sim=10)


def test_pathway_effects_rejects_direct_io_edge(snowshoe_io_with_direct_edge):
    with pytest.raises(ValueError, match="Direct input to output edge"):
        pathway_effects(snowshoe_io_with_direct_edge, "Inp1", "Out1", n_sim=10)


def test_pathway_effects_empty_observe_matches_unconditional_snowshoe_io(snowshoe_io):
    kwargs = dict(source="Inp1", target="Out1", n_sim=200, seed=1)
    result = pathway_effects(snowshoe_io, observe="", **kwargs)
    expected = pathway_effects(snowshoe_io, **kwargs)
    pd.testing.assert_frame_equal(result, expected)


def test_pathway_effects_observe_out1_positive_raises_positive_share_snowshoe_io(snowshoe_io):
    kwargs = dict(source="Inp1", target="Out1", n_sim=400, seed=1)
    prior = pathway_effects(snowshoe_io, **kwargs)
    posterior = pathway_effects(snowshoe_io, observe="Out1:+", **kwargs)
    prior_pos = prior.loc[prior["Sign"] == "+", "Contribution"].sum()
    posterior_pos = posterior.loc[posterior["Sign"] == "+", "Contribution"].sum()
    assert posterior["Contribution"].sum() == pytest.approx(1.0)
    assert posterior_pos > prior_pos


def test_pathway_effects_observe_no_matches_snowshoe_io(snowshoe_io):
    with pytest.raises(RuntimeError, match="Maximum iterations reached"):
        pathway_effects(snowshoe_io, "Inp1", "Out1", n_sim=50, seed=1, observe="Out1:0")


def test_pathway_effects_observe_unknown_node_snowshoe_io(snowshoe_io):
    with pytest.raises(ValueError, match="Unknown observation node"):
        pathway_effects(snowshoe_io, "Inp1", "Out1", n_sim=10, observe="Missing:+")


def test_tables_are_fresh_objects_snowshoe_io(snowshoe_io):
    first = cycles_table(snowshoe_io)
    first["Cycle"] = "edited"
    assert list(cycles_table(snowshoe_io)["Cycle"]) != ["edited"] * len(first)
    paths = paths_table(snowshoe_io, "Inp1", "Out1")
    paths["Path"] = "edited"
    assert list(paths_table(snowshoe_io, "Inp1", "Out1")["Path"]) != ["edited"] * len(paths)


def test_pathway_effects_no_route_returns_empty_table(self_limited_pair):
    G = self_limited_pair
    result = pathway_effects(G, "A", "B", n_sim=10)
    assert result.empty and list(result.columns) == ["Length", "Path", "Sign", "Present", "Positive", "Negative", "Zero", "Contribution"]


def test_pathway_effects_reflects_changed_edge_sign(self_limited_pair):
    G = self_limited_pair
    G.add_edge("A", "B", sign=1)
    assert pathway_effects(G, "A", "B", n_sim=10).loc[0, "Positive"] == 1
    G["A"]["B"]["sign"] = -1
    result = pathway_effects(G, "A", "B", n_sim=10)
    assert result.loc[0, "Sign"] == "−"
    assert result.loc[0, "Negative"] == 1


def test_system_paths_sum_to_adjoint_with_cycle_expansion_complements_snowshoe_rp(snowshoe_rp):
    states = get_nodes(snowshoe_rp, 'state')
    adjoint = cumulative_effects(snowshoe_rp)
    result = []
    expected = []
    for source in states:
        for target in states:
            paths = system_paths(snowshoe_rp, source, target)
            complements = complementary_feedback(snowshoe_rp, source, target)
            result.append((sp.expand(sum(paths['Effect'])), [sp.expand(f) for f in complements['Feedback']]))
            remaining = [[n for n in states if n not in path] for path in complements['Path']]
            expected.append((sp.expand(adjoint[states.index(target), states.index(source)]),
                             [cycle_expansion(snowshoe_rp.subgraph(nodes))[0][-1] for nodes in remaining]))
    assert result == expected


def test_pathway_effects_terms_match_system_paths_numerators_snowshoe_rp(snowshoe_rp):
    A = create_matrix(snowshoe_rp)
    symbols = sorted(A.free_symbols, key=str)
    paths, terms, sims = _simulate_pathway_effects(snowshoe_rp, 'R', 'P', 5, 'uniform', 1, "sample", True)
    numerators = system_paths(snowshoe_rp, 'R', 'P')['Effect']
    expected = []
    for draw in range(5):
        values = {symbol: float(sims['samples'][str(symbol)][draw]) for symbol in symbols}
        determinant = float((-A).det().subs(values))
        expected.append([float(numerator.subs(values)) / determinant for numerator in numerators])
    assert np.allclose(terms, expected, rtol=1e-9, atol=1e-12)


def test_pathway_effects_sims_link_fixed_to_zero_present(self_limited_pair):
    G = self_limited_pair
    G.add_edge("A", "B", sign=1)
    sims = get_simulations(G, n_sim=5, seed=1, perturb=("A", 1), return_samples=True,
                           presample=lambda symbols: {sp.Symbol("a_B,A"): 0})
    result = pathway_effects(G, "A", "B", sims=sims).loc[0, ["Present", "Zero", "Contribution"]].tolist()
    expected = [1.0, 1.0, 0.0]
    assert result == expected


def test_pathway_effects_sims_observe_contradicting_captured_observation(self_limited_pair):
    G = self_limited_pair
    G.add_edge("A", "B", sign=1)
    sims = get_simulations(G, n_sim=5, seed=1, perturb=("A", 1), observe=(("B", 1),), return_samples=True)
    with pytest.raises(ValueError, match="No simulations match"):
        pathway_effects(G, "A", "B", observe="B:-", sims=sims)


def test_pathway_effects_sims_observe_filters_same_draws_snowshoe_dashed(snowshoe_dashed):
    sims = get_simulations(snowshoe_dashed, n_sim=300, seed=2, perturb=("R", 1), return_samples=True)
    direct_kept = np.array(sims["structures"]) & 1 == 1
    valid = np.array([effect[1] < 0 for effect in sims["effects"]])
    tables = [pathway_effects(snowshoe_dashed, "R", "P", observe=observe, sims=sims) for observe in ("", "C:-")]
    result = [table.loc[table["Length"] == 1, "Present"].item() for table in tables]
    expected = [direct_kept.mean(), direct_kept[valid].mean()]
    assert 0 < valid.sum() < len(valid)
    assert result == pytest.approx(expected)


def test_pathway_effects_sims_observe_matches_fresh_run_snowshoe_dashed(snowshoe_dashed):
    sims = get_simulations(snowshoe_dashed, n_sim=300, seed=2, perturb=("R", 1), observe=(("C", -1),),
                           return_samples=True)
    result = pathway_effects(snowshoe_dashed, "R", "P", observe="C:-", sims=sims)
    expected = pathway_effects(snowshoe_dashed, "R", "P", n_sim=300, seed=2, observe="C:-")
    assert result.equals(expected)


def test_pathway_effects_sims_observe_requires_every_observation_snowshoe_dashed(snowshoe_dashed):
    sims = get_simulations(snowshoe_dashed, n_sim=50, seed=2, perturb=("R", 1), return_samples=True)
    with pytest.raises(ValueError, match="No simulations match"):
        pathway_effects(snowshoe_dashed, "R", "P", observe="C:-,C:+", sims=sims)


def test_pathway_effects_sims_observe_unknown_node_snowshoe(snowshoe):
    sims = get_simulations(snowshoe, n_sim=10, seed=1, perturb=("R", 1), return_samples=True)
    with pytest.raises(ValueError, match="Unknown observation node"):
        pathway_effects(snowshoe, "R", "P", observe="Missing:+", sims=sims)


@pytest.mark.parametrize("observe, observed", [("Out1:-", (("Out1", -1),)), ("C:-,Out1:+", (("C", -1), ("Out1", 1)))])
def test_pathway_effects_sims_observe_matches_valid_draws_snowshoe_io(snowshoe_io, observe, observed):
    sims = next(iter_simulations(snowshoe_io, 300, seed=1, perturb=("Inp1", 1), observe=observed, condition=False,
                                 return_samples=True))
    keep = np.array(sims["valid_sims"])
    subset = {**sims, "effects": np.array(sims["effects"])[keep], "structures": list(np.array(sims["structures"])[keep]),
              "samples": {name: values[keep] for name, values in sims["samples"].items()}}
    result = pathway_effects(snowshoe_io, "Inp1", "Out1", observe=observe, sims=sims)
    expected = pathway_effects(snowshoe_io, "Inp1", "Out1", sims=subset)
    assert 0 < keep.sum() < len(keep)
    assert result.equals(expected)


def test_pathway_effects_sims_matrix_observe_requires_perturbation_snowshoe(snowshoe):
    sims = get_simulations(snowshoe, n_sim=10, seed=1, return_samples=True)
    with pytest.raises(ValueError, match="require a perturbation"):
        pathway_effects(snowshoe, "R", "P", observe="C:+", sims=sims)


def test_pathway_effects_sims_negative_press(self_limited_pair):
    G = self_limited_pair
    G.add_edge("A", "B", sign=1)
    sims = get_simulations(G, n_sim=5, seed=1, perturb=("A", -1), return_samples=True)
    tables = [pathway_effects(G, "A", "B", observe=observe, sims=sims) for observe in ("", "B:-")]
    result = [table.loc[0, ["Positive", "Negative"]].tolist() for table in tables]
    expected = [[0.0, 1.0], [0.0, 1.0]]
    assert result == expected


def test_pathway_effects_terms_sum_sims_negative_press_snowshoe_dashed(snowshoe_dashed):
    sims = get_simulations(snowshoe_dashed, n_sim=200, seed=1, perturb=("R", -1), return_samples=True)
    paths, terms, _ = _simulate_pathway_effects(snowshoe_dashed, "R", "P", 200, "uniform", 1, "sample", True, sims=sims)
    result = terms.sum(axis=1)
    expected = np.array([effect[2] for effect in sims["effects"]])
    assert len(set(sims["structures"])) == 4
    assert np.allclose(result, expected)


def test_pathway_effects_terms_sum_sims_matrix_mesocosm(mesocosm):
    nodes = get_nodes(mesocosm, "state")
    sims = get_simulations(mesocosm, n_sim=100, seed=3, return_samples=True)
    paths, terms, _ = _simulate_pathway_effects(mesocosm, "P", "C2", 100, "uniform", 3, "sample", True, sims=sims)
    result = terms.sum(axis=1)
    expected = np.array([effect[nodes.index("C2"), nodes.index("P")] for effect in sims["effects"]])
    assert np.allclose(result, expected)


@pytest.mark.parametrize("perturb", [(("R", 1), ("P", -1)), ("P", 1), (("R", 1), ("R", 1))])
def test_pathway_effects_sims_rejects_other_press_snowshoe(snowshoe, perturb):
    sims = get_simulations(snowshoe, n_sim=10, seed=1, perturb=perturb, return_samples=True)
    with pytest.raises(ValueError, match="press only R"):
        pathway_effects(snowshoe, "R", "C", sims=sims)


@pytest.mark.parametrize("simulated, analysed", [("snowshoe_rp", "snowshoe"), ("snowshoe", "snowshoe_rp")])
def test_pathway_effects_sims_rejects_other_graph(request, simulated, analysed):
    sims = get_simulations(request.getfixturevalue(simulated), n_sim=10, seed=1, perturb=("R", 1), return_samples=True)
    with pytest.raises(ValueError, match="different graph: R -> P"):
        pathway_effects(request.getfixturevalue(analysed), "R", "P", sims=sims)


def test_pathway_effects_sims_accepts_edge_without_sign(self_limited_pair):
    G = self_limited_pair
    G.add_edge("A", "B")
    sims = get_simulations(G, n_sim=5, seed=1, perturb=("A", 1), return_samples=True)
    result = pathway_effects(G, "A", "B", sims=sims).loc[0, "Positive"]
    expected = 1.0
    assert result == expected


def test_pathway_effects_sims_rejects_changed_edge_sign(self_limited_pair):
    G = self_limited_pair
    G.add_edge("A", "B", sign=1)
    sims = get_simulations(G, n_sim=5, seed=1, perturb=("A", 1), return_samples=True)
    G["A"]["B"]["sign"] = -1
    with pytest.raises(ValueError, match="different graph: A -> B"):
        pathway_effects(G, "A", "B", sims=sims)


def test_pathway_effects_sims_rejects_other_node_order_snowshoe(snowshoe):
    sims = get_simulations(snowshoe, n_sim=10, seed=1, perturb=("R", 1), return_samples=True)
    G = nx.DiGraph()
    G.add_nodes_from(reversed(list(snowshoe.nodes(data=True))))
    G.add_edges_from(snowshoe.edges(data=True))
    with pytest.raises(ValueError, match="different graph: P, R"):
        pathway_effects(G, "R", "P", sims=sims)


def test_pathway_effects_sims_rejects_extra_node_snowshoe(snowshoe):
    sims = get_simulations(snowshoe, n_sim=10, seed=1, perturb=("R", 1), return_samples=True)
    G = nx.DiGraph(snowshoe)
    G.add_node("X", category="state")
    G.add_edge("X", "X", sign=-1)
    with pytest.raises(ValueError, match="different graph: X"):
        pathway_effects(G, "R", "P", sims=sims)


@pytest.mark.parametrize("key", ["structures", "samples"])
def test_pathway_effects_sims_rejects_misaligned_arrays_snowshoe(snowshoe, key):
    sims = get_simulations(snowshoe, n_sim=10, seed=1, perturb=("R", 1), return_samples=True)
    if key == "structures":
        sims["structures"] = sims["structures"][:1]
    else:
        sims["samples"]["a_P,C"] = sims["samples"]["a_P,C"][:1]
    with pytest.raises(ValueError, match="differ in length"):
        pathway_effects(snowshoe, "R", "P", sims=sims)


def test_pathway_effects_sims_require_samples_snowshoe(snowshoe):
    sims = get_simulations(snowshoe, n_sim=10, seed=1, perturb=("R", 1))
    with pytest.raises(ValueError, match="return_samples"):
        pathway_effects(snowshoe, "R", "P", sims=sims)


def test_pathway_effects_dropped_link_absent_and_zero_strength_link_present(self_limited_pair):
    G = self_limited_pair
    G.add_edge("A", "B", sign=1, dashes=True)
    sims = get_simulations(G, n_sim=5, seed=1, perturb=("A", 1), return_samples=True, uncertain_interactions="enumerate",
                           presample=lambda symbols: {sp.Symbol("a_B,A"): 0})
    result = pathway_effects(G, "A", "B", sims=sims).loc[0, ["Present", "Zero"]].tolist()
    expected = [0.5, 1.0]
    assert result == expected


@pytest.mark.parametrize("pair_reciprocal", [True, False])
def test_pathway_effects_present_matches_nonzero_response_reciprocal_dashes(self_limited_pair, pair_reciprocal):
    G = self_limited_pair
    G.add_edge("A", "B", sign=1, dashes=True)
    G.add_edge("B", "A", sign=-1, dashes=True)
    sims = get_simulations(G, n_sim=200, seed=1, perturb=("B", 1), return_samples=True, pair_reciprocal=pair_reciprocal)
    result = pathway_effects(G, "B", "A", sims=sims).loc[0, "Present"]
    expected = np.mean([effect[0] != 0 for effect in sims["effects"]])
    assert 0 < expected < 1
    assert result == expected


def test_pathway_effects_present_requires_every_uncertain_link_fork(fork):
    fork["A"]["C"]["dashes"] = True
    fork["C"]["B"]["dashes"] = True
    table = pathway_effects(fork, "A", "B", n_sim=5, seed=1, uncertain_interactions="enumerate")
    result = table.set_index("Path")["Present"].to_dict()
    expected = {("A", "B"): 1.0, ("A", "C", "B"): 0.25}
    assert result == expected


def test_pathway_effects_present_with_64_uncertain_links_chain():
    nodes = [f"N{i:02d}" for i in range(65)]
    G = nx.DiGraph()
    G.add_nodes_from(nodes, category="state")
    G.add_edges_from(zip(nodes, nodes), sign=-1)
    G.add_edges_from(zip(nodes, nodes[1:]), sign=1, dashes=True)
    sims = get_simulations(G, n_sim=5, seed=4, perturb=("N00", 1), return_samples=True)
    result = pathway_effects(G, "N00", "N01", sims=sims).loc[0, "Present"]
    expected = np.mean([effect[1] != 0 for effect in sims["effects"]])
    assert max(code.bit_length() for code in sims["structures"]) > 63 and 0 < expected < 1
    assert result == expected


def test_pathway_effects_present_path_without_complementary_feedback_snowshoe_rp(snowshoe_rp):
    table = pathway_effects(snowshoe_rp, "R", "P", n_sim=50, seed=1)
    result = table.loc[table["Length"] == 1, ["Present", "Zero", "Contribution"]].values.tolist()
    expected = [[1.0, 1.0, 0.0]]
    assert result == expected


def test_pathway_effects_structurally_zero_response_singular_complement(singular_complement):
    table = pathway_effects(singular_complement, "S", "S", n_sim=200, seed=1)
    result = table.loc[0, ["Positive", "Negative", "Zero", "Contribution"]].tolist()
    expected = [0.0, 0.0, 1.0, 0.0]
    assert result == expected


def test_pathway_effects_zero_share_matches_zero_response_uncertain_self_effects(singular_complement):
    G = singular_complement
    G.add_edge("X", "X", sign=-1, dashes=True)
    G.add_edge("Y", "Y", sign=-1, dashes=True)
    sims = get_simulations(G, n_sim=300, seed=1, perturb=("S", 1), return_samples=True)
    result = pathway_effects(G, "S", "S", sims=sims).loc[0, "Zero"]
    expected = np.mean([effect[0] == 0 for effect in sims["effects"]])
    assert len(set(sims["structures"])) == 4 and 0 < expected < 1
    assert result == expected


def test_pathway_effects_tiny_present_term_keeps_sign_fork(fork):
    sims = get_simulations(fork, n_sim=5, seed=1, perturb=("A", 1), return_samples=True,
                           presample=lambda symbols: {sp.Symbol("a_B,A"): 1e-12})
    table = pathway_effects(fork, "A", "B", sims=sims)
    result = table.loc[table["Length"] == 1, ["Positive", "Zero"]].values.tolist()
    expected = [[1.0, 0.0]]
    assert result == expected


def test_pathway_effects_enumerate_snowshoe_dashed(snowshoe_dashed):
    result = pathway_effects(snowshoe_dashed, "R", "P", n_sim=50, seed=2, uncertain_interactions="enumerate")
    direct = result[result["Length"] == 1].iloc[0]
    assert direct["Zero"] == 0.75
    assert direct["Present"] == 0.5
    assert result["Contribution"].sum() == pytest.approx(1.0)
