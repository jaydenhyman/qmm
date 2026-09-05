"""Tests for qmm.extensions.paths module using snowshoe_io fixture."""

import numpy as np
import pytest
import pandas as pd
import sympy as sp
import networkx as nx

from qmm.core.helper import get_nodes
from qmm.core.stability import system_feedback
from qmm.extensions.effects import cumulative_effects
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
    _pathway_terms,
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
    assert list(result["Path"]) == expected_paths
    assert list(result["Complementary subsystem"]) == expected_complements
    assert list(result["Sign"]) == ["+", "\u2212", "\u2212", "+"]


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
        paths, terms, sims = _pathway_terms(mesocosm, "P", target, 200, "uniform", 3, False)
        result = (terms.shape, terms.sum(axis=1))
        expected = ((200, len(paths)), np.array([effect[responders.index(target)] for effect in sims["effects"]]))
        assert result[0] == expected[0]
        assert np.allclose(result[1], expected[1])


def test_pathway_effects_terms_sum_inp1_r_snowshoe_io(snowshoe_io):
    responders = get_nodes(snowshoe_io, "state") + get_nodes(snowshoe_io, "output")
    paths, terms, sims = _pathway_terms(snowshoe_io, "Inp1", "R", 150, "uniform", 5, False)
    result = terms.sum(axis=1)
    expected = np.array([effect[responders.index("R")] for effect in sims["effects"]])
    assert len(paths) > 0
    assert np.allclose(result, expected)


def test_pathway_effects_terms_sum_inp1_out1_snowshoe_io(snowshoe_io):
    responders = get_nodes(snowshoe_io, "state") + get_nodes(snowshoe_io, "output")
    paths, terms, sims = _pathway_terms(snowshoe_io, "Inp1", "Out1", 150, "uniform", 5, False)
    result = (len(paths), terms.sum(axis=1))
    expected = (4, np.array([effect[responders.index("Out1")] for effect in sims["effects"]]))
    assert result[0] == expected[0]
    assert np.allclose(result[1], expected[1])


def test_pathway_effects_table_inp1_out1_snowshoe_io(snowshoe_io):
    result = pathway_effects(snowshoe_io, "Inp1", "Out1", n_sim=300, seed=1)
    expected_cols = ["Length", "Path", "Sign", "Positive", "Negative", "Zero", "Contribution"]
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
    result = pathway_effects(snowshoe_dashed, "R", "P", n_sim=300, seed=2, average_uncertain=True)
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
    with pytest.raises(ValueError, match="No simulations matched the observations"):
        pathway_effects(snowshoe_io, "Inp1", "Out1", n_sim=50, seed=1, observe="Out1:0")


def test_pathway_effects_observe_unknown_node_snowshoe_io(snowshoe_io):
    with pytest.raises(ValueError, match="Unknown observation node"):
        pathway_effects(snowshoe_io, "Inp1", "Out1", n_sim=10, observe="Missing:+")

