"""Tests for qmm.extensions.paths module using snowshoe_io fixture."""

import pytest
import pandas as pd
import sympy as sp

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
    expected = sp.Matrix([
        [       -a_RR],
        [       -a_PP],
        [-a_CR * a_RC],
        [-a_CP * a_PC]])
    assert len(result) == len(expected)
    assert set(result) == set(expected)


# =============================================================================
# cycles_table
# =============================================================================

def test_cycles_table_symbolic_loops_snowshoe_io(snowshoe_io):
    result = len(cycles_table(snowshoe_io))
    expected = 4
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
    expected = sp.Matrix([
        [a_CR * a_PC * b_RI * c_OP],
        [      -a_CR * b_RI * c_OC],
        [      -a_PC * b_CI * c_OP],
        [              b_CI * c_OC]])
    assert result == expected


def test_get_paths_form_signed_inp1_out1_snowshoe_io(snowshoe_io):
    result = get_paths(snowshoe_io, "Inp1", "Out1", form="signed")
    expected = sp.Matrix([
        [ 1],
        [-1],
        [-1],
        [ 1]])
    assert result == expected


def test_get_paths_form_binary_inp1_out1_snowshoe_io(snowshoe_io):
    result = get_paths(snowshoe_io, "Inp1", "Out1", form="binary")
    expected = sp.Matrix([
        [1],
        [1],
        [1],
        [1]])
    assert result == expected


@pytest.mark.parametrize("source", ["R", "C", "P"])
def test_get_paths_no_path_to_new_state_snowshoe_io_na_source(snowshoe_io_na, source):
    result = get_paths(snowshoe_io_na, source, "N", form="signed")
    expected = sp.Matrix([sp.Integer(0)])
    assert result == expected


def test_get_paths_direct_input_output_symbolic_snowshoe_io_with_direct_edge(snowshoe_io_with_direct_edge):
    result = get_paths(snowshoe_io_with_direct_edge, "Inp1", "Out1", form="symbolic")
    d_Inp1_Out1 = sp.Symbol('d_Out1,Inp1')
    expected = any(d_Inp1_Out1 in path.free_symbols for path in result)
    assert expected


def test_system_paths_rejects_direct_io_edge(snowshoe_io_with_direct_edge):
    with pytest.raises(ValueError) as exc_info:
        system_paths(snowshoe_io_with_direct_edge, "Inp1", "Out1")
    assert "Direct input to output edge" in str(exc_info.value)


def test_weighted_paths_rejects_direct_io_edge(snowshoe_io_with_direct_edge):
    with pytest.raises(ValueError) as exc_info:
        weighted_paths(snowshoe_io_with_direct_edge, "Inp1", "Out1")
    assert "Direct input to output edge" in str(exc_info.value)


def test_path_metrics_rejects_direct_io_edge(snowshoe_io_with_direct_edge):
    with pytest.raises(ValueError) as exc_info:
        path_metrics(snowshoe_io_with_direct_edge, "Inp1", "Out1")
    assert "Direct input to output edge" in str(exc_info.value)


def test_get_paths_symbolic_output_to_output_edge_output_to_output_graph(output_to_output_graph):
    result = len(get_paths(output_to_output_graph, 'A', 'Out2', form='symbolic'))
    expected = 1
    assert result == expected


def test_get_paths_invalid_source_snowshoe_io(snowshoe_io):
    with pytest.raises(ValueError):
        get_paths(snowshoe_io, "Invalid", "Out1")


def test_get_paths_invalid_target_snowshoe_io(snowshoe_io):
    with pytest.raises(ValueError):
        get_paths(snowshoe_io, "Inp1", "Invalid")


def test_get_paths_source_eq_target_snowshoe_io(snowshoe_io):
    assert get_paths(snowshoe_io, "R", "R") == sp.Matrix([1])


# =============================================================================
# paths_table
# =============================================================================

def test_paths_table_inp1_out1_snowshoe_io(snowshoe_io):
    result = len(paths_table(snowshoe_io, "Inp1", "Out1"))
    expected = 4
    assert result == expected


def test_paths_table_no_path_available_snowshoe_io_na(snowshoe_io_na):
    result = paths_table(snowshoe_io_na, "R", "N")
    expected = None
    assert result == expected


def test_paths_table_invalid_source_snowshoe_io(snowshoe_io):
    with pytest.raises(ValueError):
        paths_table(snowshoe_io, "Invalid", "Out1")


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
    assert result == expected


def test_complementary_feedback_invalid_source_snowshoe_io(snowshoe_io):
    with pytest.raises(ValueError):
        complementary_feedback(snowshoe_io, "Invalid", "Out1")


def test_complementary_feedback_source_eq_target_snowshoe_io(snowshoe_io):
    result = complementary_feedback(snowshoe_io, "R", "R", form="symbolic")
    expected = system_feedback(snowshoe_io.subgraph(['C', 'P']).copy(), level=2)
    assert result == expected


def test_complementary_feedback_invalid_form_feedback_test_graph(feedback_test_graph):
    with pytest.raises(ValueError):
        complementary_feedback(feedback_test_graph, 'A', 'B', form='invalid')


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
    assert result == expected


def test_system_paths_form_signed_inp1_out1_snowshoe_io(snowshoe_io):
    result = system_paths(snowshoe_io, "Inp1", "Out1", form="signed")
    expected = sp.Matrix([
        [ 1],
        [-1],
        [-1],
        [ 1]])
    assert result == expected


def test_system_paths_form_binary_inp1_out1_snowshoe_io(snowshoe_io):
    result = system_paths(snowshoe_io, "Inp1", "Out1", form="binary")
    expected = sp.Matrix([
        [1],
        [1],
        [1],
        [1]])
    assert result == expected


def test_system_paths_source_eq_target_snowshoe_io(snowshoe_io):
    states = get_nodes(snowshoe_io, "state")
    cum = cumulative_effects(snowshoe_io, form="symbolic")
    for i, node in enumerate(states):
        result = sp.expand(sum(system_paths(snowshoe_io, node, node, form="symbolic")))
        assert sp.simplify(result - sp.expand(cum[i, i])) == 0


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
    assert result == expected


def test_weighted_paths_nan_feedback_snowshoe_io(snowshoe_io):
    result = weighted_paths(snowshoe_io, "Inp1", "Out1")
    expected = (isinstance(result, sp.Matrix), result.shape[0])
    assert expected == (True, 4)


def test_weighted_paths_nan_feedback_replacement_nan_feedback_graph(nan_feedback_graph):
    result = weighted_paths(nan_feedback_graph, 'A', 'B')
    assert result.shape == (1, 1)
    assert result[0] == 0


def test_weighted_paths_invalid_source_snowshoe_io(snowshoe_io):
    with pytest.raises(ValueError):
        weighted_paths(snowshoe_io, "Invalid", "Out1")


def test_weighted_paths_source_eq_target_snowshoe_io(snowshoe_io):
    with pytest.raises(ValueError):
        weighted_paths(snowshoe_io, "R", "R")


# =============================================================================
# path_metrics
# =============================================================================

def test_path_metrics_inp1_out1_snowshoe_io(snowshoe_io):
    result = len(path_metrics(snowshoe_io, "Inp1", "Out1"))
    expected = 4
    assert result == expected


def test_path_metrics_no_path_available_snowshoe_io_na(snowshoe_io_na):
    result = path_metrics(snowshoe_io_na, "R", "N")
    expected = pd.DataFrame()
    assert result.equals(expected)


def test_path_metrics_invalid_source_snowshoe_io(snowshoe_io):
    with pytest.raises(ValueError):
        path_metrics(snowshoe_io, "Invalid", "Out1")


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
            result = sp.expand(sum(system_paths(snowshoe_io_na, src, tgt, form=form)))
            assert sp.simplify(result - sp.expand(cum[i, j])) == 0
