"""Tests for qmm.extensions.senstability module."""

import networkx as nx
import pytest
import sympy as sp

from qmm.core.helper import get_nodes
from qmm.core.structure import create_matrix
from qmm.core.stability import system_feedback
from qmm.extensions.senstability import (
    structural_sensitivity,
    net_structural_sensitivity,
    absolute_structural_sensitivity,
    weighted_structural_sensitivity,
)

# =============================================================================
# structural_sensitivity
# =============================================================================

def test_structural_sensitivity_form_symbolic_all_levels_snowshoe(snowshoe):
    result = structural_sensitivity(snowshoe)
    a_RR = sp.Symbol('a_R,R')
    a_RC = sp.Symbol('a_R,C')
    a_CR = sp.Symbol('a_C,R')
    a_CP = sp.Symbol('a_C,P')
    a_PC = sp.Symbol('a_P,C')
    a_PP = sp.Symbol('a_P,P')
    expected = sp.Matrix([
        [-a_CP*a_PC*a_RR, -a_CR*a_PP*a_RC,               0],
        [-a_CR*a_PP*a_RC,               0, -a_CP*a_PC*a_RR],
        [              0, -a_CP*a_PC*a_RR, -a_CR*a_PP*a_RC]])
    assert result == expected

def test_structural_sensitivity_form_symbolic_level_2_snowshoe(snowshoe):
    result = structural_sensitivity(snowshoe, level=2)
    a_RR = sp.Symbol('a_R,R')
    a_RC = sp.Symbol('a_R,C')
    a_CR = sp.Symbol('a_C,R')
    a_CP = sp.Symbol('a_C,P')
    a_PC = sp.Symbol('a_P,C')
    a_PP = sp.Symbol('a_P,P')
    expected = sp.Matrix([
        [-a_PP*a_RR, -a_CR*a_RC,          0],
        [-a_CR*a_RC,          0, -a_CP*a_PC],
        [         0, -a_CP*a_PC, -a_PP*a_RR]])
    assert result == expected

def test_structural_sensitivity_level_none_defaults_highest_snowshoe(snowshoe):
    n = len(get_nodes(snowshoe, 'state'))
    result = structural_sensitivity(snowshoe, level=None)
    expected = structural_sensitivity(snowshoe, level=n)
    assert result == expected

# =============================================================================
# net_structural_sensitivity
# =============================================================================

def test_net_structural_sensitivity_form_symbolic_snowshoe(snowshoe):
    result = net_structural_sensitivity(snowshoe)
    expected = sp.Matrix([
        [-1, -1,  0],
        [-1,  0, -1],
        [ 0, -1, -1]])
    assert result == expected

def test_net_structural_sensitivity_level_1_snowshoe(snowshoe):
    result = net_structural_sensitivity(snowshoe, level=1)
    expected = sp.Matrix([
        [-1, 0,  0],
        [ 0, 0,  0],
        [ 0, 0, -1]])
    assert result == expected

def test_net_structural_sensitivity_form_symbolic_chain(chain):
    result = net_structural_sensitivity(chain)
    expected = sp.Matrix([
        [-5, -3,  0,  0,  0],
        [-3, -3, -2,  0,  0],
        [ 0, -2, -4, -2,  0],
        [ 0,  0, -2, -3, -3],
        [ 0,  0,  0, -3, -5]])
    assert result == expected

# =============================================================================
# absolute_structural_sensitivity
# =============================================================================

def test_absolute_structural_sensitivity_form_symbolic_snowshoe(snowshoe):
    result = absolute_structural_sensitivity(snowshoe)
    expected = sp.Matrix([
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1]])
    assert result == expected

def test_absolute_structural_sensitivity_level_1_snowshoe(snowshoe):
    result = absolute_structural_sensitivity(snowshoe, level=1)
    expected = sp.Matrix([
        [1, 0, 0],
        [0, 0, 0],
        [0, 0, 1]])
    assert result == expected

def test_absolute_structural_sensitivity_form_symbolic_chain(chain):
    result = absolute_structural_sensitivity(chain)
    expected = sp.Matrix([
        [5, 3, 0, 0, 0],
        [3, 3, 2, 0, 0],
        [0, 2, 4, 2, 0],
        [0, 0, 2, 3, 3],
        [0, 0, 0, 3, 5]])
    assert result == expected

# =============================================================================
# weighted_structural_sensitivity
# =============================================================================

def test_weighted_structural_sensitivity_form_symbolic_snowshoe(snowshoe):
    result = weighted_structural_sensitivity(snowshoe)
    expected = sp.Matrix([
        [    -1,     -1, sp.nan],
        [    -1, sp.nan,     -1],
        [sp.nan,     -1,     -1]])
    assert result == expected

def test_weighted_structural_sensitivity_form_symbolic_chain(chain):
    result = weighted_structural_sensitivity(chain)
    expected = sp.Matrix([
        [    -1,     -1, sp.nan, sp.nan, sp.nan],
        [    -1,     -1,     -1, sp.nan, sp.nan],
        [sp.nan,     -1,     -1,     -1, sp.nan],
        [sp.nan, sp.nan,     -1,     -1,     -1],
        [sp.nan, sp.nan, sp.nan,     -1,     -1]])
    assert result == expected


# =============================================================================
# Additional coverage tests
# =============================================================================

def test_structural_sensitivity_collects_feedback_terms_with_each_link_snowshoe_rp(snowshoe_rp):
    A = create_matrix(snowshoe_rp)
    feedback = system_feedback(snowshoe_rp)
    result = []
    expected = []
    for level in range(1, A.rows + 1):
        terms = sp.expand(feedback[level]).as_ordered_terms()
        result.append((structural_sensitivity(snowshoe_rp, level=level).applyfunc(sp.expand),
                       net_structural_sensitivity(snowshoe_rp, level=level),
                       absolute_structural_sensitivity(snowshoe_rp, level=level)))
        symbolic = sp.zeros(A.rows, A.cols)
        net = sp.zeros(A.rows, A.cols)
        absolute = sp.zeros(A.rows, A.cols)
        for i in range(A.rows):
            for j in range(A.cols):
                if A[i, j] == 0:
                    continue
                symbol = next(iter(A[i, j].free_symbols))
                with_link = [term for term in terms if symbol in term.free_symbols]
                symbolic[i, j] = sp.expand(sum(with_link))
                net[i, j] = sum(term.as_coeff_Mul()[0] for term in with_link)
                absolute[i, j] = len(with_link)
        expected.append((symbolic, net, absolute))
    assert result == expected


def test_weighted_structural_sensitivity_keystone_predator(keystone_predator):
    result = [weighted_structural_sensitivity(keystone_predator, level=level) for level in (1, 2, 3)]
    expected = [
        sp.Matrix([[-1, sp.nan, sp.nan], [sp.nan, -1, sp.nan], [sp.nan, sp.nan, sp.nan]]),
        sp.Matrix([[-1, 1, -1], [1, -1, -1], [-1, -1, sp.nan]]),
        sp.Matrix([[-1, 1, 0], [1, -1, 0], [0, 0, sp.nan]]),
    ]
    assert result == expected


def test_sensitivity_results_follow_graph_edits_and_are_independent(snowshoe):
    graph = nx.DiGraph(snowshoe)
    for function in (structural_sensitivity, net_structural_sensitivity,
                     absolute_structural_sensitivity, weighted_structural_sensitivity):
        expected = function(graph).copy()
        function(graph)[0, 0] = 999
        assert function(graph) == expected
    assert net_structural_sensitivity(graph)[0, 0] == -1
    graph['R']['R']['sign'] = 1
    assert net_structural_sensitivity(graph)[0, 0] == 1


def test_structural_sensitivity_invalid_level_low(snowshoe):
    with pytest.raises(ValueError, match="Level must be between"):
        structural_sensitivity(snowshoe, level=0)


def test_net_structural_sensitivity_invalid_level_low(snowshoe):
    with pytest.raises(ValueError, match="Level must be between"):
        net_structural_sensitivity(snowshoe, level=0)


def test_absolute_structural_sensitivity_invalid_level_low(snowshoe):
    with pytest.raises(ValueError, match="Level must be between"):
        absolute_structural_sensitivity(snowshoe, level=0)


def test_weighted_structural_sensitivity_invalid_level_low(snowshoe):
    with pytest.raises(ValueError, match="Level must be between"):
        weighted_structural_sensitivity(snowshoe, level=0)
