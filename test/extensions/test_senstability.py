"""Tests for qmm.extensions.senstability module."""

import pytest
import sympy as sp

from qmm.core.helper import get_nodes
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
