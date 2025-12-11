"""Tests for qmm.extensions.senstability module."""

import sympy as sp

from qmm import get_nodes
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
    """Test structural_sensitivity with symbolic form across all levels for snowshoe."""
    result = structural_sensitivity(snowshoe)
    a_RR = sp.Symbol('a_R,R')
    a_RC = sp.Symbol('a_R,C')
    a_CR = sp.Symbol('a_C,R')
    a_CP = sp.Symbol('a_C,P')
    a_PC = sp.Symbol('a_P,C')
    a_PP = sp.Symbol('a_P,P')
    expected = sp.Matrix([
        [-a_CP*a_PC*a_RR, -a_CR*a_PP*a_RC, 0],
        [-a_CR*a_PP*a_RC, 0, -a_CP*a_PC*a_RR],
        [0, -a_CP*a_PC*a_RR, -a_CR*a_PP*a_RC]
    ])
    assert result == expected

def test_structural_sensitivity_form_symbolic_level_2_snowshoe(snowshoe):
    """Test structural_sensitivity with symbolic form at level 2 for snowshoe."""
    result = structural_sensitivity(snowshoe, level=2)
    a_RR = sp.Symbol('a_R,R')
    a_RC = sp.Symbol('a_R,C')
    a_CR = sp.Symbol('a_C,R')
    a_CP = sp.Symbol('a_C,P')
    a_PC = sp.Symbol('a_P,C')
    a_PP = sp.Symbol('a_P,P')
    expected = sp.Matrix([
        [-a_PP*a_RR, -a_CR*a_RC, 0],
        [-a_CR*a_RC, 0, -a_CP*a_PC],
        [0, -a_CP*a_PC, -a_PP*a_RR]
    ])
    assert result == expected

def test_structural_sensitivity_level_none_defaults_highest_snowshoe(snowshoe):
    """Test structural_sensitivity uses highest level when level is None."""
    n = len(get_nodes(snowshoe, 'state'))
    result = structural_sensitivity(snowshoe, level=None)
    expected = structural_sensitivity(snowshoe, level=n)
    assert result == expected

# =============================================================================
# net_structural_sensitivity
# =============================================================================

def test_net_structural_sensitivity_form_symbolic_snowshoe(snowshoe):
    """Test net_structural_sensitivity with symbolic form for snowshoe."""
    result = net_structural_sensitivity(snowshoe)
    expected = sp.Matrix([
        [-1, -1, 0],
        [-1, 0, -1],
        [0, -1, -1]
    ])
    assert result == expected

def test_net_structural_sensitivity_level_1_snowshoe(snowshoe):
    """Test net_structural_sensitivity at level 1 for snowshoe."""
    result = net_structural_sensitivity(snowshoe, level=1)
    expected = sp.Matrix([
        [-1, 0, 0],
        [0, 0, 0],
        [0, 0, -1]
    ])
    assert result == expected

def test_net_structural_sensitivity_form_symbolic_chain(chain):
    """Test net_structural_sensitivity with symbolic form for chain."""
    result = net_structural_sensitivity(chain)
    expected = sp.Matrix([
        [-5, -3, 0, 0, 0],
        [-3, -3, -2, 0, 0],
        [0, -2, -4, -2, 0],
        [0, 0, -2, -3, -3],
        [0, 0, 0, -3, -5]
    ])
    assert result == expected

# =============================================================================
# absolute_structural_sensitivity
# =============================================================================

def test_absolute_structural_sensitivity_form_symbolic_snowshoe(snowshoe):
    """Test absolute_structural_sensitivity with symbolic form for snowshoe."""
    result = absolute_structural_sensitivity(snowshoe)
    expected = sp.Matrix([
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1]
    ])
    assert result == expected

def test_absolute_structural_sensitivity_level_1_snowshoe(snowshoe):
    """Test absolute_structural_sensitivity at level 1 for snowshoe."""
    result = absolute_structural_sensitivity(snowshoe, level=1)
    expected = sp.Matrix([
        [1, 0, 0],
        [0, 0, 0],
        [0, 0, 1]
    ])
    assert result == expected

def test_absolute_structural_sensitivity_form_symbolic_chain(chain):
    """Test absolute_structural_sensitivity with symbolic form for chain."""
    result = absolute_structural_sensitivity(chain)
    expected = sp.Matrix([
        [5, 3, 0, 0, 0],
        [3, 3, 2, 0, 0],
        [0, 2, 4, 2, 0],
        [0, 0, 2, 3, 3],
        [0, 0, 0, 3, 5]
    ])
    assert result == expected

# =============================================================================
# weighted_structural_sensitivity
# =============================================================================

def test_weighted_structural_sensitivity_form_symbolic_snowshoe(snowshoe):
    """Test weighted_structural_sensitivity with symbolic form for snowshoe."""
    result = weighted_structural_sensitivity(snowshoe)
    expected = sp.Matrix([
        [-1, -1, sp.nan],
        [-1, sp.nan, -1],
        [sp.nan, -1, -1]
    ])
    assert result == expected

def test_weighted_structural_sensitivity_form_symbolic_chain(chain):
    """Test weighted_structural_sensitivity with symbolic form for chain."""
    result = weighted_structural_sensitivity(chain)
    expected = sp.Matrix([
        [-1, -1, sp.nan, sp.nan, sp.nan],
        [-1, -1, -1, sp.nan, sp.nan],
        [sp.nan, -1, -1, -1, sp.nan],
        [sp.nan, sp.nan, -1, -1, -1],
        [sp.nan, sp.nan, sp.nan, -1, -1]
    ])
    assert result == expected
