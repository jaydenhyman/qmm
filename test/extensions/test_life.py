"""Tests for qmm.extensions.life module."""

import pytest
import sympy as sp

from qmm.core.helper import get_nodes
from qmm.extensions.life import (
    birth_matrix,
    death_matrix,
    life_expectancy_change,
    net_life_expectancy_change,
    absolute_life_expectancy_change,
    weighted_predictions_life_expectancy,
)

# =============================================================================
# birth_matrix
# =============================================================================

def test_birth_matrix_form_signed_snowshoe(snowshoe):
    result = birth_matrix(snowshoe, form='signed')
    expected = sp.Matrix([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0]])
    assert result == expected


def test_birth_matrix_form_symbolic_snowshoe(snowshoe):
    result = birth_matrix(snowshoe, form='symbolic')
    expected = sp.Matrix([
        [                 0,                  0, 0],
        [sp.Symbol('a_C,R'),                  0, 0],
        [                 0, sp.Symbol('a_P,C'), 0]])
    assert result == expected


def test_birth_matrix_form_signed_perturb_first_state_snowshoe(snowshoe):
    nodes = get_nodes(snowshoe, 'state')
    result = birth_matrix(snowshoe, form='signed', perturb=nodes[0])
    expected = sp.Matrix([0, 1, 0])
    assert result == expected


def test_birth_matrix_form_symbolic_perturb_first_state_snowshoe(snowshoe):
    nodes = get_nodes(snowshoe, 'state')
    result = birth_matrix(snowshoe, form='symbolic', perturb=nodes[0])
    expected = sp.Matrix([0, sp.Symbol('a_C,R'), 0])
    assert result == expected


def test_birth_matrix_form_signed_chain(chain):
    result = birth_matrix(chain, form='signed')
    expected = sp.Matrix([
        [0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0]])
    assert result == expected


def test_birth_matrix_form_signed_mesocosm(mesocosm):
    result = birth_matrix(mesocosm, form='signed')
    expected = sp.Matrix([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 0]])
    assert result == expected

# =============================================================================
# death_matrix
# =============================================================================

def test_death_matrix_form_signed_snowshoe(snowshoe):
    result = death_matrix(snowshoe, form='signed')
    expected = sp.Matrix([
        [1, 1, 0],
        [0, 0, 1],
        [0, 0, 1]])
    assert result == expected


def test_death_matrix_form_symbolic_snowshoe(snowshoe):
    result = death_matrix(snowshoe, form='symbolic')
    expected = sp.Matrix([
        [sp.Symbol('a_R,R'), sp.Symbol('a_R,C'),                  0],
        [                 0,                  0, sp.Symbol('a_C,P')],
        [                 0,                  0, sp.Symbol('a_P,P')]])
    assert result == expected


def test_death_matrix_form_signed_perturb_R_snowshoe(snowshoe):
    result = death_matrix(snowshoe, form='signed', perturb='R')
    expected = sp.Matrix([1, 0, 0])
    assert result == expected


def test_death_matrix_form_symbolic_perturb_first_state_snowshoe(snowshoe):
    nodes = get_nodes(snowshoe, 'state')
    result = death_matrix(snowshoe, form='symbolic', perturb=nodes[0])
    expected = sp.Matrix([sp.Symbol('a_R,R'), 0, 0])
    assert result == expected


def test_death_matrix_form_signed_chain(chain):
    result = death_matrix(chain, form='signed')
    expected = sp.Matrix([
        [1, 1, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 0, 1]])
    assert result == expected


def test_death_matrix_form_signed_mesocosm(mesocosm):
    result = death_matrix(mesocosm, form='signed')
    expected = sp.Matrix([
        [1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 1]])
    assert result == expected

# =============================================================================
# life_expectancy_change
# =============================================================================

def test_life_expectancy_change_form_signed_birth_snowshoe(snowshoe):
    result = life_expectancy_change(snowshoe, form='signed', type='birth')
    expected = sp.Matrix([
        [-2,  0,  0],
        [-1, -1, -1],
        [-1, -1, -1]])
    assert result == expected


def test_life_expectancy_change_form_signed_death_snowshoe(snowshoe):
    result = life_expectancy_change(snowshoe, form='signed', type='death')
    expected = sp.Matrix([
        [ 0,  0,  0],
        [-1,  1, -1],
        [-1, -1,  1]])
    assert result == expected


def test_life_expectancy_change_form_signed_birth_perturb_first_state_snowshoe(snowshoe):
    nodes = get_nodes(snowshoe, 'state')
    result = life_expectancy_change(snowshoe, form='signed', type='birth', perturb=nodes[0])
    expected = sp.Matrix([-2, -1, -1])
    assert result == expected


def test_life_expectancy_change_invalid_perturb_node(snowshoe):
    with pytest.raises(ValueError, match="Perturbation node must be one of"):
        life_expectancy_change(snowshoe, form='signed', type='birth', perturb='InvalidNode')


def test_life_expectancy_change_form_signed_birth_chain(chain):
    result = life_expectancy_change(chain, form='signed', type='birth')
    expected = sp.Matrix([
        [-8,  0,  0,  0,  0],
        [-5, -5, -2,  1, -1],
        [-3, -3, -6, -1,  1],
        [-2, -2, -4, -6, -2],
        [-1, -1, -2, -3, -5]])
    assert result == expected


def test_life_expectancy_change_form_symbolic_birth_snowshoe(snowshoe):
    result = life_expectancy_change(snowshoe, form='symbolic', type='birth')
    expected = sp.Matrix([
        [-sp.Symbol('a_C,P')*sp.Symbol('a_P,C')*sp.Symbol('a_R,R') - sp.Symbol('a_C,R')*sp.Symbol('a_P,P')*sp.Symbol('a_R,C'),                                                         0,                                                         0],
        [                                                           -sp.Symbol('a_C,P')*sp.Symbol('a_C,R')*sp.Symbol('a_P,C'), -sp.Symbol('a_C,P')*sp.Symbol('a_P,C')*sp.Symbol('a_R,R'), -sp.Symbol('a_C,P')*sp.Symbol('a_C,R')*sp.Symbol('a_R,C')],
        [                                                           -sp.Symbol('a_C,R')*sp.Symbol('a_P,C')*sp.Symbol('a_P,P'), -sp.Symbol('a_P,C')*sp.Symbol('a_P,P')*sp.Symbol('a_R,R'), -sp.Symbol('a_C,R')*sp.Symbol('a_P,P')*sp.Symbol('a_R,C')]])
    assert result == expected


def test_life_expectancy_change_form_symbolic_death_snowshoe(snowshoe):
    result = life_expectancy_change(snowshoe, form='symbolic', type='death')
    expected = sp.Matrix([
        [                                                        0,                                                         0,                                                         0],
        [-sp.Symbol('a_C,P')*sp.Symbol('a_C,R')*sp.Symbol('a_P,C'),  sp.Symbol('a_C,R')*sp.Symbol('a_P,P')*sp.Symbol('a_R,C'), -sp.Symbol('a_C,P')*sp.Symbol('a_C,R')*sp.Symbol('a_R,C')],
        [-sp.Symbol('a_C,R')*sp.Symbol('a_P,C')*sp.Symbol('a_P,P'), -sp.Symbol('a_P,C')*sp.Symbol('a_P,P')*sp.Symbol('a_R,R'),  sp.Symbol('a_C,P')*sp.Symbol('a_P,C')*sp.Symbol('a_R,R')]])
    assert result == expected

# =============================================================================
# net_life_expectancy_change
# =============================================================================

def test_net_life_expectancy_change_birth_snowshoe(snowshoe):
    result = net_life_expectancy_change(snowshoe, type='birth')
    expected = sp.Matrix([
        [-2,  0,  0],
        [-1, -1, -1],
        [-1, -1, -1]])
    assert result == expected


def test_net_life_expectancy_change_death_snowshoe(snowshoe):
    result = net_life_expectancy_change(snowshoe, type='death')
    expected = sp.Matrix([
        [ 0,  0,  0],
        [-1,  1, -1],
        [-1, -1,  1]])
    assert result == expected


def test_net_life_expectancy_change_birth_chain(chain):
    result = net_life_expectancy_change(chain, type='birth')
    expected = sp.Matrix([
        [-8,  0,  0,  0,  0],
        [-5, -5, -2,  1, -1],
        [-3, -3, -6, -1,  1],
        [-2, -2, -4, -6, -2],
        [-1, -1, -2, -3, -5]])
    assert result == expected


def test_net_life_expectancy_change_birth_mesocosm(mesocosm):
    result = net_life_expectancy_change(mesocosm, type='birth')
    expected = sp.Matrix([
        [-2,  0,  0,  0, 0,  0,  0, 0],
        [-1, -1, -1,  1, 0, -1,  1, 0],
        [-1,  1, -3,  1, 0, -1,  1, 0],
        [-1,  1, -1, -1, 0, -1,  1, 0],
        [ 1, -3,  5, -1, 0,  1, -5, 2],
        [ 0, -2,  2,  0, 0,  0, -2, 0],
        [ 0, -2,  2,  0, 0,  0, -2, 0],
        [ 0, -2,  2,  0, 0,  0, -2, 0]])
    assert result == expected


def test_net_life_expectancy_change_death_mesocosm(mesocosm):
    result = net_life_expectancy_change(mesocosm, type='death')
    expected = sp.Matrix([
        [ 0,  0,  0,  0, 0,  0,  0, 0],
        [-1,  1, -1,  1, 0, -1,  1, 0],
        [-1,  1, -1,  1, 0, -1,  1, 0],
        [-1,  1, -1,  1, 0, -1,  1, 0],
        [ 1, -3,  5, -1, 2,  1, -5, 2],
        [ 0, -2,  2,  0, 0,  2, -2, 0],
        [ 0, -2,  2,  0, 0,  0,  0, 0],
        [ 0, -2,  2,  0, 0,  0, -2, 2]])
    assert result == expected


# =============================================================================
# absolute_life_expectancy_change
# =============================================================================

def test_absolute_life_expectancy_change_birth_snowshoe(snowshoe):
    result = absolute_life_expectancy_change(snowshoe, type='birth')
    expected = sp.Matrix([
        [2, 0, 0],
        [1, 1, 1],
        [1, 1, 1]])
    assert result == expected


def test_absolute_life_expectancy_change_death_snowshoe(snowshoe):
    result = absolute_life_expectancy_change(snowshoe, type='death')
    expected = sp.Matrix([
        [0, 0, 0],
        [1, 1, 1],
        [1, 1, 1]])
    assert result == expected


def test_absolute_life_expectancy_change_death_chain(chain):
    result = absolute_life_expectancy_change(chain, type='death')
    expected = sp.Matrix([
        [0, 0, 0, 0, 0],
        [5, 3, 2, 1, 1],
        [3, 3, 2, 1, 1],
        [2, 2, 4, 2, 2],
        [1, 1, 2, 3, 3]])
    assert result == expected


def test_absolute_life_expectancy_change_birth_mesocosm(mesocosm):
    result = absolute_life_expectancy_change(mesocosm, type='birth')
    expected = sp.Matrix([
        [18,  0, 0,  0,  0, 0, 0, 0],
        [ 1, 11, 9,  1,  2, 1, 5, 2],
        [ 1,  7, 9,  1,  2, 1, 5, 2],
        [ 1,  7, 9, 17,  2, 1, 5, 2],
        [ 7,  7, 9,  7, 14, 7, 7, 4],
        [ 2,  4, 4,  2,  4, 2, 8, 4],
        [ 2,  4, 4,  2,  4, 2, 8, 4],
        [ 2,  4, 4,  2,  4, 2, 8, 4]])
    assert result == expected


def test_absolute_life_expectancy_change_death_mesocosm(mesocosm):
    result = absolute_life_expectancy_change(mesocosm, type='death')
    expected = sp.Matrix([
        [0, 0, 0, 0, 0,  0,  0,  0],
        [1, 7, 9, 1, 2,  1,  5,  2],
        [1, 7, 9, 1, 2,  1,  5,  2],
        [1, 7, 9, 1, 2,  1,  5,  2],
        [7, 7, 9, 7, 4,  7,  7,  4],
        [2, 4, 4, 2, 4, 16,  8,  4],
        [2, 4, 4, 2, 4,  2, 10,  4],
        [2, 4, 4, 2, 4,  2,  8, 14]])
    assert result == expected


# =============================================================================
# weighted_predictions_life_expectancy
# =============================================================================

def test_weighted_predictions_life_expectancy_birth_snowshoe(snowshoe):
    result = weighted_predictions_life_expectancy(snowshoe, type='birth')
    expected = sp.Matrix([
        [-1, sp.nan, sp.nan],
        [-1,     -1,     -1],
        [-1,     -1,     -1]])
    assert result == expected


def test_weighted_predictions_life_expectancy_birth_as_nan_false_snowshoe(snowshoe):
    result = weighted_predictions_life_expectancy(snowshoe, type='birth', as_nan=False)
    expected = sp.Matrix([
        [-1,  1,  1],
        [-1, -1, -1],
        [-1, -1, -1]])
    assert result == expected


def test_weighted_predictions_life_expectancy_birth_as_abs_true_snowshoe(snowshoe):
    result = weighted_predictions_life_expectancy(snowshoe, type='birth', as_abs=True)
    expected = sp.Matrix([
        [1, sp.nan, sp.nan],
        [1,      1,      1],
        [1,      1,      1]])
    assert result == expected


def test_weighted_predictions_life_expectancy_death_snowshoe(snowshoe):
    result = weighted_predictions_life_expectancy(snowshoe, type='death')
    expected = sp.Matrix([
        [sp.nan, sp.nan, sp.nan],
        [    -1,      1,     -1],
        [    -1,     -1,      1]])
    assert result == expected


def test_weighted_predictions_life_expectancy_birth_chain(chain):
    result = weighted_predictions_life_expectancy(chain, type='birth')
    expected = sp.Matrix([
        [-1, sp.nan, sp.nan, sp.nan, sp.nan],
        [-1,     -1,     -1,      1,     -1],
        [-1,     -1,     -1,     -1,      1],
        [-1,     -1,     -1,     -1,     -1],
        [-1,     -1,     -1,     -1,     -1]])
    assert result == expected


def test_weighted_predictions_life_expectancy_birth_mesocosm(mesocosm):
    result = weighted_predictions_life_expectancy(mesocosm, type='birth')
    expected = sp.Matrix([
        [sp.Rational(-1, 9),              sp.nan,             sp.nan,              sp.nan, sp.nan,            sp.nan,             sp.nan,            sp.nan],
        [                -1, sp.Rational(-1, 11), sp.Rational(-1, 9),                   1,      0,                -1,  sp.Rational(1, 5),                 0],
        [                -1,   sp.Rational(1, 7), sp.Rational(-1, 3),                   1,      0,                -1,  sp.Rational(1, 5),                 0],
        [                -1,   sp.Rational(1, 7), sp.Rational(-1, 9), sp.Rational(-1, 17),      0,                -1,  sp.Rational(1, 5),                 0],
        [ sp.Rational(1, 7),  sp.Rational(-3, 7),  sp.Rational(5, 9),  sp.Rational(-1, 7),      0, sp.Rational(1, 7), sp.Rational(-5, 7), sp.Rational(1, 2)],
        [                 0,  sp.Rational(-1, 2),  sp.Rational(1, 2),                   0,      0,                 0, sp.Rational(-1, 4),                 0],
        [                 0,  sp.Rational(-1, 2),  sp.Rational(1, 2),                   0,      0,                 0, sp.Rational(-1, 4),                 0],
        [                 0,  sp.Rational(-1, 2),  sp.Rational(1, 2),                   0,      0,                 0, sp.Rational(-1, 4),                 0]])
    assert result == expected


def test_weighted_predictions_life_expectancy_death_mesocosm(mesocosm):
    result = weighted_predictions_life_expectancy(mesocosm, type='death')
    expected = sp.Matrix([
        [           sp.nan,             sp.nan,             sp.nan,             sp.nan,            sp.nan,            sp.nan,             sp.nan,            sp.nan],
        [               -1,  sp.Rational(1, 7), sp.Rational(-1, 9),                  1,                 0,                -1,  sp.Rational(1, 5),                 0],
        [               -1,  sp.Rational(1, 7), sp.Rational(-1, 9),                  1,                 0,                -1,  sp.Rational(1, 5),                 0],
        [               -1,  sp.Rational(1, 7), sp.Rational(-1, 9),                  1,                 0,                -1,  sp.Rational(1, 5),                 0],
        [sp.Rational(1, 7), sp.Rational(-3, 7),  sp.Rational(5, 9), sp.Rational(-1, 7), sp.Rational(1, 2), sp.Rational(1, 7), sp.Rational(-5, 7), sp.Rational(1, 2)],
        [                0, sp.Rational(-1, 2),  sp.Rational(1, 2),                  0,                 0, sp.Rational(1, 8), sp.Rational(-1, 4),                 0],
        [                0, sp.Rational(-1, 2),  sp.Rational(1, 2),                  0,                 0,                 0,                  0,                 0],
        [                0, sp.Rational(-1, 2),  sp.Rational(1, 2),                  0,                 0,                 0, sp.Rational(-1, 4), sp.Rational(1, 7)]])
    assert result == expected


def test_weighted_predictions_life_expectancy_birth_as_nan_false_mesocosm(mesocosm):
    result = weighted_predictions_life_expectancy(mesocosm, type='birth', as_nan=False)
    expected = sp.Matrix([
        [sp.Rational(-1, 9),                   1,                  1,                   1, 1,                 1,                  1,                 1],
        [                -1, sp.Rational(-1, 11), sp.Rational(-1, 9),                   1, 0,                -1,  sp.Rational(1, 5),                 0],
        [                -1,   sp.Rational(1, 7), sp.Rational(-1, 3),                   1, 0,                -1,  sp.Rational(1, 5),                 0],
        [                -1,   sp.Rational(1, 7), sp.Rational(-1, 9), sp.Rational(-1, 17), 0,                -1,  sp.Rational(1, 5),                 0],
        [ sp.Rational(1, 7),  sp.Rational(-3, 7),  sp.Rational(5, 9),  sp.Rational(-1, 7), 0, sp.Rational(1, 7), sp.Rational(-5, 7), sp.Rational(1, 2)],
        [                 0,  sp.Rational(-1, 2),  sp.Rational(1, 2),                   0, 0,                 0, sp.Rational(-1, 4),                 0],
        [                 0,  sp.Rational(-1, 2),  sp.Rational(1, 2),                   0, 0,                 0, sp.Rational(-1, 4),                 0],
        [                 0,  sp.Rational(-1, 2),  sp.Rational(1, 2),                   0, 0,                 0, sp.Rational(-1, 4),                 0]])
    assert result == expected


def test_weighted_predictions_life_expectancy_birth_as_abs_true_mesocosm(mesocosm):
    result = weighted_predictions_life_expectancy(mesocosm, type='birth', as_abs=True)
    expected = sp.Matrix([
        [sp.Rational(1, 9),             sp.nan,            sp.nan,             sp.nan, sp.nan,            sp.nan,            sp.nan,            sp.nan],
        [                1, sp.Rational(1, 11), sp.Rational(1, 9),                  1,      0,                 1, sp.Rational(1, 5),                 0],
        [                1,  sp.Rational(1, 7), sp.Rational(1, 3),                  1,      0,                 1, sp.Rational(1, 5),                 0],
        [                1,  sp.Rational(1, 7), sp.Rational(1, 9), sp.Rational(1, 17),      0,                 1, sp.Rational(1, 5),                 0],
        [sp.Rational(1, 7),  sp.Rational(3, 7), sp.Rational(5, 9),  sp.Rational(1, 7),      0, sp.Rational(1, 7), sp.Rational(5, 7), sp.Rational(1, 2)],
        [                0,  sp.Rational(1, 2), sp.Rational(1, 2),                  0,      0,                 0, sp.Rational(1, 4),                 0],
        [                0,  sp.Rational(1, 2), sp.Rational(1, 2),                  0,      0,                 0, sp.Rational(1, 4),                 0],
        [                0,  sp.Rational(1, 2), sp.Rational(1, 2),                  0,      0,                 0, sp.Rational(1, 4),                 0]])
    assert result == expected


def test_weighted_predictions_life_expectancy_death_as_nan_false_mesocosm(mesocosm):
    result = weighted_predictions_life_expectancy(mesocosm, type='death', as_nan=False)
    expected = sp.Matrix([
        [                1,                  1,                  1,                  1,                 1,                 1,                  1,                 1],
        [               -1,  sp.Rational(1, 7), sp.Rational(-1, 9),                  1,                 0,                -1,  sp.Rational(1, 5),                 0],
        [               -1,  sp.Rational(1, 7), sp.Rational(-1, 9),                  1,                 0,                -1,  sp.Rational(1, 5),                 0],
        [               -1,  sp.Rational(1, 7), sp.Rational(-1, 9),                  1,                 0,                -1,  sp.Rational(1, 5),                 0],
        [sp.Rational(1, 7), sp.Rational(-3, 7),  sp.Rational(5, 9), sp.Rational(-1, 7), sp.Rational(1, 2), sp.Rational(1, 7), sp.Rational(-5, 7), sp.Rational(1, 2)],
        [                0, sp.Rational(-1, 2),  sp.Rational(1, 2),                  0,                 0, sp.Rational(1, 8), sp.Rational(-1, 4),                 0],
        [                0, sp.Rational(-1, 2),  sp.Rational(1, 2),                  0,                 0,                 0,                  0,                 0],
        [                0, sp.Rational(-1, 2),  sp.Rational(1, 2),                  0,                 0,                 0, sp.Rational(-1, 4), sp.Rational(1, 7)]])
    assert result == expected


def test_weighted_predictions_life_expectancy_death_as_abs_true_mesocosm(mesocosm):
    result = weighted_predictions_life_expectancy(mesocosm, type='death', as_abs=True)
    expected = sp.Matrix([
        [           sp.nan,            sp.nan,            sp.nan,            sp.nan,            sp.nan,            sp.nan,            sp.nan,            sp.nan],
        [                1, sp.Rational(1, 7), sp.Rational(1, 9),                 1,                 0,                 1, sp.Rational(1, 5),                 0],
        [                1, sp.Rational(1, 7), sp.Rational(1, 9),                 1,                 0,                 1, sp.Rational(1, 5),                 0],
        [                1, sp.Rational(1, 7), sp.Rational(1, 9),                 1,                 0,                 1, sp.Rational(1, 5),                 0],
        [sp.Rational(1, 7), sp.Rational(3, 7), sp.Rational(5, 9), sp.Rational(1, 7), sp.Rational(1, 2), sp.Rational(1, 7), sp.Rational(5, 7), sp.Rational(1, 2)],
        [                0, sp.Rational(1, 2), sp.Rational(1, 2),                 0,                 0, sp.Rational(1, 8), sp.Rational(1, 4),                 0],
        [                0, sp.Rational(1, 2), sp.Rational(1, 2),                 0,                 0,                 0,                 0,                 0],
        [                0, sp.Rational(1, 2), sp.Rational(1, 2),                 0,                 0,                 0, sp.Rational(1, 4), sp.Rational(1, 7)]])
    assert result == expected


def test_weighted_predictions_life_expectancy_invalid_type_snowshoe(snowshoe):
    with pytest.raises(ValueError) as exc_info:
        weighted_predictions_life_expectancy(snowshoe, type='invalid')
    result = str(exc_info.value)
    expected = "type must be either 'birth' or 'death'"
    assert result == expected


# =============================================================================
# Additional coverage tests
# =============================================================================

def test_birth_matrix_invalid_perturb_node(snowshoe):
    with pytest.raises(ValueError, match="Perturbation node must be one of"):
        birth_matrix(snowshoe, perturb="Invalid")


def test_death_matrix_invalid_perturb_node(snowshoe):
    with pytest.raises(ValueError, match="Perturbation node must be one of"):
        death_matrix(snowshoe, perturb="Invalid")


def test_life_expectancy_change_invalid_type(snowshoe):
    with pytest.raises(ValueError, match="type must be either 'birth' or 'death'"):
        life_expectancy_change(snowshoe, type="invalid")


def test_weighted_predictions_life_expectancy_invalid_type_coverage(snowshoe):
    with pytest.raises(ValueError, match="type must be either 'birth' or 'death'"):
        weighted_predictions_life_expectancy(snowshoe, type="invalid")

