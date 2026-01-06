"""Tests for qmm.core.press module."""

import pytest
import numpy as np
import sympy as sp
from unittest.mock import patch

from qmm.core.press import (
    adjoint_matrix,
    absolute_feedback_matrix,
    weighted_predictions_matrix,
    sign_determinacy_matrix,
    numerical_simulations,
)


# =============================================================================
# adjoint_matrix()
# =============================================================================

def test_adjoint_matrix_form_signed_snowshoe(snowshoe):
    result = adjoint_matrix(snowshoe, form='signed')
    expected = sp.Matrix([
        [1, -1,  1],
        [1,  1, -1],
        [1,  1,  1]])
    assert result == expected


def test_adjoint_matrix_form_signed_chain(chain):
    result = adjoint_matrix(chain, form='signed')
    expected = sp.Matrix([
        [5, -3,  2, -1,  1],
        [3,  3, -2,  1, -1],
        [2,  2,  4, -2,  2],
        [1,  1,  2,  3, -3],
        [1,  1,  2,  3,  5]])
    assert result == expected


def test_adjoint_matrix_form_signed_mesocosm(mesocosm):
    result = adjoint_matrix(mesocosm, form='signed')
    expected = sp.Matrix([
        [ 1, -1,  1, -1,  0,  1, -1,  0],
        [-1,  3, -5,  1, -2, -1,  5, -2],
        [ 1, -1,  3, -1,  2, -1, -3,  2],
        [ 1, -1,  1,  1,  0,  1, -1,  0],
        [ 0,  2, -2,  0,  0,  0,  0,  0],
        [ 1, -1,  3, -1,  0,  1, -1,  0],
        [-1,  1, -3,  1,  0, -1,  3, -2],
        [ 0,  2, -2,  0,  0,  0,  2,  0]])
    assert result == expected


def test_adjoint_matrix_form_symbolic_snowshoe(snowshoe):
    result = adjoint_matrix(snowshoe, form='symbolic')
    a_RR = sp.Symbol('a_R,R')
    a_RC = sp.Symbol('a_R,C')
    a_CR = sp.Symbol('a_C,R')
    a_CP = sp.Symbol('a_C,P')
    a_PC = sp.Symbol('a_P,C')
    a_PP = sp.Symbol('a_P,P')
    expected = sp.Matrix([
        [a_CP*a_PC, -a_PP*a_RC,  a_CP*a_RC],
        [a_CR*a_PP,  a_PP*a_RR, -a_CP*a_RR],
        [a_CR*a_PC,  a_PC*a_RR,  a_CR*a_RC]])
    assert result == expected

def test_adjoint_matrix_form_signed_perturb_R_snowshoe(snowshoe):
    result = adjoint_matrix(snowshoe, form='signed', perturb='R')
    assert result.shape == (3, 1)
    expected = sp.Matrix([[1], [1], [1]])
    assert result == expected


def test_adjoint_matrix_form_signed_perturb_1_chain(chain):
    result = adjoint_matrix(chain, form='signed', perturb='1')
    assert result.shape == (5, 1)
    expected = sp.Matrix([[5], [3], [2], [1], [1]])
    assert result == expected


def test_adjoint_matrix_form_signed_perturb_P_mesocosm(mesocosm):
    result = adjoint_matrix(mesocosm, form='signed', perturb='P')
    assert result.shape == (8, 1)
    expected = sp.Matrix([[1], [-1], [1], [1], [0], [1], [-1], [0]])
    assert result == expected


def test_adjoint_matrix_form_symbolic_perturb_R_snowshoe(snowshoe):
    result = adjoint_matrix(snowshoe, form='symbolic', perturb='R')
    a_RR = sp.Symbol('a_R,R')
    a_CR = sp.Symbol('a_C,R')
    a_CP = sp.Symbol('a_C,P')
    a_PC = sp.Symbol('a_P,C')
    a_PP = sp.Symbol('a_P,P')
    expected = sp.Matrix([
        [a_CP*a_PC],
        [a_CR*a_PP],
        [a_CR*a_PC]])
    assert result == expected


def test_adjoint_matrix_form_symbolic_perturb_3_chain(chain):
    result = adjoint_matrix(chain, form='symbolic', perturb='3')
    a_11 = sp.Symbol('a_1,1')
    a_12 = sp.Symbol('a_1,2')
    a_21 = sp.Symbol('a_2,1')
    a_22 = sp.Symbol('a_2,2')
    a_23 = sp.Symbol('a_2,3')
    a_43 = sp.Symbol('a_4,3')
    a_44 = sp.Symbol('a_4,4')
    a_45 = sp.Symbol('a_4,5')
    a_54 = sp.Symbol('a_5,4')
    a_55 = sp.Symbol('a_5,5')

    expected = sp.Matrix([
        [              a_12*a_23*(a_44*a_55 + a_45*a_54)],
        [             -a_11*a_23*(a_44*a_55 + a_45*a_54)],
        [(a_11*a_22 + a_12*a_21)*(a_44*a_55 + a_45*a_54)],
        [              a_43*a_55*(a_11*a_22 + a_12*a_21)],
        [              a_43*a_54*(a_11*a_22 + a_12*a_21)]])
    assert result == expected


# =============================================================================
# absolute_feedback_matrix()
# =============================================================================

def test_absolute_feedback_matrix_default_snowshoe(snowshoe):
    result = absolute_feedback_matrix(snowshoe)
    expected = sp.Matrix([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]])
    assert result == expected


def test_absolute_feedback_matrix_default_chain(chain):
    result = absolute_feedback_matrix(chain)
    expected = sp.Matrix([
        [5, 3, 2, 1, 1],
        [3, 3, 2, 1, 1],
        [2, 2, 4, 2, 2],
        [1, 1, 2, 3, 3],
        [1, 1, 2, 3, 5]])
    assert result == expected


def test_absolute_feedback_matrix_default_mesocosm(mesocosm):
    result = absolute_feedback_matrix(mesocosm)
    expected = sp.Matrix([
        [1, 7,  9,  1,  2, 1,  5, 2],
        [7, 7,  9,  7,  4, 7,  7, 4],
        [9, 9, 11,  9,  4, 9,  9, 4],
        [1, 7,  9, 17,  2, 1,  5, 2],
        [2, 4,  4,  2,  4, 2, 10, 4],
        [1, 7,  9,  1,  2, 1,  5, 2],
        [5, 7,  9,  5, 10, 5, 11, 8],
        [2, 4,  4,  2,  4, 2,  8, 4]])
    assert result == expected


def test_absolute_feedback_matrix_perturb_R_snowshoe(snowshoe):
    result = absolute_feedback_matrix(snowshoe, perturb='R')
    expected = sp.Matrix([[1], [1], [1]])
    assert result == expected


def test_absolute_feedback_matrix_perturb_1_chain(chain):
    result = absolute_feedback_matrix(chain, perturb='1')
    expected = sp.Matrix([[5], [3], [2], [1], [1]])
    assert result == expected


def test_absolute_feedback_matrix_perturb_3_chain(chain):
    result = absolute_feedback_matrix(chain, perturb='3')
    expected = sp.Matrix([[2], [2], [4], [2], [2]])
    assert result == expected


def test_absolute_feedback_matrix_perturb_P_mesocosm(mesocosm):
    result = absolute_feedback_matrix(mesocosm, perturb='P')
    expected = sp.Matrix([[1], [7], [9], [1], [2], [1], [5], [2]])
    assert result == expected


# =============================================================================
# weighted_predictions_matrix()
# =============================================================================

def test_weighted_predictions_matrix_as_nan_false_abs_true_snowshoe(snowshoe):
    result = weighted_predictions_matrix(snowshoe, as_nan=False, as_abs=True)
    expected = sp.Matrix([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]])
    assert result == expected


def test_weighted_predictions_matrix_as_nan_true_abs_true_snowshoe(snowshoe):
    result = weighted_predictions_matrix(snowshoe, as_nan=True, as_abs=True)
    expected = sp.Matrix([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]])
    assert result == expected


def test_weighted_predictions_matrix_as_nan_false_abs_true_repeat_snowshoe(snowshoe):
    result = weighted_predictions_matrix(snowshoe, as_nan=False, as_abs=True)
    expected = sp.Matrix([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]])
    assert result == expected


def test_weighted_predictions_matrix_as_nan_true_signed_snowshoe(snowshoe):
    result = weighted_predictions_matrix(snowshoe, as_nan=True, as_abs=False)
    expected = sp.Matrix([
        [1, -1,  1],
        [1,  1, -1],
        [1,  1,  1]])
    assert result == expected


def test_weighted_predictions_matrix_perturb_R_snowshoe(snowshoe):
    result = weighted_predictions_matrix(snowshoe, perturb='R')
    assert result.shape == (3, 1)
    expected = sp.Matrix([[1], [1], [1]])
    assert result == expected


def test_weighted_predictions_matrix_as_nan_false_abs_true_chain(chain):
    result = weighted_predictions_matrix(chain, as_nan=False, as_abs=True)
    expected = sp.Matrix([
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1]])
    assert result == expected


def test_weighted_predictions_matrix_as_nan_true_signed_chain(chain):
    result = weighted_predictions_matrix(chain, as_nan=True, as_abs=False)
    expected = sp.Matrix([
        [1, -1,  1, -1,  1],
        [1,  1, -1,  1, -1],
        [1,  1,  1, -1,  1],
        [1,  1,  1,  1, -1],
        [1,  1,  1,  1,  1]])
    assert result == expected


def test_weighted_predictions_matrix_perturb_3_chain(chain):
    result = weighted_predictions_matrix(chain, perturb='3', as_nan=True, as_abs=True)
    assert result.shape == (5, 1)
    expected = sp.Matrix([[1], [1], [1], [1], [1]])
    assert result == expected


def test_weighted_predictions_matrix_as_nan_false_abs_true_mesocosm(mesocosm):
    result = weighted_predictions_matrix(mesocosm, as_nan=False, as_abs=True)
    expected = sp.Matrix([
        [                1, sp.Rational(1, 7),  sp.Rational(1, 9),                  1,                 0,                 1,  sp.Rational(1, 5),                 0],
        [sp.Rational(1, 7), sp.Rational(3, 7),  sp.Rational(5, 9),  sp.Rational(1, 7), sp.Rational(1, 2), sp.Rational(1, 7),  sp.Rational(5, 7), sp.Rational(1, 2)],
        [sp.Rational(1, 9), sp.Rational(1, 9), sp.Rational(3, 11),  sp.Rational(1, 9), sp.Rational(1, 2), sp.Rational(1, 9),  sp.Rational(1, 3), sp.Rational(1, 2)],
        [                1, sp.Rational(1, 7),  sp.Rational(1, 9), sp.Rational(1, 17),                 0,                 1,  sp.Rational(1, 5),                 0],
        [                0, sp.Rational(1, 2),  sp.Rational(1, 2),                  0,                 0,                 0,                  0,                 0],
        [                1, sp.Rational(1, 7),  sp.Rational(1, 3),                  1,                 0,                 1,  sp.Rational(1, 5),                 0],
        [sp.Rational(1, 5), sp.Rational(1, 7),  sp.Rational(1, 3),  sp.Rational(1, 5),                 0, sp.Rational(1, 5), sp.Rational(3, 11), sp.Rational(1, 4)],
        [                0, sp.Rational(1, 2),  sp.Rational(1, 2),                  0,                 0,                 0,  sp.Rational(1, 4),                 0]])
    assert result == expected


def test_weighted_predictions_matrix_as_nan_true_signed_mesocosm(mesocosm):
    result = weighted_predictions_matrix(mesocosm, as_nan=True, as_abs=False)
    expected = np.array([
        [  1.0, -0.14,  0.11,  -1.0,   0.0,   1.0, -0.20,   0.0],
        [-0.14,  0.43, -0.56,  0.14, -0.50, -0.14,  0.71, -0.50],
        [ 0.11, -0.11,  0.27, -0.11,  0.50, -0.11, -0.33,  0.50],
        [  1.0, -0.14,  0.11, 0.059,   0.0,   1.0, -0.20,   0.0],
        [  0.0,  0.50, -0.50,   0.0,   0.0,   0.0,   0.0,   0.0],
        [  1.0, -0.14,  0.33,  -1.0,   0.0,   1.0, -0.20,   0.0],
        [-0.20,  0.14, -0.33,  0.20,   0.0, -0.20,  0.27, -0.25],
        [  0.0,  0.50, -0.50,   0.0,   0.0,   0.0,  0.25,   0.0]
    ])
    assert np.allclose(np.array(result.tolist(), dtype=float), expected, atol=5e-3)


def test_weighted_predictions_matrix_perturb_P_mesocosm(mesocosm):
    result = weighted_predictions_matrix(mesocosm, perturb='P', as_nan=True, as_abs=True)
    expected = sp.Matrix([[1], [sp.Rational(1, 7)], [sp.Rational(1, 9)], [1], [0], [1], [sp.Rational(1, 5)], [0]])
    assert result == expected


def test_weighted_predictions_matrix_as_nan_true_missing_paths_snowshoe_io_na(snowshoe_io_na):
    result = weighted_predictions_matrix(snowshoe_io_na, as_nan=True, as_abs=False)
    expected = sp.Matrix([
        [     1,     -1,      1, 1],
        [     1,      1,     -1, 1],
        [     1,      1,      1, 1],
        [sp.nan, sp.nan, sp.nan, 1]])
    assert result == expected


def test_weighted_predictions_matrix_fill_missing_paths_snowshoe_io_na(snowshoe_io_na):
    result = weighted_predictions_matrix(snowshoe_io_na, as_nan=False, as_abs=False)
    expected = sp.Matrix([
        [1, -1,  1, 1],
        [1,  1, -1, 1],
        [1,  1,  1, 1],
        [1,  1,  1, 1]])
    assert result == expected


def test_weighted_predictions_matrix_as_nan_true_as_abs_false_snowshoe_na(snowshoe_na):
    result = weighted_predictions_matrix(snowshoe_na, as_nan=True, as_abs=False)
    expected = sp.Matrix([
        [     1,     -1,  0],
        [sp.nan,      1, -1],
        [sp.nan, sp.nan,  1]])
    assert result == expected


def test_weighted_predictions_matrix_as_nan_true_as_abs_true_snowshoe_na(snowshoe_na):
    result = weighted_predictions_matrix(snowshoe_na, as_nan=True, as_abs=True)
    expected = sp.Matrix([
        [     1,      1, 0],
        [sp.nan,      1, 1],
        [sp.nan, sp.nan, 1]])
    assert result == expected


def test_weighted_predictions_matrix_as_nan_false_as_abs_false_snowshoe_na(snowshoe_na):
    result = weighted_predictions_matrix(snowshoe_na, as_nan=False, as_abs=False)
    expected = sp.Matrix([
        [1, -1,  0],
        [1,  1, -1],
        [1,  1,  1]])
    assert result == expected


def test_weighted_predictions_matrix_as_nan_false_as_abs_true_snowshoe_na(snowshoe_na):
    result = weighted_predictions_matrix(snowshoe_na, as_nan=False, as_abs=True)
    expected = sp.Matrix([
        [1, 1, 0],
        [1, 1, 1],
        [1, 1, 1]])
    assert result == expected


# =============================================================================
# sign_determinacy_matrix()
# =============================================================================

def test_sign_determinacy_matrix_method_average_snowshoe(snowshoe):
    result = sign_determinacy_matrix(snowshoe, method='average')
    expected = sp.Matrix([
        [1, -1,  1],
        [1,  1, -1],
        [1,  1,  1]])
    assert result == expected


def test_sign_determinacy_matrix_method_95_bound_snowshoe(snowshoe):
    result = sign_determinacy_matrix(snowshoe, method='95_bound')
    expected = sp.Matrix([
        [1, -1,  1],
        [1,  1, -1],
        [1,  1,  1]])
    assert result == expected


def test_sign_determinacy_matrix_method_average_chain(chain):
    result = sign_determinacy_matrix(chain, method='average')
    expected = sp.Matrix([
        [1, -1,  1, -1,  1],
        [1,  1, -1,  1, -1],
        [1,  1,  1, -1,  1],
        [1,  1,  1,  1, -1],
        [1,  1,  1,  1,  1]])
    assert result == expected


def test_sign_determinacy_matrix_method_95_bound_chain(chain):
    result = sign_determinacy_matrix(chain, method='95_bound')
    expected = sp.Matrix([
        [1, -1,  1, -1,  1],
        [1,  1, -1,  1, -1],
        [1,  1,  1, -1,  1],
        [1,  1,  1,  1, -1],
        [1,  1,  1,  1,  1]])
    assert result == expected


def test_sign_determinacy_matrix_method_average_mesocosm(mesocosm):
    result = sign_determinacy_matrix(mesocosm, method='average')
    expected = np.array([
        [  1.0, -0.63,  0.60,  -1.0,  0.50,   1.0, -0.67,  0.50],
        [-0.63,  0.83, -0.89,  0.63, -0.86, -0.63,  0.93, -0.86],
        [ 0.60, -0.60,  0.74, -0.60,  0.86, -0.60, -0.78,  0.86],
        [  1.0, -0.63,  0.60,  0.56,  0.50,   1.0, -0.67,  0.50],
        [ 0.50,  0.86, -0.86,  0.50,  0.50,  0.50,  0.50,  0.50],
        [  1.0, -0.63,  0.78,  -1.0,  0.50,   1.0, -0.67,  0.50],
        [-0.67,  0.63, -0.78,  0.67,  0.50, -0.67,  0.74, -0.72],
        [ 0.50,  0.86, -0.86,  0.50,  0.50,  0.50,  0.72,  0.50]
    ])
    assert np.allclose(np.array(result.tolist(), dtype=float), expected, atol=5e-3)


def test_sign_determinacy_matrix_method_95_bound_mesocosm(mesocosm):
    result = sign_determinacy_matrix(mesocosm, method='95_bound')
    expected = np.array([
        [  1.0, -0.50,  0.50,  -1.0,  0.50,   1.0, -0.50,  0.50],
        [-0.50,  0.50, -0.50,  0.50, -0.50, -0.50,  0.63, -0.50],
        [ 0.50, -0.50,  0.50, -0.50,  0.50, -0.50, -0.50,  0.50],
        [  1.0, -0.50,  0.50,  0.50,  0.50,   1.0, -0.50,  0.50],
        [ 0.50,  0.50, -0.50,  0.50,  0.50,  0.50,  0.50,  0.50],
        [  1.0, -0.50,  0.50,  -1.0,  0.50,   1.0, -0.50,  0.50],
        [-0.50,  0.50, -0.50,  0.50,  0.50, -0.50,  0.50, -0.50],
        [ 0.50,  0.50, -0.50,  0.50,  0.50,  0.50,  0.50,  0.50]
    ])
    assert np.allclose(np.array(result.tolist(), dtype=float), expected, atol=5e-3)


def test_sign_determinacy_matrix_as_nan_false_snowshoe(snowshoe):
    result = sign_determinacy_matrix(snowshoe, as_nan=False)
    expected = sp.Matrix([
        [1, -1,  1],
        [1,  1, -1],
        [1,  1,  1]])
    assert result == expected


def test_sign_determinacy_matrix_as_abs_true_chain(chain):
    result = sign_determinacy_matrix(chain, as_abs=True)
    expected = sp.Matrix([
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1]])
    assert result == expected


def test_sign_determinacy_matrix_as_nan_true_as_abs_false_snowshoe_na(snowshoe_na):
    result = sign_determinacy_matrix(snowshoe_na, as_nan=True, as_abs=False)
    expected = sp.Matrix([
        [     1,     -1, sp.Rational(1, 2)],
        [sp.nan,      1,            -1],
        [sp.nan, sp.nan,             1]])
    assert result == expected


def test_sign_determinacy_matrix_as_nan_true_as_abs_true_snowshoe_na(snowshoe_na):
    result = sign_determinacy_matrix(snowshoe_na, as_nan=True, as_abs=True)
    expected = sp.Matrix([
        [     1,      1, sp.Rational(1, 2)],
        [sp.nan,      1,             1],
        [sp.nan, sp.nan,             1]])
    assert result == expected


def test_sign_determinacy_matrix_as_nan_false_as_abs_false_snowshoe_na(snowshoe_na):
    result = sign_determinacy_matrix(snowshoe_na, as_nan=False, as_abs=False)
    expected = sp.Matrix([
        [1, -1, sp.Rational(1, 2)],
        [1,  1,            -1],
        [1,  1,             1]])
    assert result == expected


def test_sign_determinacy_matrix_as_nan_false_as_abs_true_snowshoe_na(snowshoe_na):
    result = sign_determinacy_matrix(snowshoe_na, as_nan=False, as_abs=True)
    expected = sp.Matrix([
        [1, 1, sp.Rational(1, 2)],
        [1, 1,             1],
        [1, 1,             1]])
    assert result == expected


def test_sign_determinacy_matrix_perturb_R_snowshoe(snowshoe):
    result = sign_determinacy_matrix(snowshoe, perturb='R')
    expected = sp.Matrix([[1], [1], [1]])
    assert result == expected


def test_sign_determinacy_matrix_perturb_1_chain(chain):
    result = sign_determinacy_matrix(chain, perturb='1')
    expected = sp.Matrix([[1], [1], [1], [1], [1]])
    assert result == expected


def test_sign_determinacy_matrix_perturb_3_chain(chain):
    result = sign_determinacy_matrix(chain, perturb='3')
    expected = sp.Matrix([[1], [-1], [1], [1], [1]])
    assert result == expected


def test_sign_determinacy_matrix_perturb_P_mesocosm(mesocosm):
    result = sign_determinacy_matrix(mesocosm, perturb='P')
    expected = np.array([
        [  1.0],
        [-0.63],
        [ 0.60],
        [  1.0],
        [ 0.50],
        [  1.0],
        [-0.67],
        [ 0.50]
    ])
    assert np.allclose(np.array(result.tolist(), dtype=float), expected, atol=5e-3)


# =============================================================================
# numerical_simulations()
# =============================================================================

def test_numerical_simulations_signed_default_snowshoe(snowshoe):
    numerical_simulations.cache_clear()
    result = numerical_simulations(snowshoe, n_sim=100, seed=42)
    expected = sp.Matrix([
        [1.0, -1.0,  1.0],
        [1.0,  1.0, -1.0],
        [1.0,  1.0,  1.0]])
    assert result == expected


def test_numerical_simulations_signed_default_chain(chain):
    numerical_simulations.cache_clear()
    result = numerical_simulations(chain, n_sim=100, seed=42)
    expected = sp.Matrix([
        [1.0, -1.0,  1.0, -1.0,  1.0],
        [1.0,  1.0, -1.0,  1.0, -1.0],
        [1.0,  1.0,  1.0, -1.0,  1.0],
        [1.0,  1.0,  1.0,  1.0, -1.0],
        [1.0,  1.0,  1.0,  1.0,  1.0]])
    assert result == expected


def test_numerical_simulations_signed_default_mesocosm(mesocosm):
    result = numerical_simulations(mesocosm, n_sim=100, seed=42)
    expected = sp.Matrix([
        [  1.0, -0.73,  0.64,  -1.0,   0.5,   1.0, -0.68,   0.5],
        [-0.71,  0.88, -0.95,  0.71, -0.87, -0.71,  0.95, -0.87],
        [ 0.83, -0.56,  0.79, -0.83,  0.92, -0.80, -0.77,  0.92],
        [  1.0, -0.73,  0.64,  0.88,   0.5,   1.0, -0.68,   0.5],
        [ 0.57,  0.95, -0.86, -0.57,  0.59,  0.57, -0.65,  0.59],
        [  1.0, -0.73,  0.89,  -1.0,   0.5,   1.0, -0.68,   0.5],
        [ -0.8,  0.65, -0.83,   0.8,   0.5,  -0.8,  0.93, -0.93],
        [ 0.57,  0.95, -0.86, -0.57,  0.59,  0.57,  0.82,  0.59]])
    assert result == expected


def test_numerical_simulations_positive_only_true_snowshoe(snowshoe):
    numerical_simulations.cache_clear()
    result = numerical_simulations(snowshoe, n_sim=100, seed=42, positive_only=True)
    expected = sp.Matrix([
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0]])
    assert result == expected


def test_numerical_simulations_as_nan_false_snowshoe(snowshoe):
    result = numerical_simulations(snowshoe, n_sim=100, seed=42, as_nan=False)
    expected = sp.Matrix([
        [1.0, -1.0,  1.0],
        [1.0,  1.0, -1.0],
        [1.0,  1.0,  1.0]])
    assert result == expected


def test_numerical_simulations_as_abs_true_snowshoe(snowshoe):
    result = numerical_simulations(snowshoe, n_sim=100, seed=42, as_abs=True)
    expected = sp.Matrix([
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0]])
    assert result == expected


@pytest.mark.parametrize("dist", ['uniform', 'weak', 'moderate', 'strong'])
def test_numerical_simulations_distribution_options_snowshoe_dist(snowshoe, dist):
    result = numerical_simulations(snowshoe, n_sim=100, dist=dist, seed=42)
    expected = (3, 3)
    assert result.shape == expected


def test_numerical_simulations_reproducible_seed_snowshoe(snowshoe):
    numerical_simulations.cache_clear()
    result = numerical_simulations(snowshoe, n_sim=100, seed=42)
    numerical_simulations.cache_clear()
    expected = numerical_simulations(snowshoe, n_sim=100, seed=42)
    assert result == expected

def test_numerical_simulations_missing_paths_default_nan_snowshoe_io_na(snowshoe_io_na):
    result = numerical_simulations(snowshoe_io_na, n_sim=100, seed=42)
    expected = sp.Matrix([
        [   1.0,   -1.0,    1.0, 1.0],
        [   1.0,    1.0,   -1.0, 1.0],
        [   1.0,    1.0,    1.0, 1.0],
        [sp.nan, sp.nan, sp.nan, 1.0]])
    assert result == expected


def test_numerical_simulations_missing_paths_fill_zeros_snowshoe_io_na(snowshoe_io_na):
    result = numerical_simulations(snowshoe_io_na, n_sim=100, seed=42, as_nan=False)
    expected = sp.Matrix([
        [1.0, -1.0,  1.0, 1.0],
        [1.0,  1.0, -1.0, 1.0],
        [1.0,  1.0,  1.0, 1.0],
        [  0,    0,    0, 1.0]])
    assert result == expected


def test_numerical_simulations_missing_paths_as_abs_true_snowshoe_io_na(snowshoe_io_na):
    result = numerical_simulations(snowshoe_io_na, n_sim=100, seed=42, as_abs=True)
    expected = sp.Matrix([
        [   1.0,    1.0,    1.0, 1.0],
        [   1.0,    1.0,    1.0, 1.0],
        [   1.0,    1.0,    1.0, 1.0],
        [sp.nan, sp.nan, sp.nan, 1.0]])
    assert result == expected


def test_numerical_simulations_missing_paths_positive_only_true_snowshoe_io_na(snowshoe_io_na):
    result = numerical_simulations(snowshoe_io_na, n_sim=100, seed=42, positive_only=True)
    expected = sp.Matrix([
        [   1.0,    0.0,    1.0, 1.0],
        [   1.0,    1.0,    0.0, 1.0],
        [   1.0,    1.0,    1.0, 1.0],
        [sp.nan, sp.nan, sp.nan, 1.0]])
    assert result == expected


def test_numerical_simulations_as_nan_true_as_abs_false_snowshoe_na(snowshoe_na):
    numerical_simulations.cache_clear()
    result = numerical_simulations(snowshoe_na, n_sim=10000, seed=42, as_nan=True, as_abs=False)
    expected = np.array([
        [   1.0,   -1.0, -0.5],
        [np.nan,    1.0, -1.0],
        [np.nan, np.nan,  1.0]
    ])
    result_arr = np.array(result.tolist(), dtype=float)
    nan_mask = np.isnan(expected)
    assert np.all(np.isnan(result_arr[nan_mask]))
    assert np.allclose(result_arr[~nan_mask], expected[~nan_mask], atol=0.1)


def test_numerical_simulations_as_nan_true_as_abs_true_snowshoe_na(snowshoe_na):
    numerical_simulations.cache_clear()
    result = numerical_simulations(snowshoe_na, n_sim=10000, seed=42, as_nan=True, as_abs=True)
    expected = np.array([
        [   1.0,    1.0, 0.5],
        [np.nan,    1.0, 1.0],
        [np.nan, np.nan, 1.0]
    ])
    result_arr = np.array(result.tolist(), dtype=float)
    nan_mask = np.isnan(expected)
    assert np.all(np.isnan(result_arr[nan_mask]))
    assert np.allclose(result_arr[~nan_mask], expected[~nan_mask], atol=0.1)


def test_numerical_simulations_as_nan_false_as_abs_false_snowshoe_na(snowshoe_na):
    numerical_simulations.cache_clear()
    result = numerical_simulations(snowshoe_na, n_sim=10000, seed=42, as_nan=False, as_abs=False)
    expected = np.array([
        [1.0, -1.0, -0.5],
        [0.0,  1.0, -1.0],
        [0.0,  0.0,  1.0]
    ])
    assert np.allclose(np.array(result.tolist(), dtype=float), expected, atol=0.1)


def test_numerical_simulations_positive_only_true_snowshoe_na(snowshoe_na):
    numerical_simulations.cache_clear()
    result = numerical_simulations(snowshoe_na, n_sim=10000, seed=42, positive_only=True)
    expected = np.array([
        [   1.0,    0.0, 0.5],
        [np.nan,    1.0, 0.0],
        [np.nan, np.nan, 1.0]
    ])
    result_arr = np.array(result.tolist(), dtype=float)
    nan_mask = np.isnan(expected)
    assert np.all(np.isnan(result_arr[nan_mask]))
    assert np.allclose(result_arr[~nan_mask], expected[~nan_mask], atol=0.1)


def test_numerical_simulations_positive_only_requires_as_nan_true(snowshoe):
    with pytest.raises(ValueError, match="positive_only=True requires as_nan=True"):
        numerical_simulations(snowshoe, n_sim=100, seed=42, positive_only=True, as_nan=False)


def test_numerical_simulations_as_abs_requires_as_nan_true(snowshoe):
    with pytest.raises(ValueError, match="as_abs=True requires as_nan=True"):
        numerical_simulations(snowshoe, n_sim=100, seed=42, as_abs=True, as_nan=False)


def test_numerical_simulations_linalg_retries_snowshoe(snowshoe):
    original = np.linalg.inv
    count = [0]
    def mock_inv(x):
        count[0] += 1
        if count[0] <= 2:
            raise np.linalg.LinAlgError()
        return original(x)
    with patch('numpy.linalg.inv', side_effect=mock_inv):
        result = numerical_simulations(snowshoe, n_sim=100, seed=42)
        expected = (3, 3)
        assert result.shape == expected


def test_numerical_simulations_no_stable_matrices_snowshoe(snowshoe):
    result = numerical_simulations(snowshoe, n_sim=0, seed=42)
    assert result.shape == (3, 3)
    for i in range(3):
        for j in range(3):
            assert result[i, j] is sp.nan


# =============================================================================
# Additional coverage tests
# =============================================================================

def test_adjoint_matrix_invalid_perturb_node(snowshoe):
    with pytest.raises(ValueError, match="Perturbation node must be one of"):
        adjoint_matrix(snowshoe, perturb="Invalid")


def test_absolute_feedback_matrix_invalid_perturb_node(snowshoe):
    with pytest.raises(ValueError, match="Perturbation node must be one of"):
        absolute_feedback_matrix(snowshoe, perturb="Invalid")
