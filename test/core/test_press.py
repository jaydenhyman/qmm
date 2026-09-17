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
from qmm.core.helper import list_to_digraph, get_positive, get_negative


def test_numerical_simulations_rejects_nonfinite_inverse_draws(monkeypatch):
    graph = list_to_digraph([[-1]])
    monkeypatch.setattr('qmm.core.press._random_sampler',
                        lambda dist, size, rng: np.full(size, 1e-320))
    with pytest.raises(RuntimeError, match='Maximum iterations reached'):
        numerical_simulations(graph, n_sim=1)


# =============================================================================
# adjoint_matrix()
# =============================================================================

def test_absolute_feedback_matrix_beyond_63_states():
    graph = list_to_digraph(-np.eye(65, dtype=int))
    assert absolute_feedback_matrix(graph, perturb="65") == sp.eye(65)[:, 64]


def test_press_results_follow_graph_edits_and_are_independent():
    graph = list_to_digraph([[-1, 0], [1, -1]], ids=["A", "B"])
    for function in (adjoint_matrix, absolute_feedback_matrix,
                     weighted_predictions_matrix, sign_determinacy_matrix):
        expected = function(graph).copy()
        function(graph)[0, 0] = 999
        assert function(graph) == expected
    assert adjoint_matrix(graph, form="signed")[1, 0] == 1
    graph["A"]["B"]["sign"] = -1
    assert adjoint_matrix(graph, form="signed")[1, 0] == -1
    assert weighted_predictions_matrix(graph)[1, 0] == -1


@pytest.mark.parametrize("n_sim", [True, -1, 1.5])
def test_numerical_simulations_rejects_invalid_counts(snowshoe, n_sim):
    with pytest.raises(ValueError, match="nonnegative integer"):
        numerical_simulations(snowshoe, n_sim=n_sim)


def test_numerical_simulations_does_not_reuse_mutable_results():
    graph = list_to_digraph([[-1, 0], [1, -1]], ids=["A", "B"])
    numerical_simulations(graph, n_sim=2)[0, 0] = 999
    assert numerical_simulations(graph, n_sim=2)[0, 0] == 1.0
    graph["A"]["B"]["sign"] = -1
    assert numerical_simulations(graph, n_sim=2)[1, 0] == -1.0

def test_numerical_simulations_accepts_stable_diagonals_at_different_rates(monkeypatch):
    monkeypatch.setattr("qmm.core.press._random_sampler", lambda *args: np.array([1, 1e-16]))
    graph = list_to_digraph(-np.eye(2, dtype=int))
    result = numerical_simulations(graph, n_sim=1, as_nan=False)
    assert result == sp.eye(2).evalf()


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


def test_weighted_predictions_terms_count_adjoint_monomials_omnivory(omnivory):
    adjoint = adjoint_matrix(omnivory)
    net = adjoint_matrix(omnivory, form='signed')
    absolute = absolute_feedback_matrix(omnivory)
    terms = [[adjoint[i, j].as_ordered_terms() if adjoint[i, j] else [] for j in range(3)] for i in range(3)]
    positive = sp.Matrix(3, 3, lambda i, j: sum(bool(term.as_coeff_Mul()[0] > 0) for term in terms[i][j]))
    result = (get_positive(net, absolute), get_negative(net, absolute), absolute, weighted_predictions_matrix(omnivory))
    expected = (positive, sp.Matrix(3, 3, lambda i, j: len(terms[i][j])) - positive,
                sp.Matrix(3, 3, lambda i, j: len(terms[i][j])),
                sp.Matrix(3, 3, lambda i, j: (2 * positive[i, j] - len(terms[i][j])) / sp.Integer(len(terms[i][j]))))
    assert result == expected


def test_weighted_predictions_certify_signs_snowshoe_rp(snowshoe_rp):
    adjoint = adjoint_matrix(snowshoe_rp)
    weights = weighted_predictions_matrix(snowshoe_rp)
    symbols = sorted(adjoint.free_symbols, key=str)
    strengths = np.random.default_rng(42).uniform(0.01, 1, (200, len(symbols)))
    result = {(i, j): {int(np.sign(adjoint[i, j].subs(dict(zip(symbols, row))))) for row in strengths}
              for i in range(3) for j in range(3)}
    expected = {(i, j): {int(weights[i, j])} if abs(weights[i, j]) == 1 else {-1, 1} for i in range(3) for j in range(3)}
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
    result = numerical_simulations(snowshoe, n_sim=100, seed=42)
    expected = sp.Matrix([
        [1.0, -1.0,  1.0],
        [1.0,  1.0, -1.0],
        [1.0,  1.0,  1.0]])
    assert result == expected


def test_numerical_simulations_signed_default_chain(chain):
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


def test_numerical_simulations_mode_positive_snowshoe(snowshoe):
    result = numerical_simulations(snowshoe, n_sim=100, seed=42, mode="positive")
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


def test_numerical_simulations_mode_absolute_snowshoe(snowshoe):
    result = numerical_simulations(snowshoe, n_sim=100, seed=42, mode="absolute")
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
    result = numerical_simulations(snowshoe, n_sim=100, seed=42)
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


def test_numerical_simulations_missing_paths_mode_absolute_snowshoe_io_na(snowshoe_io_na):
    result = numerical_simulations(snowshoe_io_na, n_sim=100, seed=42, mode="absolute")
    expected = sp.Matrix([
        [   1.0,    1.0,    1.0, 1.0],
        [   1.0,    1.0,    1.0, 1.0],
        [   1.0,    1.0,    1.0, 1.0],
        [sp.nan, sp.nan, sp.nan, 1.0]])
    assert result == expected


def test_numerical_simulations_missing_paths_mode_positive_snowshoe_io_na(snowshoe_io_na):
    result = numerical_simulations(snowshoe_io_na, n_sim=100, seed=42, mode="positive")
    expected = sp.Matrix([
        [   1.0,    0.0,    1.0, 1.0],
        [   1.0,    1.0,    0.0, 1.0],
        [   1.0,    1.0,    1.0, 1.0],
        [sp.nan, sp.nan, sp.nan, 1.0]])
    assert result == expected


def test_numerical_simulations_as_nan_true_mode_dominant_snowshoe_na(snowshoe_na):
    result = numerical_simulations(snowshoe_na, n_sim=10000, seed=42, as_nan=True, mode="dominant")
    expected = np.array([
        [   1.0,   -1.0, -0.5],
        [np.nan,    1.0, -1.0],
        [np.nan, np.nan,  1.0]
    ])
    result_arr = np.array(result.tolist(), dtype=float)
    nan_mask = np.isnan(expected)
    assert np.all(np.isnan(result_arr[nan_mask]))
    assert np.allclose(result_arr[~nan_mask], expected[~nan_mask], atol=0.1)


def test_numerical_simulations_as_nan_true_mode_absolute_snowshoe_na(snowshoe_na):
    result = numerical_simulations(snowshoe_na, n_sim=10000, seed=42, as_nan=True, mode="absolute")
    expected = np.array([
        [   1.0,    1.0, 0.5],
        [np.nan,    1.0, 1.0],
        [np.nan, np.nan, 1.0]
    ])
    result_arr = np.array(result.tolist(), dtype=float)
    nan_mask = np.isnan(expected)
    assert np.all(np.isnan(result_arr[nan_mask]))
    assert np.allclose(result_arr[~nan_mask], expected[~nan_mask], atol=0.1)


def test_numerical_simulations_as_nan_false_mode_dominant_snowshoe_na(snowshoe_na):
    result = numerical_simulations(snowshoe_na, n_sim=10000, seed=42, as_nan=False, mode="dominant")
    expected = np.array([
        [1.0, -1.0, -0.5],
        [0.0,  1.0, -1.0],
        [0.0,  0.0,  1.0]
    ])
    assert np.allclose(np.array(result.tolist(), dtype=float), expected, atol=0.1)


def test_numerical_simulations_mode_positive_snowshoe_na(snowshoe_na):
    result = numerical_simulations(snowshoe_na, n_sim=10000, seed=42, mode="positive")
    expected = np.array([
        [   1.0,    0.0, 0.5],
        [np.nan,    1.0, 0.0],
        [np.nan, np.nan, 1.0]
    ])
    result_arr = np.array(result.tolist(), dtype=float)
    nan_mask = np.isnan(expected)
    assert np.all(np.isnan(result_arr[nan_mask]))
    assert np.allclose(result_arr[~nan_mask], expected[~nan_mask], atol=0.1)


@pytest.mark.parametrize("mode", ["positive", "absolute"])
def test_numerical_simulations_mode_requires_as_nan_true(snowshoe, mode):
    with pytest.raises(ValueError, match=f"mode='{mode}' requires as_nan=True"):
        numerical_simulations(snowshoe, n_sim=100, seed=42, mode=mode, as_nan=False)


def test_numerical_simulations_invalid_mode(snowshoe):
    with pytest.raises(ValueError, match="Invalid mode"):
        numerical_simulations(snowshoe, n_sim=100, seed=42, mode="signed")


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


def test_numerical_simulations_mode_match_adjoint_mesocosm(mesocosm):
    result = numerical_simulations(mesocosm, n_sim=100, seed=42, mode="match_adjoint")
    expected = sp.Matrix([
        [1.0, 0.73, 0.64, 1.0, 0.5, 1.0, 0.68, 0.5],
        [0.71, 0.88, 0.95, 0.71, 0.87, 0.71, 0.95, 0.87],
        [0.83, 0.56, 0.79, 0.83, 0.92, 0.8, 0.77, 0.92],
        [1.0, 0.73, 0.64, 0.88, 0.5, 1.0, 0.68, 0.5],
        [0.5, 0.95, 0.86, 0.5, 0.5, 0.5, 0.5, 0.5],
        [1.0, 0.73, 0.89, 1.0, 0.5, 1.0, 0.68, 0.5],
        [0.8, 0.65, 0.83, 0.8, 0.5, 0.8, 0.93, 0.93],
        [0.5, 0.95, 0.86, 0.5, 0.5, 0.5, 0.82, 0.5]])
    assert result == expected


def test_numerical_simulations_mode_match_adjoint_as_nan_false_mesocosm(mesocosm):
    result = numerical_simulations(mesocosm, n_sim=100, seed=42, mode="match_adjoint", as_nan=False)
    result_arr = np.array(result.tolist(), dtype=float)
    assert not np.any(np.isnan(result_arr))


# =============================================================================
# Additional coverage tests
# =============================================================================

def test_adjoint_matrix_invalid_perturb_node(snowshoe):
    with pytest.raises(ValueError, match="Perturbation node must be one of"):
        adjoint_matrix(snowshoe, perturb="Invalid")


def test_absolute_feedback_matrix_invalid_perturb_node(snowshoe):
    with pytest.raises(ValueError, match="Perturbation node must be one of"):
        absolute_feedback_matrix(snowshoe, perturb="Invalid")
