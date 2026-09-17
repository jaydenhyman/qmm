"""Tests for qmm.core.stability module."""

from qmm.core.stability import _has_cycle_cover
from math import factorial

import itertools
from test.feedback import cycle_expansion
import pytest
import sympy as sp
import numpy as np
import pandas as pd
import networkx as nx

from qmm.core.helper import list_to_digraph, perm
from qmm.core.structure import create_matrix
from qmm.core.stability import (
    sign_stability,
    feedback_metrics,
    determinants_metrics,
    conditional_stability,
    simulation_stability,
    stability_analysis,
    system_feedback,
    net_feedback,
    absolute_feedback,
    weighted_feedback,
    hurwitz_determinants,
    net_determinants,
    absolute_determinants,
    weighted_determinants,
    _hurwitz_matrix,
    _colour_test,
    _create_model_c,
)


def test_simulation_stability_zero_strengths_fail_both_hurwitz_criteria():
    graph = list_to_digraph(-np.eye(3, dtype=int))
    result = simulation_stability(graph, n_sim=1, presample=np.zeros((1, 3, 3)))
    results = result.set_index('Test')['Result']
    assert results['Stable matrices'] == '0.00%'
    assert results['Unstable matrices'] == '100.00%'
    assert results['Hurwitz criterion i'] == results['Hurwitz criterion ii'] == '100.00%'
    assert results['Hurwitz criterion i only'] == results['Hurwitz criterion ii only'] == '0.00%'


def test_stability_analysis_reports_all_sections_with_presample(snowshoe):
    result = stability_analysis(snowshoe, n_sim=2, presample=np.ones((2, 3, 3)))
    assert list(result.columns) == ['Test', 'Definition', 'Result']
    assert result.index.tolist() == list(range(17))
    assert result['Test'].is_unique
    results = result.set_index('Test')['Result']
    assert bool(results['Sign stable'])
    assert 'Model class' in results
    assert results['Stable matrices'] == '100.00%'
    assert results['Unstable matrices'] == '0.00%'


# =============================================================================
# _colour_test()
# =============================================================================

def test_colour_test_pass(colour_pass):
    result = _colour_test(colour_pass)
    expected = 'Pass'
    assert result == expected


def test_colour_test_fail(colour_fail):
    result = _colour_test(colour_fail)
    expected = 'Fail'
    assert result == expected


# =============================================================================
# sign_stability()
# =============================================================================

def test_sign_stability_columns_snowshoe(snowshoe):
    df = sign_stability(snowshoe)
    result = list(df.columns)
    expected = ['Test', 'Definition', 'Result']
    assert result == expected


def test_sign_stability_classification_snowshoe(sign_stable_snowshoe):
    df = sign_stability(sign_stable_snowshoe)
    result = df.loc[df['Test'] == 'Sign stable', 'Result'].iloc[0]
    expected = True
    assert result == expected


def test_sign_stability_classification_chain(sign_stable_chain):
    df = sign_stability(sign_stable_chain)
    result = df.loc[df['Test'] == 'Sign stable', 'Result'].iloc[0]
    expected = True
    assert result == expected


def test_sign_stability_classification_class_ii(class_ii):
    df = sign_stability(class_ii)
    result = df.loc[df['Test'] == 'Sign stable', 'Result'].iloc[0]
    expected = False
    assert result == expected

def test_sign_stability_fail_condition_i(snowshoe):
    G = snowshoe.copy()
    G.add_edge('C', 'C', sign=1)
    df = sign_stability(G)
    result = {
        'condition_i': df.loc[df['Test'] == 'Condition i', 'Result'].iloc[0],
        'sign_stable': df.loc[df['Test'] == 'Sign stable', 'Result'].iloc[0]
    }
    expected = {'condition_i': False, 'sign_stable': False}
    assert result == expected


def test_sign_stability_fail_condition_ii(snowshoe):
    G = snowshoe.copy()
    G.remove_edge('R', 'R')
    G.remove_edge('P', 'P')
    df = sign_stability(G)
    result = {
        'condition_ii': df.loc[df['Test'] == 'Condition ii', 'Result'].iloc[0],
        'sign_stable': df.loc[df['Test'] == 'Sign stable', 'Result'].iloc[0]
    }
    expected = {'condition_ii': False, 'sign_stable': False}
    assert result == expected


def test_sign_stability_fail_condition_iii(snowshoe):
    G = snowshoe.copy()
    G.remove_edge('C', 'R')
    G.add_edge('C', 'R', sign=1)
    df = sign_stability(G)
    result = {
        'condition_iii': df.loc[df['Test'] == 'Condition iii', 'Result'].iloc[0],
        'sign_stable': df.loc[df['Test'] == 'Sign stable', 'Result'].iloc[0]
    }
    expected = {'condition_iii': False, 'sign_stable': False}
    assert result == expected


def test_sign_stability_fail_condition_iv(snowshoe):
    G = snowshoe.copy()
    G.add_edge('R', 'P', sign=1)
    df = sign_stability(G)
    result = {
        'condition_iv': df.loc[df['Test'] == 'Condition iv', 'Result'].iloc[0],
        'sign_stable': df.loc[df['Test'] == 'Sign stable', 'Result'].iloc[0]
    }
    expected = {'condition_iv': False, 'sign_stable': False}
    assert result == expected


def test_sign_stability_fail_condition_v(snowshoe):
    G = snowshoe.copy()
    G.remove_edge('P', 'C')
    G.remove_edge('P', 'P')
    df = sign_stability(G)
    result = {
        'condition_v': df.loc[df['Test'] == 'Condition v', 'Result'].iloc[0],
        'sign_stable': df.loc[df['Test'] == 'Sign stable', 'Result'].iloc[0]
    }
    expected = {'condition_v': False, 'sign_stable': False}
    assert result == expected


def test_sign_stability_fail_colour_test(colour_pass):
    df = sign_stability(colour_pass)
    result = {
        'colour_test': df.loc[df['Test'] == 'Colour test', 'Result'].iloc[0],
        'sign_stable': df.loc[df['Test'] == 'Sign stable', 'Result'].iloc[0]
    }
    expected = {'colour_test': False, 'sign_stable': False}
    assert result == expected


# =============================================================================
# system_feedback()
# =============================================================================

def test_system_feedback_form_signed_level_0_snowshoe(snowshoe):
    result = system_feedback(snowshoe, level=0, form='signed')
    expected = sp.Matrix([[-1]])
    assert result == expected


def test_system_feedback_form_signed_level_2_snowshoe(snowshoe):
    result = system_feedback(snowshoe, level=2, form='signed')
    expected = sp.Matrix([[-3]])
    assert result == expected


def test_system_feedback_form_symbolic_level_2_snowshoe(snowshoe):
    result = system_feedback(snowshoe, level=2, form='symbolic')
    a_RR = sp.Symbol('a_R,R')
    a_RC = sp.Symbol('a_R,C')
    a_CR = sp.Symbol('a_C,R')
    a_CP = sp.Symbol('a_C,P')
    a_PC = sp.Symbol('a_P,C')
    a_PP = sp.Symbol('a_P,P')
    expected = sp.Matrix([[-(a_CR*a_RC + a_RR*a_PP + a_CP*a_PC)]])
    assert result == expected


def test_system_feedback_form_signed_all_levels_snowshoe(snowshoe):
    result = system_feedback(snowshoe, level=None, form='signed')
    expected = sp.Matrix([[-1], [-2], [-3], [-2]])
    assert result == expected


def test_system_feedback_form_signed_all_levels_chain(chain):
    result = system_feedback(chain, level=None, form='signed')
    expected = sp.Matrix([[-1], [-5], [-14], [-22], [-20], [-8]])
    assert result == expected


def test_system_feedback_form_symbolic_all_levels_snowshoe(snowshoe):
    result = system_feedback(snowshoe, level=None, form='symbolic')
    a_RR = sp.Symbol('a_R,R')
    a_RC = sp.Symbol('a_R,C')
    a_CR = sp.Symbol('a_C,R')
    a_CP = sp.Symbol('a_C,P')
    a_PC = sp.Symbol('a_P,C')
    a_PP = sp.Symbol('a_P,P')
    expected = sp.Matrix([
        [                                  -1],
        [                      -(a_RR + a_PP)],
        [-(a_CR*a_RC + a_RR*a_PP + a_CP*a_PC)],
        [  -(a_CR*a_RC*a_PP + a_RR*a_CP*a_PC)]])
    assert result == expected


def test_system_feedback_matches_disjoint_cycle_expansion_snowshoe_rp(snowshoe_rp):
    feedback, counts = cycle_expansion(snowshoe_rp)
    result = (system_feedback(snowshoe_rp), absolute_feedback(snowshoe_rp), absolute_feedback(snowshoe_rp, method='polynomial'))
    expected = (feedback, counts, counts)
    assert result == expected


def test_absolute_feedback_counts_cancelled_terms_mutualism():
    G = list_to_digraph([[-1, 1], [1, -1]], ['1', '2'])
    result = (net_feedback(G), absolute_feedback(G), weighted_feedback(G))
    expected = (sp.Matrix([-1, -2, 0]), sp.Matrix([1, 2, 2]), sp.Matrix([-1, -1, 0]))
    assert result == expected


# =============================================================================
# net_feedback()
# =============================================================================

def test_net_feedback_signed_default_snowshoe(snowshoe):
    result = net_feedback(snowshoe)
    expected = sp.Matrix([[-1], [-2], [-3], [-2]])
    assert result == expected


def test_net_feedback_signed_default_chain(chain):
    result = net_feedback(chain)
    expected = sp.Matrix([[-1], [-5], [-14], [-22], [-20], [-8]])
    assert result == expected


# =============================================================================
# absolute_feedback()
# =============================================================================

def test_absolute_feedback_level_0_snowshoe(snowshoe):
    result = absolute_feedback(snowshoe, level=0)
    expected = sp.Matrix([[1]])
    assert result == expected


def test_absolute_feedback_all_levels_snowshoe(snowshoe):
    result = absolute_feedback(snowshoe)
    expected = sp.Matrix([[1], [2], [3], [2]])
    assert result == expected


def test_absolute_feedback_method_combinations_level_2_snowshoe(snowshoe):
    result = absolute_feedback(snowshoe, level=2, method="combinations")
    expected = sp.Matrix([[3]])
    assert result == expected


def test_absolute_feedback_method_polynomial_level_2_snowshoe(snowshoe):
    result = absolute_feedback(snowshoe, level=2, method="polynomial")
    expected = sp.Matrix([[3]])
    assert result == expected


def test_absolute_feedback_method_polynomial_all_levels_snowshoe(snowshoe):
    result = absolute_feedback(snowshoe, method="polynomial")
    expected = sp.Matrix([[1], [2], [3], [2]])
    assert result == expected


def test_absolute_feedback_exact_for_large_counts():
    n = 20
    G = list_to_digraph([[-1 if i == j else 1 for j in range(n)] for i in range(n)])
    assert absolute_feedback(G, level=n)[0] == factorial(n)


def test_absolute_feedback_all_levels_chain(chain):
    result = absolute_feedback(chain)
    expected = sp.Matrix([[1], [5], [14], [22], [20], [8]])
    assert result == expected


# =============================================================================
# weighted_feedback()
# =============================================================================

def test_weighted_feedback_default_snowshoe(snowshoe):
    result = weighted_feedback(snowshoe)
    expected = sp.Matrix([[-1], [-1], [-1], [-1]])
    assert result == expected


def test_weighted_feedback_default_chain(chain):
    result = weighted_feedback(chain)
    expected = sp.Matrix([[-1], [-1], [-1], [-1], [-1], [-1]])
    assert result == expected


# =============================================================================
# _hurwitz_matrix()
# =============================================================================

def test_hurwitz_matrix_level_0_snowshoe(snowshoe):
    fb = system_feedback(snowshoe, level=None, form='symbolic')
    result = _hurwitz_matrix(fb, level=0)
    expected = sp.Matrix([[1]])
    assert result == expected


def test_hurwitz_matrix_level_2_snowshoe(snowshoe):
    fb = system_feedback(snowshoe, level=None, form='symbolic')
    result = _hurwitz_matrix(fb, level=2)
    a_RR = sp.Symbol('a_R,R')
    a_RC = sp.Symbol('a_R,C')
    a_CR = sp.Symbol('a_C,R')
    a_CP = sp.Symbol('a_C,P')
    a_PC = sp.Symbol('a_P,C')
    a_PP = sp.Symbol('a_P,P')
    f1 = a_RR + a_PP
    f2 = a_CR*a_RC + a_RR*a_PP + a_CP*a_PC
    f3 = a_CR*a_RC*a_PP + a_RR*a_CP*a_PC
    expected = sp.Matrix([
        [f1, f3],
        [ 1, f2]])
    assert result == expected


# =============================================================================
# hurwitz_determinants()
# =============================================================================

def test_hurwitz_determinants_form_signed_snowshoe(snowshoe):
    result = hurwitz_determinants(snowshoe, form='signed')
    expected = sp.Matrix([[1], [2], [4], [8]])
    assert result == expected


def test_hurwitz_determinants_form_symbolic_snowshoe(snowshoe):
    result = hurwitz_determinants(snowshoe, form='symbolic')
    a_RR = sp.Symbol('a_R,R')
    a_RC = sp.Symbol('a_R,C')
    a_CR = sp.Symbol('a_C,R')
    a_CP = sp.Symbol('a_C,P')
    a_PC = sp.Symbol('a_P,C')
    a_PP = sp.Symbol('a_P,P')
    expected = sp.Matrix([
        [                                                                                                                                                                                                                          1],
        [                                                                                                                                                                                                                a_PP + a_RR],
        [                                                                                                                                                              a_CP*a_PC*a_PP + a_CR*a_RC*a_RR + a_PP**2*a_RR + a_PP*a_RR**2],
        [a_CP**2*a_PC**2*a_PP*a_RR + a_CP*a_CR*a_PC*a_PP**2*a_RC + a_CP*a_CR*a_PC*a_RC*a_RR**2 + a_CP*a_PC*a_PP**2*a_RR**2 + a_CP*a_PC*a_PP*a_RR**3 + a_CR**2*a_PP*a_RC**2*a_RR + a_CR*a_PP**3*a_RC*a_RR + a_CR*a_PP**2*a_RC*a_RR**2]])
    assert result == expected


def test_hurwitz_determinants_form_signed_level_1_snowshoe(snowshoe):
    result = hurwitz_determinants(snowshoe, level=1, form='signed')
    expected = sp.Matrix([[2]])
    assert result == expected


def test_hurwitz_determinants_form_signed_chain(chain):
    result = hurwitz_determinants(chain, form='signed')
    expected = sp.Matrix([[1], [5], [48], [596], [7280], [58240]])
    assert result == expected


def test_hurwitz_determinants_form_symbolic_rejects_large_six_node(large_six_node):
    result = None
    expected = ValueError
    with pytest.raises(expected):
        hurwitz_determinants(large_six_node, form='symbolic')
        result = "No exception"
    assert result is None


def _hurwitz_from_coefficients(coefficients, level):
    n = len(coefficients) - 1
    return sp.Matrix(level, level, lambda i, j: coefficients[2 * j - i + 1] if 0 <= 2 * j - i + 1 <= n else 0)


def test_hurwitz_determinants_match_polynomial_coefficients(snowshoe_rp, mesocosm):
    lam = sp.Symbol('lambda')
    symbolic = create_matrix(snowshoe_rp)
    signed = create_matrix(mesocosm, 'signed')
    result = (hurwitz_determinants(snowshoe_rp).applyfunc(sp.expand), hurwitz_determinants(mesocosm, form='signed'))
    expected = tuple(
        sp.Matrix([_hurwitz_from_coefficients(sp.Poly((lam * sp.eye(A.rows) - A).det(), lam).all_coeffs(), k).det()
                   for k in range(A.rows + 1)]).applyfunc(sp.expand)
        for A in (symbolic, signed))
    assert result == expected


def test_hurwitz_determinants_certify_stability_per_sample_mesocosm(mesocosm):
    signed = np.array(create_matrix(mesocosm, 'signed'), dtype=float)
    strengths = np.random.default_rng(0).uniform(0.05, 1, (500, *signed.shape))
    result = []
    expected = []
    for sample in signed * strengths:
        a = np.poly(sample)
        n = len(a) - 1
        hurwitz = np.array([[a[2 * j - i + 1] if 0 <= 2 * j - i + 1 <= n else 0.0 for j in range(n)] for i in range(n)])
        result.append(all(np.linalg.det(hurwitz[:k, :k]) > 0 for k in range(1, n + 1)))
        expected.append(bool(np.all(np.linalg.eigvals(sample).real < 0)))
    assert result == expected


def _jeffries_colouring_exists(signed):
    n = signed.shape[0]
    predation = {i: [j for j in range(n) if j != i and signed[i, j] * signed[j, i] < 0] for i in range(n)}
    candidates = [i for i in range(n) if signed[i, i] >= 0]
    for bits in itertools.product([0, 1], repeat=len(candidates)):
        white = {node for node, bit in zip(candidates, bits) if bit}
        if not white:
            continue
        if all(any(j in white for j in predation[v]) for v in white) and \
                all(sum(j in white for j in predation[b]) != 1 for b in range(n) if b not in white):
            return True
    return False


def test_colour_test_matches_exhaustive_colouring():
    rng = np.random.default_rng(3)
    result = []
    expected = []
    for _ in range(300):
        n = int(rng.integers(2, 7))
        signed = np.zeros((n, n), dtype=int)
        for i in range(n):
            signed[i, i] = -int(rng.uniform() < 0.5)
            for j in range(i + 1, n):
                if rng.uniform() < 0.5:
                    signed[i, j] = int(rng.choice([-1, 1]))
                    signed[j, i] = -signed[i, j]
        G = list_to_digraph(signed.tolist(), [str(i) for i in range(n)])
        result.append(_colour_test(G) == 'Pass')
        expected.append(_jeffries_colouring_exists(signed))
    assert result == expected


def test_sign_stability_neutral_chain_has_imaginary_eigenvalues():
    A = [[0, 1, 0, 0, 0], [-1, 0, 1, 0, 0], [0, -1, -1, 1, 0], [0, 0, -1, 0, 1], [0, 0, 0, -1, 0]]
    G = list_to_digraph(A, ['1', '2', '3', '4', '5'])
    df = sign_stability(G)
    result = (dict(zip(df['Test'], df['Result'])), sp.I in sp.Matrix(A).eigenvals())
    expected = ({'Condition i': True, 'Condition ii': True, 'Condition iii': True, 'Condition iv': True,
                 'Condition v': True, 'Colour test': False, 'Sign stable': False}, True)
    assert result == expected


# =============================================================================
# net_determinants()
# =============================================================================

def test_net_determinants_all_levels_snowshoe(snowshoe):
    result = net_determinants(snowshoe)
    expected = sp.Matrix([[1], [2], [4], [8]])
    assert result == expected


def test_net_determinants_all_levels_chain(chain):
    result = net_determinants(chain)
    expected = sp.Matrix([[1], [5], [48], [596], [7280], [58240]])
    assert result == expected


def test_net_determinants_level_0_chain(chain):
    result = net_determinants(chain, level=0)
    expected = sp.Matrix([[1]])
    assert result == expected


def test_net_determinants_level_2_chain(chain):
    result = net_determinants(chain, level=2)
    expected = sp.Matrix([[48]])
    assert result == expected


def test_net_determinants_rejects_invalid_level(chain):
    with pytest.raises(ValueError):
        net_determinants(chain, level=100)


# =============================================================================
# absolute_determinants()
# =============================================================================

def test_absolute_determinants_all_levels_snowshoe(snowshoe):
    result = absolute_determinants(snowshoe)
    expected = sp.Matrix([[1], [2], [8], [16]])
    assert result == expected


def test_absolute_determinants_level_0_snowshoe(snowshoe):
    result = absolute_determinants(snowshoe, level=0)
    expected = sp.Matrix([[1]])
    assert result == expected


def test_absolute_determinants_level_2_snowshoe(snowshoe):
    result = absolute_determinants(snowshoe, level=2)
    expected = sp.Matrix([[8]])
    assert result == expected


def test_absolute_determinants_rejects_invalid_level(snowshoe):
    with pytest.raises(ValueError):
        absolute_determinants(snowshoe, level=100)


# =============================================================================
# weighted_determinants()
# =============================================================================

def test_weighted_determinants_all_levels_snowshoe(snowshoe):
    result = weighted_determinants(snowshoe)
    expected = sp.Matrix([
        [                1],
        [                1],
        [sp.Rational(1, 2)],
        [sp.Rational(1, 2)]])
    assert result == expected


# =============================================================================
# feedback_metrics()
# =============================================================================

def test_feedback_metrics_columns_snowshoe(snowshoe):
    df = feedback_metrics(snowshoe)
    result = list(df.columns)
    expected = ['Feedback level', 'Net', 'Absolute', 'Positive', 'Negative', 'Weighted']
    assert result == expected


def test_feedback_metrics_values_snowshoe(snowshoe):
    df = feedback_metrics(snowshoe)
    result = (list(df['Net']), list(df['Absolute']))
    expected = ([-1, -2, -3, -2], [1, 2, 3, 2])
    assert result == expected


def test_feedback_metrics_values_chain(chain):
    df = feedback_metrics(chain)
    result = (list(df['Net']), list(df['Absolute']))
    expected = ([-1, -5, -14, -22, -20, -8], [1, 5, 14, 22, 20, 8])
    assert result == expected


def test_feedback_metrics_columns_mesocosm(mesocosm):
    df = feedback_metrics(mesocosm)
    result = list(df.columns)
    expected = ['Feedback level', 'Net', 'Absolute', 'Positive', 'Negative', 'Weighted']
    assert result == expected


def test_feedback_metrics_values_mesocosm(mesocosm):
    df = feedback_metrics(mesocosm)
    result = {
        'Net': list(df['Net']),
        'Absolute': list(df['Absolute']),
        'Positive': list(df['Positive']),
        'Negative': list(df['Negative']),
        'Weighted': list(df['Weighted'])
    }
    expected = {
        'Net': [-1, -3, -13, -24, -41, -39, -30, -12, -2],
        'Absolute': [1, 3, 13, 26, 53, 69, 74, 56, 18],
        'Positive': [0, 0, 0, 1, 6, 15, 22, 22, 8],
        'Negative': [1, 3, 13, 25, 47, 54, 52, 34, 10],
        'Weighted': [-1, -1, -1, sp.Rational(-12, 13), sp.Rational(-41, 53), sp.Rational(-13, 23), sp.Rational(-15, 37), sp.Rational(-3, 14), sp.Rational(-1, 9)]
    }
    assert result == expected


# =============================================================================
# determinants_metrics()
# =============================================================================

def test_determinants_metrics_columns_snowshoe(snowshoe):
    df = determinants_metrics(snowshoe)
    result = list(df.columns)
    expected = ['Hurwitz determinant', 'Net', 'Absolute', 'Weighted']
    assert result == expected


def test_determinants_metrics_values_snowshoe(snowshoe):
    df = determinants_metrics(snowshoe)
    result = (list(df['Net']), list(df['Absolute']))
    expected = ([1, 2, 4, 8], [1, 2, 8, 16])
    assert result == expected


# =============================================================================
# conditional_stability()
# =============================================================================

def test_create_model_c_is_the_dambacher_chain():
    chain = [[0, 1, 0, 0, 0], [-1, 0, 1, 0, 0], [0, -1, 0, 1, 0], [0, 0, -1, 0, 1], [0, 0, 0, -1, -1]]
    result = nx.is_isomorphic(_create_model_c(5), list_to_digraph(chain, ['1', '2', '3', '4', '5']),
                              edge_match=lambda a, b: a['sign'] == b['sign'])
    expected = True
    assert result == expected


def test_conditional_stability_sign_stable_snowshoe(snowshoe):
    df = conditional_stability(snowshoe)
    result = df.loc[df['Test'] == 'Model class', 'Result'].iloc[0]
    expected = 'Sign stable'
    assert result == expected


def test_conditional_stability_sign_stable_chain(chain):
    df = conditional_stability(chain)
    result = df.loc[df['Test'] == 'Model class', 'Result'].iloc[0]
    expected = 'Sign stable'
    assert result == expected


def test_conditional_stability_class_ii(class_ii):
    df = conditional_stability(class_ii)
    result = df.loc[df['Test'] == 'Model class', 'Result'].iloc[0]
    expected = 'Class II'
    assert result == expected


def test_conditional_stability_class_ii_colour_pass(colour_pass):
    df = conditional_stability(colour_pass)
    result = df.loc[df['Test'] == 'Model class', 'Result'].iloc[0]
    expected = 'Class II'
    assert result == expected


def test_conditional_stability_class_i_mesocosm(mesocosm):
    df = conditional_stability(mesocosm)
    result = df.loc[df['Test'] == 'Model class', 'Result'].iloc[0]
    expected = 'Class I'
    assert result == expected


# =============================================================================
# simulation_stability()
# =============================================================================

def test_simulation_stability_metrics_snowshoe(snowshoe):
    np.random.seed(42)
    result = simulation_stability(snowshoe, n_sim=100)
    metrics = {
        'Stable matrices': result.loc[result['Test'] == 'Stable matrices', 'Result'].iloc[0],
        'Unstable matrices': result.loc[result['Test'] == 'Unstable matrices', 'Result'].iloc[0],
        'Hurwitz criterion i': result.loc[result['Test'] == 'Hurwitz criterion i', 'Result'].iloc[0],
        'Hurwitz criterion ii': result.loc[result['Test'] == 'Hurwitz criterion ii', 'Result'].iloc[0],
    }
    expected = {
        'Stable matrices': '100.00%',
        'Unstable matrices': '0.00%',
        'Hurwitz criterion i': '0.00%',
        'Hurwitz criterion ii': '0.00%',
    }
    assert metrics == expected


def test_simulation_stability_metrics_class_ii(class_ii):
    np.random.seed(42)
    result = simulation_stability(class_ii, n_sim=100)
    metrics = {
        'Stable matrices': result.loc[result['Test'] == 'Stable matrices', 'Result'].iloc[0],
        'Unstable matrices': result.loc[result['Test'] == 'Unstable matrices', 'Result'].iloc[0],
        'Hurwitz criterion i': result.loc[result['Test'] == 'Hurwitz criterion i', 'Result'].iloc[0],
        'Hurwitz criterion ii': result.loc[result['Test'] == 'Hurwitz criterion ii', 'Result'].iloc[0],
        'Hurwitz criterion i only': result.loc[result['Test'] == 'Hurwitz criterion i only', 'Result'].iloc[0],
        'Hurwitz criterion ii only': result.loc[result['Test'] == 'Hurwitz criterion ii only', 'Result'].iloc[0],
    }
    expected = {
        'Stable matrices': '6.00%',
        'Unstable matrices': '94.00%',
        'Hurwitz criterion i': '94.00%',
        'Hurwitz criterion ii': '13.00%',
        'Hurwitz criterion i only': '81.00%',
        'Hurwitz criterion ii only': '0.00%',
    }
    assert metrics == expected


def test_simulation_stability_metrics_mesocosm(mesocosm):
    np.random.seed(42)
    result = simulation_stability(mesocosm, n_sim=100)
    metrics = {
        'Stable matrices': result.loc[result['Test'] == 'Stable matrices', 'Result'].iloc[0],
        'Unstable matrices': result.loc[result['Test'] == 'Unstable matrices', 'Result'].iloc[0],
        'Hurwitz criterion i': result.loc[result['Test'] == 'Hurwitz criterion i', 'Result'].iloc[0],
        'Hurwitz criterion ii': result.loc[result['Test'] == 'Hurwitz criterion ii', 'Result'].iloc[0],
        'Hurwitz criterion i only': result.loc[result['Test'] == 'Hurwitz criterion i only', 'Result'].iloc[0],
        'Hurwitz criterion ii only': result.loc[result['Test'] == 'Hurwitz criterion ii only', 'Result'].iloc[0],
    }
    expected = {
        'Stable matrices': '36.00%',
        'Unstable matrices': '64.00%',
        'Hurwitz criterion i': '42.00%',
        'Hurwitz criterion ii': '44.00%',
        'Hurwitz criterion i only': '20.00%',
        'Hurwitz criterion ii only': '22.00%',
    }
    assert metrics == expected


# =============================================================================
# Additional coverage tests
# =============================================================================

def test_simulation_stability_unstable_equals_hurwitz_failures(snowshoe_rp):
    table = simulation_stability(snowshoe_rp, n_sim=500, seed=42)
    rates = {test: float(value.rstrip('%')) for test, value in zip(table['Test'], table['Result'])}
    result = rates['Unstable matrices']
    expected = rates['Hurwitz criterion ii'] + rates['Hurwitz criterion i only']
    assert result == pytest.approx(expected)


def test_simulation_stability_presample(snowshoe):
    n_sim = 5
    presample = np.random.uniform(0.01, 1, (n_sim, 3, 3))
    result = simulation_stability(snowshoe, n_sim=n_sim, presample=presample)
    assert isinstance(result, pd.DataFrame)
    assert 'Test' in result.columns
    assert 'Result' in result.columns
    assert len(result) == 6


def test_simulation_stability_rejects_presample_wrong_shape(snowshoe):
    presample = np.random.uniform(0.01, 1, (5, 2, 2))
    with pytest.raises(ValueError, match="presample must have shape"):
        simulation_stability(snowshoe, n_sim=5, presample=presample)


def test_system_feedback_rejects_invalid_form(snowshoe):
    with pytest.raises(ValueError, match="^Invalid form"):
        system_feedback(snowshoe, form="invalid")


def test_system_feedback_rejects_negative_level(snowshoe):
    with pytest.raises(ValueError, match="Level must be between"):
        system_feedback(snowshoe, level=-1)


def test_absolute_feedback_rejects_negative_level(snowshoe):
    with pytest.raises(ValueError, match="Level must be between"):
        absolute_feedback(snowshoe, level=-1)


def test_absolute_feedback_rejects_invalid_method(snowshoe):
    with pytest.raises(ValueError, match="method must be either 'combinations' or 'polynomial'"):
        absolute_feedback(snowshoe, method="invalid")


def test_hurwitz_determinants_rejects_invalid_form(snowshoe):
    with pytest.raises(ValueError, match="^Invalid form"):
        hurwitz_determinants(snowshoe, form="invalid")


def test_sign_stability_jeffries_counterexample():
    G = list_to_digraph([[-1, 1, 0], [0, 0, -1], [0, 1, 0]], ["1", "2", "3"])
    result = sign_stability(G).set_index("Test")["Result"]
    assert all(result[f"Condition {c}"] for c in ["i", "ii", "iii", "iv", "v"])
    assert not result["Colour test"]
    assert not result["Sign stable"]


def test_simulation_stability_positive_self_effect_fails_criterion_i():
    G = list_to_digraph([[1, 0], [0, -1]], ["a", "b"])
    result = simulation_stability(G, n_sim=200).set_index("Test")["Result"]
    assert result["Stable matrices"] == "0.00%"
    assert result["Hurwitz criterion i"] == "100.00%"


def test_conditional_stability_tied_maximum_is_class_ii():
    G = list_to_digraph([[-1, 1, -1], [1, -1, -1], [1, 1, -1]], ["A", "B", "C"])
    assert conditional_stability(G)["Result"].iloc[-1] == "Class II"


def test_conditional_stability_rejects_missing_feedback():
    with pytest.raises(ValueError, match="^No feedback terms at level 1$"):
        conditional_stability(list_to_digraph([[0, -1], [1, 0]], ["A", "B"]))


def test_stability_analysis_keeps_sections_when_conditional_unavailable():
    G = list_to_digraph([[0, -1], [1, 0]], ["A", "B"])
    presample = np.ones((2, 2, 2))
    result = stability_analysis(G, n_sim=2, presample=presample).set_index("Test")["Result"]
    expected = simulation_stability(G, n_sim=2, presample=presample).set_index("Test")["Result"]
    assert result["Sign stable"] == sign_stability(G)["Result"].iloc[-1]
    assert result["Stable matrices"] == expected["Stable matrices"]
    assert result["Weighted feedback"] == "Unavailable: No feedback terms at level 1"
    assert result["Model class"] == "Unavailable: No feedback terms at level 1"


def test_sign_stability_beyond_63_states():
    G = list_to_digraph((-np.eye(64)).astype(int).tolist(), [f"n{i}" for i in range(64)])
    assert sign_stability(G)["Result"].iloc[-1]
    assert absolute_feedback(G, level=0) == sp.Matrix([1])


def test_absolute_feedback_beyond_63_states():
    G = list_to_digraph(-np.eye(64, dtype=int))
    assert absolute_feedback(G, level=1) == sp.Matrix([64])
    assert absolute_feedback(G, level=64) == sp.Matrix([1])


def test_absolute_feedback_level_1_uses_diagonal(monkeypatch):
    def unexpected_count(*args, **kwargs):
        pytest.fail("First-level feedback only needs diagonal entries")
    monkeypatch.setattr("qmm.core.stability.perm", unexpected_count)
    G = list_to_digraph(-np.ones((16, 16), dtype=int))
    assert absolute_feedback(G, level=1) == sp.Matrix([16])


def test_has_cycle_cover_agrees_with_exact_count():
    rng = np.random.RandomState(0)
    for _ in range(300):
        n = rng.randint(1, 7)
        A = (rng.uniform(size=(n, n)) < 0.35).astype(int)
        assert _has_cycle_cover(A) == (perm(A) > 0)


def test_weighted_feedback_follows_graph_edits():
    G = list_to_digraph([[-1]])
    result = weighted_feedback(G)
    result[1] = 999
    assert weighted_feedback(G) == sp.Matrix([-1, -1])
    G["1"]["1"]["sign"] = 1
    assert weighted_feedback(G) == sp.Matrix([-1, 1])


def test_hurwitz_determinants_size_limit_precedes_expansion(monkeypatch):
    def unexpected_expansion(*args, **kwargs):
        pytest.fail("The unsupported symbolic system should be rejected before expansion")

    monkeypatch.setattr("qmm.core.stability.system_feedback", unexpected_expansion)
    with pytest.raises(ValueError, match="five or fewer"):
        hurwitz_determinants(list_to_digraph(-np.eye(6, dtype=int)))


@pytest.mark.parametrize("strengths", [[1, 1e-16], [1, 1e-6, 1e-6, 1e-6]])
def test_simulation_stability_accepts_stable_diagonals_at_different_rates(strengths):
    n = len(strengths)
    result = simulation_stability(
        list_to_digraph(-np.eye(n, dtype=int)), n_sim=1,
        presample=np.diag(strengths)[None],
    ).set_index("Test")["Result"]
    assert result["Stable matrices"] == "100.00%"
    assert result["Hurwitz criterion i"] == "0.00%"
    assert result["Hurwitz criterion ii"] == "0.00%"


@pytest.mark.parametrize("scale", [1e-100, 1.0, 1e100])
def test_simulation_stability_hurwitz_checks_independent_of_overall_scale(scale):
    result = simulation_stability(
        list_to_digraph(-np.eye(5, dtype=int)), n_sim=1,
        presample=np.full((1, 5, 5), scale),
    ).set_index("Test")["Result"]
    assert result["Stable matrices"] == "100.00%"
    assert result["Hurwitz criterion i"] == "0.00%"
    assert result["Hurwitz criterion ii"] == "0.00%"


@pytest.mark.parametrize("n_sim", [0, -1, 1.5, True])
def test_simulation_stability_rejects_invalid_counts(n_sim):
    with pytest.raises(ValueError, match="n_sim must be a positive integer"):
        simulation_stability(list_to_digraph([[-1]]), n_sim=n_sim)


@pytest.mark.parametrize("strength", [-1, np.inf, np.nan])
def test_simulation_stability_rejects_invalid_presampled_strengths(strength):
    with pytest.raises(ValueError, match="finite nonnegative strengths"):
        simulation_stability(
            list_to_digraph([[-1]]), n_sim=1, presample=np.full((1, 1, 1), strength)
        )
