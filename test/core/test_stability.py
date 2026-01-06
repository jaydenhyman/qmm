"""Tests for qmm.core.stability module."""

import pytest
import sympy as sp
import numpy as np
import pandas as pd

from qmm.core.stability import (
    sign_stability,
    feedback_metrics,
    determinants_metrics,
    conditional_stability,
    simulation_stability,
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
)


# =============================================================================
# _colour_test()
# =============================================================================

def test_colour_test_pass_colour_pass(colour_pass):
    result = _colour_test(colour_pass)
    expected = 'Pass'
    assert result == expected


def test_colour_test_fail_colour_fail(colour_fail):
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

def test_sign_stability_fail_condition_i_snowshoe(snowshoe):
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


def test_system_feedback_form_signed_chain_repeat_chain(chain):
    result = system_feedback(chain, level=None, form='signed')
    expected = sp.Matrix([[-1], [-5], [-14], [-22], [-20], [-8]])
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

def test_absolute_feedback_level_0_default_form_snowshoe(snowshoe):
    result = absolute_feedback(snowshoe, level=0)
    expected = sp.Matrix([[1]])
    assert result == expected


def test_absolute_feedback_all_levels_default_form_snowshoe(snowshoe):
    result = absolute_feedback(snowshoe)
    expected = sp.Matrix([[1], [2], [3], [2]])
    assert result == expected


def test_absolute_feedback_combinations_method_level_2_snowshoe(snowshoe):
    result = absolute_feedback(snowshoe, level=2, method="combinations")
    expected = sp.Matrix([[3]])
    assert result == expected


def test_absolute_feedback_polynomial_method_level_2_snowshoe(snowshoe):
    result = absolute_feedback(snowshoe, level=2, method="polynomial")
    expected = sp.Matrix([[3]])
    assert result == expected


def test_absolute_feedback_polynomial_all_levels_snowshoe(snowshoe):
    result = absolute_feedback(snowshoe, method="polynomial")
    expected = sp.Matrix([[1], [2], [3], [2]])
    assert result == expected


def test_absolute_feedback_all_levels_default_form_chain(chain):
    result = absolute_feedback(chain)
    expected = sp.Matrix([[1], [5], [14], [22], [20], [8]])
    assert result == expected


# =============================================================================
# weighted_feedback()
# =============================================================================

def test_weighted_feedback_default_form_signed_snowshoe(snowshoe):
    result = weighted_feedback(snowshoe)
    expected = sp.Matrix([[-1], [-1], [-1], [-1]])
    assert result == expected


def test_weighted_feedback_default_form_signed_chain(chain):
    result = weighted_feedback(chain)
    expected = sp.Matrix([[-1], [-1], [-1], [-1], [-1], [-1]])
    assert result == expected


# =============================================================================
# _hurwitz_matrix()
# =============================================================================

def test_hurwitz_matrix_level_0_symbolic_feedback_snowshoe(snowshoe):
    fb = system_feedback(snowshoe, level=None, form='symbolic')
    result = _hurwitz_matrix(fb, level=0)
    expected = sp.Matrix([[1]])
    assert result == expected


def test_hurwitz_matrix_level_2_symbolic_snowshoe(snowshoe):
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


def test_hurwitz_determinants_form_symbolic_large_graph_large_six_node(large_six_node):
    result = None
    expected = ValueError
    with pytest.raises(expected):
        hurwitz_determinants(large_six_node, form='symbolic')
        result = "No exception"
    assert result is None


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


def test_net_determinants_invalid_level_chain(chain):
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


def test_absolute_determinants_invalid_level_snowshoe(snowshoe):
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

def test_conditional_stability_sign_class_snowshoe(snowshoe):
    df = conditional_stability(snowshoe)
    result = df.loc[df['Test'] == 'Model class', 'Result'].iloc[0]
    expected = 'Sign stable'
    assert result == expected


def test_conditional_stability_sign_class_chain(chain):
    df = conditional_stability(chain)
    result = df.loc[df['Test'] == 'Model class', 'Result'].iloc[0]
    expected = 'Sign stable'
    assert result == expected


def test_conditional_stability_class_ii_class_ii(class_ii):
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

def test_simulation_stability_with_presample(snowshoe):
    n_sim = 5
    presample = np.random.uniform(0.01, 1, (n_sim, 3, 3))
    result = simulation_stability(snowshoe, n_sim=n_sim, presample=presample)
    assert isinstance(result, pd.DataFrame)
    assert 'Test' in result.columns
    assert 'Result' in result.columns
    assert len(result) == 6


def test_simulation_stability_presample_wrong_shape(snowshoe):
    presample = np.random.uniform(0.01, 1, (5, 2, 2))
    with pytest.raises(ValueError, match="presample must have shape"):
        simulation_stability(snowshoe, n_sim=5, presample=presample)


def test_system_feedback_invalid_form(snowshoe):
    with pytest.raises(ValueError, match="form must be either 'symbolic' or 'signed'"):
        system_feedback(snowshoe, form="invalid")


def test_system_feedback_invalid_level_negative(snowshoe):
    with pytest.raises(ValueError, match="Level must be between"):
        system_feedback(snowshoe, level=-1)


def test_absolute_feedback_invalid_level_negative(snowshoe):
    with pytest.raises(ValueError, match="Level must be between"):
        absolute_feedback(snowshoe, level=-1)


def test_absolute_feedback_invalid_method(snowshoe):
    with pytest.raises(ValueError, match="method must be either 'combinations' or 'polynomial'"):
        absolute_feedback(snowshoe, method="invalid")


def test_hurwitz_determinants_invalid_form(snowshoe):
    with pytest.raises(ValueError, match="form must be either 'symbolic' or 'signed'"):
        hurwitz_determinants(snowshoe, form="invalid")
