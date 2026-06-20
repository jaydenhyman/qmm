"""Tests for qmm.extensions.indicators module."""

import pandas as pd

from qmm.extensions.indicators import mutual_information


def test_mutual_information_perturb_P_positive_mesocosm_alt_models(mesocosm_alt_models):
    result = mutual_information(mesocosm_alt_models, perturb='P:+', n_sim=100, seed=42)
    result['Mutual Information'] = result['Mutual Information'].round(6)
    expected = pd.DataFrame({
        'Node': ['A1', 'A2', 'C1', 'H1', 'C2', 'AP', 'H2', 'P'],
        'Mutual Information': [0.046425, 0.025838, 0.024157, 0.014534, 0.01416, 0.0, 0.0, 0.0],
    })
    assert result.equals(expected)


def test_mutual_information_perturb_P_negative_mesocosm_alt_models(mesocosm_alt_models):
    result = mutual_information(mesocosm_alt_models, perturb='P:-', n_sim=100, seed=42)
    result['Mutual Information'] = result['Mutual Information'].round(6)
    expected = pd.DataFrame({
        'Node': ['A1', 'A2', 'C1', 'H1', 'C2', 'AP', 'H2', 'P'],
        'Mutual Information': [0.046425, 0.025838, 0.024157, 0.014534, 0.01416, 0.0, 0.0, 0.0],
    })
    assert result.equals(expected)


def test_mutual_information_multiple_perturbations_mesocosm_alt_models(mesocosm_alt_models):
    result = mutual_information(mesocosm_alt_models, perturb='A1:+, H1:-', n_sim=100, seed=42)
    result['Mutual Information'] = result['Mutual Information'].round(6)
    expected = pd.DataFrame({
        'Node': ['C1', 'C2', 'A2', 'A1', 'H1', 'AP', 'H2', 'P'],
        'Mutual Information': [0.042498, 0.010158, 0.008817, 0.007682, 0.003365, 0.001912, 0.001912, 0.001912],
    })
    assert result.equals(expected)


def test_mutual_information_include_null_mesocosm(mesocosm):
    result = mutual_information(mesocosm, perturb='P:+', n_sim=100, seed=42, include_null=True)
    result['Mutual Information'] = result['Mutual Information'].round(6)
    expected = pd.DataFrame({
        'Node': ['H2', 'AP', 'P', 'A2', 'H1', 'A1', 'C1', 'C2'],
        'Mutual Information': [0.320574, 0.293565, 0.280667, 0.172393, 0.164049, 0.16089, 0.141143, 0.120571],
    })
    assert result.equals(expected)


def test_mutual_information_nan_effects_snowshoe(snowshoe):
    G1 = snowshoe.copy()
    G2 = snowshoe.copy()
    G2.remove_node('P')
    result = mutual_information((G1, G2), perturb='R:+', n_sim=100, seed=42)
    assert 'P' in result['Node'].values
    nan_node_mi = result[result['Node'] == 'P']['Mutual Information'].iloc[0]
    assert nan_node_mi == 0.0
