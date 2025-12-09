"""Tests for qmm.extensions.validation module."""

import pytest
import sympy as sp
import pandas as pd
import numpy as np
import networkx as nx
from qmm.extensions.validation import (
    marginal_likelihood,
    model_validation,
    posterior_predictions,
    diagnose_observations,
    bayes_factors,
)

# =============================================================================
# marginal_likelihood
# =============================================================================

def test_marginal_likelihood(mesocosm):
    """Test basic functionality of marginal_likelihood."""
    result = marginal_likelihood(mesocosm, perturb='P:+', observe='A1:+', seed=42)
    expected = 0.3461
    assert np.allclose(result, expected)

def test_marginal_likelihood_three_observations(mesocosm):
    """Test marginal_likelihood with three observations."""
    result = marginal_likelihood(mesocosm, perturb='P:+', n_sim=100, observe='AP:+, C2:+, H2:+', distribution='uniform', seed=42)
    expected = 0.57
    assert np.allclose(result, expected)

def test_marginal_likelihood_four_observations(mesocosm):
    """Test marginal_likelihood with four observations."""
    result = marginal_likelihood(mesocosm, perturb='P:+', n_sim=100, observe='A2:+, AP:+, C2:+, H2:+', distribution='uniform', seed=42)
    expected = 0.47
    assert np.allclose(result, expected)

def test_marginal_likelihood_reproducibility(mesocosm):
    """Test that marginal_likelihood is reproducible with same seed."""
    result = marginal_likelihood(mesocosm, perturb='P:+', observe='A1:+', seed=42)
    expected = marginal_likelihood(mesocosm, perturb='P:+', observe='A1:+', seed=42)
    assert np.allclose(result, expected)

def test_marginal_likelihood_invalid_perturbation(mesocosm):
    """Test that marginal_likelihood handles invalid perturbation."""
    with pytest.raises(Exception) as exc_info:
        marginal_likelihood(mesocosm, perturb='Invalid:+', observe='A1:+')
    result = exc_info.type
    expected = KeyError
    assert result == expected

# =============================================================================
# model_validation
# =============================================================================

def test_model_validation_alternative_structure(snowshoe_dashed):
    """Test model_validation with alternative structure."""
    df = model_validation(snowshoe_dashed, perturb='C:+', observe='P:-', n_sim=100, seed=42, combinations=True)
    expected_data = {
        'Marginal likelihood': ['0.520', '0.470', '0.430', '0.430', '0.000', '0.000', '0.000', '0.000'],
        'R → P': ['\u2713', '\u2713', '\u2713', '\u2713', '', '', '', ''],
        'C ⊸ C': ['\u2713', '', '', '\u2713', '', '\u2713', '', '\u2713'],
        'P ⊸ R': ['\u2713', '\u2713', '', '', '', '', '\u2713', '\u2713']
    }
    result = df.to_dict('list')
    expected = expected_data
    assert result == expected
    
def test_model_validation_combinations_false(snowshoe_dashed):
    """Test model_validation with combinations=False."""
    df = model_validation(snowshoe_dashed, perturb='C:+', observe='P:-', n_sim=100, seed=42, combinations=False)
    expected_data = {
        'Marginal likelihood': ['0.520', '0.000'],
        'R → P': ['\u2713', ''],
        'C ⊸ C': ['\u2713', ''],
        'P ⊸ R': ['\u2713', '']
    }
    result = df.to_dict('list')
    expected = expected_data
    assert result == expected

def test_model_validation_no_dashed_edges(snowshoe):
    """Test model_validation with no dashed edges."""
    df = model_validation(snowshoe, perturb='R:+', observe='C:+')
    result = (len(df), 'Marginal likelihood' in df.columns)
    expected = (1, True)
    assert result == expected



# =============================================================================
# posterior_predictions
# =============================================================================

def test_posterior_predictions(mesocosm):
    """Test functionality of posterior_predictions."""
    result = posterior_predictions(mesocosm, perturb='P:+', observe='A1:+', seed=42)
    expected = [
        1.0,
        1.0,
        -0.551863623230280,
        1.0,
        0.983819705287489,
        1.0,
        0.587113550996822,
        0.983819705287489,
    ]
    assert np.allclose(np.array(result.tolist(), dtype=float).ravel(), expected)

def test_posterior_predictions_complex_observations(mesocosm):
    """Test posterior_predictions with multiple observations."""
    result = posterior_predictions(mesocosm, perturb='P:+', n_sim=100, observe='AP:+, C2:+, H2:+', dist='uniform', seed=42)
    expected = [
        1.0,
        -0.508771929824561,
        0.824561403508772,
        1.0,
        1.0,
        1.0,
        -0.701754385964912,
        1.0,
    ]
    assert np.allclose(np.array(result.tolist(), dtype=float).ravel(), expected)

def test_posterior_predictions_positive_only(mesocosm):
    """Test posterior_predictions with positive_only=True."""
    result = posterior_predictions(mesocosm, perturb='P:+', n_sim=100, observe='A2:+, AP:+, C2:+, H2:+', dist='uniform', seed=42, positive_only=True)
    expected = [
        1.0,
        0.382978723404255,
        1.0,
        1.0,
        1.0,
        1.0,
        0.191489361702128,
        1.0,
    ]
    assert np.allclose(np.array(result.tolist(), dtype=float).ravel(), expected)


def test_posterior_predictions_no_matching_simulations(mesocosm):
    """Test posterior_predictions with observations that match no simulations."""
    result = posterior_predictions(mesocosm, perturb='P:+', observe='A1:-, A2:-, AP:-, H1:-, H2:-, C1:-, C2:-', n_sim=100, seed=42)
    expected = (8, 1)
    assert result.shape == expected

# =============================================================================
# diagnose_observations
# =============================================================================

def test_diagnose_observations_input_only(snowshoe_io):
    """Test diagnose_observations with perturb_nodes='input'."""
    df = diagnose_observations(snowshoe_io, observe='R:+, C:+', n_sim=100, perturb_nodes='input', seed=42)
    expected_data = {
        'Input': ['Inp1', 'Inp1', 'Inp2', 'Inp2'],
        'Sign': ['+', '-', '+', '-'],
        'Marginal likelihood': [0.57, 0.0, 0.0, 0.0]
    }
    result = df.to_dict('list')
    expected = expected_data
    assert result == expected


def test_diagnose_observations_state_only(snowshoe_io):
    """Test diagnose_observations with perturb_nodes='state'."""
    df = diagnose_observations(snowshoe_io, observe='R:+', n_sim=100, perturb_nodes='state', seed=42)
    result = set(df['Input'].unique())
    expected = {'R', 'C', 'P'}
    assert result == expected


def test_diagnose_observations_comma_separated_nodes(snowshoe_io):
    """Test diagnose_observations with comma-separated node list."""
    df = diagnose_observations(snowshoe_io, observe='R:+', n_sim=100, perturb_nodes='R, C', seed=42)
    result = set(df['Input'].unique())
    expected = {'R', 'C'}
    assert result == expected


def test_diagnose_observations_default_nodes(snowshoe_io):
    """Test diagnose_observations with default perturb_nodes."""
    df = diagnose_observations(snowshoe_io, observe='R:+', n_sim=100, seed=42)
    result = set(df['Input'].unique())
    expected = {'R', 'C', 'P', 'Inp1', 'Inp2'}
    assert result == expected


def test_diagnose_observations_with_errors(snowshoe_io):
    """Test diagnose_observations handles errors gracefully."""
    df = diagnose_observations(snowshoe_io, observe='InvalidNode:+', n_sim=100, perturb_nodes='input', seed=42)
    result = isinstance(df, pd.DataFrame)
    expected = True
    assert result == expected


def test_diagnose_observations_all_errors_empty_df(minimal_error_graph):
    """Test diagnose_observations returns empty DataFrame when all nodes error."""
    df = diagnose_observations(minimal_error_graph, observe='A:+', n_sim=10, perturb_nodes='A', seed=42)
    result = isinstance(df, pd.DataFrame)
    expected = True
    assert result == expected


# =============================================================================
# bayes_factors
# =============================================================================

def test_bayes_factors(mesocosm_alt_models):
    """Test basic functionality of bayes_factors."""
    G, G_alt = mesocosm_alt_models
    df = bayes_factors((G, G_alt), perturb='P:+', n_sim=100, observe='AP:+, C2:+, H2:+', distribution='uniform', seed=42)
    result = (
        df['Model comparison'].tolist(),
        np.allclose(df['Likelihood 1'].to_numpy(), [0.57]),
        np.allclose(df['Likelihood 2'].to_numpy(), [0.73]),
        np.allclose(df['Bayes factor'].to_numpy(), [0.781], atol=0.01)
    )
    expected = (['Model A/Model B'], True, True, True)
    assert result == expected

def test_bayes_factors_more_observations(mesocosm_alt_models):
    """Test bayes_factors with additional observations."""
    G, G_alt = mesocosm_alt_models
    df = bayes_factors((G, G_alt), perturb='P:+', n_sim=100, observe='A2:+, AP:+, C2:+, H2:+', distribution='uniform', seed=42)
    result = (
        df['Model comparison'].tolist(),
        np.allclose(df['Likelihood 1'].to_numpy(), [0.47]),
        np.allclose(df['Likelihood 2'].to_numpy(), [0.36]),
        np.allclose(df['Bayes factor'].to_numpy(), [1.306], atol=0.01)
    )
    expected = (['Model A/Model B'], True, True, True)
    assert result == expected

def test_bayes_factors_custom_names(bayes_models):
    """Test bayes_factors with custom model names."""
    df = bayes_factors(bayes_models, perturb='R:+', observe='P:+', names=['ModA', 'ModB'])
    result = 'ModA/ModB' in df['Model comparison'].values
    expected = True
    assert result == expected
