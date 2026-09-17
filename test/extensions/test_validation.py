"""Tests for qmm.extensions.validation module."""

import pytest
import pandas as pd
import numpy as np
import networkx as nx
from qmm.extensions.effects import get_simulations
from qmm.core.structure import define_input_output
import sympy as sp
from qmm.extensions.validation import (
    marginal_likelihood,
    compare_model_alternatives,
    posterior_predictions,
    diagnose_observations,
    bayes_factors,
)

# =============================================================================
# marginal_likelihood
# =============================================================================

def test_marginal_likelihood_default_mesocosm(mesocosm):
    result = marginal_likelihood(mesocosm, perturb='P:+', observe='A1:+', seed=42)
    expected = 0.3461
    assert np.allclose(result, expected)

def test_marginal_likelihood_three_observations(mesocosm):
    result = marginal_likelihood(mesocosm, perturb='P:+', n_sim=100, observe='AP:+, C2:+, H2:+', dist='uniform', seed=42)
    expected = 0.57
    assert np.allclose(result, expected)

def test_marginal_likelihood_four_observations(mesocosm):
    result = marginal_likelihood(mesocosm, perturb='P:+', n_sim=100, observe='A2:+, AP:+, C2:+, H2:+', dist='uniform', seed=42)
    expected = 0.47
    assert np.allclose(result, expected)

def test_marginal_likelihood_reproducible_seed(mesocosm):
    result = marginal_likelihood(mesocosm, perturb='P:+', observe='A1:+', seed=42)
    expected = marginal_likelihood(mesocosm, perturb='P:+', observe='A1:+', seed=42)
    assert np.allclose(result, expected)

def test_marginal_likelihood_rejects_invalid_perturbation(mesocosm):
    with pytest.raises(ValueError) as exc_info:
        marginal_likelihood(mesocosm, perturb='Invalid:+', observe='A1:+')
    assert "Unknown perturbation node" in str(exc_info.value)

def test_marginal_likelihood_zero_observation_dashed_edge(snowshoe_io_na):
    G = snowshoe_io_na.copy()
    G.add_edge('R', 'N', sign=1, dashes=True)
    sampled = marginal_likelihood(G, perturb='P:+', observe='N:0', n_sim=100, dist='uniform', seed=42)
    G['R']['N']['dashes'] = False
    result = (0.0 < sampled < 1.0, marginal_likelihood(G, perturb='P:+', observe='N:0', n_sim=100, dist='uniform', seed=42))
    expected = (True, 0.0)
    assert result == expected

def test_marginal_likelihood_zero_observation_no_edge(snowshoe_io_na):
    result = marginal_likelihood(snowshoe_io_na, perturb='P:+', observe='N:0', n_sim=100, dist='uniform', seed=42)
    expected = 1.0
    assert np.allclose(result, expected)

# =============================================================================
# compare_model_alternatives
# =============================================================================

def test_compare_model_alternatives_combinations(snowshoe_dashed):
    df = compare_model_alternatives(snowshoe_dashed, perturb='C:+', observe='P:-', n_sim=100, seed=42, combinations=True)
    expected_data = {
        'Marginal likelihood': [0.52, 0.47, 0.0, 0.0],
        ('R', 'P'): ['\u2713', '\u2713', '', ''],
        ('P', 'R'): ['\u2713', '\u2713', '', ''],
        ('C', 'C'): ['\u2713', '', '', '\u2713']
    }
    result = df.to_dict('list')
    expected = expected_data
    assert result == expected

def test_compare_model_alternatives_combinations_false(snowshoe_dashed):
    df = compare_model_alternatives(snowshoe_dashed, perturb='C:+', observe='P:-', n_sim=100, seed=42, combinations=False)
    expected_data = {
        'Marginal likelihood': [0.47, 0.0, 0.0],
        ('R', 'P'): ['\u2713', '', ''],
        ('P', 'R'): ['\u2713', '', ''],
        ('C', 'C'): ['', '', '\u2713']
    }
    result = df.to_dict('list')
    expected = expected_data
    assert result == expected

def test_compare_model_alternatives_rejects_category_change(dashed_role_change):
    with pytest.raises(ValueError, match="^Nodes change category: C$"):
        compare_model_alternatives(dashed_role_change, perturb="A:+", observe="B:+", n_sim=10)


def test_compare_model_alternatives_no_dashed_edges(snowshoe):
    G = snowshoe.copy()
    G.nodes['C']['category'] = 'input'
    original = G.copy()
    result = compare_model_alternatives(G, perturb='R:+', observe='C:+', n_sim=5)
    expected = pd.DataFrame({'Marginal likelihood': [1.0]})
    assert result.equals(expected)
    assert nx.utils.graphs_equal(G, original)



# =============================================================================
# posterior_predictions
# =============================================================================

def test_posterior_predictions_default_mesocosm(mesocosm):
    result = posterior_predictions(mesocosm, perturb='P:+', observe='A1:+', seed=42)
    expected = [
        1.0,
        1.0,
        -0.5574,
        1.0,
        0.9835,
        1.0,
        0.5942,
        0.9835,
    ]
    assert np.allclose(np.array(result.tolist(), dtype=float).ravel(), expected)
    assert posterior_predictions(mesocosm, perturb='P:+', observe='A1:+', seed=42, mode="dominant") == result
    for mode in ("absolute", "match_adjoint", "invalid", True, False, None):
        with pytest.raises(ValueError, match="Invalid mode"):
            posterior_predictions(None, perturb='P:+', mode=mode)

def test_posterior_predictions_three_observations(mesocosm):
    result = posterior_predictions(mesocosm, perturb='P:+', n_sim=100, observe='AP:+, C2:+, H2:+', dist='uniform', seed=42)
    expected = [
        1.0,
        0.57,
        0.72,
        1.0,
        1.0,
        1.0,
        -0.62,
        1.0,
    ]
    assert np.allclose(np.array(result.tolist(), dtype=float).ravel(), expected)

def test_posterior_predictions_mode_positive(mesocosm):
    result = posterior_predictions(mesocosm, perturb='P:+', n_sim=100, observe='A2:+, AP:+, C2:+, H2:+', dist='uniform', seed=42, mode="positive")
    expected = [
        1.0,
        0.38,
        1.0,
        1.0,
        1.0,
        1.0,
        0.21,
        1.0,
    ]
    assert np.allclose(np.array(result.tolist(), dtype=float).ravel(), expected)


def test_posterior_predictions_rejects_no_matching_simulations(mesocosm):
    with pytest.raises(RuntimeError, match="Maximum iterations reached"):
        posterior_predictions(mesocosm, perturb='P:+', observe='A1:-, A2:-, AP:-, H1:-, H2:-, C1:-, C2:-', n_sim=100, seed=42)


def test_posterior_predictions_structural_no_path_stays_nan(self_limited_pair):
    G = self_limited_pair
    for mode in ("dominant", "positive"):
        result = posterior_predictions(G, perturb="A:+", observe="A:+", n_sim=20, seed=1, max_attempts=20, mode=mode)
        assert float(result[0]) == 1.0 and result[1] is sp.nan
    with pytest.raises(RuntimeError, match="Matched 19/20 draws"):
        posterior_predictions(G, perturb="A:+", observe="A:+", n_sim=20, max_attempts=19)

# =============================================================================
# diagnose_observations
# =============================================================================

def test_diagnose_observations_perturb_nodes_input(snowshoe_io):
    df = diagnose_observations(snowshoe_io, observe='R:+, C:+', n_sim=100, perturb_nodes='input', seed=42)
    expected_data = {
        'Input': ['Inp1', 'Inp1', 'Inp2', 'Inp2'],
        'Sign': ['+', '-', '+', '-'],
        'Marginal likelihood': [0.52, 0.0, 0.0, 0.0]
    }
    result = df.to_dict('list')
    expected = expected_data
    assert result == expected


def test_diagnose_observations_perturb_nodes_state(snowshoe_io):
    df = diagnose_observations(snowshoe_io, observe='R:+', n_sim=100, perturb_nodes='state', seed=42)
    result = set(df['Input'].unique())
    expected = {'R', 'C', 'P'}
    assert result == expected


def test_diagnose_observations_perturb_nodes_comma_separated(snowshoe_io):
    df = diagnose_observations(snowshoe_io, observe='R:+', n_sim=100, perturb_nodes='R, C', seed=42)
    result = set(df['Input'].unique())
    expected = {'R', 'C'}
    assert result == expected


def test_diagnose_observations_perturb_nodes_default(snowshoe_io):
    df = diagnose_observations(snowshoe_io, observe='R:+', n_sim=100, seed=42)
    result = set(df['Input'].unique())
    expected = {'R', 'C', 'P', 'Inp1', 'Inp2'}
    assert result == expected


def test_diagnose_observations_rejects_unknown_observe_node(snowshoe_io):
    with pytest.raises(ValueError, match="Unknown observation node"):
        diagnose_observations(snowshoe_io, observe='InvalidNode:+', n_sim=100, perturb_nodes='input', seed=42)


def test_diagnose_observations_all_errors_returns_dataframe(minimal_error_graph):
    df = diagnose_observations(minimal_error_graph, observe='A:+', n_sim=10, perturb_nodes='A', seed=42)
    result = isinstance(df, pd.DataFrame)
    expected = True
    assert result == expected


# =============================================================================
# bayes_factors
# =============================================================================

def test_bayes_factors_three_observations(mesocosm_alt_models):
    G, G_alt = mesocosm_alt_models
    df = bayes_factors((G, G_alt), perturb='P:+', n_sim=100, observe='AP:+, C2:+, H2:+', dist='uniform', seed=42)
    result = (
        df['Model comparison'].tolist(),
        np.allclose(df['Likelihood 1'].to_numpy(), [0.57]),
        np.allclose(df['Likelihood 2'].to_numpy(), [0.73]),
        np.allclose(df['Bayes factor'].to_numpy(), [0.781], atol=0.01)
    )
    expected = (['Model A/Model B'], True, True, True)
    assert result == expected

def test_bayes_factors_four_observations(mesocosm_alt_models):
    G, G_alt = mesocosm_alt_models
    df = bayes_factors((G, G_alt), perturb='P:+', n_sim=100, observe='A2:+, AP:+, C2:+, H2:+', dist='uniform', seed=42)
    result = (
        df['Model comparison'].tolist(),
        np.allclose(df['Likelihood 1'].to_numpy(), [0.47]),
        np.allclose(df['Likelihood 2'].to_numpy(), [0.36]),
        np.allclose(df['Bayes factor'].to_numpy(), [1.306], atol=0.01)
    )
    expected = (['Model A/Model B'], True, True, True)
    assert result == expected

def test_bayes_factors_rejects_different_nodes(snowshoe):
    other = nx.DiGraph(snowshoe)
    other.remove_node('P')
    with pytest.raises(ValueError, match=r"Model B has different nodes: \['P'\]"):
        bayes_factors([snowshoe, other], perturb='R:+', observe='C:+', n_sim=10)


def test_bayes_factors_rejects_category_change(snowshoe):
    other = nx.DiGraph(snowshoe)
    other.remove_edges_from([('C', 'R'), ('C', 'P')])
    with pytest.raises(ValueError, match="^Model B: Nodes change category: C$"):
        bayes_factors([snowshoe, other], perturb='R:+', observe='C:+', n_sim=10)


def test_bayes_factors_custom_names(bayes_models):
    df = bayes_factors(bayes_models, perturb='R:+', observe='P:+', names=['ModA', 'ModB'])
    result = 'ModA/ModB' in df['Model comparison'].values
    expected = True
    assert result == expected


def test_bayes_factors_undefined_when_both_likelihoods_zero():
    G = nx.DiGraph()
    G.add_node('A', category='state')
    G.add_edge('A', 'A', sign=-1)
    result = bayes_factors([G, G.copy()], 'A:+', 'A:-', n_sim=5)
    assert result['Likelihood 1'].iloc[0] == result['Likelihood 2'].iloc[0] == 0
    assert np.isnan(result['Bayes factor'].iloc[0])


@pytest.mark.parametrize('perturb_nodes', [[], 'input'])
def test_diagnose_observations_no_candidates_returns_empty_table(snowshoe, perturb_nodes):
    result = diagnose_observations(snowshoe, 'R:+', n_sim=1, perturb_nodes=perturb_nodes)
    assert result.empty
    assert list(result.columns) == ['Input', 'Sign', 'Marginal likelihood']


def test_diagnose_observations_perturb_nodes_list(snowshoe):
    result = diagnose_observations(snowshoe, 'R:+', n_sim=3, perturb_nodes=['R'])
    assert result['Input'].tolist() == ['R', 'R']
    assert result['Sign'].tolist() == ['+', '-']
    assert result['Marginal likelihood'].tolist() == [1.0, 0.0]


def test_marginal_likelihood_structural_zero_cell(structural_zero_chain):
    result = tuple(marginal_likelihood(structural_zero_chain, 'A:+', observe, n_sim=200, seed=42) for observe in ('B:0', 'B:+', 'C:+'))
    expected = (1.0, 0.0, 1.0)
    assert result == expected


def test_marginal_likelihood_is_half_by_symmetry_fork(fork):
    result = marginal_likelihood(fork, 'A:+', 'B:+', n_sim=4000, seed=42)
    expected = 0.5
    assert abs(result - expected) <= 3 * (0.25 / 4000) ** 0.5


def test_marginal_likelihood_certain_response_is_one_fork(fork):
    result = marginal_likelihood(fork, 'A:+', 'C:+', n_sim=200, seed=42)
    expected = 1.0
    assert result == expected


def test_marginal_likelihood_divides_by_stable_draws(mesocosm):
    result = marginal_likelihood(mesocosm, 'P:+', 'P:+', n_sim=200, seed=42)
    expected = 1.0
    assert result == expected


def test_compare_model_alternatives_dashed_route_fork(fork):
    G = fork.copy()
    G['C']['B']['dashes'] = True
    G.nodes['B']['category'] = 'input'
    original = G.copy()
    df = compare_model_alternatives(G, 'A:+', 'B:+', n_sim=4000, seed=42, combinations=False)
    result = (df['Marginal likelihood'][0], df[('C', 'B')][0], df[('C', 'B')][1])
    expected = (1.0, '', '✓')
    assert result == expected
    assert abs(float(df['Marginal likelihood'][1]) - 0.5) <= 3 * (0.25 / 4000) ** 0.5
    assert nx.utils.graphs_equal(G, original)


def test_posterior_predictions_conditioned_on_observation_fork(fork):
    result = [posterior_predictions(fork, 'A:+', observe, n_sim=200, seed=42).tolist() for observe in ('B:+', 'B:-')]
    expected = [[[1.0], [1.0], [1.0]], [[1.0], [-1.0], [1.0]]]
    assert result == expected


def test_diagnose_observations_ranks_certain_press_first_fork(fork):
    df = diagnose_observations(fork, 'B:-', perturb_nodes=['A', 'C'], n_sim=4000, seed=42)
    result = (df['Input'][0], df['Sign'][0], df['Marginal likelihood'][0], df['Marginal likelihood'][3])
    expected = ('C', '+', 1.0, 0.0)
    assert result == expected
    assert all(abs(x - 0.5) <= 3 * (0.25 / 4000) ** 0.5 for x in df.loc[df['Input'] == 'A', 'Marginal likelihood'])


def test_bayes_factors_against_single_route_fork(fork):
    other = fork.copy()
    other.remove_edge('C', 'B')
    df = bayes_factors([fork, other], 'A:+', 'B:+', n_sim=4000, seed=42)
    result = (df['Likelihood 2'][0], df['Bayes factor'][0])
    expected = (1.0, df['Likelihood 1'][0])
    assert result == expected
    assert abs(df['Likelihood 1'][0] - 0.5) <= 3 * (0.25 / 4000) ** 0.5


@pytest.mark.parametrize('uncertain_interactions', ['sample', 'enumerate'])
@pytest.mark.parametrize('stale_first', [False, True])
def test_bayes_factors_align_models_by_node_identity(mesocosm_alt_models, snowshoe_io, uncertain_interactions, stale_first):
    G, G_alt = mesocosm_alt_models
    reordered = nx.DiGraph()
    reordered.add_nodes_from(reversed(list(G_alt.nodes(data=True))))
    reordered.add_edges_from(G_alt.edges(data=True))
    result = bayes_factors([G, reordered], 'P:+', 'AP:+, C2:+', n_sim=100, seed=42)
    expected = bayes_factors([G, G_alt], 'P:+', 'AP:+, C2:+', n_sim=100, seed=42)
    assert result.equals(expected)

    G = nx.DiGraph(snowshoe_io)
    G.add_edge('C', 'C', sign=-1, dashes=True)
    stale = G.copy()
    stale.nodes['Out2']['category'] = 'input'
    original = stale.copy()
    models = [stale, G] if stale_first else [G, stale]
    result = bayes_factors(models, 'R:+', 'Out2:+', n_sim=10, seed=42, uncertain_interactions=uncertain_interactions)
    expected = pd.DataFrame({'Model comparison': ['Model A/Model B'], 'Likelihood 1': [1.0], 'Likelihood 2': [1.0], 'Bayes factor': [1.0]})
    assert result.equals(expected)
    assert nx.utils.graphs_equal(stale, original)
    assert stale in models


def test_bayes_factors_accepts_alternative_that_isolates_a_self_limited_node(snowshoe):
    joined = nx.DiGraph(snowshoe)
    joined.add_edge('Z', 'Z', sign=-1)
    apart = nx.DiGraph(joined)
    joined.add_edge('Z', 'R', sign=-1)
    table = bayes_factors([define_input_output(joined), define_input_output(apart)], perturb='R:+', observe='C:+', n_sim=50, seed=1)
    result = (len(table), table["Likelihood 2"][0])
    expected = (1, 1.0)
    assert result == expected


def test_posterior_predictions_enumerate_averages_structures_equally(fork):
    G = nx.DiGraph(fork)
    G['A']['C']['dashes'] = True
    G['C']['B']['dashes'] = True
    G = define_input_output(G)
    sims = get_simulations(G, n_sim=100, seed=1, perturb=('A', 1), observe=(('B', 1),), uncertain_interactions="enumerate")
    valid = np.array(sims["valid_sims"])
    pooled = np.mean(np.array(sims["effects"])[valid][:, 2] > 0)
    predictions = posterior_predictions(G, 'A:+', 'B:+', n_sim=100, seed=1, mode="positive", uncertain_interactions="enumerate")
    counts = [sum(v for v, c in zip(sims["valid_sims"], sims["structures"]) if c == code) for code in range(4)]
    result = ([float(x) for x in predictions], pooled, counts)
    expected = ([1.0, 1.0, 0.5], 0.5, [100] * 4)
    assert result == expected


def test_posterior_predictions_enumerate_skips_structure_without_matches_fork(fork):
    G = nx.DiGraph(fork)
    G['A']['B']['dashes'] = True
    with pytest.warns(UserWarning, match="No matching draws for structure 0"):
        result = posterior_predictions(G, 'A:+', 'B:+', n_sim=20, seed=1, uncertain_interactions="enumerate")
    expected = sp.Matrix([1.0, 1.0, 1.0])
    assert result == expected


def test_compare_model_alternatives_pairs_reciprocal_edges_snowshoe_dashed(snowshoe_dashed):
    df = compare_model_alternatives(snowshoe_dashed, perturb='C:+', observe='P:-', n_sim=50, seed=1)
    result = (len(df), all((row[('R', 'P')] == "\u2713") == (row[('P', 'R')] == "\u2713") for _, row in df.iterrows()))
    expected = (4, True)
    assert result == expected
