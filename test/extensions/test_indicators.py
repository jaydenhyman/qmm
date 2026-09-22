"""Tests for qmm.extensions.indicators module."""

import pandas as pd
import networkx as nx
import numpy as np
import pytest

from qmm.core.structure import define_input_output
from qmm.extensions.indicators import mutual_information


def test_mutual_information_perturb_P_positive_mesocosm_alt_models(mesocosm_alt_models):
    result = mutual_information(mesocosm_alt_models, perturb='P:+', n_sim=100, seed=42)
    result['Mutual information'] = result['Mutual information'].round(6)
    expected = pd.DataFrame({
        'Node': ['A1', 'A2', 'C1', 'H1', 'C2', 'AP', 'H2', 'P'],
        'Mutual information': [0.046425, 0.025838, 0.024157, 0.014534, 0.01416, 0.0, 0.0, 0.0],
    })
    assert result.equals(expected)


def test_mutual_information_perturb_P_negative_mesocosm_alt_models(mesocosm_alt_models):
    result = mutual_information(mesocosm_alt_models, perturb='P:-', n_sim=100, seed=42)
    result['Mutual information'] = result['Mutual information'].round(6)
    expected = pd.DataFrame({
        'Node': ['A1', 'A2', 'C1', 'H1', 'C2', 'AP', 'H2', 'P'],
        'Mutual information': [0.046425, 0.025838, 0.024157, 0.014534, 0.01416, 0.0, 0.0, 0.0],
    })
    assert result.equals(expected)


def test_mutual_information_multiple_perturbations_mesocosm_alt_models(mesocosm_alt_models):
    result = mutual_information(mesocosm_alt_models, perturb='A1:+, H1:-', n_sim=100, seed=42)
    result['Mutual information'] = result['Mutual information'].round(6)
    expected = pd.DataFrame({
        'Node': ['A2', 'C1', 'C2', 'A1', 'AP', 'H2', 'P', 'H1'],
        'Mutual information': [0.024017, 0.020364, 0.006702, 0.005268, 0.000471, 0.000471, 0.000471, 0.000080],
    })
    assert result.equals(expected)


def test_mutual_information_include_null_mesocosm(mesocosm):
    result = mutual_information(mesocosm, perturb='P:+', n_sim=100, seed=42, include_null=True)
    result['Mutual information'] = result['Mutual information'].round(6)
    expected = pd.DataFrame({
        'Node': ['H2', 'AP', 'P', 'A2', 'H1', 'A1', 'C1', 'C2'],
        'Mutual information': [0.320574, 0.293565, 0.280667, 0.172393, 0.164049, 0.16089, 0.141143, 0.120571],
    })
    assert result.equals(expected)


def test_mutual_information_rejects_different_nodes(snowshoe):
    G2 = snowshoe.copy()
    G2.remove_node('P')
    with pytest.raises(ValueError, match=r"Model B has different nodes: \['P'\]"):
        mutual_information((snowshoe, G2), perturb='R:+', n_sim=100, seed=42)


def test_mutual_information_rejects_category_change(snowshoe):
    G2 = snowshoe.copy()
    G2.remove_edges_from([('C', 'R'), ('C', 'P')])
    with pytest.raises(ValueError, match="^Model B: Nodes change category: C$"):
        mutual_information((snowshoe, G2), perturb='R:+', n_sim=100, seed=42)


def test_mutual_information_accepts_a_list_of_models(snowshoe_rp):
    alternative = nx.DiGraph(snowshoe_rp)
    alternative.remove_edge("C", "P")
    result = mutual_information([snowshoe_rp, alternative], perturb="R:+", n_sim=100, seed=42)
    assert set(result["Node"]) == {"R", "C", "P"}


@pytest.mark.parametrize('uncertain_interactions', ['sample', 'enumerate'])
@pytest.mark.parametrize('weights', ['equal', 'posterior'])
def test_mutual_information_weights_options(monkeypatch, uncertain_interactions, weights):
    models = [nx.DiGraph(), nx.DiGraph()]
    for model in models:
        model.add_node('X', category='state')
        model.add_edge('X', 'X', sign=-1)
    draws = iter([
        [{'effects': [np.array([1.0])] * 4, 'prop_stable': 0.25}],
        [{'effects': [np.array([-1.0])] * 4, 'prop_stable': stability}
         for stability in ((0.5, 1.0) if uncertain_interactions == 'enumerate' else (0.75,))],
    ])
    calls = []
    monkeypatch.setattr('qmm.extensions.indicators._simulate', lambda *args, **kwargs: (calls.append(kwargs), iter(next(draws)))[1])
    result = mutual_information(models, 'X:+', n_sim=4, dist='strong', uncertain_interactions=uncertain_interactions, weights=weights, base=2)
    assert [call['dist'] for call in calls] == ['strong', 'strong']
    expected = -(0.25 * np.log2(0.25) + 0.75 * np.log2(0.75)) if weights == 'posterior' else 1.0
    assert result['Mutual information'].iloc[0] == pytest.approx(expected)


def test_mutual_information_dist_strong_mesocosm_alt_models(mesocosm_alt_models):
    default = mutual_information(mesocosm_alt_models, 'P:+', n_sim=100, seed=42)
    result = mutual_information(mesocosm_alt_models, 'P:+', n_sim=100, seed=42, dist='strong')
    assert list(result['Node']) == list(default['Node']) or set(result['Node']) == set(default['Node'])
    assert not result.set_index('Node').equals(default.set_index('Node'))
    assert mutual_information(mesocosm_alt_models, 'P:+', n_sim=100, seed=42, dist='uniform').equals(default)


def test_mutual_information_bits_are_nats_over_log_two(mesocosm_alt_models):
    nats = mutual_information(mesocosm_alt_models, 'P:+', n_sim=100)
    bits = mutual_information(mesocosm_alt_models, 'P:+', n_sim=100, base=2)
    np.testing.assert_allclose(bits['Mutual information'], nats['Mutual information'] / np.log(2))


def test_mutual_information_melbourne_thomas_2012_table_1(mesocosm_alt_models):
    """Reproduce the indicator information in Melbourne-Thomas et al. (2012).

    Ecological Monographs 82:505–519, Table 1, reports bits after conditioning
    the two mesocosm models on stability and increasing phosphorus. The paper
    uses 100,000 simulated samples. This test uses 20,000 accepted simulations
    per model and the validation suite's 0.01-bit Monte Carlo tolerance.
    https://doi.org/10.1890/12-0207.1
    """
    result = mutual_information(
        mesocosm_alt_models, perturb='P:+', n_sim=20_000, seed=1,
        weights='posterior', base=2,
    ).set_index('Node')['Mutual information']
    expected = pd.Series({
        'A1': 0.078, 'C2': 0.064, 'C1': 0.048, 'A2': 0.016,
        'H1': 0.002, 'H2': 0.0, 'P': 0.0, 'AP': 0.0,
    })
    assert set(result.index) == set(expected.index)
    np.testing.assert_allclose(result.reindex(expected.index), expected, atol=0.01, rtol=0)


@pytest.mark.parametrize('kwargs', [{'weights': 'invalid'}, {'base': 1}, {'base': 0}, {'base': np.inf}, {'base': np.nan}])
def test_mutual_information_rejects_invalid_weight_or_base(snowshoe, kwargs):
    with pytest.raises(ValueError):
        mutual_information(snowshoe, 'R:+', n_sim=1, **kwargs)


@pytest.mark.parametrize('weights', ['equal', 'posterior'])
def test_mutual_information_opposite_signs_is_one_bit(weights):
    models = []
    for sign in (1, -1):
        G = nx.DiGraph()
        G.add_edge('A', 'A', sign=-1)
        G.add_edge('B', 'B', sign=-1)
        G.add_edge('A', 'B', sign=sign)
        models.append(G)
    result = mutual_information(models, 'A:+', n_sim=100, seed=42, weights=weights, base=2)
    expected = pd.DataFrame({'Node': ['B', 'A'], 'Mutual information': [1.0, 0.0]})
    assert result.equals(expected)


@pytest.mark.parametrize('stale_first', [False, True])
def test_mutual_information_aligns_models_by_node_identity(mesocosm_alt_models, snowshoe_io, stale_first):
    G, G_alt = mesocosm_alt_models
    reordered = nx.DiGraph()
    reordered.add_nodes_from(reversed(list(G_alt.nodes(data=True))))
    reordered.add_edges_from(G_alt.edges(data=True))
    result = mutual_information([G, reordered], 'P:+', n_sim=100, seed=42)
    expected = mutual_information([G, G_alt], 'P:+', n_sim=100, seed=42)
    assert result.equals(expected)

    stale = snowshoe_io.copy()
    stale.nodes['Out2']['category'] = 'input'
    original = stale.copy()
    models = [stale, snowshoe_io] if stale_first else [snowshoe_io, stale]
    result = mutual_information(models, 'R:+', n_sim=10, seed=42)
    expected = pd.DataFrame({'Node': ['C', 'Out1', 'Out2', 'P', 'R'], 'Mutual information': [0] * 5})
    assert result.equals(expected)
    assert nx.utils.graphs_equal(stale, original)


def test_mutual_information_accepts_alternative_that_isolates_a_self_limited_node(snowshoe):
    joined = nx.DiGraph(snowshoe)
    joined.add_edge('Z', 'Z', sign=-1)
    apart = nx.DiGraph(joined)
    joined.add_edge('Z', 'R', sign=-1)
    table = mutual_information([define_input_output(joined), define_input_output(apart)], perturb='Z:-', n_sim=50, seed=1)
    result = dict(zip(table["Node"], table["Mutual information"]))["Z"]
    expected = 0.0
    assert result == expected


def test_mutual_information_enumerate_runs_every_structure_snowshoe_dashed(snowshoe_dashed):
    stale = snowshoe_dashed.copy()
    stale.nodes['C']['category'] = 'input'
    original = stale.copy()
    table = mutual_information([stale, snowshoe_dashed], perturb='C:+', n_sim=25, seed=1, uncertain_interactions="enumerate")
    result = (len(table), table["Mutual information"].max())
    expected = (3, 0.0)
    assert result == expected
    assert nx.utils.graphs_equal(stale, original)
