"""Tests for qmm.extensions.effects module."""

import pytest
import sympy as sp
import numpy as np
import pandas as pd
import networkx as nx

from qmm.core.helper import get_nodes
from qmm.extensions.effects import define_input_output
from qmm.extensions.effects import (
    cumulative_effects,
    net_effects,
    absolute_effects,
    weighted_effects,
    sign_determinacy_effects,
    get_simulations,
    simulation_effects,
    simulations_table,
    direct_effects,
    table_of_direct_effects,
    table_of_effects,
)
from qmm.core.press import (
    adjoint_matrix,
    numerical_simulations,
    absolute_feedback_matrix,
    weighted_predictions_matrix,
    sign_determinacy_matrix,
)

# =============================================================================
# define_input_output
# =============================================================================

def test_define_input_output_categories_snowshoe_io(snowshoe_io):
    categorized = define_input_output(snowshoe_io)
    result = {node: data['category'] for node, data in categorized.nodes(data=True)}
    expected = {
        'R': 'state',
        'C': 'state',
        'P': 'state',
        'Inp1': 'input',
        'Inp2': 'input',
        'Out1': 'output',
        'Out2': 'output',
    }
    assert result == expected


def test_define_input_output_remove_disconnected_true_disconnected_graph(disconnected_graph):
    result = sorted(define_input_output(disconnected_graph, remove_disconnected=True).nodes())
    expected = ['A', 'B']
    assert result == expected


def test_define_input_output_remove_disconnected_false_disconnected_graph(disconnected_graph):
    result = sorted(define_input_output(disconnected_graph, remove_disconnected=False).nodes())
    expected = ['A', 'B', 'C']
    assert result == expected


def test_define_input_output_invalid_input_type_no_fixture():
    with pytest.raises(TypeError) as exc_info:
        define_input_output("not a graph")
    result = str(exc_info.value)
    expected = "Input must be a networkx.DiGraph."
    assert result == expected


def test_define_input_output_cyclic_inputs_become_state(cyclic_inputs_graph):
    # a cycle among would-be inputs is feedback, so it classifies as state
    G = define_input_output(cyclic_inputs_graph)
    assert set(get_nodes(G, "state")) >= {"I1", "I2"}


def test_define_input_output_rejects_feedthrough(snowshoe_io_with_direct_edge):
    with pytest.raises(ValueError, match="Direct input to output edge"):
        define_input_output(snowshoe_io_with_direct_edge)


def test_define_input_output_rejects_feedback_free_chain():
    # a pure cascade has no dynamic core, so its input->output transition is
    # feedthrough and is rejected (a QMM model needs a feedback core)
    G = nx.DiGraph()
    for a, b in [('A', 'B'), ('B', 'C'), ('C', 'D')]:
        G.add_edge(a, b, sign=1)
    with pytest.raises(ValueError, match="Direct input to output edge"):
        define_input_output(G)


def test_define_input_output_rejects_non_unit_signs():
    G = nx.DiGraph()
    G.add_edge('A', 'B', sign=0.5)
    with pytest.raises(ValueError, match="Edge signs must be"):
        define_input_output(G)


def test_define_input_output_overwrites_preset_categories():
    # pre-set categories are ignored; classification is purely topological
    G = nx.DiGraph()
    G.add_edge('R', 'R', sign=-1)
    G.add_edge('Inp', 'R', sign=1)
    G.add_edge('R', 'Out', sign=1)
    G.nodes['Inp']['category'] = 'output'  # deliberately wrong; topology says input
    Gd = define_input_output(G)
    assert (Gd.nodes['Inp']['category'], Gd.nodes['R']['category'], Gd.nodes['Out']['category']) == ('input', 'state', 'output')

# =============================================================================
# cumulative_effects
# =============================================================================

def test_cumulative_effects_form_signed_snowshoe_io(snowshoe_io):
    result = cumulative_effects(snowshoe_io, form='signed')
    expected = sp.Matrix([
        [1, -1,  1, 2, -1],
        [1,  1, -1, 0,  1],
        [1,  1,  1, 0, -1],
        [0,  0,  2, 0, -2],
        [1,  1, -1, 0,  1]])
    assert result == expected


def test_cumulative_effects_form_binary_snowshoe_io(snowshoe_io):
    result = cumulative_effects(snowshoe_io, form='binary')
    expected = sp.Matrix([
        [1, 1, 1, 2, 1],
        [1, 1, 1, 2, 1],
        [1, 1, 1, 2, 1],
        [2, 2, 2, 4, 2],
        [1, 1, 1, 2, 1]])
    assert result == expected


def test_cumulative_effects_form_symbolic_snowshoe_io(snowshoe_io):
    result = cumulative_effects(snowshoe_io, form='symbolic')
    a_RR = sp.Symbol('a_R,R')
    a_RC = sp.Symbol('a_R,C')
    a_CR = sp.Symbol('a_C,R')
    a_CP = sp.Symbol('a_C,P')
    a_PC = sp.Symbol('a_P,C')
    a_PP = sp.Symbol('a_P,P')
    b_R_Inp1 = sp.Symbol('b_R,Inp1')
    b_C_Inp1 = sp.Symbol('b_C,Inp1')
    b_P_Inp2 = sp.Symbol('b_P,Inp2')
    c_Out1_C = sp.Symbol('c_Out1,C')
    c_Out1_P = sp.Symbol('c_Out1,P')
    c_Out2_C = sp.Symbol('c_Out2,C')
    expected = sp.Matrix([
        [                              a_CP*a_PC,                              -a_PP*a_RC,                               a_CP*a_RC,                                                                               a_CP*a_PC*b_R_Inp1 + a_PP*a_RC*b_C_Inp1,                                        -a_CP*a_RC*b_P_Inp2],
        [                              a_CR*a_PP,                               a_PP*a_RR,                              -a_CP*a_RR,                                                                               a_CR*a_PP*b_R_Inp1 - a_PP*a_RR*b_C_Inp1,                                         a_CP*a_RR*b_P_Inp2],
        [                              a_CR*a_PC,                               a_PC*a_RR,                               a_CR*a_RC,                                                                               a_CR*a_PC*b_R_Inp1 - a_PC*a_RR*b_C_Inp1,                                        -a_CR*a_RC*b_P_Inp2],
        [a_CR*a_PC*c_Out1_P - a_CR*a_PP*c_Out1_C, a_PC*a_RR*c_Out1_P - a_PP*a_RR*c_Out1_C, a_CP*a_RR*c_Out1_C + a_CR*a_RC*c_Out1_P, a_CR*a_PC*b_R_Inp1*c_Out1_P - a_CR*a_PP*b_R_Inp1*c_Out1_C - a_PC*a_RR*b_C_Inp1*c_Out1_P + a_PP*a_RR*b_C_Inp1*c_Out1_C, -a_CP*a_RR*b_P_Inp2*c_Out1_C - a_CR*a_RC*b_P_Inp2*c_Out1_P],
        [                     a_CR*a_PP*c_Out2_C,                      a_PP*a_RR*c_Out2_C,                     -a_CP*a_RR*c_Out2_C,                                                             a_CR*a_PP*b_R_Inp1*c_Out2_C - a_PP*a_RR*b_C_Inp1*c_Out2_C,                                a_CP*a_RR*b_P_Inp2*c_Out2_C]])
    assert result == expected


def test_cumulative_effects_form_signed_snowshoe(snowshoe):
    G = define_input_output(snowshoe)
    result = cumulative_effects(G, form='signed')
    expected = sp.Matrix([
        [1, -1,  1],
        [1,  1, -1],
        [1,  1,  1]])
    assert result == expected


def test_cumulative_effects_symbolic_matches_adjoint_snowshoe_snowshoe_io(snowshoe, snowshoe_io):
    result = cumulative_effects(snowshoe_io, form='symbolic')[:3, :3]
    expected = adjoint_matrix(snowshoe, form='symbolic')
    assert result == expected


def test_cumulative_effects_invalid_form_snowshoe_io(snowshoe_io):
    with pytest.raises(ValueError) as exc_info:
        cumulative_effects(snowshoe_io, form='invalid')
    result = str(exc_info.value)
    expected = "Invalid form. Choose 'symbolic', 'signed', 'binary'."
    assert result == expected

# =============================================================================
# absolute_effects
# =============================================================================

def test_absolute_effects_default_snowshoe_io(snowshoe_io):
    result = absolute_effects(snowshoe_io)
    expected = sp.Matrix([
        [1, 1, 1, 2, 1],
        [1, 1, 1, 2, 1],
        [1, 1, 1, 2, 1],
        [2, 2, 2, 4, 2],
        [1, 1, 1, 2, 1]])
    assert result == expected


def test_absolute_effects_match_binary_cumulative_snowshoe_io(snowshoe_io):
    result = absolute_effects(snowshoe_io)
    expected = cumulative_effects(snowshoe_io, form='binary')
    assert result == expected


def test_absolute_effects_vs_absolute_feedback_snowshoe_snowshoe_io(snowshoe, snowshoe_io):
    result = absolute_effects(snowshoe_io)[:3, :3]
    expected = absolute_feedback_matrix(snowshoe)
    assert result == expected

# =============================================================================
# weighted_effects
# =============================================================================

def test_weighted_effects_snowshoe_io(snowshoe_io):
    result = weighted_effects(snowshoe_io)
    expected = sp.Matrix([
        [1, -1,  1, 1, -1],
        [1,  1, -1, 0,  1],
        [1,  1,  1, 0, -1],
        [0,  0,  1, 0, -1],
        [1,  1, -1, 0,  1]])
    assert result == expected

def test_weighted_predictions_vs_weighted_effects(snowshoe, snowshoe_io):
    result = weighted_effects(snowshoe_io)[:3, :3]
    expected = weighted_predictions_matrix(snowshoe)
    assert result == expected

def test_weighted_effects_nan_for_missing_paths(snowshoe_io_na):
    result = weighted_effects(snowshoe_io_na)
    expected = sp.Matrix([
        [     1,     -1,      1, 1,      1,     -1],
        [     1,      1,     -1, 1,      0,      1],
        [     1,      1,      1, 1,      0,     -1],
        [sp.nan, sp.nan, sp.nan, 1, sp.nan, sp.nan],
        [     0,      0,      1, 0,      0,     -1],
        [     1,      1,     -1, 1,      0,      1]])
    assert result == expected

# =============================================================================
# sign_determinacy_effects
# =============================================================================

def test_sign_determinacy_effects_average(snowshoe_io):
    result = sign_determinacy_effects(snowshoe_io, method='average')
    expected = sp.Matrix([
        [                1,                -1,  1,                 1, -1],
        [                1,                 1, -1, sp.Rational(1, 2),  1],
        [                1,                 1,  1, sp.Rational(1, 2), -1],
        [sp.Rational(1, 2), sp.Rational(1, 2),  1, sp.Rational(1, 2), -1],
        [                1,                 1, -1, sp.Rational(1, 2),  1]])
    assert result == expected

def test_sign_determinacy_effects_95_bound(snowshoe_io):
    result = sign_determinacy_effects(snowshoe_io, method='95_bound')
    expected = sp.Matrix([
        [                1,                -1,  1,                 1, -1],
        [                1,                 1, -1, sp.Rational(1, 2),  1],
        [                1,                 1,  1, sp.Rational(1, 2), -1],
        [sp.Rational(1, 2), sp.Rational(1, 2),  1, sp.Rational(1, 2), -1],
        [                1,                 1, -1, sp.Rational(1, 2),  1]])
    assert result == expected

def test_sign_determinacy_effects_vs_matrix(snowshoe, snowshoe_io):
    result = sign_determinacy_effects(snowshoe_io, method='average')[:3, :3]
    expected = sign_determinacy_matrix(snowshoe, method='average')
    assert result == expected

def test_sign_determinacy_effects_nan_for_missing_paths(snowshoe_io_na):
    result = sign_determinacy_effects(snowshoe_io_na, method='average')
    half = sp.Rational(1, 2)
    expected = sp.Matrix([
        [     1,     -1,      1,    1,      1,     -1],
        [     1,      1,     -1,    1,   half,      1],
        [     1,      1,      1,    1,   half,     -1],
        [sp.nan, sp.nan, sp.nan,    1, sp.nan, sp.nan],
        [  half,   half,      1, half,   half,     -1],
        [     1,      1,     -1,    1,   half,      1]])
    assert result == expected

# =============================================================================
# get_simulations
# =============================================================================

def test_get_simulations(snowshoe_io):
    result = set(get_simulations(snowshoe_io, n_sim=100, seed=42).keys())
    expected = {'effects', 'valid_sims', 'all_nodes', 'tmat', 'prop_stable', 'attempts', 'n_stable'}
    assert result == expected

def test_get_simulations_effects_length(snowshoe_io):
    n_sim = 100
    result = len(get_simulations(snowshoe_io, n_sim=n_sim, seed=42)['effects'])
    expected = n_sim
    assert result == expected


def test_get_simulations_reproducibility(snowshoe_io):
    result_data = get_simulations(snowshoe_io, n_sim=100, seed=42)
    expected_data = get_simulations(snowshoe_io, n_sim=100, seed=42)
    result = {
        'effects': [effect.tolist() for effect in result_data['effects']],
        'valid_sims': result_data['valid_sims'],
        'all_nodes': result_data['all_nodes'],
        'tmat': result_data['tmat'].tolist(),
    }
    expected = {
        'effects': [effect.tolist() for effect in expected_data['effects']],
        'valid_sims': expected_data['valid_sims'],
        'all_nodes': expected_data['all_nodes'],
        'tmat': expected_data['tmat'].tolist(),
    }
    assert result == expected


@pytest.mark.parametrize("dist", ['uniform', 'weak', 'moderate', 'strong'])
def test_get_simulations_distributions(snowshoe_io, dist):
    result = len(get_simulations(snowshoe_io, n_sim=100, dist=dist, seed=42)['effects'])
    expected = 100
    assert result == expected

def test_get_simulations_uniform_two_oom(snowshoe_io):
    sims = get_simulations(snowshoe_io, n_sim=50, dist="uniform_two_oom", seed=42)
    assert len(sims["effects"]) == 50


def test_get_simulations_presample_applied_before_sampling(snowshoe):
    def presample(symbols):
        return {sp.Symbol('a_R,R'): 1}

    sims = get_simulations(snowshoe, n_sim=100, seed=42, presample=presample, return_samples=True)
    assert 'a_R,R' in sims['samples']
    assert np.all(sims['samples']['a_R,R'] == 1)


def test_get_simulations_presample_symbols_available(snowshoe_io):
    def presample(symbols):
        expected = {sp.Symbol('a_R,R'), sp.Symbol('b_R,Inp1')}
        assert set(symbols) >= expected
        return {sp.Symbol('a_R,R'): 1, sp.Symbol('b_R,Inp1'): 0.5}

    sims = get_simulations(snowshoe_io, n_sim=100, seed=42, presample=presample, return_samples=True)
    assert 'a_R,R' in sims['samples']
    assert 'b_R,Inp1' in sims['samples']
    assert np.all(sims['samples']['a_R,R'] == 1)
    assert np.all(sims['samples']['b_R,Inp1'] == 0.5)


def test_get_simulations_return_samples(snowshoe):
    sims = get_simulations(snowshoe, n_sim=100, seed=42, return_samples=True)
    assert 'samples' in sims
    assert all(len(v) == 100 for v in sims['samples'].values())
    assert 'a_R,R' in sims['samples']


def test_get_simulations_prop_stable(snowshoe):
    sims = get_simulations(snowshoe, n_sim=100, seed=42)
    assert sims['prop_stable'] == pytest.approx(1.0)


def test_get_simulations_with_perturb(snowshoe_io):
    state_nodes = get_nodes(snowshoe_io, 'state')
    perturb = (state_nodes[0], 1)
    effects = get_simulations(snowshoe_io, n_sim=100, seed=42, perturb=perturb)['effects']
    result = [effect.ndim for effect in effects]
    expected = [1] * len(effects)
    assert result == expected

def test_get_simulations_with_perturb_negative(snowshoe_io):
    state_nodes = get_nodes(snowshoe_io, 'state')
    perturb = (state_nodes[0], -1)
    result = 'effects' in get_simulations(snowshoe_io, n_sim=100, seed=42, perturb=perturb)
    expected = True
    assert result == expected

def test_get_simulations_with_observe(snowshoe_io):
    state_nodes = get_nodes(snowshoe_io, 'state')
    perturb = (state_nodes[0], 1)
    observe = ((state_nodes[1], 1),)
    result = 'valid_sims' in get_simulations(snowshoe_io, n_sim=100, seed=42, perturb=perturb, observe=observe)
    expected = True
    assert result == expected

def test_get_simulations_all_nodes_includes_all(snowshoe_io):
    sim_data = get_simulations(snowshoe_io, n_sim=100, seed=42)
    state_nodes = get_nodes(snowshoe_io, 'state')
    input_nodes = get_nodes(snowshoe_io, 'input')
    output_nodes = get_nodes(snowshoe_io, 'output')
    all_expected = state_nodes + input_nodes + output_nodes
    result = set(sim_data['all_nodes'])
    expected = set(all_expected)
    assert result == expected

def test_get_simulations_invalid_perturb_node(snowshoe_io):
    with pytest.raises(ValueError, match="Perturbation node 'InvalidNode' not found."):
        get_simulations(snowshoe_io, n_sim=100, perturb=('InvalidNode', 1))


def test_get_simulations_no_state_variables(io_only_graph):
    with pytest.raises(ValueError, match="Direct input to output edge"):
        get_simulations(io_only_graph, n_sim=50, perturb=('I', 1), seed=42)


def test_get_simulations_runtime_error_max_iterations(positive_loop_graph):
    with pytest.raises(RuntimeError) as exc_info:
        get_simulations(positive_loop_graph, n_sim=100, seed=42)
    message = str(exc_info.value)
    result = message.split(' Stable')[0]
    expected = "Maximum iterations reached."
    assert result == expected



# =============================================================================
# simulation_effects
# =============================================================================

def test_simulation_effects_full_matrix(snowshoe_io):
    result = simulation_effects(snowshoe_io, n_sim=100, seed=42)
    expected = sp.Matrix([
        [  1.0,  -1.0,  1.0,   1.0, -1.0],
        [  1.0,   1.0, -1.0,  0.52,  1.0],
        [  1.0,   1.0,  1.0,  0.52, -1.0],
        [-0.63, -0.63,  1.0, -0.53, -1.0],
        [  1.0,   1.0, -1.0,  0.52,  1.0]])
    assert result == expected


def test_simulation_effects_positive_only(snowshoe_io):
    result = simulation_effects(snowshoe_io, n_sim=100, seed=42, positive_only=True)
    expected = sp.Matrix([
        [ 1.0,  0.0, 1.0,  1.0, 0.0],
        [ 1.0,  1.0, 0.0, 0.52, 1.0],
        [ 1.0,  1.0, 1.0, 0.52, 0.0],
        [0.37, 0.37, 1.0, 0.47, 0.0],
        [ 1.0,  1.0, 0.0, 0.52, 1.0]])
    assert result == expected


def test_simulation_effects_presample_full_matrix(snowshoe_rp):
    def presample(symbols):
        return {sp.Symbol('a_P,R'): 1}

    result = simulation_effects(snowshoe_rp, n_sim=100, seed=42, presample=presample, positive_only=False)
    expected = sp.Matrix([
        [  1.0,  -1.0,  1.0],
        [-0.59,   1.0, -1.0],
        [  1.0, -0.54,  1.0]])
    assert result == expected


@pytest.mark.parametrize("dist", ['uniform', 'uniform_two_oom', 'weak', 'moderate', 'strong'])
def test_simulation_effects_distributions(snowshoe_io, dist):
    expected_mats = {
        'uniform': sp.Matrix([
            [  1.0,  -1.0,  1.0,   1.0, -1.0],
            [  1.0,   1.0, -1.0,  0.52,  1.0],
            [  1.0,   1.0,  1.0,  0.52, -1.0],
            [-0.63, -0.63,  1.0, -0.53, -1.0],
            [  1.0,   1.0, -1.0,  0.52,  1.0]]),
        'uniform_two_oom': sp.Matrix([
            [  1.0,  -1.0,  1.0,   1.0, -1.0],
            [  1.0,   1.0, -1.0,  0.52,  1.0],
            [  1.0,   1.0,  1.0,  0.52, -1.0],
            [-0.63, -0.63,  1.0, -0.53, -1.0],
            [  1.0,   1.0, -1.0,  0.52,  1.0]]),
        'weak': sp.Matrix([
            [  1.0,  -1.0,  1.0,   1.0, -1.0],
            [  1.0,   1.0, -1.0,  0.52,  1.0],
            [  1.0,   1.0,  1.0,  0.52, -1.0],
            [-0.57, -0.57,  1.0, -0.63, -1.0],
            [  1.0,   1.0, -1.0,  0.52,  1.0]]),
        'moderate': sp.Matrix([
            [  1.0,  -1.0,  1.0,  1.0, -1.0],
            [  1.0,   1.0, -1.0, 0.57,  1.0],
            [  1.0,   1.0,  1.0, 0.57, -1.0],
            [-0.53, -0.53,  1.0,  0.5, -1.0],
            [  1.0,   1.0, -1.0, 0.57,  1.0]]),
        'strong': sp.Matrix([
            [  1.0,  -1.0,  1.0,   1.0, -1.0],
            [  1.0,   1.0, -1.0, -0.51,  1.0],
            [  1.0,   1.0,  1.0, -0.51, -1.0],
            [  0.5,   0.5,  1.0, -0.53, -1.0],
            [  1.0,   1.0, -1.0, -0.51,  1.0]]),
    }
    result = simulation_effects(snowshoe_io, n_sim=100, dist=dist, seed=42)
    expected = expected_mats[dist]
    assert result == expected

def test_net_effects_returns_signed_cumulative(snowshoe_io):
    result = net_effects(snowshoe_io)
    expected = cumulative_effects(snowshoe_io, form='signed')
    assert result == expected


def test_net_effects_vs_adjoint_signed(snowshoe, snowshoe_io):
    result = cumulative_effects(snowshoe_io, form='signed')[:3, :3]
    expected = adjoint_matrix(snowshoe, form='signed')
    assert result == expected

def test_simulation_effects_vs_numerical_simulations(snowshoe, snowshoe_io):
    seed = 42
    n_sim = 100
    result = simulation_effects(snowshoe_io, n_sim=n_sim, seed=seed)[:3, :3]
    expected = numerical_simulations(snowshoe, n_sim=n_sim, seed=seed)
    assert result == expected


def test_simulation_effects_nan_for_no_path(snowshoe_io_na):
    result = simulation_effects(snowshoe_io_na, n_sim=100, seed=42)
    expected = sp.Matrix([
        [   1.0,   -1.0,    1.0,   1.0,    1.0,   -1.0],
        [   1.0,    1.0,   -1.0,   1.0,  -0.51,    1.0],
        [   1.0,    1.0,    1.0,   1.0,  -0.51,   -1.0],
        [sp.nan, sp.nan, sp.nan,   1.0, sp.nan, sp.nan],
        [ -0.54,  -0.54,    1.0, -0.54,  -0.57,   -1.0],
        [   1.0,    1.0,   -1.0,   1.0,  -0.51,    1.0]])
    assert result == expected


def test_simulation_effects_positive_only_nan_for_no_path(snowshoe_io_na):
    result = simulation_effects(snowshoe_io_na, n_sim=100, seed=42, positive_only=True)
    expected = sp.Matrix([
        [   1.0,    0.0,    1.0,  1.0,    1.0,    0.0],
        [   1.0,    1.0,    0.0,  1.0,   0.49,    1.0],
        [   1.0,    1.0,    1.0,  1.0,   0.49,    0.0],
        [sp.nan, sp.nan, sp.nan,  1.0, sp.nan, sp.nan],
        [  0.46,   0.46,    1.0, 0.46,   0.43,    0.0],
        [   1.0,    1.0,    0.0,  1.0,   0.49,    1.0]])
    assert result == expected


# =============================================================================
# simulations_table
# =============================================================================

def test_simulations_table_no_response_nodes():
    # all-input graph (hand-built, bypassing define_input_output's layer checks)
    # has no state/output response nodes -> empty table from the defensive guard
    G = nx.DiGraph()
    G.add_node('A', category='input')
    G.add_node('B', category='input')
    G.add_edge('A', 'B', sign=1)
    nx.freeze(G)
    result = simulations_table(G, perturb="A:+", n_sim=5, seed=42)
    expected_columns = [
        "model",
        "effect_on",
        "negative",
        "no_effect",
        "positive",
        "valid_sims",
        "stable_sims",
        "attempts",
    ]
    assert result.columns.tolist() == expected_columns
    assert result.empty


def test_simulations_table_no_valid_sims(snowshoe_io):
    result = simulations_table(snowshoe_io, perturb="P:+", observe="C:0", n_sim=50, seed=42)
    assert result["valid_sims"].eq(0).all()
    assert result["negative"].eq(0).all()
    assert result["positive"].eq(0).all()


def test_simulations_table_counts_match_structure(snowshoe_io_na):
    result = simulations_table(snowshoe_io_na, perturb="P:+", n_sim=100, seed=42)
    expected_columns = [
        "model",
        "effect_on",
        "negative",
        "no_effect",
        "positive",
        "valid_sims",
        "stable_sims",
        "attempts",
    ]
    assert result.columns.tolist() == expected_columns
    assert (result["model"] == 1).all()

    tmat = sp.matrix2numpy(absolute_effects(snowshoe_io_na)).astype(int)
    response_nodes = get_nodes(snowshoe_io_na, "state") + get_nodes(snowshoe_io_na, "output")
    perturb_nodes = get_nodes(snowshoe_io_na, "state") + get_nodes(snowshoe_io_na, "input")
    p_idx = perturb_nodes.index("P")

    for _, row in result.iterrows():
        node_idx = response_nodes.index(row["effect_on"])
        has_effect = tmat[node_idx, p_idx] != 0
        if has_effect:
            assert row["no_effect"] == 0
            assert row["negative"] + row["positive"] == row["valid_sims"]
        else:
            assert row["no_effect"] == row["valid_sims"]
            assert row["negative"] == 0
            assert row["positive"] == 0


def test_simulations_table_importable():
    from qmm.extensions import simulations_table as from_extensions
    assert simulations_table is from_extensions


# =============================================================================
# Additional coverage tests
# =============================================================================

def test_cumulative_effects_binary_form(snowshoe_io):
    result = cumulative_effects(snowshoe_io, form="binary")
    assert result.shape[0] > 0
    assert result.shape[1] > 0


def test_simulations_table_with_observe(snowshoe):
    result = simulations_table(snowshoe, perturb='R:+', observe='C:+', n_sim=100, seed=42)
    assert result is not None


# =============================================================================
# direct_effects() and table_of_direct_effects()
# =============================================================================

def test_direct_effects_net_form(snowshoe_io):
    result = direct_effects(snowshoe_io, form="net")
    assert isinstance(result, sp.MatrixBase)
    assert result.shape == (5, 5)


def test_direct_effects_absolute_form(snowshoe_io):
    result = direct_effects(snowshoe_io, form="absolute")
    assert isinstance(result, sp.MatrixBase)
    assert result.shape == (5, 5)


def test_direct_effects_positive_form(snowshoe_io):
    result = direct_effects(snowshoe_io, form="positive")
    assert isinstance(result, sp.MatrixBase)
    assert result.shape == (5, 5)


def test_direct_effects_negative_form(snowshoe_io):
    result = direct_effects(snowshoe_io, form="negative")
    assert isinstance(result, sp.MatrixBase)
    assert result.shape == (5, 5)


def test_direct_effects_invalid_form(snowshoe_io):
    with pytest.raises(ValueError, match="Invalid form"):
        direct_effects(snowshoe_io, form="invalid")


def test_table_of_direct_effects(snowshoe_io):
    result = table_of_direct_effects(snowshoe_io)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (5, 5)


# =============================================================================
# table_of_effects()
# =============================================================================

def test_table_of_effects_with_string_net_effects(snowshoe_io):
    result = table_of_effects(snowshoe_io, generator="net_effects")
    assert isinstance(result, pd.DataFrame)


def test_table_of_effects_with_string_absolute_effects(snowshoe_io):
    result = table_of_effects(snowshoe_io, generator="absolute_effects")
    assert isinstance(result, pd.DataFrame)


def test_table_of_effects_with_string_weighted_effects(snowshoe_io):
    result = table_of_effects(snowshoe_io, generator="weighted_effects")
    assert isinstance(result, pd.DataFrame)


def test_table_of_effects_with_string_sign_determinacy_effects(snowshoe_io):
    result = table_of_effects(snowshoe_io, generator="sign_determinacy_effects")
    assert isinstance(result, pd.DataFrame)


def test_table_of_effects_with_string_simulation_effects(snowshoe_io):
    result = table_of_effects(snowshoe_io, generator="simulation_effects")
    assert isinstance(result, pd.DataFrame)


def test_table_of_effects_invalid_string_generator(snowshoe_io):
    with pytest.raises(ValueError, match="Generator must be callable"):
        table_of_effects(snowshoe_io, generator="invalid_generator")


def test_table_of_effects_with_lambda_no_name(snowshoe_io):
    result = table_of_effects(snowshoe_io, generator=lambda G: net_effects(G))
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (5, 5)


def test_table_of_effects_forwards_kwargs(snowshoe_io):
    default = table_of_effects(snowshoe_io, simulation_effects, n_sim=200, seed=42)
    positive = table_of_effects(snowshoe_io, simulation_effects, n_sim=200, seed=42, positive_only=True)
    assert isinstance(positive, pd.DataFrame)
    assert not default.equals(positive)


def test_table_of_effects_decimals_rounds(snowshoe_io):
    full = table_of_effects(snowshoe_io, simulation_effects, n_sim=200, seed=42)
    rounded = table_of_effects(snowshoe_io, simulation_effects, n_sim=200, seed=42, decimals=2)
    assert isinstance(rounded, pd.DataFrame)
    assert not full.equals(rounded)


def test_simulation_effects_handles_singular_matrices(snowshoe_io):
    from unittest.mock import patch
    original_inv = np.linalg.inv
    call_count = [0]

    def selective_inv(A):
        call_count[0] += 1
        if call_count[0] % 3 == 0:
            raise np.linalg.LinAlgError("Singular matrix")
        return original_inv(A)

    with patch('numpy.linalg.inv', side_effect=selective_inv):
        result = simulation_effects(snowshoe_io, n_sim=5, seed=42)
        assert isinstance(result, sp.Matrix)
        assert call_count[0] > 5
