"""Tests for qmm.extensions.effects module."""

import pytest
import sympy as sp
import numpy as np
from unittest.mock import patch

from qmm import define_input_output, get_nodes
from qmm.extensions.effects import (
    cumulative_effects,
    absolute_effects,
    weighted_effects,
    sign_determinacy_effects,
    get_simulations,
    simulation_effects,
    _observation_matches,
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
    """Test define_input_output categorizes snowshoe_io nodes into state, input, and output."""
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
    """Test define_input_output removes disconnected components when requested."""
    result = sorted(define_input_output(disconnected_graph, remove_disconnected=True).nodes())
    expected = ['A', 'B']
    assert result == expected


def test_define_input_output_remove_disconnected_false_disconnected_graph(disconnected_graph):
    """Test define_input_output keeps disconnected components when requested."""
    result = sorted(define_input_output(disconnected_graph, remove_disconnected=False).nodes())
    expected = ['A', 'B', 'C']
    assert result == expected


def test_define_input_output_invalid_input_type_no_fixture():
    """Test define_input_output raises TypeError for non-graph input."""
    with pytest.raises(TypeError) as exc_info:
        define_input_output("not a graph")
    result = str(exc_info.value)
    expected = "Input must be a networkx.DiGraph."
    assert result == expected

# =============================================================================
# cumulative_effects
# =============================================================================

def test_cumulative_effects_form_signed_snowshoe_io(snowshoe_io):
    """Test cumulative_effects with signed form for snowshoe_io."""
    result = cumulative_effects(snowshoe_io, form='signed')
    expected = sp.Matrix([
        [1, -1, 1, 2, -1],
        [1, 1, -1, 0, 1],
        [1, 1, 1, 0, -1],
        [0, 0, 2, 0, -2],
        [1, 1, -1, 0, 1]
    ])
    assert result == expected


def test_cumulative_effects_form_binary_snowshoe_io(snowshoe_io):
    """Test cumulative_effects with binary form for snowshoe_io."""
    result = cumulative_effects(snowshoe_io, form='binary')
    expected = sp.Matrix([
        [1, 1, 1, 2, 1],
        [1, 1, 1, 2, 1],
        [1, 1, 1, 2, 1],
        [2, 2, 2, 4, 2],
        [1, 1, 1, 2, 1]
    ])
    assert result == expected


def test_cumulative_effects_form_symbolic_snowshoe_io(snowshoe_io):
    """Test cumulative_effects with symbolic form for snowshoe_io."""
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
        [a_CP*a_PC, -a_PP*a_RC, a_CP*a_RC, a_CP*a_PC*b_R_Inp1 + a_PP*a_RC*b_C_Inp1, -a_CP*a_RC*b_P_Inp2],
        [a_CR*a_PP, a_PP*a_RR, -a_CP*a_RR, a_CR*a_PP*b_R_Inp1 - a_PP*a_RR*b_C_Inp1, a_CP*a_RR*b_P_Inp2],
        [a_CR*a_PC, a_PC*a_RR, a_CR*a_RC, a_CR*a_PC*b_R_Inp1 - a_PC*a_RR*b_C_Inp1, -a_CR*a_RC*b_P_Inp2],
        [a_CR*a_PC*c_Out1_P - a_CR*a_PP*c_Out1_C, a_PC*a_RR*c_Out1_P - a_PP*a_RR*c_Out1_C, a_CP*a_RR*c_Out1_C + a_CR*a_RC*c_Out1_P, a_CR*a_PC*b_R_Inp1*c_Out1_P - a_CR*a_PP*b_R_Inp1*c_Out1_C - a_PC*a_RR*b_C_Inp1*c_Out1_P + a_PP*a_RR*b_C_Inp1*c_Out1_C, -a_CP*a_RR*b_P_Inp2*c_Out1_C - a_CR*a_RC*b_P_Inp2*c_Out1_P],
        [a_CR*a_PP*c_Out2_C, a_PP*a_RR*c_Out2_C, -a_CP*a_RR*c_Out2_C, a_CR*a_PP*b_R_Inp1*c_Out2_C - a_PP*a_RR*b_C_Inp1*c_Out2_C, a_CP*a_RR*b_P_Inp2*c_Out2_C]
    ])
    assert result == expected


def test_cumulative_effects_form_signed_snowshoe(snowshoe):
    """Test cumulative_effects with signed form on snowshoe without IO nodes."""
    G = define_input_output(snowshoe)
    result = cumulative_effects(G, form='signed')
    expected = sp.Matrix([
        [1, -1, 1],
        [1, 1, -1],
        [1, 1, 1]
    ])
    assert result == expected


def test_cumulative_effects_symbolic_matches_adjoint_snowshoe_snowshoe_io(snowshoe, snowshoe_io):
    """Test cumulative_effects symbolic form matches adjoint_matrix symbolic output."""
    result = cumulative_effects(snowshoe_io, form='symbolic')[:3, :3]
    expected = adjoint_matrix(snowshoe, form='symbolic')
    assert result == expected


def test_cumulative_effects_invalid_form_snowshoe_io(snowshoe_io):
    """Test cumulative_effects raises ValueError for invalid form argument."""
    with pytest.raises(ValueError) as exc_info:
        cumulative_effects(snowshoe_io, form='invalid')
    result = str(exc_info.value)
    expected = "Invalid form. Choose 'symbolic', 'signed', 'binary'."
    assert result == expected

# =============================================================================
# absolute_effects
# =============================================================================

def test_absolute_effects_default_snowshoe_io(snowshoe_io):
    """Test absolute_effects default computation for snowshoe_io."""
    result = absolute_effects(snowshoe_io)
    expected = sp.Matrix([
        [1, 1, 1, 2, 1],
        [1, 1, 1, 2, 1],
        [1, 1, 1, 2, 1],
        [2, 2, 2, 4, 2],
        [1, 1, 1, 2, 1]
    ])
    assert result == expected


def test_absolute_effects_match_binary_cumulative_snowshoe_io(snowshoe_io):
    """Test absolute_effects output matches binary cumulative_effects for snowshoe_io."""
    result = absolute_effects(snowshoe_io)
    expected = cumulative_effects(snowshoe_io, form='binary')
    assert result == expected


def test_absolute_effects_vs_absolute_feedback_snowshoe_snowshoe_io(snowshoe, snowshoe_io):
    """Test absolute_effects for state nodes equals absolute_feedback_matrix for snowshoe."""
    result = absolute_effects(snowshoe_io)[:3, :3]
    expected = absolute_feedback_matrix(snowshoe)
    assert result == expected

# =============================================================================
# weighted_effects
# =============================================================================

def test_weighted_effects_snowshoe_io(snowshoe_io):
    """Test calculation of weighted effects."""
    result = weighted_effects(snowshoe_io)
    expected = sp.Matrix([
        [1, -1, 1, 1, -1],
        [1, 1, -1, 0, 1],
        [1, 1, 1, 0, -1],
        [0, 0, 1, 0, -1],
        [1, 1, -1, 0, 1]
    ])
    assert result == expected

def test_weighted_predictions_vs_weighted_effects(snowshoe, snowshoe_io):
    """Compare weighted_effects with weighted_predictions_matrix."""
    result = weighted_effects(snowshoe_io)[:3, :3]
    expected = weighted_predictions_matrix(snowshoe)
    assert result == expected

def test_weighted_effects_nan_for_missing_paths(snowshoe_io_na):
    """Weighted effects should be NaN where no paths reach N."""
    result = weighted_effects(snowshoe_io_na)
    expected = sp.Matrix([
        [1, -1, 1, 1, 1, -1],
        [1, 1, -1, 1, 0, 1],
        [1, 1, 1, 1, 0, -1],
        [sp.nan, sp.nan, sp.nan, 1, sp.nan, sp.nan],
        [0, 0, 1, 0, 0, -1],
        [1, 1, -1, 1, 0, 1],
    ])
    assert result == expected

# =============================================================================
# sign_determinacy_effects
# =============================================================================

def test_sign_determinacy_effects_average(snowshoe_io):
    """Test sign determinacy using average method."""
    result = sign_determinacy_effects(snowshoe_io, method='average')
    expected = sp.Matrix([
        [1, -1, 1, 1, -1],
        [1, 1, -1, sp.Rational(1, 2), 1],
        [1, 1, 1, sp.Rational(1, 2), -1],
        [sp.Rational(1, 2), sp.Rational(1, 2), 1, sp.Rational(1, 2), -1],
        [1, 1, -1, sp.Rational(1, 2), 1]
    ])
    assert result == expected

def test_sign_determinacy_effects_95_bound(snowshoe_io):
    """Test sign determinacy using 95% bound method."""
    result = sign_determinacy_effects(snowshoe_io, method='95_bound')
    expected = sp.Matrix([
        [1, -1, 1, 1, -1],
        [1, 1, -1, sp.Rational(1, 2), 1],
        [1, 1, 1, sp.Rational(1, 2), -1],
        [sp.Rational(1, 2), sp.Rational(1, 2), 1, sp.Rational(1, 2), -1],
        [1, 1, -1, sp.Rational(1, 2), 1]
    ])
    assert result == expected

def test_sign_determinacy_effects_vs_matrix(snowshoe, snowshoe_io):
    """Compare sign_determinacy_effects with sign_determinacy_matrix."""
    result = sign_determinacy_effects(snowshoe_io, method='average')[:3, :3]
    expected = sign_determinacy_matrix(snowshoe, method='average')
    assert result == expected

def test_sign_determinacy_effects_nan_for_missing_paths(snowshoe_io_na):
    """Sign determinacy should return NaN for unreachable nodes."""
    result = sign_determinacy_effects(snowshoe_io_na, method='average')
    half = sp.Rational(1, 2)
    expected = sp.Matrix([
        [1, -1, 1, 1, 1, -1],
        [1, 1, -1, 1, half, 1],
        [1, 1, 1, 1, half, -1],
        [sp.nan, sp.nan, sp.nan, 1, sp.nan, sp.nan],
        [half, half, 1, half, half, -1],
        [1, 1, -1, 1, half, 1],
    ])
    assert result == expected

# =============================================================================
# get_simulations
# =============================================================================

def test_get_simulations(snowshoe_io):
    """Test basic functionality of get_simulations."""
    result = set(get_simulations(snowshoe_io, n_sim=100, seed=42).keys())
    expected = {'effects', 'valid_sims', 'all_nodes', 'tmat'}
    assert result == expected

def test_get_simulations_effects_length(snowshoe_io):
    """Test that correct number of simulations are returned."""
    n_sim = 100
    result = len(get_simulations(snowshoe_io, n_sim=n_sim, seed=42)['effects'])
    expected = n_sim
    assert result == expected


def test_get_simulations_reproducibility(snowshoe_io):
    """Test that simulations are reproducible with same seed."""
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


@pytest.mark.parametrize("dist", ['uniform', 'weak', 'moderate', 'strong',
                                   'normal_weak', 'normal_moderate', 'normal_strong'])
def test_get_simulations_distributions(snowshoe_io, dist):
    """Test simulations with different distributions."""
    result = len(get_simulations(snowshoe_io, n_sim=100, dist=dist, seed=42)['effects'])
    expected = 100
    assert result == expected


def test_get_simulations_with_perturb(snowshoe_io):
    """Test simulations with specific perturbation."""
    state_nodes = get_nodes(snowshoe_io, 'state')
    perturb = (state_nodes[0], 1)
    effects = get_simulations(snowshoe_io, n_sim=100, seed=42, perturb=perturb)['effects']
    result = [effect.ndim for effect in effects]
    expected = [1] * len(effects)
    assert result == expected

def test_get_simulations_with_perturb_negative(snowshoe_io):
    """Test simulations with negative perturbation."""
    state_nodes = get_nodes(snowshoe_io, 'state')
    perturb = (state_nodes[0], -1)
    result = 'effects' in get_simulations(snowshoe_io, n_sim=100, seed=42, perturb=perturb)
    expected = True
    assert result == expected

def test_get_simulations_with_observe(snowshoe_io):
    """Test simulations with observations."""
    state_nodes = get_nodes(snowshoe_io, 'state')
    perturb = (state_nodes[0], 1)
    observe = ((state_nodes[1], 1),)
    result = 'valid_sims' in get_simulations(snowshoe_io, n_sim=100, seed=42, perturb=perturb, observe=observe)
    expected = True
    assert result == expected

def test_get_simulations_all_nodes_includes_all(snowshoe_io):
    """Test that all_nodes includes state, input, and output nodes."""
    sim_data = get_simulations(snowshoe_io, n_sim=100, seed=42)
    state_nodes = get_nodes(snowshoe_io, 'state')
    input_nodes = get_nodes(snowshoe_io, 'input')
    output_nodes = get_nodes(snowshoe_io, 'output')
    all_expected = state_nodes + input_nodes + output_nodes
    result = set(sim_data['all_nodes'])
    expected = set(all_expected)
    assert result == expected

def test_get_simulations_no_state_variables(io_only_graph):
    """Test get_simulations rejects direct input to output edges."""
    with pytest.raises(ValueError) as exc_info:
        get_simulations(io_only_graph, n_sim=100, perturb=('I', 1))
    assert "Direct input to output edge" in str(exc_info.value)

def test_get_simulations_linalg_error(positive_loop_graph):
    """Test that LinAlgError is caught and simulation continues."""
    with patch('numpy.linalg.eigvals', return_value=np.array([-1.0])):
        side_effect = [np.linalg.LinAlgError("Singular matrix")] * 5 + [np.array([[1.0]])] * 100
        with patch('numpy.linalg.inv', side_effect=side_effect):
            result = len(get_simulations(positive_loop_graph, n_sim=100, seed=42)['effects'])
            expected = 100
            assert result == expected


def test_get_simulations_runtime_error_max_iterations(positive_loop_graph):
    """Test that RuntimeError is raised when max iterations are reached."""
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
    """Test simulation_effects with full matrix assertion."""
    result = simulation_effects(snowshoe_io, n_sim=100, seed=42)
    expected = sp.Matrix([
        [1.0, -1.0, 1.0, 1.0, -1.0],
        [1.0, 1.0, -1.0, 0.57, 1.0],
        [1.0, 1.0, 1.0, 0.57, -1.0],
        [-0.54, -0.54, 1.0, 0.51, -1.0],
        [1.0, 1.0, -1.0, 0.57, 1.0],
    ])
    assert result == expected


def test_simulation_effects_positive_only(snowshoe_io):
    """Test simulation effects with positive_only=True."""
    result = simulation_effects(snowshoe_io, n_sim=100, seed=42, positive_only=True)
    expected = sp.Matrix([
        [1.0, 0.0, 1.0, 1.0, 0.0],
        [1.0, 1.0, 0.0, 0.57, 1.0],
        [1.0, 1.0, 1.0, 0.57, 0.0],
        [0.46, 0.46, 1.0, 0.51, 0.0],
        [1.0, 1.0, 0.0, 0.57, 1.0],
    ])
    assert result == expected


@pytest.mark.parametrize("dist", ['uniform', 'weak', 'moderate', 'strong'])
def test_simulation_effects_distributions(snowshoe_io, dist):
    """Test simulation effects with different distributions."""
    expected_mats = {
        'uniform': sp.Matrix([
            [1.0, -1.0, 1.0, 1.0, -1.0],
            [1.0, 1.0, -1.0, 0.57, 1.0],
            [1.0, 1.0, 1.0, 0.57, -1.0],
            [-0.54, -0.54, 1.0, 0.51, -1.0],
            [1.0, 1.0, -1.0, 0.57, 1.0],
        ]),
        'weak': sp.Matrix([
            [1.0, -1.0, 1.0, 1.0, -1.0],
            [1.0, 1.0, -1.0, 0.58, 1.0],
            [1.0, 1.0, 1.0, 0.58, -1.0],
            [-0.51, -0.51, 1.0, 0.57, -1.0],
            [1.0, 1.0, -1.0, 0.58, 1.0],
        ]),
        'moderate': sp.Matrix([
            [1.0, -1.0, 1.0, 1.0, -1.0],
            [1.0, 1.0, -1.0, 0.51, 1.0],
            [1.0, 1.0, 1.0, 0.51, -1.0],
            [-0.52, -0.52, 1.0, 0.53, -1.0],
            [1.0, 1.0, -1.0, 0.51, 1.0],
        ]),
        'strong': sp.Matrix([
            [1.0, -1.0, 1.0, 1.0, -1.0],
            [1.0, 1.0, -1.0, -0.54, 1.0],
            [1.0, 1.0, 1.0, -0.54, -1.0],
            [-0.56, -0.56, 1.0, 0.52, -1.0],
            [1.0, 1.0, -1.0, -0.54, 1.0],
        ]),
    }
    result = simulation_effects(snowshoe_io, n_sim=100, dist=dist, seed=42)
    expected = expected_mats[dist]
    assert result == expected

def test_net_effects_vs_adjoint_signed(snowshoe, snowshoe_io):
    """Compare cumulative_effects(signed) with adjoint_matrix(signed)."""
    result = cumulative_effects(snowshoe_io, form='signed')[:3, :3]
    expected = adjoint_matrix(snowshoe, form='signed')
    assert result == expected

def test_simulation_effects_vs_numerical_simulations(snowshoe, snowshoe_io):
    """Compare simulation_effects with numerical_simulations."""
    seed = 42
    n_sim = 100
    result = simulation_effects(snowshoe_io, n_sim=n_sim, seed=seed)[:3, :3]
    expected = numerical_simulations(snowshoe, n_sim=n_sim, seed=seed)
    assert result == expected


def test_simulation_effects_nan_for_no_path(snowshoe_io_na):
    """Test that NaN is returned when there is no path (zero absolute effect)."""
    result = simulation_effects(snowshoe_io_na, n_sim=100, seed=42)
    expected = sp.Matrix([
        [1.0, -1.0, 1.0, 1.0, 1.0, -1.0],
        [1.0, 1.0, -1.0, 1.0, 0.56, 1.0],
        [1.0, 1.0, 1.0, 1.0, 0.56, -1.0],
        [sp.nan, sp.nan, sp.nan, 1.0, sp.nan, sp.nan],
        [0.52, 0.52, 1.0, 0.52, 0.50, -1.0],
        [1.0, 1.0, -1.0, 1.0, 0.56, 1.0],
    ])
    assert result == expected


def test_simulation_effects_positive_only_nan_for_no_path(snowshoe_io_na):
    """Positive-only simulation effects should still return NaN for unreachable nodes."""
    result = simulation_effects(snowshoe_io_na, n_sim=100, seed=42, positive_only=True)
    expected = sp.Matrix([
        [1.0, 0.0, 1.0, 1.0, 1.0, 0.0],
        [1.0, 1.0, 0.0, 1.0, 0.56, 1.0],
        [1.0, 1.0, 1.0, 1.0, 0.56, 0.0],
        [sp.nan, sp.nan, sp.nan, 1.0, sp.nan, sp.nan],
        [0.52, 0.52, 1.0, 0.52, 0.50, 0.0],
        [1.0, 1.0, 0.0, 1.0, 0.56, 1.0],
    ])
    assert result == expected


# =============================================================================
# _observation_matches()
# =============================================================================

def test_observation_matches_no_expected_effect_no_fixture():
    """Test _observation_matches returns True when no effect expected and obs is 0."""
    result = _observation_matches(obs=0, effect_val=0.5, effect_expected=False)
    expected = True
    assert result == expected


def test_observation_matches_no_expected_effect_nonzero_obs_no_fixture():
    """Test _observation_matches returns False when no effect expected but obs is nonzero."""
    result = _observation_matches(obs=1, effect_val=0.5, effect_expected=False)
    expected = False
    assert result == expected
