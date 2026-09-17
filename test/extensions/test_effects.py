"""Tests for qmm.extensions.effects module."""

from qmm.extensions import simulations_table as from_extensions
from unittest.mock import patch
from qmm.extensions.validation import marginal_likelihood

import pytest
import sympy as sp
import numpy as np
import pandas as pd
import networkx as nx

from qmm.core.helper import get_nodes
from qmm.core.structure import create_matrix
from qmm.extensions.effects import define_input_output
from qmm.extensions.effects import (
    cumulative_effects,
    net_effects,
    absolute_effects,
    weighted_effects,
    sign_determinacy_effects,
    get_simulations,
    iter_simulations,
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


def _scaled_equilibrium_derivative(G):
    states, inputs, outputs = (get_nodes(G, kind) for kind in ('state', 'input', 'output'))
    A, B, C, D = (create_matrix(G, 'symbolic', block) for block in 'ABCD')
    x = sp.Matrix(sp.symbols(' '.join(f'x_{n}' for n in states), seq=True))
    p = sp.Matrix(sp.symbols(' '.join(f'p_{n}' for n in states), seq=True))
    u = sp.Matrix(sp.symbols(' '.join(f'u_{n}' for n in inputs), seq=True))
    equilibrium = sp.solve(list(A * x + B * u + p), list(x))
    derivative = sp.Matrix.vstack(x, C * x + D * u).subs(equilibrium).jacobian(sp.Matrix.vstack(p, u))
    return ((-A).det() * derivative).applyfunc(sp.cancel).applyfunc(sp.expand)


def test_cumulative_effects_equals_scaled_equilibrium_derivative_snowshoe_io(snowshoe_io):
    result = cumulative_effects(snowshoe_io)
    expected = _scaled_equilibrium_derivative(snowshoe_io)
    assert result == expected


def test_absolute_effects_count_numerator_terms_io_chain(io_chain):
    numerator = _scaled_equilibrium_derivative(io_chain)
    result = absolute_effects(io_chain)
    expected = numerator.applyfunc(lambda e: len(e.as_ordered_terms()) if e else 0)
    assert result == expected


def test_cumulative_effects_invalid_form_snowshoe_io(snowshoe_io):
    with pytest.raises(ValueError) as exc_info:
        cumulative_effects(snowshoe_io, form='invalid')
    result = str(exc_info.value)
    expected = "Invalid form. Choose 'symbolic', 'signed', 'binary'."
    assert result == expected


@pytest.mark.parametrize('form', ['symbolic', 'signed', 'binary'])
def test_cumulative_effects_rejects_direct_io_edge(snowshoe_io_with_direct_edge, form):
    with pytest.raises(ValueError, match="Direct input to output edge"):
        cumulative_effects(snowshoe_io_with_direct_edge, form=form)

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
    expected = {'effects', 'valid_sims', 'all_nodes', 'tmat', 'prop_stable', 'attempts', 'n_stable', 'structures', 'perturb', 'interactions', 'signs'}
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


@pytest.mark.parametrize('dashed', [False, True])
@pytest.mark.parametrize('strength', [sp.Integer(1), sp.Symbol('a_R,R') / 2 + sp.Rational(1, 4)])
def test_get_simulations_presample_applied_before_sampling(snowshoe, dashed, strength):
    G = nx.DiGraph(snowshoe)
    G['R']['R']['dashes'] = dashed
    a_rr = sp.Symbol('a_R,R')
    sims = get_simulations(G, n_sim=10, seed=42, return_samples=True,
                           presample=lambda symbols: {a_rr: strength})
    baseline = get_simulations(G, n_sim=10, seed=42, return_samples=True)
    raw = baseline['samples']['a_R,R']
    expected = np.where(raw == 0, 0, sp.lambdify(a_rr, strength)(raw))
    result = sims['samples']['a_R,R']
    assert np.array_equal(result, expected)
    matrix = create_matrix(G, form='symbolic')
    for i, effect in enumerate(sims['effects']):
        samples = {sp.Symbol(name): values[i] for name, values in sims['samples'].items()}
        expected = np.linalg.inv(-np.asarray(matrix.subs(samples), dtype=float))
        result = effect
        assert np.allclose(result, expected)
    present = sims['samples']['a_R,R'][sims['samples']['a_R,R'] != 0]
    assert np.all((present >= 0.25) & (present <= 1))


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

@pytest.mark.parametrize('presses', [
    (('R', 1), ('C', -1)),
    (('Inp1', 1), ('Inp2', -1)),
    (('R', 1), ('Inp2', 1)),
])
def test_simultaneous_presses_match_direct_solve(snowshoe_io, presses):
    sims = get_simulations(snowshoe_io, n_sim=10, perturb=presses, return_samples=True)
    baseline = get_simulations(snowshoe_io, n_sim=10, return_samples=True)
    assert sims['attempts'] == baseline['attempts']
    assert sims['samples'].keys() == baseline['samples'].keys()
    for symbol, values in sims['samples'].items():
        np.testing.assert_array_equal(values, baseline['samples'][symbol])

    state, inputs = (get_nodes(snowshoe_io, category) for category in ('state', 'input'))
    press = dict(presses)
    p_x = np.array([press.get(node, 0) for node in state])
    p_u = np.array([press.get(node, 0) for node in inputs])
    matrices = [create_matrix(snowshoe_io, form='symbolic', matrix_type=m) for m in 'ABC']
    for i, effect in enumerate(sims['effects']):
        subs = {sp.Symbol(symbol): values[i] for symbol, values in sims['samples'].items()}
        A, B, C = [np.array(matrix.subs(subs), dtype=float) for matrix in matrices]
        x = np.linalg.solve(-A, p_x + B @ p_u)
        np.testing.assert_allclose(effect, np.concatenate([x, C @ x]))


def test_single_press_tuple_remains_compatible(snowshoe_io):
    old = get_simulations(snowshoe_io, n_sim=10, perturb=('Inp1', -1))
    nested = get_simulations(snowshoe_io, n_sim=10, perturb=(('Inp1', -1),))
    np.testing.assert_array_equal(old['effects'], nested['effects'])


def test_get_simulations_with_observe(snowshoe_io):
    sims = get_simulations(snowshoe_io, n_sim=100, seed=42,
                           perturb=('Inp1', 1), observe=(('Out1', 1),))
    result = (sum(sims['valid_sims']), sims['n_stable'] > 100)
    expected = (100, True)
    assert result == expected
    sims = get_simulations(snowshoe_io, n_sim=100, seed=42,
                           perturb=('Inp1', 1), observe=(('Out1', 1),), condition=False)
    result = (sims['n_stable'], 0 < sum(sims['valid_sims']) < 100)
    expected = (100, True)
    assert result == expected
    sims = get_simulations(snowshoe_io, n_sim=100, seed=42,
                           perturb=('Inp1', 1), observe=(('Out1', 0),), condition=False)
    assert sims['n_stable'] == 100
    assert not any(sims['valid_sims'])


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


@pytest.mark.parametrize('max_attempts', [None, 1])
def test_get_simulations_runtime_error_max_iterations(positive_loop_graph, max_attempts):
    with pytest.raises(RuntimeError) as exc_info:
        get_simulations(positive_loop_graph, n_sim=100, seed=42, max_attempts=max_attempts)
    assert str(exc_info.value).startswith('Maximum iterations reached.')



def test_get_simulations_presampled_draw_matches_response_blocks(snowshoe_io, snowshoe_io_strengths):
    A, B, C, D = (np.array(create_matrix(snowshoe_io, "symbolic", m).subs(snowshoe_io_strengths), dtype=float) for m in "ABCD")
    inverse = np.linalg.inv(-A)
    expected = np.block([[inverse, inverse @ B], [C @ inverse, C @ inverse @ B + D]])
    result = get_simulations(snowshoe_io, n_sim=1, presample=lambda symbols: snowshoe_io_strengths)["effects"][0]
    assert np.allclose(result, expected, atol=1e-12)


def test_get_simulations_simultaneous_presses_sum_signed_columns(snowshoe_io, snowshoe_io_strengths):
    single = get_simulations(snowshoe_io, n_sim=1, presample=lambda symbols: snowshoe_io_strengths)
    columns = single["all_nodes"]
    expected = single["effects"][0][:, columns.index("Inp1")] - single["effects"][0][:, columns.index("Inp2")]
    result = get_simulations(snowshoe_io, n_sim=1, presample=lambda symbols: snowshoe_io_strengths,
                             perturb=(("Inp1", 1), ("Inp2", -1)))["effects"][0]
    assert np.array_equal(result, expected)


# =============================================================================
# simulation_effects
# =============================================================================

def test_simulation_effects_sign_determined_cells_are_exact(snowshoe_io):
    weights = weighted_predictions_matrix(snowshoe_io)
    simulated = simulation_effects(snowshoe_io, n_sim=200, seed=42)
    determined = [(i, j) for i in range(weights.rows) for j in range(weights.cols) if weights[i, j] in (1, -1)]
    result = [float(simulated[i, j]) for i, j in determined]
    expected = [float(weights[i, j]) for i, j in determined]
    assert result == expected


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

def test_simulations_table_rejects_invalid_nodes():
    G = nx.DiGraph()
    G.add_node('A', category='input')
    G.add_node('B', category='input')
    G.add_edge('A', 'B', sign=1)
    with pytest.raises(ValueError, match=r"Invalid nodes: \['A', 'B'\]"):
        simulations_table(G, perturb="A:+", n_sim=5, seed=42)


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


def test_simulations_table_rejects_category_change(dashed_role_change):
    with pytest.raises(ValueError, match="^Nodes change category: C$"):
        simulations_table(dashed_role_change, perturb="A:+", n_sim=10)


def test_simulations_table_importable():
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


def test_zero_observation_matches_draws_where_uncertain_link_is_absent(self_limited_pair):
    G = self_limited_pair
    G.add_edge("A", "B", sign=1, dashes=True)
    averaged = marginal_likelihood(G, "A:+", "B:0", n_sim=400, seed=3, uncertain_interactions="sample")
    assert 0.3 < averaged < 0.7
    certain = G.copy()
    certain["A"]["B"]["dashes"] = False
    assert marginal_likelihood(certain, "A:+", "B:0", n_sim=100, seed=3) == 0.0
    H = G.copy()
    H.remove_edge("A", "B")
    assert marginal_likelihood(H, "A:+", "B:0", n_sim=100, seed=3) == 1.0
    with pytest.raises(ValueError, match="require a perturbation"):
        get_simulations(G, n_sim=10, observe=(("B", 0),))


@pytest.mark.parametrize('option', ['n_sim', 'max_attempts'])
@pytest.mark.parametrize('value', [0, -1, 1.5, True, np.bool_(False)])
def test_get_simulations_requires_a_positive_integer(snowshoe, option, value):
    with pytest.raises(ValueError, match='positive integer'):
        get_simulations(snowshoe, **{option: value})


def test_fixed_uncertain_strength_is_sampled_in_and_out(self_limited_pair):
    G = self_limited_pair
    G.add_edge('A', 'B', sign=1, dashes=True)
    sims = get_simulations(
        G, n_sim=50, seed=42, uncertain_interactions="sample", return_samples=True,
        presample=lambda symbols: {sp.Symbol('a_B,A'): 0.5},
    )
    assert set(sims['samples']['a_B,A']) == {0.0, 0.5}
    assert np.array_equal(sims['samples']['a_B,A'] > 0,
                          np.array([effect[1, 0] > 0 for effect in sims['effects']]))


def test_tied_presampled_strengths_report_actual_draws(snowshoe):
    sims = get_simulations(snowshoe, n_sim=10, return_samples=True,
                           presample=lambda symbols: {sp.Symbol('a_R,R'): sp.Symbol('a_P,P')})
    assert np.array_equal(sims['samples']['a_R,R'], sims['samples']['a_P,P'])


def test_simulations_table_counts_cancelled_presses_as_no_effect(self_limited_pair):
    G = self_limited_pair
    G.add_edge('A', 'B', sign=1)
    result = simulations_table(G, 'A:+, B:-', n_sim=5,
                               presample=lambda symbols: {symbol: 1 for symbol in symbols})
    row = result.set_index('effect_on').loc['B']
    assert row['no_effect'] == row['valid_sims'] == 5
    assert result[['negative', 'no_effect', 'positive']].sum(axis=1).equals(result['valid_sims'])


@pytest.mark.parametrize('category', ['output', 'input', 'state'])
def test_simulations_table_combines_existing_input_presses(category):
    G = nx.DiGraph()
    G.add_edge('X', 'X', sign=-1)
    G.add_edge('I1', 'X', sign=1)
    G.add_edge('I2', 'X', sign=1)
    G.add_edge('X', 'Y', sign=1)
    G = define_input_output(G)
    G.nodes['Y']['category'] = category
    original = G.copy()
    result = simulations_table(G, 'I1:+, I2:+', n_sim=5)
    assert list(result['effect_on']) == ['X', 'Y']
    assert list(result['positive']) == [5, 5]
    assert list(result['no_effect']) == [0, 0]
    assert nx.utils.graphs_equal(G, original)


def test_nonfinite_responses_are_rejected():
    G = nx.DiGraph()
    G.add_node('A', category='state')
    G.add_edge('A', 'A', sign=-1)
    with pytest.raises(RuntimeError, match='Maximum iterations'):
        get_simulations(G, n_sim=1,
                        presample=lambda symbols: {sp.Symbol('a_A,A'): 1e-320})


def test_simulation_results_and_graph_edits_do_not_reuse_stale_cache():
    G = nx.DiGraph()
    G.add_node('A', category='state')
    G.add_edge('A', 'A', sign=-1)
    sims = get_simulations(G, n_sim=np.int64(1))
    original = sims['effects'][0].copy()
    sims['effects'][0][:] = -999
    assert np.array_equal(get_simulations(G, n_sim=np.int64(1))['effects'][0], original)
    G.add_node('B', category='state')
    G.add_edge('B', 'B', sign=-1)
    assert get_simulations(G, n_sim=np.int64(1))['effects'][0].shape == (2, 2)


def test_structure_averaging_rejects_changes_to_node_roles():
    graph = nx.DiGraph()
    graph.add_node('A', category='state')
    graph.add_node('B', category='state')
    graph.add_edge('A', 'A', sign=-1, dashes=True)
    graph.add_edge('A', 'B', sign=1)
    graph.add_edge('B', 'B', sign=-1)
    with pytest.raises(ValueError, match=r"^Nodes change category: A$"):
        get_simulations(graph, n_sim=20, uncertain_interactions="sample", seed=42)


def test_zero_presampled_output_link_disconnects_the_whole_output_chain():
    graph = nx.DiGraph()
    graph.add_edge('X', 'X', sign=-1)
    graph.add_edge('X', 'Y0', sign=1)
    graph.add_edge('Y0', 'Y1', sign=1)
    graph = define_input_output(graph)
    result = get_simulations(
        graph, n_sim=3, return_samples=True,
        presample=lambda symbols: {sp.Symbol('c_Y0,X'): 0},
    )
    assert all(np.array_equal(effect[1:], np.zeros((2, 1))) for effect in result['effects'])
    assert np.array_equal(result['samples']['c_Y0,X'], np.zeros(3))
    assert 'c_Y1,Y0' not in result['samples']


def test_get_simulations_structural_zero_cell(structural_zero_chain):
    sims = get_simulations(structural_zero_chain, n_sim=200, seed=42, perturb=('A', 1))
    result = np.array(sims['effects'])[:, sims['all_nodes'].index('B')]
    expected = np.zeros(200)
    assert np.array_equal(result, expected)


def test_get_simulations_fork_is_always_stable(fork):
    sims = get_simulations(fork, n_sim=200, seed=42, perturb=('A', 1))
    result = (sims['prop_stable'], sims['attempts'], sims['n_stable'])
    expected = (1.0, 200, 200)
    assert result == expected


def test_get_simulations_sample_allows_isolating_a_self_limited_node(snowshoe):
    G = nx.DiGraph(snowshoe)
    G.add_edge('Z', 'Z', sign=-1)
    G.add_edge('Z', 'R', sign=-1, dashes=True)
    sims = get_simulations(define_input_output(G), n_sim=50, seed=1, uncertain_interactions="sample", return_samples=True)
    result = (sims["samples"]["a_R,Z"] == 0.0).any(), len(sims["effects"])
    expected = (True, 50)
    assert result == expected


def test_get_simulations_rejects_unknown_uncertain_mode_snowshoe_dashed(snowshoe_dashed):
    with pytest.raises(ValueError, match="uncertain_interactions must be 'sample' or 'enumerate'"):
        get_simulations(snowshoe_dashed, n_sim=10, uncertain_interactions="unknown")


def test_iter_simulations_enumerate_yields_each_structure_snowshoe_dashed(snowshoe_dashed):
    strengths = {symbol: 0.5 for symbol in create_matrix(snowshoe_dashed, "symbolic").free_symbols}
    batches = list(iter_simulations(snowshoe_dashed, n_sim=10, seed=1, uncertain_interactions="enumerate",
                                    max_attempts=np.int64(10), presample=lambda symbols: strengths))
    result = [(len(batch["effects"]), len(set(batch["structures"]))) for batch in batches]
    expected = [(10, 1)] * 4
    assert result == expected
    assert len({batch["structures"][0] for batch in batches}) == 4
    assert all(batch["attempts"] == 10 for batch in batches)
    with pytest.raises(RuntimeError, match="Matched 9/10 draws"):
        get_simulations(snowshoe_dashed, n_sim=10, uncertain_interactions="enumerate", max_attempts=9,
                        presample=lambda symbols: strengths)


def test_get_simulations_enumerate_with_individual_edges_has_eight_structures_snowshoe_dashed(snowshoe_dashed):
    sims = get_simulations(snowshoe_dashed, n_sim=5, seed=1, uncertain_interactions="enumerate", pair_reciprocal=False)
    result = (len(sims["effects"]), len(set(sims["structures"])))
    expected = (40, 8)
    assert result == expected


def test_get_simulations_sample_drops_reciprocal_dashed_edges_together_snowshoe_dashed(snowshoe_dashed):
    sims = get_simulations(snowshoe_dashed, n_sim=200, seed=1, return_samples=True)
    dropped_rp, dropped_pr = sims["samples"]["a_P,R"] == 0.0, sims["samples"]["a_R,P"] == 0.0
    result = (np.array_equal(dropped_rp, dropped_pr), dropped_rp.any(), (~dropped_rp).any())
    expected = (True, True, True)
    assert result == expected


def test_get_simulations_sample_can_drop_reciprocal_dashed_edges_separately_snowshoe_dashed(snowshoe_dashed):
    sims = get_simulations(snowshoe_dashed, n_sim=200, seed=1, return_samples=True, pair_reciprocal=False)
    result = np.array_equal(sims["samples"]["a_P,R"] == 0.0, sims["samples"]["a_R,P"] == 0.0)
    expected = False
    assert result == expected


@pytest.mark.parametrize('uncertain_interactions', ['sample', 'enumerate'])
@pytest.mark.parametrize('perturb', [None, ('A', 1)])
def test_get_simulations_zeroes_cells_after_dropping_a_self_effect(structural_zero_chain, uncertain_interactions, perturb):
    G = nx.DiGraph(structural_zero_chain)
    G.add_edge('C', 'C', sign=-1, dashes=True)
    sims = get_simulations(define_input_output(G), n_sim=100, seed=1,
                           perturb=perturb, uncertain_interactions=uncertain_interactions, return_samples=True)
    dropped = sims['samples']['a_C,C'] == 0.0
    responses = np.asarray(sims['effects'])
    b = responses[:, 1] if perturb else responses[:, 1, 0]
    result = (dropped.any(), (~dropped).any(), np.all(b[dropped] == 0.0), np.all(b[~dropped] > 0.0))
    expected = (True, True, True, True)
    assert result == expected


def test_simulations_table_pairs_reciprocal_dashed_edges_snowshoe_dashed(snowshoe_dashed):
    result = simulations_table(snowshoe_dashed, perturb="C:+", n_sim=20, seed=1)["model"].nunique()
    expected = 4
    assert result == expected


def test_simulation_effects_enumerate_divides_by_all_stable_draws_snowshoe_dashed(snowshoe_dashed):
    result = simulation_effects(snowshoe_dashed, n_sim=50, seed=1, positive_only=True, uncertain_interactions="enumerate")[0, 0]
    expected = 1.0
    assert result == expected
