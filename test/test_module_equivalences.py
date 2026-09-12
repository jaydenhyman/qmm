"""Equivalences between qmm modules."""

from test.feedback import cycle_expansion

import networkx as nx
import numpy as np
import pytest
import sympy as sp

from qmm.core.helper import _edge_prefix, get_nodes, get_weight, load_digraph
from qmm.core.press import (
    absolute_feedback_matrix,
    adjoint_matrix,
    sign_determinacy_matrix,
    weighted_predictions_matrix,
)
from qmm.core.stability import absolute_feedback, net_feedback, system_feedback
from qmm.core.structure import create_matrix
from qmm.extensions.effects import cumulative_effects, direct_effects, get_simulations
from qmm.extensions.indicators import mutual_information
from qmm.extensions.life import birth_matrix, death_matrix, life_expectancy_change
from qmm.extensions.paths import complementary_feedback, system_paths, weighted_paths
from qmm.extensions.senstability import (
    absolute_structural_sensitivity,
    net_structural_sensitivity,
    structural_sensitivity,
)
from qmm.extensions.validation import marginal_likelihood


@pytest.fixture
def io_and_state_model(request):
    """Input-output model, the same model with self-limited input and output states, and their symbol map."""
    G = request.getfixturevalue(request.param)
    H = nx.DiGraph()
    H.add_nodes_from((node, {**data, "category": "state"}) for node, data in G.nodes(data=True))
    H.add_edges_from(G.edges(data=True))
    H.add_edges_from((node, node, {"sign": -1}) for node in get_nodes(G, "input") + get_nodes(G, "output"))
    symbols = {sp.Symbol(f"a_{v},{u}"): sp.Symbol(f"{_edge_prefix(G, u, v)}_{v},{u}")
               for u, v in G.edges() if _edge_prefix(G, u, v) != "a"}
    symbols.update({sp.Symbol(f"a_{node},{node}"): sp.Integer(1)
                    for node in get_nodes(G, "input") + get_nodes(G, "output")})
    return G, H, symbols


# =============================================================================
# effects and structure: inputs and outputs as self-limited states
# =============================================================================

@pytest.mark.parametrize("io_and_state_model, form", [
    ("snowshoe_io", "symbolic"),
    ("snowshoe_io", "signed"),
    ("snowshoe_io", "binary"),
    ("io_chain", "symbolic"),
    ("io_chain", "signed"),
    ("io_chain", "binary"),
    ("io_long_chain", "symbolic"),
    ("io_long_chain", "signed"),
    ("io_long_chain", "binary"),
    ("io_branched", "signed"),
    ("io_branched", "binary"),
], indirect=["io_and_state_model"])
def test_cumulative_effects_match_self_limited_state_adjoint_io_and_state_model(io_and_state_model, form):
    G, H, symbols = io_and_state_model
    nodes = get_nodes(H, "state")
    rows = [nodes.index(node) for node in get_nodes(G, "state") + get_nodes(G, "output")]
    cols = [nodes.index(node) for node in get_nodes(G, "state") + get_nodes(G, "input")]
    effects = absolute_feedback_matrix(H) if form == "binary" else adjoint_matrix(H, form=form)
    result = sp.expand(effects.subs(symbols)[rows, cols])
    expected = sp.expand(cumulative_effects(G, form=form))
    assert result == expected


@pytest.mark.parametrize("io_and_state_model",
                         ["snowshoe_io", "io_chain", "io_long_chain", "io_branched"], indirect=True)
def test_input_and_output_blocks_match_self_limited_state_series_io_and_state_model(io_and_state_model):
    G, H, symbols = io_and_state_model
    states, inputs, outputs = (get_nodes(G, kind) for kind in ("state", "input", "output"))
    index = {node: i for i, node in enumerate(get_nodes(H, "state"))}
    A = create_matrix(H, form="symbolic").subs(symbols)
    b_direct = sp.Matrix([[A[index[state], index[source]] for source in inputs] for state in states])
    c_direct = sp.Matrix([[A[index[output], index[state]] for state in states] for output in outputs])
    input_block = sp.Matrix([[A[index[i], index[j]] for j in inputs] for i in inputs])
    output_block = sp.Matrix([[A[index[i], index[j]] for j in outputs] for i in outputs])
    result = (sp.expand(b_direct * (-input_block).inv()), sp.expand((-output_block).inv() * c_direct))
    expected = (create_matrix(G, form="symbolic", matrix_type="B"), create_matrix(G, form="symbolic", matrix_type="C"))
    assert result == expected


# =============================================================================
# senstability and stability
# =============================================================================

@pytest.mark.parametrize("sensitivity, feedback", [
    (structural_sensitivity, system_feedback),
    (net_structural_sensitivity, net_feedback),
    (absolute_structural_sensitivity, absolute_feedback),
])
def test_structural_sensitivity_sums_to_level_times_feedback_snowshoe_rp(snowshoe_rp, sensitivity, feedback):
    levels = range(1, len(get_nodes(snowshoe_rp, "state")) + 1)
    result = [sp.expand(sum(sensitivity(snowshoe_rp, level=level))) for level in levels]
    expected = [sp.expand(level * feedback(snowshoe_rp)[level]) for level in levels]
    assert result == expected


@pytest.mark.parametrize("sensitivity, feedback", [
    (net_structural_sensitivity, net_feedback),
    (absolute_structural_sensitivity, absolute_feedback),
])
def test_structural_sensitivity_sums_to_level_times_feedback_chain(chain, sensitivity, feedback):
    levels = range(1, len(get_nodes(chain, "state")) + 1)
    result = [sum(sensitivity(chain, level=level)) for level in levels]
    expected = [level * feedback(chain)[level] for level in levels]
    assert result == expected


# =============================================================================
# press and stability
# =============================================================================

@pytest.mark.parametrize("model", ["snowshoe", "snowshoe_rp", "chain"])
def test_adjoint_diagonal_matches_feedback_of_the_remaining_subsystem(model):
    G = load_digraph(model)
    nodes = get_nodes(G, "state")
    adjoint = adjoint_matrix(G, form="symbolic")
    result = [sp.expand(adjoint[i, i]) for i in range(len(nodes))]
    expected = [
        sp.expand(-system_feedback(G.subgraph([n for n in nodes if n != node]).copy(), level=len(nodes) - 1)[0])
        for node in nodes
    ]
    assert result == expected


@pytest.mark.parametrize("model", ["snowshoe_rp", "chain"])
def test_perturb_argument_matches_the_full_matrix_column(model):
    G = load_digraph(model)
    nodes = get_nodes(G, "state")
    matrices = [
        adjoint_matrix,
        absolute_feedback_matrix,
        weighted_predictions_matrix,
        sign_determinacy_matrix,
        birth_matrix,
        death_matrix,
        life_expectancy_change,
    ]
    result = [sp.expand(matrix(G, perturb=node)) for matrix in matrices for node in nodes]
    expected = [sp.expand(matrix(G).col(j)) for matrix in matrices for j in range(len(nodes))]
    assert result == expected


# =============================================================================
# paths, stability and helper
# =============================================================================

@pytest.mark.parametrize("model", ["snowshoe_rp", "chain"])
def test_disjoint_cycle_combinations_match_system_feedback(model):
    G = load_digraph(model)
    result, _ = cycle_expansion(G)
    expected = system_feedback(G).applyfunc(sp.expand)
    assert result == expected


@pytest.mark.parametrize("model, source, target", [
    ("snowshoe_io", "Inp1", "Out1"),
    ("snowshoe_io", "Inp2", "Out1"),
    ("mesocosm", "P", "A1"),
    ("mesocosm", "A1", "C2"),
])
def test_weighted_paths_match_the_weight_of_system_paths(model, source, target):
    G = load_digraph(model)
    result = sp.Matrix(weighted_paths(G, source, target)["Weight"].tolist())
    expected = get_weight(
        sp.Matrix(system_paths(G, source, target, form="signed")["Effect"].tolist()),
        sp.Matrix(system_paths(G, source, target, form="binary")["Effect"].tolist()),
        sp.Integer(0),
    )
    assert result == expected


@pytest.mark.parametrize("model, source, target", [
    ("snowshoe_io", "Inp1", "Out1"),
    ("mesocosm", "P", "A1"),
    ("mesocosm", "A1", "C2"),
    ("chain", "1", "3"),
])
def test_binary_complementary_feedback_counts_symbolic_terms(model, source, target):
    G = load_digraph(model)
    symbolic = complementary_feedback(G, source, target, form="symbolic")["Feedback"]
    result = [len(feedback.as_ordered_terms()) if feedback else 0 for feedback in symbolic]
    expected = [int(feedback) for feedback in complementary_feedback(G, source, target, form="binary")["Feedback"]]
    assert result == expected


# =============================================================================
# life, structure and effects
# =============================================================================

@pytest.mark.parametrize("model", ["snowshoe_rp", "chain", "mesocosm"])
def test_birth_and_death_matrices_decompose_the_interaction_matrix(model):
    G = load_digraph(model)
    result = (
        sp.expand(birth_matrix(G) - death_matrix(G)),
        birth_matrix(G, form="signed") - death_matrix(G, form="signed"),
        birth_matrix(G, form="signed") + death_matrix(G, form="signed"),
    )
    expected = (
        create_matrix(G, form="symbolic"),
        create_matrix(G, form="signed"),
        create_matrix(G, form="binary"),
    )
    assert result == expected


def test_direct_effects_signs_match_birth_and_death_matrices_snowshoe_io(snowshoe_io):
    n = len(get_nodes(snowshoe_io, "state"))
    result = (direct_effects(snowshoe_io, form="positive")[:n, :n], direct_effects(snowshoe_io, form="negative")[:n, :n])
    expected = (birth_matrix(snowshoe_io, form="signed"), death_matrix(snowshoe_io, form="signed"))
    assert result == expected


# =============================================================================
# validation, indicators and effects
# =============================================================================

@pytest.mark.parametrize("observe, sign", [("C2:+", 1), ("H1:-", -1)])
def test_marginal_likelihood_matches_the_simulated_sign_proportion_mesocosm(mesocosm, observe, sign):
    node = observe.split(":")[0]
    sims = get_simulations(mesocosm, n_sim=400, seed=11, perturb=("P", 1))
    responses = np.array([effect[sims["all_nodes"].index(node)] for effect in sims["effects"]]).ravel()
    result = marginal_likelihood(mesocosm, "P:+", observe, n_sim=400, seed=11)
    expected = float(np.mean(responses * sign > 0))
    assert result == expected


def test_mutual_information_of_identical_models_is_zero_snowshoe(snowshoe):
    result = mutual_information([snowshoe, snowshoe], "R:+", n_sim=200)["Mutual Information"].tolist()
    expected = [0] * len(get_nodes(snowshoe, "state"))
    assert result == expected
