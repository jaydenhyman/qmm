"""Shared test fixtures for qmm tests."""

import pytest
import networkx as nx
import subprocess
from pathlib import Path

from qmm.core.helper import list_to_digraph
from qmm.extensions.effects import define_input_output


# =============================================================================
# Test models
# =============================================================================

@pytest.fixture
def snowshoe():
    """Simple 3-node predator-prey model (snowshoe_rp hare system)."""
    A = [
        [-1, -1, 0],
        [1, 0, -1],
        [0, 1, -1]
    ]
    labels = ['R', 'C', 'P']
    return list_to_digraph(A, labels)


@pytest.fixture
def snowshoe_rp():
    """Snowshoe model with an added positive R->P link."""
    A = [
        [-1, -1, 0],
        [1, 0, -1],
        [1, 1, -1],
    ]
    labels = ['R', 'C', 'P']
    return list_to_digraph(A, labels)


@pytest.fixture
def snowshoe_na():
    """3-node model to test as_nan feature - distinguishes ambiguous (0.5) from no effects (nan)."""
    A = [
        [-1, -1, -1],
        [0, -1, -1],
        [0, 0, -1]
    ]
    labels = ['1', '2', '3']
    return list_to_digraph(A, labels)


@pytest.fixture
def mesocosm():
    """8-node mesocosm model with complex interactions."""
    A = [
        [-1, -1, -1, -1, 0, 0, 0, 0],
        [1, 0, 0, 0, -1, -1, 0, 0],
        [1, 0, 0, 0, 0, -1, 0, 0],
        [1, 0, 0, -1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, -1, -1],
        [0, 1, 1, 0, 0, 0, 0, -1],
        [0, 0, 0, 0, 1, 0, 0, -1],
        [0, 0, 0, 0, 1, 1, 1, -1],
    ]
    labels = ['P', 'A1', 'A2', 'AP', 'H1', 'H2', 'C1', 'C2']
    return list_to_digraph(A, labels)


@pytest.fixture
def chain():
    """5-node straight chain model with self-effects."""
    A = [
        [-1, -1, 0, 0, 0],
        [1, -1, -1, 0, 0],
        [0, 1, -1, -1, 0],
        [0, 0, 1, -1, -1],
        [0, 0, 0, 1, -1]
    ]
    labels = ['1', '2', '3', '4', '5']
    return list_to_digraph(A, labels)


@pytest.fixture
def snowshoe_io():
    """System with input and output nodes (snowshoe_rp hare system)."""
    G = nx.DiGraph()
    G.add_node('R')
    G.add_node('C')
    G.add_node('P')
    G.add_node('Inp1')
    G.add_node('Inp2')
    G.add_node('Out1')
    G.add_node('Out2')
    G.add_edge('R', 'R', sign=-1)
    G.add_edge('R', 'C', sign=1)
    G.add_edge('C', 'R', sign=-1)
    G.add_edge('C', 'P', sign=1)
    G.add_edge('P', 'C', sign=-1)
    G.add_edge('P', 'P', sign=-1)
    G.add_edge('Inp1', 'R', sign=1)
    G.add_edge('Inp1', 'C', sign=-1)
    G.add_edge('Inp2', 'P', sign=-1)
    G.add_edge('C', 'Out1', sign=-1)
    G.add_edge('C', 'Out2', sign=1)
    G.add_edge('P', 'Out1', sign=1)
    G = define_input_output(G)
    return G


@pytest.fixture
def snowshoe_io_na(snowshoe_io):
    """snowshoe_io graph with an extra state node that creates NaN entries."""
    G = snowshoe_io.copy()
    G.add_node('N', category='state')
    G.add_edge('N', 'N', sign=-1)
    G.add_edge('N', 'R', sign=1)
    return G


@pytest.fixture
def sign_stable_chain(chain):
    """Chain model for sign stable true test."""
    return chain


@pytest.fixture
def sign_stable_snowshoe(snowshoe):
    """Snowshoe_rp model for sign stable true test."""
    return snowshoe


@pytest.fixture
def class_ii():
    """3-node model that is Class II."""
    A = [
        [-1, 1, 1],
        [1, -1, 1],
        [1, 1, -1]
    ]
    labels = ['A', 'B', 'C']
    return list_to_digraph(A, labels)


@pytest.fixture
def colour_pass():
    """5-node model that passes Jeffries' colour test."""
    A = [
        [0, -1, 0, 0, 0],
        [1, 0, -1, 0, 0],
        [0, 1, -1, -1, 0],
        [0, 0, 1, 0, -1],
        [0, 0, 0, 1, 0]
    ]
    labels = ['1', '2', '3', '4', '5']
    return list_to_digraph(A, labels)


@pytest.fixture
def disconnected_graph():
    """Graph with disconnected components (A->B, C)."""
    G = nx.DiGraph()
    G.add_edge('A', 'B', sign=1)
    G.add_node('C')
    return G


@pytest.fixture
def io_only_graph():
    """Graph with only input/output nodes (I->O) and no state variables."""
    G = nx.DiGraph()
    G.add_node('I', category='input')
    G.add_node('O', category='output')
    G.add_edge('I', 'O', sign=1)
    return G


@pytest.fixture
def positive_loop_graph():
    """Single node with a positive self-loop (A->A, +)."""
    G = nx.DiGraph()
    G.add_node('A', category='state')
    G.add_edge('A', 'A', sign=1)
    return G


@pytest.fixture
def snowshoe_dashed():
    """Snowshoe_rp graph with dashed edges for model validation tests."""
    G = nx.DiGraph()
    G.add_node('R', category='state')
    G.add_node('C', category='state')
    G.add_node('P', category='state')
    G.add_edge('R', 'R', sign=-1)
    G.add_edge('R', 'C', sign=1)
    G.add_edge('C', 'R', sign=-1)
    G.add_edge('C', 'P', sign=1)
    G.add_edge('P', 'C', sign=-1)
    G.add_edge('P', 'P', sign=-1)
    G.add_edge('R', 'P', sign=1, dashes=True)
    G.add_edge('C', 'C', sign=-1, dashes=True)
    G.add_edge('P', 'R', sign=-1, dashes=True)
    return G


@pytest.fixture
def bayes_models():
    """Two similar models for Bayes factor comparison."""
    A1 = [
        [-1, -1, 0],
        [1, 0, -1],
        [0, 1, -1]
    ]
    A2 = [
        [-1, -1, 0],
        [1, -1, -1],
        [0, 1, -1]
    ]
    G1 = list_to_digraph(A1, ['R', 'C', 'P'])
    G2 = list_to_digraph(A2, ['R', 'C', 'P'])
    return G1, G2


@pytest.fixture
def mesocosm_alt_models(mesocosm):
    """Two alternative mesocosm models for Bayes factor comparison."""
    G = mesocosm
    G_alt = G.copy()
    G_alt.remove_edge('C1', 'C2')
    G_alt.remove_edge('C2', 'C1')
    G_alt.add_edge('C1', 'C1', sign=-1)
    return G, G_alt


@pytest.fixture
def simple_two_node():
    """Simple 2-node graph for basic tests."""
    return list_to_digraph([[0, 1], [-1, 0]])


@pytest.fixture
def simple_ab_positive():
    """Simple graph with A->B positive edge."""
    G = nx.DiGraph()
    G.add_node('A', category='state')
    G.add_node('B', category='state')
    G.add_edge('A', 'B', sign=1)
    return G


@pytest.fixture
def simple_xy_negative():
    """Simple graph with X->Y negative edge."""
    G = nx.DiGraph()
    G.add_node('X', category='state')
    G.add_node('Y', category='state')
    G.add_edge('X', 'Y', sign=-1)
    return G


@pytest.fixture
def colour_fail():
    """5-node model that fails Jeffries' colour test."""
    A = [
        [-1, 1, 0, 0, 0],
        [-1, 0, 0, 0, 0],
        [0, 0, -1, 1, 0],
        [0, 0, -1, 0, 0],
        [0, 0, 0, 0, -1]
    ]
    labels = ['1', '2', '3', '4', '5']
    return list_to_digraph(A, labels)


@pytest.fixture
def large_six_node():
    """Large 6-node model with all self-effects for hurwitz test."""
    return list_to_digraph([[-1]*6 for _ in range(6)], [str(i) for i in range(6)])


@pytest.fixture
def output_to_output_graph():
    """Graph with output to output edge."""
    G = nx.DiGraph()
    G.add_node('A', category='state')
    G.add_node('Out1', category='output')
    G.add_node('Out2', category='output')
    G.add_edge('A', 'Out1', sign=1)
    G.add_edge('Out1', 'Out2', sign=1)
    return G


@pytest.fixture
def feedback_test_graph():
    """Graph for testing complementary feedback."""
    G = nx.DiGraph()
    G.add_node('A', category='state')
    G.add_node('B', category='state')
    G.add_node('C', category='state')
    G.add_edge('A', 'B', sign=1)
    G.add_edge('B', 'C', sign=1)
    G.add_edge('C', 'C', sign=-1)
    return G


@pytest.fixture
def nan_feedback_graph():
    """Graph with disconnected components for NaN feedback test."""
    G = nx.DiGraph()
    G.add_node('A', category='state')
    G.add_node('B', category='state')
    G.add_node('C', category='state')
    G.add_node('D', category='state')
    G.add_edge('A', 'B', sign=1)
    G.add_edge('C', 'D', sign=1)
    return G


@pytest.fixture
def minimal_error_graph():
    """Minimal graph for error testing."""
    G = nx.DiGraph()
    G.add_node('A', category='state')
    return G


@pytest.fixture
def snowshoe_io_with_direct_edge(snowshoe_io):
    """Snowshoe IO graph with direct input to output edge."""
    G = snowshoe_io.copy()
    G.add_edge('Inp1', 'Out1', sign=1)
    return G


@pytest.fixture
def direct_input_output_graph():
    """Graph with direct input to output edge for D matrix testing."""
    G = nx.DiGraph()
    G.add_node('X', category='state')
    G.add_node('U', category='input')
    G.add_node('Y', category='output')
    G.add_edge('U', 'X', sign=1)
    G.add_edge('U', 'Y', sign=1)
    G.add_edge('X', 'Y', sign=1)
    return G


@pytest.fixture
def non_standard_sign_graph():
    """Graph with non-standard sign value."""
    G = nx.DiGraph()
    G.add_node('A')
    G.add_node('B')
    G.add_edge('A', 'B', sign=0.5)
    return G


@pytest.fixture
def cyclic_inputs_graph():
    """Graph with cyclic input nodes for error testing."""
    G = nx.DiGraph()
    G.add_node('S', category='state')
    G.add_edge('S', 'S', sign=-1)
    G.add_node('I1', category='input')
    G.add_node('I2', category='input')
    G.add_edge('I1', 'I2', sign=1)
    G.add_edge('I2', 'I1', sign=1)
    G.add_edge('I1', 'S', sign=1)
    return G


@pytest.fixture
def cyclic_outputs_graph():
    """Graph with cyclic output nodes for error testing."""
    G = nx.DiGraph()
    G.add_node('S', category='state')
    G.add_edge('S', 'S', sign=-1)
    G.add_node('O1', category='output')
    G.add_node('O2', category='output')
    G.add_edge('O1', 'O2', sign=1)
    G.add_edge('O2', 'O1', sign=1)
    G.add_edge('S', 'O1', sign=1)
    return G
