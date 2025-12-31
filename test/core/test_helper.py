"""Tests for qmm.core.helper module."""

import pytest
import networkx as nx
import numpy as np
import sympy as sp

from qmm import (
    list_to_digraph,
    load_digraph,
    digraph_to_list,
    get_nodes,
    get_weight,
    get_positive,
    get_negative,
    sign_determinacy,
)
from qmm.core.helper import _arrows, _sign_string, _NodeSign, _parse_perturbations, _parse_observations, _random_sampler, perm, _perm_ryser, _perm_bbfg, get_dashed_alternatives
from qmm.core.stability import net_feedback, absolute_feedback


# =============================================================================
# list_to_digraph()
# =============================================================================

def test_list_to_digraph_signed_matrix_snowshoe(snowshoe):
    """Test list_to_digraph builds the snowshoe DiGraph from a signed matrix."""
    result = (
        isinstance(snowshoe, nx.DiGraph),
        snowshoe.number_of_nodes(),
        snowshoe.number_of_edges(),
        snowshoe['R']['R']['sign'],
        snowshoe['R']['C']['sign'],
        snowshoe['C']['R']['sign']
    )
    expected = (True, 3, 6, -1, 1, -1)
    assert result == expected


def test_list_to_digraph_default_labels_simple_two_node(simple_two_node):
    """Test list_to_digraph assigns default numeric labels for unnamed nodes."""
    result = ('1' in simple_two_node.nodes(), '2' in simple_two_node.nodes())
    expected = (True, True)
    assert result == expected


def test_list_to_digraph_explicit_labels_no_fixture():
    """Test list_to_digraph honors explicit node labels for custom snowshoe matrix."""
    A = [[-1, -1, 0], [1, 0, -1], [0, 1, -1]]
    G = list_to_digraph(A, ['R', 'C', 'P'])
    result = (list(G.nodes()), G['R']['R']['sign'], G['C']['R']['sign'], G['P']['C']['sign'])
    expected = (['R', 'C', 'P'], -1, -1, -1)
    assert result == expected


def test_list_to_digraph_chain_structure_chain(chain):
    """Test list_to_digraph builds the chain graph structure."""
    result = (list(chain.nodes()), chain.number_of_edges())
    expected = (['1', '2', '3', '4', '5'], 13)
    assert result == expected


@pytest.mark.parametrize("bad_input", ["string", [[1, 2, 3], [4, 5, 6]]])
def test_list_to_digraph_invalid_input_bad_input(bad_input):
    """Test list_to_digraph raises ValueError for invalid matrix inputs."""
    result = None
    expected = ValueError
    with pytest.raises(expected):
        list_to_digraph(bad_input)
        result = "No exception"
    assert result is None


def test_list_to_digraph_label_mismatch_no_fixture():
    """Test list_to_digraph raises ValueError when labels mismatch matrix shape."""
    result = None
    expected = ValueError
    with pytest.raises(expected):
        list_to_digraph([[0, 1], [-1, 0]], ['A', 'B', 'C'])
        result = "No exception"
    assert result is None


# =============================================================================
# load_digraph()
# =============================================================================

def test_load_digraph_snowshoe_nodes():
    """Test load_digraph loads snowshoe model with correct nodes."""
    G = load_digraph("snowshoe")
    result = list(G.nodes())
    expected = ['V', 'H', 'P']
    assert result == expected


def test_load_digraph_snowshoe_edges():
    """Test load_digraph loads snowshoe model with correct number of edges."""
    G = load_digraph("snowshoe")
    result = G.number_of_edges()
    expected = 7
    assert result == expected


def test_load_digraph_snowshoe_i_nodes():
    """Test load_digraph loads snowshoe_i model with correct nodes and categories."""
    G = load_digraph("snowshoe_i")
    result = (list(G.nodes()), G.nodes['I']['category'], G.nodes['V']['category'])
    expected = (['V', 'H', 'P', 'I'], 'input', 'state')
    assert result == expected


def test_load_digraph_snowshoe_i_edges():
    """Test load_digraph loads snowshoe_i model with correct edges."""
    G = load_digraph("snowshoe_i")
    result = G.number_of_edges()
    expected = 10
    assert result == expected


def test_load_digraph_snowshoe_io_categories():
    """Test load_digraph loads snowshoe_io model with correct input/state/output categories."""
    G = load_digraph("snowshoe_io")
    result = (G.nodes['I']['category'], G.nodes['V']['category'], G.nodes['O']['category'])
    expected = ('input', 'state', 'output')
    assert result == expected


def test_load_digraph_chain_nodes():
    """Test load_digraph loads chain model with correct nodes."""
    G = load_digraph("chain")
    result = (G.number_of_nodes(), list(G.nodes()))
    expected = (5, ['1', '2', '3', '4', '5'])
    assert result == expected


def test_load_digraph_mesocosm_nodes():
    """Test load_digraph loads mesocosm model with correct nodes."""
    G = load_digraph("mesocosm")
    result = G.number_of_nodes()
    expected = 8
    assert result == expected


def test_load_digraph_class_ii_nodes():
    """Test load_digraph loads class_ii model with correct nodes."""
    G = load_digraph("class_ii")
    result = (G.number_of_nodes(), sorted(G.nodes()))
    expected = (3, ['A', 'B', 'C'])
    assert result == expected


def test_load_digraph_invalid_model():
    """Test load_digraph raises ValueError for invalid model name."""
    with pytest.raises(ValueError, match="Model 'invalid' not found"):
        load_digraph("invalid")


# =============================================================================
# digraph_to_list()
# =============================================================================

def test_digraph_to_list_serialization_snowshoe(snowshoe):
    """Test digraph_to_list serializes the snowshoe graph."""
    result = digraph_to_list(snowshoe)
    expected = ('[[' in result and ']]' in result)
    assert expected is True


def test_digraph_to_list_serialization_chain(chain):
    """Test digraph_to_list serializes the chain graph."""
    result = digraph_to_list(chain)
    expected = (isinstance(result, str), '[[-1' in result)
    assert expected == (True, True)


def test_digraph_to_list_invalid_input_no_fixture():
    """Test digraph_to_list raises TypeError for non-graph input."""
    result = None
    expected = TypeError
    with pytest.raises(expected):
        digraph_to_list("not a graph")
        result = "No exception"
    assert result is None


# =============================================================================
# get_nodes()
# =============================================================================

def test_get_nodes_state_category_snowshoe(snowshoe):
    """Test get_nodes returns state nodes for the snowshoe model."""
    result = get_nodes(snowshoe, 'state')
    expected = ['R', 'C', 'P']
    assert result == expected


def test_get_nodes_all_categories_snowshoe_chain(snowshoe, chain):
    """Test get_nodes returns all nodes for snowshoe and chain graphs."""
    result = (get_nodes(snowshoe, 'all'), get_nodes(chain, 'all'))
    expected = (['R', 'C', 'P'], ['1', '2', '3', '4', '5'])
    assert result == expected


def test_get_nodes_by_category_snowshoe_io(snowshoe_io):
    """Test get_nodes returns state, input, and output lists for snowshoe_io."""
    result = (
        get_nodes(snowshoe_io, 'state'),
        get_nodes(snowshoe_io, 'input'),
        get_nodes(snowshoe_io, 'output')
    )
    expected = (
        ['R', 'C', 'P'],
        ['Inp1', 'Inp2'],
        ['Out1', 'Out2']
    )
    assert result == expected


def test_get_nodes_with_labels_flag_snowshoe_io(snowshoe_io):
    """Test get_nodes respects labels flag for snowshoe_io categories."""
    result = (
        get_nodes(snowshoe_io, 'state', labels=True),
        get_nodes(snowshoe_io, 'input', labels=True),
        get_nodes(snowshoe_io, 'output', labels=True)
    )
    expected = (
        ['R', 'C', 'P'],
        ['Inp1', 'Inp2'],
        ['Out1', 'Out2']
    )
    assert result == expected


def test_get_nodes_invalid_category_snowshoe_io(snowshoe_io):
    """Test get_nodes returns empty list for invalid category on snowshoe_io."""
    result = get_nodes(snowshoe_io, 'invalid')
    expected = []
    assert result == expected


def test_get_nodes_invalid_input_type_no_fixture():
    """Test get_nodes raises TypeError when input is not a graph."""
    result = None
    expected = TypeError
    with pytest.raises(expected):
        get_nodes("not a graph", "state")
        result = "No exception"
    assert result is None

# =============================================================================
# get_weight()
# =============================================================================

def test_get_weight_ratios_inline_matrices():
    """Test get_weight computes weight ratios from net and absolute matrices."""
    net = sp.Matrix([[2, 0], [6, 8]])
    absolute = sp.Matrix([[4, 0], [12, 16]])
    result = get_weight(net, absolute)
    expected = sp.Matrix([
        [sp.Rational(1, 2), sp.nan],
        [sp.Rational(1, 2), sp.Rational(1, 2)]
    ])
    assert result == expected


def test_get_weight_custom_no_effect_inline_matrices():
    """Test get_weight uses provided no_effect value for zero totals."""
    net = sp.Matrix([[2, 0], [6, 8]])
    absolute = sp.Matrix([[4, 0], [12, 16]])
    w = get_weight(net, absolute, no_effect=sp.Integer(1))
    result = w[0, 1]
    expected = sp.Integer(1)
    assert result == expected


def test_get_weight_shape_mismatch_no_fixture():
    """Test get_weight raises ValueError for mismatched matrix shapes."""
    result = None
    expected = ValueError
    with pytest.raises(expected):
        get_weight(sp.Matrix([[1]]), sp.Matrix([[1, 2]]))
        result = "No exception"
    assert result is None


# =============================================================================
# get_positive()
# =============================================================================

def test_get_positive_single_entry_inline_matrices():
    """Test get_positive computes positive contribution for a matrix."""
    net = sp.Matrix([[2]])
    absolute = sp.Matrix([[6]])
    result = get_positive(net, absolute)
    expected = sp.Matrix([[4]])
    assert result == expected


def test_get_positive_feedback_terms_snowshoe(snowshoe):
    """Test get_positive extracts positive feedback totals for snowshoe."""
    net = net_feedback(snowshoe)
    absolute = absolute_feedback(snowshoe)
    result = get_positive(net, absolute)
    expected = sp.Matrix([
        [0],
        [0],
        [0],
        [0]
    ])
    assert result == expected


def test_get_positive_shape_mismatch_no_fixture():
    """Test get_positive raises ValueError for mismatched matrix shapes."""
    result = None
    expected = ValueError
    with pytest.raises(expected):
        get_positive(sp.Matrix([[1]]), sp.Matrix([[1, 2]]))
        result = "No exception"
    assert result is None


# =============================================================================
# get_negative()
# =============================================================================

def test_get_negative_single_entry_inline_matrices():
    """Test get_negative computes negative contribution for a matrix."""
    net = sp.Matrix([[2]])
    absolute = sp.Matrix([[6]])
    result = get_negative(net, absolute)
    expected = sp.Matrix([[2]])
    assert result == expected


def test_get_negative_feedback_terms_snowshoe(snowshoe):
    """Test get_negative extracts negative feedback totals for snowshoe."""
    net = net_feedback(snowshoe)
    absolute = absolute_feedback(snowshoe)
    result = get_negative(net, absolute)
    expected = sp.Matrix([
        [1],
        [2],
        [3],
        [2]
    ])
    assert result == expected


def test_get_negative_shape_mismatch_no_fixture():
    """Test get_negative raises ValueError for mismatched matrix shapes."""
    result = None
    expected = ValueError
    with pytest.raises(expected):
        get_negative(sp.Matrix([[1]]), sp.Matrix([[1, 2]]))
        result = "No exception"
    assert result is None


# =============================================================================
# sign_determinacy()
# =============================================================================

def test_sign_determinacy_matrix_type_inline_matrices():
    """Test sign_determinacy returns a matrix result for valid inputs."""
    result = sign_determinacy(sp.Matrix([[sp.Rational(1, 2)]]), sp.Matrix([[10]]))
    expected = isinstance(result, sp.Matrix)
    assert expected is True


def test_sign_determinacy_zero_weight_inline_matrices():
    """Test sign_determinacy handles zero weight with nan output."""
    result = sign_determinacy(sp.Matrix([[0]]), sp.Matrix([[10]]))
    expected = sp.Matrix([[sp.Rational(1, 2)]])
    assert result == expected


def test_sign_determinacy_unit_weight_inline_matrices():
    """Test sign_determinacy returns one for unit weight and positive total."""
    result = sign_determinacy(sp.Matrix([[1]]), sp.Matrix([[10]]))
    expected = sp.Matrix([[1]])
    assert result == expected


def test_sign_determinacy_negative_weight_inline_matrices():
    """Test sign_determinacy returns negative one for unit magnitude negative weight."""
    result = sign_determinacy(sp.Matrix([[-1]]), sp.Matrix([[10]]))
    expected = sp.Matrix([[-1]])
    assert result == expected


def test_sign_determinacy_zero_total_inline_matrices():
    """Test sign_determinacy returns nan when total terms are zero."""
    result = sign_determinacy(sp.Matrix([[sp.Rational(1, 2)]]), sp.Matrix([[0]]))
    expected = sp.Matrix([[sp.nan]])
    assert result == expected


@pytest.mark.parametrize("w,t,method", [
    (0.99, 50000, "average"),
    (0.99, 50000, "95_bound"),
    (0.7, 50, "average"),
    (0.7, 50, "95_bound"),
])
def test_sign_determinacy_bounds_parameterized_inline_scalars(w, t, method):
    """Test sign_determinacy respects bounds across methods and parameter sets."""
    result = sign_determinacy(sp.Matrix([[w]]), sp.Matrix([[t]]), method=method)
    expected = (0.5 <= float(result[0, 0]) <= 1.0)
    assert expected is True


def test_sign_determinacy_overflow_guard_inline_scalars():
    """Test sign_determinacy caps probability when exponent overflows."""
    w = sp.Matrix([[0.7]])
    t = sp.Matrix([[2000]])
    result = sign_determinacy(w, t, method="average")
    expected = sp.Float('0.999999')
    assert result[0, 0] == expected


def test_sign_determinacy_overflow_guard_95_bound_inline_scalars():
    """Test sign_determinacy caps probability at MAX_PROB for 95_bound method."""
    w = sp.Matrix([[0.7]])
    t = sp.Matrix([[2000]])
    result = sign_determinacy(w, t, method="95_bound")
    expected = sp.Float('0.999999')
    assert result[0, 0] == expected


def test_sign_determinacy_invalid_method_no_fixture():
    """Test sign_determinacy raises ValueError for an unsupported method name."""
    result = None
    expected = ValueError
    with pytest.raises(expected):
        sign_determinacy(sp.Matrix([[1]]), sp.Matrix([[1]]), method="invalid")
        result = "No exception"
    assert result is None


# =============================================================================
# _arrows()
# =============================================================================

def test_arrows_positive_edge_simple_ab_positive(simple_ab_positive):
    """Test _arrows renders a positive edge string for simple_ab_positive."""
    result = _arrows(simple_ab_positive, ['A', 'B'])
    expected = 'A $\\rightarrow$ B'
    assert result == expected


def test_arrows_negative_edge_simple_xy_negative(simple_xy_negative):
    """Test _arrows renders a negative edge string for simple_xy_negative."""
    result = _arrows(simple_xy_negative, ['X', 'Y'])
    expected = 'X $\\multimap$ Y'
    assert result == expected


def test_arrows_chain_path_chain(chain):
    """Test _arrows renders a chained path string for the chain graph."""
    result = _arrows(chain, ['1', '2', '3'])
    expected = '1 $\\rightarrow$ 2 $\\rightarrow$ 3'
    assert result == expected


# =============================================================================
# _sign_string()
# =============================================================================

def test_sign_string_positive_edge_simple_ab_positive(simple_ab_positive):
    """Test _sign_string returns '+' for a positive edge."""
    result = _sign_string(simple_ab_positive, ['A', 'B'])
    expected = '+'
    assert result == expected


def test_sign_string_negative_edge_simple_xy_negative(simple_xy_negative):
    """Test _sign_string returns '−' for a negative edge."""
    result = _sign_string(simple_xy_negative, ['X', 'Y'])
    expected = '\u2212'
    assert result == expected


def test_sign_string_chain_edges_chain(chain):
    """Test _sign_string returns signs for forward and backward chain edges."""
    result = (_sign_string(chain, ['1', '2']), _sign_string(chain, ['2', '1']))
    expected = ('+', '\u2212')
    assert result == expected


def test_sign_string_zero_product_no_fixture():
    G = nx.DiGraph()
    G.add_edge('A', 'B', sign=0)
    result = _sign_string(G, ['A', 'B'])
    expected = '0'
    assert result == expected


# =============================================================================
# _NodeSign
# =============================================================================

@pytest.mark.parametrize("s,node,sign", [('A:+', 'A', 1), ('B:-', 'B', -1), ('C:0', 'C', 0)])
def test_node_sign_from_str_node_sign_params(s, node, sign):
    """Test _NodeSign.from_str parses node sign strings."""
    ns = _NodeSign.from_str(s)
    result = (ns.node, ns.sign, ns.to_tuple())
    expected = (node, sign, (node, sign))
    assert result == expected


def test_node_sign_invalid_format_no_fixture():
    """Test _NodeSign.from_str raises ValueError for malformed input."""
    result = None
    expected = ValueError
    with pytest.raises(expected):
        _NodeSign.from_str('X:invalid')
        result = "No exception"
    assert result is None


# =============================================================================
# _parse_perturbations()
# =============================================================================

def test_parse_perturbations_single_value_snowshoe(snowshoe):
    """Test _parse_perturbations handles a single perturbation string."""
    G = snowshoe
    _, pt = _parse_perturbations(G, 'R:+')
    result = pt
    expected = ('R', 1)
    assert result == expected


def test_parse_perturbations_multiple_values_snowshoe(snowshoe):
    """Test _parse_perturbations handles multiple perturbation strings."""
    G = snowshoe
    G2, pt2 = _parse_perturbations(G, 'R:+, C:-')
    result = ('_P' in G2.nodes(), pt2)
    expected = (True, ('_P', 1))
    assert result == expected


# =============================================================================
# _parse_observations()
# =============================================================================

def test_parse_observations_empty_input_no_fixture():
    """Test _parse_observations returns empty tuple for blank input."""
    result = _parse_observations('')
    expected = tuple()
    assert result == expected


def test_parse_observations_multiple_values_no_fixture():
    """Test _parse_observations parses multiple observation tokens."""
    result = _parse_observations('A:+, B:-')
    expected = (('A', 1), ('B', -1))
    assert result == expected


# =============================================================================
# _random_sampler()
# =============================================================================

def test_random_sampler_invalid_distribution_no_fixture():
    """Test _random_sampler raises ValueError for invalid distribution."""
    with pytest.raises(ValueError, match="Invalid distribution"):
        _random_sampler("invalid_dist", 10)


# =============================================================================
# perm()
# =============================================================================

def test_perm_not_array_no_fixture():
    with pytest.raises(TypeError, match="NumPy array"):
        perm([[1, 2], [3, 4]])


def test_perm_non_square_no_fixture():
    with pytest.raises(ValueError, match="square"):
        perm(np.array([[1, 2, 3], [4, 5, 6]]))


def test_perm_contains_nan_no_fixture():
    with pytest.raises(ValueError, match="NaN"):
        perm(np.array([[1, np.nan], [3, 4]]))


def test_perm_empty_matrix_no_fixture():
    A = np.array([]).reshape(0, 0)
    result = perm(A)
    expected = 1.0
    assert result == expected


def test_perm_1x1_no_fixture():
    A = np.array([[5.0]])
    result = perm(A)
    expected = 5.0
    assert result == expected


def test_perm_2x2_no_fixture():
    A = np.array([[1, 2], [3, 4]])
    result = perm(A)
    expected = 1*4 + 2*3
    assert result == expected


def test_perm_3x3_no_fixture():
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    result = perm(A)
    expected = 450.0
    assert result == expected


def test_perm_4x4_bbfg_no_fixture():
    A = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], dtype=float)
    result = perm(A, method="bbfg")
    expected = 24.0
    assert result == expected


def test_perm_4x4_ryser_no_fixture():
    A = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], dtype=float)
    result = perm(A, method="ryser")
    expected = 24.0
    assert result == expected


def test_perm_methods_agree_no_fixture():
    A = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], dtype=float)
    result_bbfg = perm(A, method="bbfg")
    result_ryser = perm(A, method="ryser")
    result = abs(result_bbfg - result_ryser) < 1e-10
    expected = True
    assert result == expected


def test_perm_ryser_empty_no_fixture():
    A = np.array([], dtype=float).reshape(0, 0)
    result = _perm_ryser(A)
    expected = 1.0
    assert result == expected


def test_perm_bbfg_empty_no_fixture():
    A = np.array([], dtype=float).reshape(0, 0)
    result = _perm_bbfg(A)
    expected = 1.0
    assert result == expected


def test_perm_ryser_py_func_empty_no_fixture():
    A = np.array([], dtype=float).reshape(0, 0)
    result = _perm_ryser.py_func(A)
    expected = 1.0
    assert result == expected


def test_perm_bbfg_py_func_empty_no_fixture():
    A = np.array([], dtype=float).reshape(0, 0)
    result = _perm_bbfg.py_func(A)
    expected = 1.0
    assert result == expected


def test_perm_ryser_py_func_2x2_no_fixture():
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = _perm_ryser.py_func(A)
    expected = 1.0 * 4.0 + 2.0 * 3.0
    assert result == expected


def test_perm_bbfg_py_func_2x2_no_fixture():
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = _perm_bbfg.py_func(A)
    expected = 1.0 * 4.0 + 2.0 * 3.0
    assert result == expected


# =============================================================================
# get_dashed_alternatives()
# =============================================================================

def test_get_dashed_alternatives_no_dashed_edges_snowshoe(snowshoe):
    """Test get_dashed_alternatives returns original graph when no dashed edges exist."""
    result = get_dashed_alternatives(snowshoe)
    assert len(result) == 1
    assert result[0].number_of_edges() == snowshoe.number_of_edges()


def test_get_dashed_alternatives_combinations_true_snowshoe_dashed(snowshoe_dashed):
    """Test get_dashed_alternatives with snowshoe model having 3 dashed edges and combinations=True."""
    result = get_dashed_alternatives(snowshoe_dashed, combinations=True)
    assert len(result) == 8  # 2^3 combinations
    assert result[0].number_of_edges() == 6  # Base: all dashed removed
    assert result[7].number_of_edges() == 9  # All edges included


def test_get_dashed_alternatives_combinations_false_snowshoe_dashed(snowshoe_dashed):
    """Test get_dashed_alternatives with snowshoe model having 3 dashed edges and combinations=False."""
    result = get_dashed_alternatives(snowshoe_dashed, combinations=False)
    assert len(result) == 4 
    assert result[0].number_of_edges() == 6
    assert result[1].number_of_edges() == 7
    assert result[2].number_of_edges() == 7
    assert result[3].number_of_edges() == 7
    base_edges = set(result[0].edges())
    assert ('R', 'R') in base_edges
    assert ('R', 'C') in base_edges
    assert ('C', 'R') in base_edges
    assert ('R', 'P') not in base_edges


# =============================================================================
# Additional coverage tests
# =============================================================================

def test_parse_perturbations_empty_string(snowshoe):
    """Test _parse_perturbations raises ValueError for empty perturbation string."""
    from qmm.extensions.validation import marginal_likelihood
    with pytest.raises(ValueError, match="Perturbation string cannot be empty"):
        marginal_likelihood(snowshoe, perturb='   ', observe='R:+')


def test_parse_perturbations_invalid_node_multi(snowshoe):
    """Test _parse_perturbations raises ValueError for unknown node in multi-perturbation."""
    from qmm import simulations_table
    with pytest.raises(ValueError, match="Unknown perturbation node"):
        simulations_table(snowshoe, perturb='R:+, Invalid:+', observe='')
