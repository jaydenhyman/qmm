"""Tests for qmm.core.helper module."""

import pytest
import networkx as nx
import sympy as sp

from qmm import (
    list_to_digraph,
    digraph_to_list,
    get_nodes,
    get_weight,
    get_positive,
    get_negative,
    sign_determinacy,
)
from qmm.core.helper import _arrows, _sign_string, _NodeSign, _parse_perturbations, _parse_observations, _random_sampler
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
    expected = 'A → B'
    assert result == expected


def test_arrows_negative_edge_simple_xy_negative(simple_xy_negative):
    """Test _arrows renders a negative edge string for simple_xy_negative."""
    result = _arrows(simple_xy_negative, ['X', 'Y'])
    expected = 'X ⊸ Y'
    assert result == expected


def test_arrows_chain_path_chain(chain):
    """Test _arrows renders a chained path string for the chain graph."""
    result = _arrows(chain, ['1', '2', '3'])
    expected = '1 → 2 → 3'
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
