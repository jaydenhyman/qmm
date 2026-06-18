"""Tests for qmm.core.helper module."""

import pytest
import networkx as nx
import numpy as np
import sympy as sp

from qmm.core.helper import (
    list_to_digraph,
    load_digraph,
    digraph_to_list,
    get_nodes,
    get_weight,
    get_positive,
    get_negative,
    sign_determinacy,
    _arrows,
    _sign_string,
    _NodeSign,
    _parse_perturbations,
    _parse_observations,
    _check_signs,
    _check_direct_io_edges,
    _random_sampler,
    perm,
    _perm_ryser,
    _perm_bbfg,
    get_dashed_alternatives,
)
from qmm.core.stability import net_feedback, absolute_feedback


# =============================================================================
# list_to_digraph()
# =============================================================================

def test_list_to_digraph_signed_matrix_snowshoe(snowshoe):
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
    result = ('1' in simple_two_node.nodes(), '2' in simple_two_node.nodes())
    expected = (True, True)
    assert result == expected


def test_list_to_digraph_explicit_labels_no_fixture():
    A = [[-1, -1, 0], [1, 0, -1], [0, 1, -1]]
    G = list_to_digraph(A, ['R', 'C', 'P'])
    result = (list(G.nodes()), G['R']['R']['sign'], G['C']['R']['sign'], G['P']['C']['sign'])
    expected = (['R', 'C', 'P'], -1, -1, -1)
    assert result == expected


def test_list_to_digraph_chain_structure_chain(chain):
    result = (list(chain.nodes()), chain.number_of_edges())
    expected = (['1', '2', '3', '4', '5'], 13)
    assert result == expected


@pytest.mark.parametrize("bad_input", ["string", [[1, 2, 3], [4, 5, 6]]])
def test_list_to_digraph_invalid_input_bad_input(bad_input):
    result = None
    expected = ValueError
    with pytest.raises(expected):
        list_to_digraph(bad_input)
        result = "No exception"
    assert result is None


def test_list_to_digraph_label_mismatch_no_fixture():
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
    G = load_digraph("snowshoe")
    result = list(G.nodes())
    expected = ['R', 'C', 'P']
    assert result == expected


def test_load_digraph_snowshoe_edges():
    G = load_digraph("snowshoe")
    result = G.number_of_edges()
    expected = 6
    assert result == expected


def test_load_digraph_snowshoe_io_structure():
    G = load_digraph("snowshoe_io")
    result = (
        G.number_of_nodes(),
        G.number_of_edges(),
        G.nodes["R"]["category"],
        G.nodes["Inp1"]["category"],
        G.nodes["Out1"]["category"],
        G["Inp1"]["R"]["sign"],
        G["C"]["Out2"]["sign"],
    )
    expected = (7, 12, "state", "input", "output", 1, 1)
    assert result == expected





def test_load_digraph_invalid_model():
    with pytest.raises(ValueError, match="Model 'invalid' not found"):
        load_digraph("invalid")


# =============================================================================
# digraph_to_list()
# =============================================================================

def test_digraph_to_list_serialization_snowshoe(snowshoe):
    result = digraph_to_list(snowshoe)
    expected = ('[[' in result and ']]' in result)
    assert expected is True


def test_digraph_to_list_serialization_chain(chain):
    result = digraph_to_list(chain)
    expected = (isinstance(result, str), '[[-1' in result)
    assert expected == (True, True)


def test_digraph_to_list_invalid_input_no_fixture():
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
    result = get_nodes(snowshoe, 'state')
    expected = ['R', 'C', 'P']
    assert result == expected


def test_get_nodes_all_categories_snowshoe_chain(snowshoe, chain):
    result = (get_nodes(snowshoe, 'all'), get_nodes(chain, 'all'))
    expected = (['R', 'C', 'P'], ['1', '2', '3', '4', '5'])
    assert result == expected


def test_get_nodes_by_category_snowshoe_io(snowshoe_io):
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
    result = get_nodes(snowshoe_io, 'invalid')
    expected = []
    assert result == expected


def test_get_nodes_invalid_input_type_no_fixture():
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
    net = sp.Matrix([[2, 0], [6, 8]])
    absolute = sp.Matrix([[4, 0], [12, 16]])
    result = get_weight(net, absolute)
    expected = sp.Matrix([
        [sp.Rational(1, 2),            sp.nan],
        [sp.Rational(1, 2), sp.Rational(1, 2)]])
    assert result == expected


def test_get_weight_custom_no_effect_inline_matrices():
    net = sp.Matrix([[2, 0], [6, 8]])
    absolute = sp.Matrix([[4, 0], [12, 16]])
    w = get_weight(net, absolute, no_effect=sp.Integer(1))
    result = w[0, 1]
    expected = sp.Integer(1)
    assert result == expected


def test_get_weight_shape_mismatch_no_fixture():
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
    net = sp.Matrix([[2]])
    absolute = sp.Matrix([[6]])
    result = get_positive(net, absolute)
    expected = sp.Matrix([[4]])
    assert result == expected


def test_get_positive_feedback_terms_snowshoe(snowshoe):
    net = net_feedback(snowshoe)
    absolute = absolute_feedback(snowshoe)
    result = get_positive(net, absolute)
    expected = sp.Matrix([
        [0],
        [0],
        [0],
        [0]])
    assert result == expected


def test_get_positive_shape_mismatch_no_fixture():
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
    net = sp.Matrix([[2]])
    absolute = sp.Matrix([[6]])
    result = get_negative(net, absolute)
    expected = sp.Matrix([[2]])
    assert result == expected


def test_get_negative_feedback_terms_snowshoe(snowshoe):
    net = net_feedback(snowshoe)
    absolute = absolute_feedback(snowshoe)
    result = get_negative(net, absolute)
    expected = sp.Matrix([
        [1],
        [2],
        [3],
        [2]])
    assert result == expected


def test_get_negative_shape_mismatch_no_fixture():
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
    result = sign_determinacy(sp.Matrix([[sp.Rational(1, 2)]]), sp.Matrix([[10]]))
    expected = isinstance(result, sp.Matrix)
    assert expected is True


def test_sign_determinacy_zero_weight_inline_matrices():
    result = sign_determinacy(sp.Matrix([[0]]), sp.Matrix([[10]]))
    expected = sp.Matrix([[sp.Rational(1, 2)]])
    assert result == expected


def test_sign_determinacy_unit_weight_inline_matrices():
    result = sign_determinacy(sp.Matrix([[1]]), sp.Matrix([[10]]))
    expected = sp.Matrix([[1]])
    assert result == expected


def test_sign_determinacy_negative_weight_inline_matrices():
    result = sign_determinacy(sp.Matrix([[-1]]), sp.Matrix([[10]]))
    expected = sp.Matrix([[-1]])
    assert result == expected


def test_sign_determinacy_zero_total_inline_matrices():
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
    result = sign_determinacy(sp.Matrix([[w]]), sp.Matrix([[t]]), method=method)
    expected = (0.5 <= float(result[0, 0]) <= 1.0)
    assert expected is True


def test_sign_determinacy_overflow_guard_inline_scalars():
    w = sp.Matrix([[0.7]])
    t = sp.Matrix([[2000]])
    result = sign_determinacy(w, t, method="average")
    expected = sp.Float('0.999999')
    assert result[0, 0] == expected


def test_sign_determinacy_overflow_guard_95_bound_inline_scalars():
    w = sp.Matrix([[0.7]])
    t = sp.Matrix([[2000]])
    result = sign_determinacy(w, t, method="95_bound")
    expected = sp.Float('0.999999')
    assert result[0, 0] == expected


def test_sign_determinacy_invalid_method_no_fixture():
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
    result = _arrows(simple_ab_positive, ['A', 'B'])
    expected = 'A $\\rightarrow$ B'
    assert result == expected


def test_arrows_negative_edge_simple_xy_negative(simple_xy_negative):
    result = _arrows(simple_xy_negative, ['X', 'Y'])
    expected = 'X $\\multimap$ Y'
    assert result == expected


def test_arrows_chain_path_chain(chain):
    result = _arrows(chain, ['1', '2', '3'])
    expected = '1 $\\rightarrow$ 2 $\\rightarrow$ 3'
    assert result == expected


# =============================================================================
# _sign_string()
# =============================================================================

def test_sign_string_positive_edge_simple_ab_positive(simple_ab_positive):
    result = _sign_string(simple_ab_positive, ['A', 'B'])
    expected = '+'
    assert result == expected


def test_sign_string_negative_edge_simple_xy_negative(simple_xy_negative):
    result = _sign_string(simple_xy_negative, ['X', 'Y'])
    expected = '\u2212'
    assert result == expected


def test_sign_string_chain_edges_chain(chain):
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
    ns = _NodeSign.from_str(s)
    result = (ns.node, ns.sign, ns.to_tuple())
    expected = (node, sign, (node, sign))
    assert result == expected


def test_node_sign_invalid_format_no_fixture():
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
    G = snowshoe
    _, pt = _parse_perturbations(G, 'R:+')
    result = pt
    expected = ('R', 1)
    assert result == expected


def test_parse_perturbations_multiple_values_snowshoe(snowshoe):
    G = snowshoe
    G2, pt2 = _parse_perturbations(G, 'R:+, C:-')
    result = ('_P' in G2.nodes(), pt2)
    expected = (True, ('_P', 1))
    assert result == expected


def test_parse_perturbations_invalid_node_snowshoe(snowshoe):
    with pytest.raises(ValueError, match="Unknown perturbation node"):
        _parse_perturbations(snowshoe, "Invalid:+")


def test_check_signs_rejects_non_unit_no_fixture():
    G = nx.DiGraph()
    G.add_edge("A", "B", sign=2)
    with pytest.raises(ValueError, match="Edge signs must be"):
        _check_signs(G)


def test_check_signs_accepts_unit_no_fixture():
    G = nx.DiGraph()
    G.add_edge("A", "B", sign=1)
    G.add_edge("B", "A", sign=-1)
    assert _check_signs(G) is None


def test_check_direct_io_edges_rejects_feedthrough_no_fixture():
    G = nx.DiGraph()
    G.add_node("Inp", category="input")
    G.add_node("Out", category="output")
    G.add_edge("Inp", "Out", sign=1)
    with pytest.raises(ValueError, match="Direct input to output edge"):
        _check_direct_io_edges(G)


# =============================================================================
# _parse_observations()
# =============================================================================

def test_parse_observations_empty_input_no_fixture():
    result = _parse_observations('')
    expected = tuple()
    assert result == expected


def test_parse_observations_multiple_values_no_fixture():
    result = _parse_observations('A:+, B:-')
    expected = (('A', 1), ('B', -1))
    assert result == expected


# =============================================================================
# _random_sampler()
# =============================================================================

def test_random_sampler_invalid_distribution_no_fixture():
    with pytest.raises(ValueError, match="Invalid distribution"):
        _random_sampler("invalid_dist", 10)


def test_random_sampler_uniform_range_no_fixture():
    result = _random_sampler("uniform", 10)
    assert result.shape == (10,)
    assert (result >= 0).all() and (result <= 1).all()


def test_random_sampler_uniform_two_oom_range_no_fixture():
    result = _random_sampler("uniform_two_oom", 10)
    assert result.shape == (10,)
    assert (result >= 0.01).all() and (result <= 1).all()


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


def test_perm_overflow_raises_no_fixture():
    with pytest.raises(OverflowError):
        perm(np.ones((17, 17)))


def test_absolute_feedback_overflow_raises_no_fixture():
    M = [[(-1 if i == j else 1) for j in range(17)] for i in range(17)]
    with pytest.raises(OverflowError):
        absolute_feedback(list_to_digraph(M))


@pytest.mark.parametrize("seed", range(8))
def test_perm_decompose_equals_plain_random(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(4, 10))
    A = (rng.random((n, n)) < rng.uniform(0.25, 0.6)).astype(float)
    assert round(perm(A, decompose=True)) == round(perm(A, decompose=False))


# =============================================================================
# get_dashed_alternatives()
# =============================================================================

def test_get_dashed_alternatives_no_dashed_edges_snowshoe(snowshoe):
    result = get_dashed_alternatives(snowshoe)
    assert len(result) == 1
    assert result[0].number_of_edges() == snowshoe.number_of_edges()


def test_get_dashed_alternatives_combinations_true_snowshoe_dashed(snowshoe_dashed):
    result = get_dashed_alternatives(snowshoe_dashed, combinations=True)
    assert len(result) == 8
    assert result[0].number_of_edges() == 6
    assert result[7].number_of_edges() == 9


def test_get_dashed_alternatives_combinations_false_snowshoe_dashed(snowshoe_dashed):
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
    from qmm.extensions.validation import marginal_likelihood
    with pytest.raises(ValueError, match="Perturbation string cannot be empty"):
        marginal_likelihood(snowshoe, perturb='   ', observe='R:+')


def test_parse_perturbations_invalid_node_multi(snowshoe):
    from qmm.extensions.effects import simulations_table
    with pytest.raises(ValueError, match="Unknown perturbation node"):
        simulations_table(snowshoe, perturb='R:+, Invalid:+', observe='')
