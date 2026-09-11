"""Tests for qmm.core.helper module."""

import pytest
import networkx as nx
import numpy as np
import sympy as sp
from itertools import combinations, permutations
from math import comb, prod

from qmm.core.helper import (
    list_to_digraph,
    load_digraph,
    digraph_to_list,
    get_nodes,
    get_weight,
    get_positive,
    get_negative,
    sign_determinacy,
    _sign_string,
    _NodeSign,
    _parse_perturbations,
    _parse_observations,
    _check_signs,
    _check_direct_io_edges,
    _random_sampler,
    perm,
    get_dashed_alternatives,
)
from qmm.core.stability import net_feedback, absolute_feedback, absolute_determinants, _hurwitz_matrix
from qmm.core.press import absolute_feedback_matrix
from qmm.core.structure import create_matrix


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


def test_get_nodes_defaults_missing_category_to_state():
    G = nx.DiGraph()
    G.add_edge('A', 'B', sign=1)
    result = get_nodes(G, 'state')
    expected = ['A', 'B']
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


def test_sign_determinacy_hosack_2008_constants_inline_matrices():
    weights = sp.Matrix([[sp.Rational(1, 2), sp.Rational(-1, 4)], [sp.Rational(3, 4), sp.Rational(1, 10)]])
    totals = sp.Matrix([[4, 8], [20, 100]])
    result = [sign_determinacy(weights, totals, method=method) for method in ("average", "95_bound")]

    def average(w, t):
        x = 3.45962 * w + 0.03417 * w * t
        return max(0.5, np.exp(x) / (1 + np.exp(x)))

    def bound(w, t):
        x = 9.766 * w + 0.139 * w * t
        return max(0.5, np.exp(x) / (1253.992 + np.exp(x)))

    expected = [
        sp.Matrix(2, 2, lambda i, j: sp.sign(weights[i, j]) * fit(abs(float(weights[i, j])), float(totals[i, j])))
        for fit in (average, bound)
    ]
    assert all(
        abs(float(result[k][i, j]) - float(expected[k][i, j])) < 1e-12
        for k in range(2) for i in range(2) for j in range(2)
    )


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


@pytest.mark.parametrize('text', ['', '  ', ':+', ' : -'])
def test_node_sign_requires_a_node_name(text):
    with pytest.raises(ValueError, match='Missing node name'):
        _NodeSign.from_str(text)

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
    expected = (('R', 1),)
    assert result == expected


def test_parse_perturbations_multiple_values_snowshoe(snowshoe):
    G = snowshoe
    G2, pt2 = _parse_perturbations(G, 'R:+, C:-')
    result = ('_P' in G2.nodes(), pt2)
    expected = (False, (('R', 1), ('C', -1)))
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
# get_dashed_alternatives()
# =============================================================================

def test_get_dashed_alternatives_no_dashed_edges_snowshoe(snowshoe):
    result = get_dashed_alternatives(snowshoe)
    assert len(result) == 1
    assert result[0].number_of_edges() == snowshoe.number_of_edges()


def test_get_dashed_alternatives_combinations_true_snowshoe_dashed(snowshoe_dashed):
    result = get_dashed_alternatives(snowshoe_dashed, combinations=True, pair_reciprocal=False)
    assert len(result) == 8
    assert result[0].number_of_edges() == 6
    assert result[7].number_of_edges() == 9


def test_get_dashed_alternatives_combinations_false_snowshoe_dashed(snowshoe_dashed):
    result = get_dashed_alternatives(snowshoe_dashed, combinations=False, pair_reciprocal=False)
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


def test_perm_not_array_no_fixture():
    with pytest.raises(TypeError, match="NumPy array"):
        perm([[1, 2], [3, 4]])


def test_perm_non_square_no_fixture():
    with pytest.raises(ValueError, match="square"):
        perm(np.array([[1, 2, 3], [4, 5, 6]]))


def test_perm_contains_nan_no_fixture():
    with pytest.raises(ValueError, match="NaN"):
        perm(np.array([[1, np.nan], [3, 4]]))


@pytest.mark.parametrize("n", [63, 64, 65, 127, 128, 129])
def test_perm_arbitrary_width_masks(n):
    A = np.eye(n, dtype=int)
    assert perm(A) == 1
    assert perm(A, levels=True) == [comb(n, k) for k in range(n + 1)]
    for source in [0, n // 2, n - 1]:
        assert perm(A, source=source) == [int(i == source) for i in range(n)]


@pytest.mark.parametrize("n", [64, 65, 129])
def test_perm_large_directed_cycle(n):
    columns = np.roll(np.arange(n), 1)
    A = np.eye(n, dtype=int)[columns]
    assert perm(A) == 1
    assert perm(A, levels=True) == [1] + [0] * (n - 1) + [1]
    for source in [0, n // 2, n - 1]:
        assert perm(A, source=source) == [int(i == columns[source]) for i in range(n)]


def test_perm_large_signed_tridiagonal():
    n = 65
    A = 2 * np.eye(n, dtype=int) + np.eye(n, k=1, dtype=int) - np.eye(n, k=-1, dtype=int)
    assert perm(A) == n + 1
    assert perm(A, levels=True) == [comb(2 * n - k + 1, k) for k in range(n + 1)]
    for source in [0, 32, 64]:
        expected = [(-1) ** max(target - source, 0) * (min(source, target) + 1) * (n - max(source, target))
                    for target in range(n)]
        assert perm(A, source=source) == expected
    rng = np.random.default_rng(65)
    assert perm(A[np.ix_(rng.permutation(n), rng.permutation(n))]) == n + 1


@pytest.mark.parametrize("source", [True, np.bool_(False), -1, 2, 0.0, [0], "0"])
def test_perm_rejects_invalid_source(source):
    with pytest.raises(ValueError, match="Source"):
        perm(np.eye(2, dtype=int), source=source)


def test_perm_rejects_combined_modes():
    with pytest.raises(ValueError, match="Source and levels"):
        perm(np.eye(2, dtype=int), source=0, levels=True)


@pytest.mark.parametrize("A", [np.array(1), np.array([1, 2]), np.zeros((1, 1, 1))])
def test_perm_rejects_wrong_dimensions(A):
    with pytest.raises(ValueError, match="square"):
        perm(A)


@pytest.mark.parametrize("value", [np.inf, -np.inf, complex(1, np.nan)])
def test_perm_rejects_nonfinite_entries(value):
    with pytest.raises(ValueError, match="NaNs or infinities"):
        perm(np.array([[value]]))


def test_perm_normalizes_matrix_subclass():
    with pytest.warns(PendingDeprecationWarning):
        A = np.matrix([[1, 2], [3, 4]])
    assert perm(A) == 10
    assert perm(A, source=np.int64(0)) == [4, 3]
    assert perm(A, levels=True) == [1, 5, 10]


def test_perm_signed_and_huge_integer_independent_oracle():
    rng = np.random.default_rng(9187)
    for n in range(6):
        A = rng.integers(-2, 3, size=(n, n)).astype(object) * (10**25 + 7)

        def direct_permanent(rows, columns):
            return sum(prod(A[i, j] for i, j in zip(rows, assignment))
                       for assignment in permutations(columns))

        assert perm(A) == direct_permanent(range(n), range(n))
        assert perm(A, levels=True) == [
            sum(direct_permanent(subset, subset) for subset in combinations(range(n), k))
            for k in range(n + 1)
        ]
        for source in range(n):
            assert perm(A, source=source) == [
                direct_permanent([i for i in range(n) if i != source], [j for j in range(n) if j != target])
                for target in range(n)
            ]


def test_perm_small_matrices_no_fixture():
    cases = [
        (np.array([]).reshape(0, 0), 1),
        (np.array([[5.0]]), 5),
        (np.array([[1, 2], [3, 4]], dtype=float), 10),
        (np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float), 450),
        (np.ones((4, 4)), 24),
    ]
    for A, expected in cases:
        assert perm(A) == expected


def test_perm_exact_beyond_float_precision_no_fixture():
    n = 19
    assert perm(np.ones((n, n), dtype=int)) == __import__("math").factorial(n)
    A = np.array([[1 if abs(i - j) <= 1 else 0 for j in range(40)] for i in range(40)])
    fib = [0, 1]
    for _ in range(41):
        fib.append(fib[-1] + fib[-2])
    assert perm(A) == fib[41]


@pytest.mark.parametrize('scalar', [np.int64, np.uint64])
def test_perm_object_array_numpy_integers_remain_exact(scalar):
    value = 2**62 + 1
    A = np.zeros((3, 3), dtype=object)
    for i in range(3):
        A[i, i] = scalar(value)
    assert perm(A) == value**3
    assert perm(A, source=1) == [0, value**2, 0]
    assert perm(A, levels=True) == [1, 3 * value, 3 * value**2, value**3]


def test_perm_object_array_numpy_booleans_remain_exact():
    A = np.kron(np.eye(63, dtype=bool), np.ones((2, 2), dtype=bool)).astype(object)
    for i, j in zip(*A.nonzero()):
        A[i, j] = np.bool_(True)
    assert perm(A) == 2**63


def test_perm_matches_sympy_permanent_no_fixture():
    rng = np.random.default_rng(7)
    for _ in range(20):
        n = int(rng.integers(1, 7))
        A = rng.integers(0, 3, size=(n, n)) * (rng.random((n, n)) < 0.6)
        assert perm(A) == int(sp.Matrix(A.tolist()).per())


@pytest.mark.parametrize("model", ["snowshoe", "snowshoe_rp", "chain", "mesocosm"])
def test_perm_source_matches_absolute_feedback_matrix(model):
    G = load_digraph(model)
    A = sp.matrix2numpy(create_matrix(G, form="binary"), dtype=int)
    n = A.shape[0]
    result = sp.Matrix([perm(A, source=j) for j in range(n)]).T
    assert result == absolute_feedback_matrix(G)


@pytest.mark.parametrize("model", ["snowshoe", "chain", "mesocosm"])
def test_perm_source_matches_absolute_feedback_matrix_perturb(model):
    G = load_digraph(model)
    A = sp.matrix2numpy(create_matrix(G, form="binary"), dtype=int)
    for j, node in enumerate(get_nodes(G, "state")):
        assert sp.Matrix(perm(A, source=j)) == absolute_feedback_matrix(G, perturb=node)


@pytest.mark.parametrize("model", ["snowshoe", "snowshoe_rp", "chain", "mesocosm"])
def test_perm_levels_matches_absolute_feedback(model):
    G = load_digraph(model)
    A = np.abs(sp.matrix2numpy(create_matrix(G, form="signed"), dtype=int))
    assert sp.Matrix(perm(A, levels=True)) == absolute_feedback(G)


@pytest.mark.parametrize("model", ["snowshoe", "chain", "mesocosm"])
def test_perm_levels_matches_absolute_feedback_polynomial(model):
    G = load_digraph(model)
    A = np.abs(sp.matrix2numpy(create_matrix(G, form="signed"), dtype=int))
    assert sp.Matrix(perm(A, levels=True)) == absolute_feedback(G, method="polynomial")


@pytest.mark.parametrize("model", ["snowshoe", "chain", "mesocosm"])
def test_perm_matches_absolute_determinants(model):
    G = load_digraph(model)
    n = absolute_feedback(G).shape[0] - 1
    h = _hurwitz_matrix(absolute_feedback(G), n)
    result = [sp.Integer(1)]
    for k in range(1, n + 1):
        H = np.array([[abs(int(x)) for x in row] for row in h[:k, :k].tolist()], dtype=object)
        result.append(sp.Integer(perm(H)))
    assert sp.Matrix(result) == absolute_determinants(G)


def test_perm_levels_matches_perm_of_principal_submatrices_no_fixture():
    from itertools import combinations
    rng = np.random.default_rng(13)
    for _ in range(10):
        n = int(rng.integers(1, 7))
        A = (rng.random((n, n)) < 0.5).astype(float)
        levels = perm(A, levels=True)
        for k in range(n + 1):
            expected = sum(int(sp.Matrix(A[np.ix_(c, c)].tolist()).per()) if k else 1 for c in combinations(range(n), k))
            assert levels[k] == expected


def test_perm_source_matches_perm_of_minors_no_fixture():
    rng = np.random.default_rng(17)
    for _ in range(10):
        n = int(rng.integers(2, 8))
        A = (rng.random((n, n)) < 0.4).astype(float)
        for j in range(n):
            minors = perm(A, source=j)
            for i in range(n):
                minor = np.delete(np.delete(A, j, 0), i, 1)
                expected = int(sp.Matrix(minor.tolist()).per()) if minor.size else 1
                assert minors[i] == expected


@pytest.mark.parametrize("value", [1.8, -0.5, np.nan, np.inf])
def test_list_to_digraph_rejects_non_sign_entries(value):
    with pytest.raises(ValueError, match="entries"):
        list_to_digraph([[value]])


def test_list_to_digraph_rejects_duplicate_ids():
    with pytest.raises(ValueError, match="unique"):
        list_to_digraph([[-1, 1], [-1, -1]], ids=["A", "A"])


def test_parse_perturbations_preserves_graph_with_existing_press_node_name():
    G = list_to_digraph([[-1, 0], [1, -1]], ids=["_P", "B"])
    modified, press = _parse_perturbations(G, "_P:+, B:-")
    assert modified is G
    assert press == (("_P", 1), ("B", -1))
    assert modified.nodes["_P"]["category"] == "state"
    assert modified["_P"]["_P"]["sign"] == -1
    assert "_P_" not in G


def test_get_dashed_alternatives_keeps_reciprocal_dashed_edges_together_snowshoe_dashed(snowshoe_dashed):
    variants = get_dashed_alternatives(snowshoe_dashed)
    result = (len(variants), sum(g.has_edge('R', 'P') == g.has_edge('P', 'R') for g in variants), len(get_dashed_alternatives(snowshoe_dashed, combinations=False)))
    expected = (4, 4, 3)
    assert result == expected


def test_get_dashed_alternatives_variants_carry_no_dashes_snowshoe_dashed(snowshoe_dashed):
    result = any(d.get("dashes") for g in get_dashed_alternatives(snowshoe_dashed) for _, _, d in g.edges(data=True))
    expected = False
    assert result == expected
