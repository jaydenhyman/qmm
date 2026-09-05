"""Utility functions for model development and analysis."""

import numpy as np
import sympy as sp
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, maximum_bipartite_matching
from typing import List, Union, Dict, Any, Optional, Tuple, Literal
from dataclasses import dataclass
from numba import jit

def list_to_digraph(matrix: Union[List[List[int]], np.ndarray], ids: Optional[List[str]] = None) -> nx.DiGraph:
    """Convert an adjacency matrix to a directed graph.

    Args:
        matrix: A square matrix (list of lists or numpy array) representing the adjacency matrix.
            Non-zero values indicate edges, where the value represents the sign of the edge.
        ids: Optional list of node identifiers. If None, nodes will be labeled 1 to n.

    Returns:
        nx.DiGraph: A NetworkX directed graph with signed edges.

    Examples:
        ```python
        from qmm import list_to_digraph
        G = list_to_digraph([[-1, -1, 0], [1, 0, -1], [1, 1, -1]])
        list(G.nodes())
        # ['1', '2', '3']

        list(G.edges(data='sign'))
        # [('1', '1', -1), ('1', '2', 1), ('1', '3', 1), ('2', '1', -1), ('2', '3', 1), ('3', '2', -1), ('3', '3', -1)]
        ```
    """
    if not isinstance(matrix, (list, np.ndarray)):
        raise ValueError("Input must be a list of lists or a numpy array")
    if isinstance(matrix, list):
        matrix = np.array(matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input must be a square matrix")
    G = nx.DiGraph()
    n = matrix.shape[0]
    if ids is None:
        node_ids = [str(i) for i in range(1, n + 1)]
    else:
        if len(ids) != n:
            raise ValueError("Number of ids must match matrix dimensions")
        node_ids = ids
    G.add_nodes_from(node_ids)
    for i in range(n):
        for j in range(n):
            if matrix[i][j] != 0:
                G.add_edge(node_ids[j], node_ids[i], sign=int(matrix[i][j]))
    _check_signs(G)
    nx.set_node_attributes(G, "state", "category")
    nx.freeze(G)
    return G


def load_digraph(model: str) -> nx.DiGraph:
    """Load a built-in example model as a signed directed graph.

    Args:
        model: Name of the built-in model to load. Available models:
            - "snowshoe": Simple 3-node predator-prey model (R, C, P)
            - "snowshoe_rp": Snowshoe model with an added positive R->P link
            - "snowshoe_io": Snowshoe_rp model with input/output nodes
            - "chain": 5-node linear chain with self-effects
            - "mesocosm": 8-node complex ecosystem model

    Returns:
        nx.DiGraph: A NetworkX directed graph with signed edges.

    Raises:
        ValueError: If model name is not recognized.

    Examples:
        ```python
        from qmm import load_digraph
        G = load_digraph("snowshoe_rp")
        list(G.nodes())
        # ['R', 'C', 'P']

        list(G.edges(data='sign'))
        # [('R', 'R', -1), ('R', 'C', 1), ('R', 'P', 1), ('C', 'R', -1), ('C', 'P', 1), ('P', 'C', -1), ('P', 'P', -1)]
        ```
    """
    models = {
        "snowshoe": {
            "matrix": [[-1, -1, 0], [1, 0, -1], [0, 1, -1]],
            "labels": ['R', 'C', 'P']
        },
        "snowshoe_rp": {
            "matrix": [[-1, -1, 0], [1, 0, -1], [1, 1, -1]],
            "labels": ['R', 'C', 'P']
        },
        "chain": {
            "matrix": [[-1, -1, 0, 0, 0], [1, -1, -1, 0, 0], [0, 1, -1, -1, 0], [0, 0, 1, -1, -1], [0, 0, 0, 1, -1]],
            "labels": ['1', '2', '3', '4', '5']
        },
        "mesocosm": {
            "matrix": [
                [-1, -1, -1, -1, 0, 0, 0, 0],
                [1, 0, 0, 0, -1, -1, 0, 0],
                [1, 0, 0, 0, 0, -1, 0, 0],
                [1, 0, 0, -1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, -1, -1],
                [0, 1, 1, 0, 0, 0, 0, -1],
                [0, 0, 0, 0, 1, 0, 0, -1],
                [0, 0, 0, 0, 1, 1, 1, -1],
            ],
            "labels": ['P', 'A1', 'A2', 'AP', 'H1', 'H2', 'C1', 'C2']
        }
    }

    if model == "snowshoe_io":
        G = nx.DiGraph()
        for node in ['R', 'C', 'P']:
            G.add_node(node, category='state')
        for node in ['Inp1', 'Inp2']:
            G.add_node(node, category='input')
        for node in ['Out1', 'Out2']:
            G.add_node(node, category='output')
        edges = [
            ('R', 'R', -1),
            ('R', 'C', 1),
            ('C', 'R', -1),
            ('C', 'P', 1),
            ('P', 'C', -1),
            ('P', 'P', -1),
            ('Inp1', 'R', 1),
            ('Inp1', 'C', -1),
            ('Inp2', 'P', -1),
            ('C', 'Out1', -1),
            ('C', 'Out2', 1),
            ('P', 'Out1', 1),
        ]
        for source, target, sign in edges:
            G.add_edge(source, target, sign=sign)
        nx.freeze(G)
        return G

    if model not in models:
        available_models = list(models.keys()) + ["snowshoe_io"]
        available = ', '.join(f'"{m}"' for m in sorted(available_models))
        raise ValueError(f"Model '{model}' not found. Available models: {available}")

    m = models[model]
    G = list_to_digraph(m["matrix"], m["labels"])
    return G


def digraph_to_list(G: nx.DiGraph) -> str:
    """Convert a directed graph to an adjacency matrix string representation.

    Args:
        G: A NetworkX directed graph with signed edges.

    Returns:
        str: String representation of the adjacency matrix.

    Examples:
        ```python
        from qmm import load_digraph, digraph_to_list
        digraph_to_list(load_digraph("snowshoe_rp"))
        # '[[0, -1, 1], [1, -1, 1], [-1, 0, -1]]'
        ```
    """
    if not isinstance(G, nx.DiGraph):
        raise TypeError("Input must be a networkx.DiGraph.")
    n = G.number_of_nodes()
    nodes = sorted(G.nodes())
    node_to_index = {node: i for i, node in enumerate(nodes)}
    matrix = [[0 for _ in range(n)] for _ in range(n)]
    for source, target, data in G.edges(data=True):
        i, j = node_to_index[source], node_to_index[target]
        sign = data.get("sign", 1)
        matrix[j][i] = sign
    return str(matrix)

def get_nodes(
    G: nx.DiGraph,
    node_type: Literal["state", "input", "output", "all"] = "state",
    labels: bool = False,
) -> List[Union[str, Dict[str, Any]]]:
    """Get nodes of a specific type from a directed graph.

    Args:
        G: NetworkX directed graph to extract nodes from.
        node_type: Type of nodes to extract ('state', 'input', 'output', or 'all').
        labels: If True, return node labels instead of node ids.

    Returns:
        List of node identifiers or dictionaries containing node data.

    Examples:
        ```python
        from qmm import load_digraph, get_nodes
        get_nodes(load_digraph("snowshoe_rp"), "state")
        # ['R', 'C', 'P']
        ```
    """
    if not isinstance(G, nx.DiGraph):
        raise TypeError("Input must be a networkx.DiGraph.")

    if node_type == "all":
        return [n if not labels else d.get("label", n) for n, d in G.nodes(data=True)]
    else:
        return [n if not labels else d.get("label", n) for n, d in G.nodes(data=True) if d.get("category") == node_type]

def get_weight(net: sp.Matrix, absolute: sp.Matrix, no_effect: Union[sp.Basic, float] = sp.nan) -> sp.Matrix:
    """Calculate weight matrix by dividing net effect by absolute effect.

    Args:
        net: Matrix of net terms.
        absolute: Matrix of absolute terms.
        no_effect: Value to use when absolute terms is 0 (default: sympy.nan).

    Returns:
        sympy.Matrix: Matrix of weights.

    Examples:
        ```python
        import sympy as sp
        from qmm import get_weight
        net = sp.Matrix([[2, -2], [1, 0]])
        absolute = sp.Matrix([[4, 2], [1, 0]])
        get_weight(net, absolute)
        # Matrix([
        # [1/2,  -1],
        # [  1, nan]])
        ```
    """
    if net.shape != absolute.shape:
        raise ValueError("Matrices must have the same shape")
    result = sp.zeros(*net.shape)
    for i in range(net.shape[0]):
        for j in range(net.shape[1]):
            if absolute[i, j] == 0:
                result[i, j] = no_effect
            else:
                result[i, j] = net[i, j] / absolute[i, j]
    return result

def get_positive(net: sp.Matrix, absolute: sp.Matrix) -> sp.Matrix:
    """Calculate matrix of positive terms.

    Args:
        net: Matrix of net terms.
        absolute: Matrix of absolute terms.

    Returns:
        sympy.Matrix: Matrix of positive terms.

    Examples:
        ```python
        import sympy as sp
        from qmm import get_positive
        net = sp.Matrix([[3, -2], [1, 0]])
        absolute = sp.Matrix([[4, 2], [1, 0]])
        get_positive(net, absolute)
        # Matrix([
        # [3, 0],
        # [1, 0]])
        ```
    """
    if net.shape != absolute.shape:
        raise ValueError("Matrices must have the same shape")
    result = sp.zeros(*net.shape)
    for i in range(net.shape[0]):
        for j in range(net.shape[1]):
            result[i, j] = (net[i, j] + absolute[i, j]) // 2
    return result

def get_negative(net: sp.Matrix, absolute: sp.Matrix) -> sp.Matrix:
    """Calculate matrix of negative terms.

    Args:
        net: Matrix of net terms.
        absolute: Matrix of absolute terms.

    Returns:
        sympy.Matrix: Matrix of negative terms.

    Examples:
        ```python
        import sympy as sp
        from qmm import get_negative
        net = sp.Matrix([[3, -2], [1, 0]])
        absolute = sp.Matrix([[4, 2], [1, 0]])
        get_negative(net, absolute)
        # Matrix([
        # [0, 2],
        # [0, 0]])
        ```
    """
    if net.shape != absolute.shape:
        raise ValueError("Matrices must have the same shape")
    result = sp.zeros(*net.shape)
    for i in range(net.shape[0]):
        for j in range(net.shape[1]):
            result[i, j] = (absolute[i, j] - net[i, j]) // 2
    return result

def sign_determinacy(
    wmat: sp.Matrix,
    tmat: sp.Matrix,
    method: Literal["average", "95_bound"] = "average",
) -> sp.Matrix:
    """Calculate sign determinacy matrix from prediction weights.

    Args:
        wmat: Matrix of prediction weights.
        tmat: Matrix of absolute feedback.
        method: Method to use for probability calculation ('average' or '95_bound').

    Returns:
        sympy.Matrix: Probability of sign determinacy.

    References:
        - Hosack, G.R., Hayes, K.R., Dambacher, J.M. (2008). Assessing Model Structure Uncertainty Through an Analysis of System Feedback and Bayesian Networks. Ecological Applications 18, 1070–1082.

    Examples:
        ```python
        from qmm import load_digraph, weighted_predictions_matrix, absolute_feedback_matrix, sign_determinacy
        G = load_digraph("snowshoe_rp")
        wmat = weighted_predictions_matrix(G)
        tmat = absolute_feedback_matrix(G)
        sign_determinacy(wmat, tmat, method='average')
        # Matrix([
        # [  1,  -1,  1],
        # [1/2,   1, -1],
        # [  1, 1/2,  1]])
        ```
    """

    MAX_PROB = sp.Float('0.999999')
    
    def compute_prob(w, t, method):
        if t == sp.Integer(0):
            return sp.nan
        return compute_prob_average(w, t) if method == "average" else compute_prob_95_bound(w, t)
    
    def compute_prob_average(w, t):
        bw = 3.45962
        bwt = 0.03417
        w_float = float(w)
        t_float = float(t)
        exponent = bw * w_float + bwt * w_float * t_float
        
        if exponent > 700:
            return MAX_PROB
            
        prob_float = np.exp(exponent) / (1 + np.exp(exponent))
        prob = sp.Float(prob_float)
        
        prob = max(sp.Rational(1, 2), prob)

        if prob >= MAX_PROB:
            prob = MAX_PROB
        return prob
    
    def compute_prob_95_bound(w, t):
        bw = 9.766
        bwt = 0.139
        w_float = float(w)
        t_float = float(t)
        exponent = bw * w_float + bwt * w_float * t_float
        
        if exponent > 700:
            return MAX_PROB
            
        prob_float = np.exp(exponent) / (1253.992 + np.exp(exponent))
        prob = sp.Float(prob_float)
        
        prob = max(sp.Rational(1, 2), prob)
        if prob >= MAX_PROB:
            prob = MAX_PROB
        return prob
    
    if method not in ["average", "95_bound"]:
        raise ValueError("Invalid method. Choose 'average' or '95_bound'.")
    rows, cols = wmat.shape
    def calc_prob(i, j):
        w, t = wmat[i, j], tmat[i, j]
        if w.is_zero:
            return sp.Rational(1, 2)
        if sp.Abs(w) == sp.Integer(1):
            return sp.sign(w) * sp.Integer(1)
        prob = compute_prob(sp.Abs(w), t, method)
        return sp.sign(w) * prob if prob is not None else sp.nan
    
    pmat = sp.Matrix(rows, cols, lambda i, j: calc_prob(i, j))
    return pmat


def _node_name(G: nx.DiGraph, node: str, labels: bool = False) -> str:
    """The node's label when labels is True and it has one, otherwise its id."""
    return str(G.nodes[node].get("label") or node) if labels else str(node)


def _arrows(G: nx.DiGraph, path: List[str], labels: bool = False) -> str:
    """Write a path as nodes joined by $\\rightarrow$ (positive link) or $\\multimap$ (negative link)."""
    parts = []
    for from_node, to_node in zip(path, path[1:]):
        arrow = "$\\rightarrow$" if G[from_node][to_node].get("sign", 1) > 0 else "$\\multimap$"
        parts.append(f"{_node_name(G, from_node, labels)} {arrow}")
    parts.append(_node_name(G, path[-1], labels))
    return " ".join(parts)


def _sign_string(G: nx.DiGraph, path: List[str]) -> str:
    product = 1
    for from_node, to_node in zip(path, path[1:]):
        product *= G[from_node][to_node].get("sign", 1)
    if product > 0:
        return "+"
    elif product < 0:
        return "\u2212"
    else:
        return "0"


@dataclass(frozen=True)
class _NodeSign:
    node: str
    sign: int
    
    @classmethod
    def from_str(cls, s: str) -> '_NodeSign':
        """Create from 'B:+', 'B: -', 'B:0', or a bare 'B' (sign defaults to +)."""
        node, sep, sign = s.strip().partition(":")
        node = node.strip()
        sign = sign.strip() if sep else "+"
        if not node:
            raise ValueError(f"Missing node name in '{s}'")
        if sign not in ["+", "-", "0"]:
            raise ValueError(f"Sign must be +, -, or 0, got '{sign}' in '{s}'")
        return cls(node, 1 if sign == "+" else (-1 if sign == "-" else 0))
    
    def to_tuple(self) -> tuple[str, int]:
        """Convert to tuple format for internal use"""
        return (self.node, self.sign)

def _parse_perturbations(G: nx.DiGraph, perturb: str) -> Tuple[nx.DiGraph, Tuple[str, int]]:
    perturbations = [p.strip() for p in perturb.split(',') if p.strip()]
    if not perturbations:
        raise ValueError("Perturbation string cannot be empty.")
    valid_nodes = set(get_nodes(G, "all"))
    if len(perturbations) > 1:
        G_mod = G.copy()
        G_mod.add_node('_P', category='input')
        for p in perturbations:
            ns = _NodeSign.from_str(p)
            if ns.node not in valid_nodes:
                raise ValueError(f"Unknown perturbation node: {ns.node}")
            G_mod.add_edge('_P', ns.node, sign=ns.sign)
        nx.freeze(G_mod)
        return G_mod, ('_P', 1)
    ns = _NodeSign.from_str(perturbations[0])
    if ns.node not in valid_nodes:
        raise ValueError(f"Unknown perturbation node: {ns.node}")
    return G, ns.to_tuple()

def _parse_observations(s: str) -> Tuple[Tuple[str, int], ...]:
    if not s:
        return tuple()
    return tuple(_NodeSign.from_str(obs.strip()).to_tuple() 
                for obs in s.split(","))


def _check_signs(G: nx.DiGraph) -> None:
    """Raise unless every edge sign is +1 or -1."""
    bad = [(u, v, d.get("sign")) for u, v, d in G.edges(data=True) if d.get("sign", 1) not in (-1, 1)]
    if bad:
        raise ValueError(f"Edge signs must be +1 or -1: {bad}")


def _edge_prefix(G: nx.DiGraph, source: str, target: str) -> str:
    """Edge symbol prefix d/b/c/a by endpoint category; shared so create_matrix,
    edges_table and get_paths always agree."""
    src_in = G.nodes[source].get("category", "state") == "input"
    tgt_out = G.nodes[target].get("category", "state") == "output"
    return "d" if src_in and tgt_out else "b" if src_in else "c" if tgt_out else "a"


def _check_direct_io_edges(G: nx.DiGraph) -> None:
    """Raise on any direct input->output edge."""
    inputs, outputs = get_nodes(G, "input"), get_nodes(G, "output")
    for inp in inputs:
        for out in outputs:
            if G.has_edge(inp, out):
                raise ValueError(f"Direct input to output edge ({inp} to {out}) not supported.")


def perm(
    A: np.ndarray, method: Literal["bbfg", "ryser"] = "bbfg", decompose: bool = True
) -> float:
    """Compute the permanent of a square matrix.

    The permanent is similar to the determinant but uses only addition
    (no sign alternation). This implementation is based on the algorithms
    from thewalrus library (https://github.com/XanaduAI/thewalrus).

    Args:
        A: A square numpy array (float or complex).
        method: Algorithm to use - "bbfg" for BBFG formula (default, faster)
                or "ryser" for Ryser formula. Any other value uses Ryser.
        decompose: Dulmage-Mendelsohn decomposition (default True) for sparse matrices.

    Returns:
        The permanent of matrix A.

    Raises:
        TypeError: If input is not a numpy array.
        ValueError: If matrix is not square or contains NaNs.

    References:
        - Ryser, H.J. (1963). Combinatorial Mathematics.
        - Glynn, D.G. (2010). The permanent of a square matrix. European Journal of Combinatorics 31, 1887–1891.

    Examples:
        ```python
        import numpy as np
        from qmm.core.helper import perm
        perm(np.array([[1, 2], [3, 4]]), method='bbfg')
        # 10
        ```
    """
    if not isinstance(A, np.ndarray):
        raise TypeError("Input matrix must be a NumPy array.")

    matshape = A.shape
    if matshape[0] != matshape[1]:
        raise ValueError("Input matrix must be square.")
    if np.isnan(A).any():
        raise ValueError("Input matrix must not contain NaNs.")

    if matshape[0] == 0:
        return A.dtype.type(1.0)
    if matshape[0] == 1:
        return A[0, 0]
    if matshape[0] == 2:
        return A[0, 0] * A[1, 1] + A[0, 1] * A[1, 0]
    if matshape[0] == 3:
        return (
            A[0, 2] * A[1, 1] * A[2, 0]
            + A[0, 1] * A[1, 2] * A[2, 0]
            + A[0, 2] * A[1, 0] * A[2, 1]
            + A[0, 0] * A[1, 2] * A[2, 1]
            + A[0, 1] * A[1, 0] * A[2, 2]
            + A[0, 0] * A[1, 1] * A[2, 2]
        )

    overflow = bool(np.prod(np.abs(A).sum(axis=0, dtype=float)) > 2.0**53)
    if overflow and int((A != 0).sum()) <= 8 * matshape[0] and not np.mod(A, 1).any():
        return _perm_int(A)

    if decompose:
        S = csr_matrix(A != 0)
        col_of = maximum_bipartite_matching(S, perm_type="column")
        if (col_of < 0).any():
            return 0
        nb, labels = connected_components(S[:, col_of], connection="strong")
        if nb > 1:
            result = 1
            for b in range(nb):
                rk = np.flatnonzero(labels == b)
                blk = np.ascontiguousarray(A[np.ix_(rk, col_of[rk])])
                result *= perm(blk, method, decompose=False)
            return result
    if overflow:
        raise OverflowError("perm exceeds float precision (2**53)")
    return _perm_bbfg(A) if method == "bbfg" else _perm_ryser(A)


@jit(nopython=True)
def _perm_ryser(M: np.ndarray) -> float:
    """Compute permanent using Ryser formula with Gray code ordering.

    Args:
        M: A square numpy array.

    Returns:
        The permanent of matrix M.
    """
    n = len(M)
    if n == 0:
        return M.dtype.type(1.0)

    row_comb = np.zeros(n, dtype=M.dtype)
    total = 0
    old_grey = 0
    sign = +1
    binary_power_dict = np.array([2**i for i in range(n)])
    num_loops = 2**n

    for k in range(num_loops):
        bin_index = (k + 1) % num_loops
        reduced = np.prod(row_comb)
        total += sign * reduced
        new_grey = bin_index ^ (bin_index // 2)
        grey_diff = old_grey ^ new_grey
        grey_diff_index = 0
        for idx in range(n):
            if binary_power_dict[idx] == grey_diff:
                grey_diff_index = idx
                break
        new_vector = M[grey_diff_index]
        direction = (old_grey > new_grey) - (old_grey < new_grey)

        for i in range(n):
            row_comb[i] += new_vector[i] * direction

        sign = -sign
        old_grey = new_grey

    return total


@jit(nopython=True)
def _perm_bbfg(M: np.ndarray) -> float:
    """Compute permanent using BBFG formula with Gray code ordering.

    This is generally faster than Ryser for most matrices.

    Args:
        M: A square numpy array.

    Returns:
        The permanent of matrix M.
    """
    n = len(M)
    if n == 0:
        return M.dtype.type(1.0)

    row_comb = np.sum(M, 0)
    total = 0
    old_gray = 0
    sign = +1
    binary_power_dict = np.array([2**i for i in range(n)])
    num_loops = 2 ** (n - 1)

    for bin_index in range(1, num_loops + 1):
        reduced = np.prod(row_comb)
        total += sign * reduced
        new_gray = bin_index ^ (bin_index // 2)
        gray_diff = old_gray ^ new_gray
        gray_diff_index = 0
        for idx in range(n):
            if binary_power_dict[idx] == gray_diff:
                gray_diff_index = idx
                break
        new_vector = M[gray_diff_index]
        direction = 2 * ((old_gray > new_gray) - (old_gray < new_gray))

        for i in range(n):
            row_comb[i] += new_vector[i] * direction

        sign = -sign
        old_gray = new_gray

    return total / num_loops


def _perm_int(A: np.ndarray) -> int:
    """Compute exact integer permanent using minor expansion.

    Args:
        A: A square numpy array.

    Returns:
        The permanent of matrix A.
    """
    n = A.shape[0]
    rows = [[(j, int(A[i, j])) for j in range(n) if A[i, j] != 0] for i in range(n)]
    rows.sort(key=len)  # most-constrained row first => fewer reachable subsets
    memo: dict = {}

    def expand(depth: int, available: int) -> int:
        if depth == n:
            return 1
        if available in memo:
            return memo[available]
        total = 0
        for col, value in rows[depth]:
            bit = 1 << col
            if available & bit:
                total += value * expand(depth + 1, available ^ bit)
        memo[available] = total
        return total

    return expand(0, (1 << n) - 1)


def cycle_products(A: np.ndarray, source: Optional[int] = None, levels: bool = False) -> Union[int, float, List]:
    """Count products of disjoint cycles in a square matrix.

    An alternative to perm() whose cost follows the sparsity of the model.
    Counts are exact integers. For a binary interaction matrix they are
    the absolute number of feedback terms.

    Args:
        A: A square numpy array with at most 63 rows.
        source: Index of the perturbed variable. Returns the permanent of
            each minor with that row removed.
        levels: If True, return the number of terms covering 0, 1, ..., n
            variables (absolute feedback at each level).

    Returns:
        The permanent of A, a list of n values if source is given, or a
        list of n + 1 values if levels is True.

    Raises:
        TypeError: If input is not a numpy array.
        ValueError: If matrix is not square, has more than 63 rows or contains NaNs.

    References:
        - Puccia, C.J., Levins, R. (1985). Qualitative Modeling of Complex Systems: An Introduction to Loop Analysis and Time Averaging. Harvard University Press.
        - Dambacher, J.M., Luh, H.-K., Li, H.W., Rossignol, P.A. (2003). Qualitative stability and ambiguity in model ecosystems. The American Naturalist 161, 876–888.
        - Björklund, A., Husfeldt, T., Kaski, P., Koivisto, M. (2010). Evaluation of permanents in rings and semirings. Information Processing Letters 110, 867–870.

    Examples:
        ```python
        import numpy as np
        from qmm.core.helper import cycle_products
        A = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]])
        cycle_products(A)
        # 2

        cycle_products(A, source=0)
        # [1, 1, 1]

        cycle_products(A, levels=True)
        # [1, 2, 3, 2]
        ```
    """
    if not isinstance(A, np.ndarray):
        raise TypeError("Input matrix must be a NumPy array.")
    n = A.shape[0]
    if A.ndim != 2 or A.shape[1] != n:
        raise ValueError("Input matrix must be square.")
    if n > 63:
        raise ValueError("Input matrix must have at most 63 rows.")
    if A.dtype.kind == "f" and np.isnan(A).any():
        raise ValueError("Input matrix must not contain NaNs.")

    links = A.tolist()
    pattern = (A != 0) | np.eye(n, dtype=bool) if levels else A != 0
    remaining = np.ones(n, dtype=bool)
    if source is not None:
        remaining[source] = False
    order = []
    reached = np.zeros(n, dtype=bool)
    while remaining.any():
        wanted = pattern[remaining].sum(0) - pattern > 0
        opens = ((reached | pattern) & wanted).sum(1)
        opens[~remaining] = n + 1
        i = int(np.lexsort((pattern.sum(1), opens))[0])
        order.append(i)
        remaining[i] = False
        reached = reached | pattern[i]

    states = np.zeros(1, dtype=np.int64)
    weights = np.zeros((1, n + 1) if levels else 1, dtype=object)
    weights.flat[0] = 1
    remaining[order] = True
    for i in order:
        remaining[i] = False
        new_states, new_weights = [], []
        for j in np.flatnonzero(A[i]):
            free = ((states >> j) & 1) == 0
            if free.any():
                new_states.append(states[free] | np.int64(1 << int(j)))
                new_weights.append(weights[free] * links[i][j])
        if levels:
            free = ((states >> i) & 1) == 0
            if free.any():
                left_off = np.zeros_like(weights[free])
                left_off[:, 1:] = weights[free][:, :-1]
                new_states.append(states[free] | np.int64(1 << i))
                new_weights.append(left_off)
        if not new_states:
            states = states[:0]
            weights = weights[:0]
            break
        states = np.concatenate(new_states)
        weights = np.concatenate(new_weights)
        idx = np.argsort(states, kind="stable")
        states = states[idx]
        weights = weights[idx]
        first = np.flatnonzero(np.concatenate(([True], states[1:] != states[:-1])))
        states = states[first]
        weights = np.add.reduceat(weights, first, axis=0)
        closed = np.int64(sum(1 << int(j) for j in np.flatnonzero(~pattern[remaining].any(0))))
        unused = closed & ~states
        keep = unused == 0 if source is None else (unused & (unused - 1)) == 0
        states = states[keep]
        weights = weights[keep]

    found = dict(zip(states.tolist(), weights.tolist()))
    full = (1 << n) - 1
    if source is not None:
        return [found.get(full ^ (1 << i), 0) for i in range(n)]
    if levels:
        left_off = found.get(full, [0] * (n + 1))
        return [left_off[n - k] for k in range(n + 1)]
    return found.get(full, 0)


def _random_sampler(dist: Literal["uniform", "weak", "moderate", "strong", "uniform_two_oom"], size: int, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    """Sample random interaction strengths from a specified distribution.

    Used for numerical simulations where interaction strengths are drawn from
    probability distributions representing different assumptions about the
    magnitude of interactions.

    Args:
        dist: Distribution type for sampling:
            - "uniform": Uniform(0, 1) - no assumption about interaction strength
            - "weak": Beta(1, 3) - weak interactions predominate
            - "moderate": Beta(2, 2) - moderate interactions predominate
            - "strong": Beta(3, 1) - strong interactions predominate
            - "uniform_two_oom": Uniform(0.01, 1)
        size: Number of samples to draw
        rng: NumPy RandomState for reproducible draws (a fresh one is used if None)

    Returns:
        np.ndarray: Array of sampled interaction strengths

    Raises:
        ValueError: If dist is not a valid distribution name
    """
    if rng is None:
        rng = np.random.RandomState()
    if dist == "uniform_two_oom":
        return rng.uniform(0.01, 1.0, size)

    samplers = {
        "uniform": lambda: rng.uniform(0, 1, size),
        "weak": lambda: rng.beta(1, 3, size),
        "moderate": lambda: rng.beta(2, 2, size),
        "strong": lambda: rng.beta(3, 1, size),
    }

    if dist not in samplers:
        raise ValueError(f"Invalid distribution '{dist}'. Must be one of: {sorted(samplers.keys())} or 'uniform_two_oom'.")

    return samplers[dist]()


def get_dashed_alternatives(G: nx.DiGraph, combinations: bool = True) -> List[nx.DiGraph]:
    """Generate all alternative model structures based on dashed edges.

    Args:
        G: NetworkX DiGraph with potentially dashed edges (edges with dashes=True attribute)
        combinations: If True, return all 2^n combinations of dashed edges.
                     If False, return base graph (no dashed edges) plus variants with each single dashed edge added.

    Returns:
        List[nx.DiGraph]: List of graph variants with different dashed edge configurations.
                         If no dashed edges exist, returns a list containing only the original graph.

    References:
        - Raymond, B., McInnes, J., Dambacher, J.M., Way, S., Bergstrom, D.M. (2011). Qualitative modelling of invasive species eradication on subantarctic Macquarie Island. Journal of Applied Ecology 48, 181–191.

    Examples:
        ```python
        from qmm import load_digraph
        from qmm.core.helper import get_dashed_alternatives
        G_mod = load_digraph("snowshoe_rp").copy()
        G_mod.remove_edge('C', 'P')
        G_mod.add_edge('C', 'P', sign=1, dashes=True)
        variants = get_dashed_alternatives(G_mod, combinations=True)

        len(variants)
        # 2
        ```
    """
    dashed_edges = [(j, i, d) for j, i, d in G.edges(data=True) if d.get("dashes", False)]

    if not dashed_edges:
        return [G]

    if combinations:
        mask_values = range(2 ** len(dashed_edges))
        variants = []
        for mask in mask_values:
            G_variant = G.copy()
            for idx, (j, i, _) in enumerate(dashed_edges):
                include_edge = bool(mask & (1 << idx))
                if not include_edge:
                    G_variant.remove_edge(j, i)
            variants.append(G_variant)
    else:
        G_base = G.copy()
        for j, i, _ in dashed_edges:
            G_base.remove_edge(j, i)

        variants = [G_base]

        for j, i, edge_data in dashed_edges:
            G_variant = G_base.copy()
            G_variant.add_edge(j, i, **edge_data)
            variants.append(G_variant)

    return variants
