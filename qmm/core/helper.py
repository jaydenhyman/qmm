"""Utility functions for model development and analysis."""

import numpy as np
import sympy as sp
import networkx as nx
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
    nx.set_node_attributes(G, "state", "category")
    nx.freeze(G)
    return G


def load_digraph(model: str) -> nx.DiGraph:
    """Load a built-in example model as a signed directed graph.

    Args:
        model: Name of the built-in model to load. Available models:
            - "snowshoe": Simple 3-node predator-prey model
            - "chain": 5-node linear chain with self-effects
            - "mesocosm": 8-node complex ecosystem model
            - "class_ii": 3-node Class II stable model

    Returns:
        nx.DiGraph: A NetworkX directed graph with signed edges.

    Raises:
        ValueError: If model name is not recognized.

    Examples:
        ```python
        from qmm import load_digraph
        G = load_digraph("snowshoe")
        list(G.nodes())
        # ['V', 'H', 'P']

        list(G.edges(data='sign'))
        # [('V', 'V', -1), ('V', 'H', 1), ('V', 'P', 1), ('H', 'V', -1), ('H', 'P', 1), ('P', 'H', -1), ('P', 'P', -1)]
        ```
    """
    models = {
        "snowshoe": {
            "matrix": [[-1, -1, 0], [1, 0, -1], [1, 1, -1]],
            "labels": ['V', 'H', 'P']
        },
        "snowshoe_i": {
            "matrix": [[-1, -1, 0, 1], [1, 0, -1, 1], [1, 1, -1, -1], [0, 0, 0, 0]],
            "labels": ['V', 'H', 'P', 'I']
        },
        "snowshoe_io": {
            "matrix": [[-1, -1, 0, 1, 0], [1, 0, -1, 1, 0], [1, 1, -1, -1, 0], [0, 0, 0, 0, 0], [1, 1, 1, 0, 0]],
            "labels": ['V', 'H', 'P', 'I', 'O']
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
        },
        "class_ii": {
            "matrix": [[-1, 1, 1], [1, -1, 1], [1, 1, -1]],
            "labels": ['A', 'B', 'C']
        }
    }

    if model not in models:
        available = ', '.join(f'"{m}"' for m in models.keys())
        raise ValueError(f"Model '{model}' not found. Available models: {available}")

    m = models[model]
    G = list_to_digraph(m["matrix"], m["labels"])

    if model in ("snowshoe_i", "snowshoe_io"):
        G_def = G.copy()
        nx.set_node_attributes(G_def, "state", "category")
        while True:
            reclassified = False
            for node in list(G_def.nodes()):
                if G_def.nodes[node]["category"] == "state":
                    if all(G_def.nodes[pred]["category"] == "input" for pred in G_def.predecessors(node)):
                        G_def.nodes[node]["category"] = "input"
                        reclassified = True
                    elif all(G_def.nodes[succ]["category"] == "output" for succ in G_def.successors(node)):
                        G_def.nodes[node]["category"] = "output"
                        reclassified = True
            if not reclassified:
                break
        nx.freeze(G_def)
        return G_def

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
        digraph_to_list(load_digraph("snowshoe"))
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
        get_nodes(load_digraph("snowshoe"), "state")
        # ['V', 'H', 'P']
        ```
    """
    if not isinstance(G, nx.DiGraph):
        raise TypeError("Input must be a networkx.DiGraph.")

    if node_type == "all":
        return list(G.nodes()) if not labels else list(G.nodes(data=True))
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
        G = load_digraph("snowshoe")
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
        
        if exponent > 700:  # exp(700) is near the float64 limit
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


def _arrows(G: nx.DiGraph, path: List[str]) -> str:
    arrows = []
    for i in range(len(path) - 1):
        if G[path[i]][path[i + 1]]["sign"] > 0:
            arrows.append(f"{path[i]} →")  # Right arrow
        else:
            arrows.append(f"{path[i]} ⊸")  # Multimap
    arrows.append(str(path[-1]))
    return " ".join(arrows)

def _sign_string(G: nx.DiGraph, path: List[str]) -> str:
    product = 1
    for from_node, to_node in zip(path, path[1:]):
        product *= G[from_node][to_node]["sign"]
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
        """Create from string like 'B:+' or 'B: +' or 'B:0'"""
        # Strip whitespace
        s = s.strip()
        node, sign = s.split(":")
        node = node.strip()
        sign = sign.strip()
        
        if sign not in ["+", "-", "0"]:
            raise ValueError(f"Sign must be +, -, or 0, got '{sign}'")
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


def _check_direct_io_edges(G: nx.DiGraph) -> None:
    """Check for unsupported direct input-to-output edges.

    Args:
        G: NetworkX DiGraph representing signed digraph model

    Raises:
        ValueError: If direct input-to-output edge exists
    """
    input_nodes = get_nodes(G, "input")
    output_nodes = get_nodes(G, "output")
    for inp in input_nodes:
        for out in output_nodes:
            if G.has_edge(inp, out):
                raise ValueError(
                    f"Direct input to output edge ({inp} to {out}) not supported."
                )


def _check_acyclic_inputs(G: nx.DiGraph) -> None:
    """Check that input subgraph is acyclic.

    Args:
        G: NetworkX DiGraph representing signed digraph model

    Raises:
        ValueError: If input subgraph contains cycles
    """
    input_nodes = get_nodes(G, "input")
    if input_nodes:
        input_subgraph = G.subgraph(input_nodes)
        if not nx.is_directed_acyclic_graph(input_subgraph):
            raise ValueError("Input subgraph contains cycles - input nodes must be acyclic")


def _check_acyclic_outputs(G: nx.DiGraph) -> None:
    """Check that output subgraph is acyclic.

    Args:
        G: NetworkX DiGraph representing signed digraph model

    Raises:
        ValueError: If output subgraph contains cycles
    """
    output_nodes = get_nodes(G, "output")
    if output_nodes:
        output_subgraph = G.subgraph(output_nodes)
        if not nx.is_directed_acyclic_graph(output_subgraph):
            raise ValueError("Output subgraph contains cycles - output nodes must be acyclic")


def perm(A: np.ndarray, method: Literal["bbfg", "ryser"] = "bbfg") -> float:
    """Compute the permanent of a square matrix.

    The permanent is similar to the determinant but uses only addition
    (no sign alternation). This implementation is based on the algorithms
    from thewalrus library (https://github.com/XanaduAI/thewalrus).

    Args:
        A: A square numpy array (float or complex).
        method: Algorithm to use - "bbfg" for BBFG formula (default, faster)
                or "ryser" for Ryser formula. Any other value uses Ryser.

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

    # Handle small matrices directly for efficiency
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

    if method == "bbfg":
        return _perm_bbfg(A)
    else:
        return _perm_ryser(A)


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
        # Find index of grey_diff in binary_power_dict
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
        # Find index of gray_diff in binary_power_dict
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


def _random_sampler(dist: Literal["uniform", "weak", "moderate", "strong", "uniform_two_oom"], size: int) -> np.ndarray:
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

    Returns:
        np.ndarray: Array of sampled interaction strengths

    Raises:
        ValueError: If dist is not a valid distribution name
    """
    if dist == "uniform_two_oom":
        return np.random.uniform(0.01, 1.0, size)

    samplers = {
        "uniform": lambda: np.random.uniform(0, 1, size),
        "weak": lambda: np.random.beta(1, 3, size),
        "moderate": lambda: np.random.beta(2, 2, size),
        "strong": lambda: np.random.beta(3, 1, size),
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
        G_mod = load_digraph("snowshoe").copy()
        G_mod.remove_edge('V', 'P')
        G_mod.add_edge('V', 'P', sign=1, dashes=True)
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
