"""Utility functions for model development and analysis."""

import numpy as np
import sympy as sp
import networkx as nx
from typing import List, Union, Dict, Any, Optional, Tuple, Literal

def list_to_digraph(matrix: Union[List[List[int]], np.ndarray], ids: Optional[List[str]] = None) -> nx.DiGraph:
    """Convert an adjacency matrix to a directed graph.

    Every node is assigned the state role, and no connectivity or feedback check is
    applied; analysis trusts a graph built this way.

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
    if not np.isin(matrix, [-1, 0, 1]).all():
        raise ValueError("Matrix entries must be -1, 0, or +1")
    G = nx.DiGraph()
    n = matrix.shape[0]
    if ids is None:
        node_ids = [str(i) for i in range(1, n + 1)]
    else:
        if len(ids) != n:
            raise ValueError("Number of ids must match matrix dimensions")
        node_ids = ids
    if len(set(node_ids)) != n:
        raise ValueError("Node ids must be unique")
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
    node_type: Literal["state", "input", "output", "invalid", "all"] = "state",
    labels: bool = False,
) -> List[Union[str, Dict[str, Any]]]:
    """Get nodes of a specific type from a directed graph.

    Args:
        G: NetworkX directed graph to extract nodes from.
        node_type: Type of nodes to extract ('state', 'input', 'output', 'invalid', or 'all').
            Nodes without a category count as 'state'.
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
        return [n if not labels else d.get("label", n) for n, d in G.nodes(data=True) if d.get("category", "state") == node_type]

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

    if method not in ["average", "95_bound"]:
        raise ValueError("Invalid method. Choose 'average' or '95_bound'.")
    MAX_PROB = sp.Float('0.999999')
    bw, bwt, offset = (3.45962, 0.03417, 1) if method == "average" else (9.766, 0.139, 1253.992)

    def compute_prob(w, t):
        if t == sp.Integer(0):
            return sp.nan
        exponent = bw * float(w) + bwt * float(w) * float(t)
        if exponent > 700:
            return MAX_PROB
        prob = max(sp.Rational(1, 2), sp.Float(np.exp(exponent) / (offset + np.exp(exponent))))
        return MAX_PROB if prob >= MAX_PROB else prob

    def calc_prob(i, j):
        w, t = wmat[i, j], tmat[i, j]
        if w.is_zero:
            return sp.Rational(1, 2)
        if sp.Abs(w) == sp.Integer(1):
            return sp.sign(w) * sp.Integer(1)
        return sp.sign(w) * compute_prob(sp.Abs(w), t)

    return sp.Matrix(*wmat.shape, calc_prob)


def _arrows(G: nx.DiGraph, path: List[str], labels: bool = False) -> str:
    """Write a path as nodes joined by $\\rightarrow$ (positive link) or $\\multimap$ (negative link)."""
    parts = []
    for from_node, to_node in zip(path, path[1:]):
        arrow = "$\\rightarrow$" if G[from_node][to_node].get("sign", 1) > 0 else "$\\multimap$"
        name = str(G.nodes[from_node].get("label") or from_node) if labels else str(from_node)
        parts.append(f"{name} {arrow}")
    parts.append(str(G.nodes[path[-1]].get("label") or path[-1]) if labels else str(path[-1]))
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


def _parse_perturbations(G: nx.DiGraph, perturb: str) -> Tuple[Tuple[str, int], ...]:
    """Parse fixed simultaneous unit presses."""
    perturbations = [p.strip() for p in perturb.split(',') if p.strip()]
    if not perturbations:
        raise ValueError("Perturbation string cannot be empty.")
    valid_nodes = set(get_nodes(G, "all"))
    presses = []
    for p in perturbations:
        node, sign = _parse_observations(p)[0]
        if node not in valid_nodes:
            raise ValueError(f"Unknown perturbation node: {node}")
        presses.append((node, sign))
    return tuple(presses)

def _parse_observations(s: str) -> Tuple[Tuple[str, int], ...]:
    """Parse 'B:+', 'B: -', 'B:0', or a bare 'B' (sign defaults to +), comma-separated."""
    if not s:
        return tuple()
    pairs = []
    for text in s.split(","):
        text = text.strip()
        node, sep, sign = text.partition(":")
        node = node.strip()
        sign = sign.strip() if sep else "+"
        if not node:
            raise ValueError(f"Missing node name in '{text}'")
        if sign not in ["+", "-", "0"]:
            raise ValueError(f"Sign must be +, -, or 0, got '{sign}' in '{text}'")
        pairs.append((node, 1 if sign == "+" else (-1 if sign == "-" else 0)))
    return tuple(pairs)


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


def perm(A: np.ndarray, source: Optional[int] = None, levels: bool = False) -> Union[int, float, List]:
    """Calculate the permanent of a square matrix.

    Args:
        A: Square NumPy array.
        source: Return minor permanents after removing this row. Entry j is the
            permanent after also removing column j.
        levels: Return sums of principal permanents for sizes 0 through n.
            A principal submatrix uses the same set of rows and columns.
            For a binary interaction matrix, these count feedback terms.

    Returns:
        One permanent, a list of n minor permanents when source is given, or
        n + 1 totals when levels is True.

    Raises:
        TypeError: If A is not a NumPy array.
        ValueError: If A is not square, has nonfinite float or complex entries,
            source is invalid, or source and levels are requested together.

    References:
        - Björklund, A., Husfeldt, T., Kaski, P., Koivisto, M. (2010).
          Evaluation of permanents in rings and semirings. Information Processing
          Letters 110, 867–870. https://doi.org/10.1016/j.ipl.2010.07.005
        - Kiah, H.M., Vardy, A., Yao, H. (2021). Computing Permanents on a Trellis,
          Definition 12 (sparse trellis). Preprint: https://arxiv.org/abs/2107.07377
          Row ordering and feasibility pruning here are additional optimizations.

    Examples:
        ```python
        import numpy as np
        from qmm.core.helper import perm
        A = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]])
        perm(A)
        # 2

        perm(A, source=0)
        # [1, 1, 1]

        perm(A, levels=True)
        # [1, 2, 3, 2]
        ```
    """
    if not isinstance(A, np.ndarray):
        raise TypeError("Input matrix must be a NumPy array.")
    A = np.asarray(A)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Input matrix must be square.")
    n = A.shape[0]
    if A.dtype.kind in "fc" and not np.isfinite(A).all():
        raise ValueError("Input matrix must not contain NaNs or infinities.")
    if source is not None:
        if isinstance(source, (bool, np.bool_)) or not isinstance(source, (int, np.integer)) or not 0 <= source < n:
            raise ValueError("Source must be an integer index within the matrix.")
        if levels:
            raise ValueError("Source and levels cannot be requested together.")

    links = A.tolist()
    if A.dtype.kind == "O":
        links = [[int(value) if isinstance(value, (np.integer, np.bool_)) else value
                  for value in row] for row in links]
    pattern = (A != 0) | (np.eye(n, dtype=bool) if levels else False)
    remaining = np.ones(n, dtype=bool)
    if source is not None:
        remaining[source] = False
    order, reached = [], np.zeros(n, dtype=bool)
    while remaining.any():
        wanted = pattern[remaining].sum(0) - pattern > 0
        opens = ((reached | pattern) & wanted).sum(1)
        opens[~remaining] = n + 1
        i = int(np.lexsort((pattern.sum(1), opens))[0])
        order.append(i)
        remaining[i] = False
        reached |= pattern[i]

    states = {0: np.array([1] + [0] * n, dtype=object) if levels else 1}
    full = (1 << n) - 1
    for k, i in enumerate(order):
        updated, closed_columns = {}, sum(1 << int(j) for j in np.flatnonzero(~pattern[order[k + 1:]].any(0)))
        options = [(1 << int(j), links[i][j], levels and i == j) for j in np.flatnonzero(pattern[i])]
        for used, weight in states.items():
            for bit, strength, diagonal in options:
                if used & bit:
                    continue
                target = used | bit
                unused = closed_columns & ~target
                if (unused if source is None else unused & (unused - 1)):
                    continue
                value = weight * strength
                if diagonal:
                    value[1:] += weight[:-1]
                updated[target] = updated.get(target, 0) + value
        states = updated
    if source is not None:
        return [states.get(full ^ (1 << j), 0) for j in range(n)]
    result = states.get(full, np.zeros(n + 1, dtype=object) if levels else 0)
    return result[::-1].tolist() if levels else result


DISTRIBUTIONS = ("uniform", "weak", "moderate", "strong", "uniform_two_oom")


def _random_sampler(dist: Union[str, Dict[str, List[float]]], size: int, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
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
            - {"beta": [a, b]}: Beta(a, b) with positive shapes
        size: Number of samples to draw
        rng: NumPy RandomState for reproducible draws (a fresh one is used if None)

    Returns:
        np.ndarray: Array of sampled interaction strengths

    Raises:
        ValueError: If dist is not a valid distribution name
    """
    if rng is None:
        rng = np.random.RandomState()
    if isinstance(dist, dict):
        return rng.beta(*_beta_shapes(dist), size)
    if dist == "uniform_two_oom":
        return rng.uniform(0.01, 1.0, size)
    if dist == "uniform":
        return rng.uniform(0, 1, size)
    shapes = {"weak": (1, 3), "moderate": (2, 2), "strong": (3, 1)}
    if dist not in shapes:
        raise ValueError(f"Invalid distribution '{dist}'. Must be one of: ['moderate', 'strong', 'uniform', 'weak'], 'uniform_two_oom' or {{'beta': [a, b]}}.")
    return rng.beta(*shapes[dist], size)


def _beta_shapes(dist: Dict[str, List[float]]) -> Tuple[float, float]:
    """Positive (a, b) from a {"beta": [a, b]} distribution."""
    shapes = dist.get("beta") if set(dist) == {"beta"} else None
    if (not isinstance(shapes, (list, tuple)) or len(shapes) != 2
            or not all(_is_number(x) and np.isfinite(x) and x > 0 for x in shapes)):
        raise ValueError(f"Invalid distribution {dist}. Use {{'beta': [a, b]}} with positive a and b.")
    return float(shapes[0]), float(shapes[1])


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_))


def _check_edge_priors(G: nx.DiGraph) -> None:
    """Raise unless every edge prior (dist, range, inclusion, stronger_than) is well formed.

    dist is a distribution name or {"beta": [a, b]}; range is [low, high] with
    0 <= low <= high and scales the strength drawn from dist onto that interval;
    inclusion is the probability that a dashed edge is present in a draw;
    stronger_than lists [from, to] edges whose strength must be smaller.
    """
    _edge_orderings(G)
    for u, v, data in G.edges(data=True):
        edge = f"{u} -> {v}"
        if data.get("dist") is not None:
            try:
                _random_sampler(data["dist"], 0, np.random.RandomState(0))
            except (ValueError, TypeError):
                raise ValueError(f"Invalid dist: {edge}") from None
        bounds = data.get("range")
        if bounds is not None and not (
                isinstance(bounds, (list, tuple)) and len(bounds) == 2
                and all(_is_number(x) and np.isfinite(x) for x in bounds)
                and 0 <= bounds[0] <= bounds[1]):
            raise ValueError(f"Invalid range: {edge}. Use [low, high] with 0 <= low <= high.")
        inclusion = data.get("inclusion")
        if inclusion is not None:
            if not (_is_number(inclusion) and 0 <= inclusion <= 1):
                raise ValueError(f"Invalid inclusion: {edge}. Use a probability from 0 to 1.")
            if not data.get("dashes", False):
                raise ValueError(f"Inclusion needs a dashed edge: {edge}")


def _edge_orderings(G: nx.DiGraph) -> nx.DiGraph:
    """Ordering graph with an arc from each edge to every edge it is stronger_than.

    References are [from, to] pairs matched by node id as text, so ids read from JSON
    as numbers still match. Raises on unknown edges, self references and cycles.
    """
    edges = {(str(u), str(v)): (u, v) for u, v in G.edges()}
    order = nx.DiGraph()
    for u, v, data in G.edges(data=True):
        weaker = data.get("stronger_than")
        if weaker is None:
            continue
        edge = f"{u} -> {v}"
        if not isinstance(weaker, (list, tuple)) or not all(
                isinstance(ref, (list, tuple)) and len(ref) == 2 for ref in weaker):
            raise ValueError(f"Invalid stronger_than: {edge}. Use a list of [from, to] edges.")
        for a, b in weaker:
            target = edges.get((str(a), str(b)))
            if target is None:
                raise ValueError(f"Unknown edge in stronger_than: {edge} names {a} -> {b}")
            if target == (u, v):
                raise ValueError(f"Edge stronger than itself: {edge}")
            order.add_edge((u, v), target)
    if not nx.is_directed_acyclic_graph(order):
        cycle = " > ".join(f"{u} -> {v}" for (u, v), _ in nx.find_cycle(order))
        raise ValueError(f"Cyclic stronger_than: {cycle}")
    return order


MAX_ORDERING_DRAWS = 10000


def _prior_sampler(G: nx.DiGraph, edges: List[Optional[Tuple[str, str]]], dist: Union[str, Dict[str, List[float]]]):
    """Sampler of strengths for edges[i] at position i, honouring each edge's dist and range.

    Returns None when no listed edge sets a prior, so callers draw from dist as before.
    Otherwise draw(rng) first draws every position from dist (the same draws as without
    priors), then redraws positions whose edge sets dist and rescales those with a range.
    Edges linked by stronger_than are then redrawn together, each from its own prior,
    until every ordering holds; orderings naming an edge outside edges are skipped.
    """
    _check_edge_priors(G)
    position = {edge: i for i, edge in enumerate(edges) if edge is not None}
    order = _edge_orderings(G).subgraph(position)
    pairs = [(position[a], position[b]) for a, b in order.edges()]
    components = [np.array(sorted(position[e] for e in c)) for c in nx.weakly_connected_components(order)
                  if len(c) > 1]
    by_dist: Dict[str, Tuple[Any, List[int]]] = {}
    scaled, lows, spans = [], [], []
    for i, edge in enumerate(edges):
        if edge is None:
            continue
        data = G.edges[edge]
        if data.get("dist") is not None:
            by_dist.setdefault(repr(data["dist"]), (data["dist"], []))[1].append(i)
        if data.get("range") is not None:
            low, high = data["range"]
            scaled.append(i)
            lows.append(float(low))
            spans.append(float(high) - float(low))
    if not by_dist and not scaled and not pairs:
        return None
    _random_sampler(dist, 0, np.random.RandomState(0))
    groups = [(edge_dist, np.array(idx)) for edge_dist, idx in by_dist.values()]
    scaled, lows, spans = np.array(scaled, dtype=int), np.array(lows), np.array(spans)
    stronger, weaker = (np.array(side, dtype=int) for side in zip(*pairs)) if pairs else ((), ())
    own_dist = {i: edge_dist for edge_dist, idx in groups for i in idx}
    scale = {i: (low, span) for i, low, span in zip(scaled, lows, spans)}
    redraws = []
    for comp in components:
        by = {}
        for i in comp:
            by.setdefault(repr(own_dist.get(i, dist)), (own_dist.get(i, dist), []))[1].append(i)
        comp_scaled = [i for i in comp if i in scale]
        redraws.append(([(d, np.array(idx)) for d, idx in by.values()], np.array(comp_scaled, dtype=int),
                        np.array([scale[i][0] for i in comp_scaled]), np.array([scale[i][1] for i in comp_scaled])))

    def draw(rng: np.random.RandomState) -> np.ndarray:
        values = _random_sampler(dist, len(edges), rng)
        for edge_dist, idx in groups:
            values[idx] = _random_sampler(edge_dist, len(idx), rng)
        values[scaled] = lows + spans * values[scaled]
        if pairs:
            for _ in range(MAX_ORDERING_DRAWS):
                if np.all(values[stronger] > values[weaker]):
                    return values
                for comp_groups, comp_scaled, comp_lows, comp_spans in redraws:
                    for edge_dist, idx in comp_groups:
                        values[idx] = _random_sampler(edge_dist, len(idx), rng)
                    values[comp_scaled] = comp_lows + comp_spans * values[comp_scaled]
            raise RuntimeError(f"No draw met the stronger_than orderings in {MAX_ORDERING_DRAWS} tries; "
                               "check they are possible within the edge ranges.")
        return values

    return draw


def _inclusion_probabilities(G: nx.DiGraph, interactions: List[List[Tuple[str, str]]]) -> np.ndarray:
    """Fixed inclusion probability per uncertain interaction, NaN where none is set."""
    probabilities = np.full(len(interactions), np.nan)
    for i, group in enumerate(interactions):
        values = {G.edges[edge].get("inclusion") for edge in group}
        if len(values) > 1:
            raise ValueError(f"Reciprocal uncertain edges need the same inclusion: {group}")
        if None not in values:
            probabilities[i] = float(values.pop())
    return probabilities


def _group_uncertain_edges(G: nx.DiGraph, pair_reciprocal: bool = True) -> List[List[Tuple[str, str]]]:
    """Dashed edges grouped into uncertain interactions; a reciprocal dashed pair is one interaction when pair_reciprocal."""
    dashed = [(u, v) for u, v, d in G.edges(data=True) if d.get("dashes", False)]
    if not pair_reciprocal:
        return [[edge] for edge in dashed]
    groups: Dict[frozenset, List[Tuple[str, str]]] = {}
    for u, v in dashed:
        groups.setdefault(frozenset((u, v)), []).append((u, v))
    return list(groups.values())


def _build_model_variant(G: nx.DiGraph, interactions: List[List[Tuple[str, str]]], present) -> nx.DiGraph:
    """Copy of G keeping the uncertain interactions flagged present, with no dashed edges left."""
    H = nx.DiGraph(G)
    H.remove_edges_from([edge for keep, group in zip(present, interactions) if not keep for edge in group])
    kept = {(str(u), str(v)) for u, v in H.edges()}
    for _, _, data in H.edges(data=True):
        data.pop("dashes", None)
        data.pop("inclusion", None)
        if data.get("stronger_than") is not None:
            data["stronger_than"] = [ref for ref in data["stronger_than"] if (str(ref[0]), str(ref[1])) in kept]
    return H


def get_dashed_alternatives(G: nx.DiGraph, combinations: bool = True, pair_reciprocal: bool = True) -> List[nx.DiGraph]:
    """Generate all alternative model structures based on dashed edges.

    Args:
        G: NetworkX DiGraph with potentially dashed edges (edges with dashes=True attribute)
        combinations: If True, return all 2^n combinations of uncertain interactions.
                     If False, return base graph (no dashed edges) plus variants with each single interaction added.
        pair_reciprocal: If True, a reciprocal pair of dashed edges is one uncertain interaction kept or dropped together.

    Returns:
        List[nx.DiGraph]: List of graph variants with different dashed edge configurations; no variant carries
                         the dashes attribute. If no dashed edges exist, returns a list containing only the original graph.

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
    interactions = _group_uncertain_edges(G, pair_reciprocal)
    if not interactions:
        return [G]
    if combinations:
        return [_build_model_variant(G, interactions, [bool(mask & (1 << i)) for i in range(len(interactions))])
                for mask in range(2 ** len(interactions))]
    none = [False] * len(interactions)
    return [_build_model_variant(G, interactions, none)] + [
        _build_model_variant(G, interactions, [j == i for j in range(len(interactions))]) for i in range(len(interactions))]
