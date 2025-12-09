"""Utility functions for model development and analysis."""

import numpy as np
import sympy as sp
import networkx as nx
from typing import List, Union, Dict, Any, Optional, Tuple, Literal
from dataclasses import dataclass
from scipy.stats import truncnorm

def list_to_digraph(matrix: Union[List[List[int]], np.ndarray], ids: Optional[List[str]] = None) -> nx.DiGraph:
    """Convert an adjacency matrix to a directed graph.
    
    Args:
        matrix: A square matrix (list of lists or numpy array) representing the adjacency matrix.
            Non-zero values indicate edges, where the value represents the sign of the edge.
        ids: Optional list of node identifiers. If None, nodes will be labeled 1 to n.
    
    Returns:
        nx.DiGraph: A NetworkX directed graph with signed edges.
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

def digraph_to_list(G: nx.DiGraph) -> str:
    """Convert a directed graph to an adjacency matrix string representation.
    
    Args:
        G: A NetworkX directed graph with signed edges.
        
    Returns:
        str: String representation of the adjacency matrix.
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

def get_nodes(G: nx.DiGraph, node_type: str = "state", labels: bool = False) -> List[Union[str, Dict[str, Any]]]:
    """Get nodes of a specific type from a directed graph.
    
    Args:
        G: NetworkX directed graph to extract nodes from.
        node_type: Type of nodes to extract ('state' or 'all').
        labels: If True, return node labels instead of node ids.
        
    Returns:
        List of node identifiers or dictionaries containing node data.
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
    """
    if net.shape != absolute.shape:
        raise ValueError("Matrices must have the same shape")
    result = sp.zeros(*net.shape)
    for i in range(net.shape[0]):
        for j in range(net.shape[1]):
            result[i, j] = (absolute[i, j] - net[i, j]) // 2
    return result

def sign_determinacy(wmat: sp.Matrix, tmat: sp.Matrix, method: str = "average") -> sp.Matrix:
    """Calculate sign determinacy matrix from prediction weights.

    See Hosack et al. 2008. Ecological Applications 18, 1070–1082. https://doi.org/10.1890/07-0482.1

    Args:
        wmat: Matrix of prediction weights.
        tmat: Matrix of absolute feedback.
        method: Method to use for probability calculation ('average' or '95_bound').
        
    Returns:
        sympy.Matrix: Probability of sign determinacy.
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
    signs = []
    for from_node, to_node in zip(path, path[1:]):
        sign = G[from_node][to_node]["sign"]
        if sign != 0:
            signs.append(int(sign))
    product = sp.prod(signs)
    if product > 0:
        return "+"
    elif product < 0:
        return "\u2212"


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
    perturbations = [p.strip() for p in perturb.split(',')]
    if len(perturbations) > 1:
        G_mod = G.copy()
        G_mod.add_node('_P', category='input')
        for p in perturbations:
            ns = _NodeSign.from_str(p)
            G_mod.add_edge('_P', ns.node, sign=ns.sign)
        nx.freeze(G_mod)
        return G_mod, ('_P', 1)
    return G, _NodeSign.from_str(perturb).to_tuple()

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


distribution_types = Literal["uniform", "weak", "moderate", "strong", "normal_weak", "normal_moderate", "normal_strong"]
valid_distributions = frozenset({"uniform", "weak", "moderate", "strong", "normal_weak", "normal_moderate", "normal_strong"})


def _random_sampler(dist: distribution_types, size: int) -> np.ndarray:
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
            - "normal_weak": Truncated normal, mean near 0
            - "normal_moderate": Truncated normal, mean at 0.5
            - "normal_strong": Truncated normal, mean near 1
        size: Number of samples to draw
        
    Returns:
        np.ndarray: Array of sampled interaction strengths
        
    Raises:
        ValueError: If dist is not a valid distribution name
    """
    if dist not in valid_distributions:
        raise ValueError(f"Invalid distribution '{dist}'. Must be one of: {sorted(valid_distributions)}")
    
    samplers = {
        "uniform": lambda: np.random.uniform(0, 1, size),
        "weak": lambda: np.random.beta(1, 3, size),
        "moderate": lambda: np.random.beta(2, 2, size),
        "strong": lambda: np.random.beta(3, 1, size),
        "normal_weak": lambda: truncnorm.rvs(a=0, b=3, loc=0, scale=1/3, size=size),
        "normal_moderate": lambda: truncnorm.rvs(a=-3, b=3, loc=0.5, scale=1/6, size=size),
        "normal_strong": lambda: truncnorm.rvs(a=-3, b=0, loc=1, scale=1/3, size=size),
    }
    return samplers[dist]()
