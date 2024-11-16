import numpy as np
import sympy as sp
import networkx as nx
from numba import jit
from typing import List, Union, Dict, Any
from dataclasses import dataclass

def list_to_digraph(matrix, ids=None) -> nx.DiGraph:
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
    return G

def digraph_to_list(G) -> str:
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

def get_weight(net, absolute, no_effect=sp.nan) -> sp.Matrix:
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

def get_positive(net, absolute) -> sp.Matrix:
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

def get_negative(net, absolute) -> sp.Matrix:
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

def sign_determinacy(wmat, tmat, method="average") -> sp.Matrix:
    """Calculate sign determinacy matrix from prediction weights.
    
    Args:
        wmat: Matrix of prediction weights.
        tmat: Matrix of absolute feedback.
        method: Method to use for probability calculation ('average' or '95_bound').
        
    Returns:
        sympy.Matrix: Probability of sign determinacy.
    """
    def compute_prob(w, t, method):
        if w == sp.Integer(0):
            return sp.Rational(1, 2)
        elif w == sp.Integer(1):
            return sp.Integer(1)
        elif w == sp.Integer(-1):
            return sp.Integer(-1)
        elif t == sp.Integer(0):
            return sp.nan
        return compute_prob_average(w, t) if method == "average" else compute_prob_95_bound(w, t)
    def compute_prob_average(w, t):
        bw = 3.45962
        bwt = 0.03417
        prob = sp.exp(bw * w + bwt * w * t) / (1 + sp.exp(bw * w + bwt * w * t))
        return max(sp.Rational(1, 2), prob)
    def compute_prob_95_bound(w, t):
        bw = 9.766
        bwt = 0.139
        prob = sp.exp(bw * w + bwt * w * t) / (1253.992 + sp.exp(bw * w + bwt * w * t))
        return max(sp.Rational(1, 2), prob)
    if method not in ["average", "95_bound"]:
        raise ValueError("Invalid method. Choose 'average' or '95_bound'.")
    rows, cols = wmat.shape
    def calc_prob(i, j):
        w, t = wmat[i, j], tmat[i, j]
        if w.is_zero:
            return sp.Rational(1, 2)
        prob = compute_prob(sp.Abs(w), t, method)
        return sp.sign(w) * prob if prob is not None else sp.nan
    pmat = sp.Matrix(rows, cols, lambda i, j: calc_prob(i, j))
    return pmat

def _arrows(G, path) -> str:
    arrows = []
    for i in range(len(path) - 1):
        if G[path[i]][path[i + 1]]["sign"] > 0:
            arrows.append(f"{path[i]} →")  # Right arrow
        else:
            arrows.append(f"{path[i]} ⊸")  # Multimap
    arrows.append(str(path[-1]))
    return " ".join(arrows)

def _sign_string(G, path) -> str:
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
    else:
        return "0"

@dataclass(frozen=True)
class _NodeSign:
    node: str
    sign: int
    
    @classmethod
    def from_str(cls, s: str) -> '_NodeSign':
        """Create from string like 'B:+' or 'B: +'"""
        # Strip whitespace
        s = s.strip()
        node, sign = s.split(":")
        node = node.strip()
        sign = sign.strip()
        
        if sign not in ["+", "-"]:
            raise ValueError(f"Sign must be + or -, got '{sign}'")
        return cls(node, 1 if sign == "+" else -1)
    
    def to_tuple(self) -> tuple[str, int]:
        """Convert to tuple format for internal use"""
        return (self.node, self.sign)

def perm(A, method="glynn"):
    """Calculate the permanent of a matrix using specified method.
    
    Args:
        A: NumPy array representing the input matrix.
        method: Algorithm to use ('glynn' or 'ryser').
        
    Returns:
        float: Permanent of the matrix.
    """
    # Permanent function is a reimplementation of the following code:
    # Brajesh Gupt, Josh Izaac and Nicolás Quesada. The Walrus: a library for the calculation of hafnians, Hermite polynomials and Gaussian boson sampling. Journal of Open Source Software, 4(44), 1705 (2019)
    # https://the-walrus.readthedocs.io/en/latest/_modules/thewalrus/_permanent.html
    #
    # The original code is licensed under the Apache License, Version 2.0
    # (http://www.apache.org/licenses/LICENSE-2.0).
    if not isinstance(A, np.ndarray):
        raise TypeError("Input matrix must be a NumPy array.")
    matshape = A.shape
    if matshape[0] != matshape[1]:
        raise ValueError("Input matrix must be square.")
    if np.isnan(A).any():
        raise ValueError("Input matrix must not contain NaNs.")
    if matshape[0] == 0:
        return 1
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
    return _ryser(A) if method != "glynn" else _glynn(A)

@jit(nopython=True)
def _ryser(A):
    n = len(A)
    if n == 0:
        return 1
    row_comb = np.zeros((n), dtype=A.dtype)
    total = 0
    old_grey = 0
    sign = +1
    binary_power_dict = [2**i for i in range(n)]
    num_loops = 2**n
    for k in range(0, num_loops):
        bin_index = (k + 1) % num_loops
        reduced = np.prod(row_comb)
        total += sign * reduced
        new_grey = bin_index ^ (bin_index // 2)
        grey_diff = old_grey ^ new_grey
        grey_diff_index = binary_power_dict.index(grey_diff)
        new_vector = A[grey_diff_index]
        direction = (old_grey > new_grey) - (old_grey < new_grey)
        for i in range(n):
            row_comb[i] += new_vector[i] * direction
        sign = -sign
        old_grey = new_grey
    return total

@jit(nopython=True)
def _glynn(A):
    n = len(A)
    if n == 0:
        return 1
    row_comb = np.sum(A, 0)
    total = 0
    old_gray = 0
    sign = +1
    binary_power_dict = [2**i for i in range(n)]
    num_loops = 2 ** (n - 1)
    for bin_index in range(1, num_loops + 1):
        reduced = np.prod(row_comb)
        total += sign * reduced
        new_gray = bin_index ^ (bin_index // 2)
        gray_diff = old_gray ^ new_gray
        gray_diff_index = binary_power_dict.index(gray_diff)
        new_vector = A[gray_diff_index]
        direction = 2 * ((old_gray > new_gray) - (old_gray < new_gray))
        for i in range(n):
            row_comb[i] += new_vector[i] * direction
        sign = -sign
        old_gray = new_gray
    return total / num_loops
