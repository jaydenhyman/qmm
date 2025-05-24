"""Analyse the sensitivity of system stability to direct effects within feedback cycles."""

import sympy as sp
import networkx as nx
from functools import cache
from ..core.structure import create_matrix
from ..core.stability import system_feedback, net_feedback, absolute_feedback
from ..core.helper import get_nodes, get_weight
from typing import Optional

@cache
def structural_sensitivity(G: nx.DiGraph, level: Optional[int] = None) -> sp.Matrix:
    """Calculate contribution of direct effects to stabilising and destabilising feedback.

    Args:
        G: NetworkX DiGraph representing signed digraph model
        level: Feedback level (None for highest level)
        
    Returns:
        sp.Matrix: Ratio of net to total feedback terms for each direct effect
    """
    A = create_matrix(G, "signed")
    n = A.shape[0]
    fcp = system_feedback(G)[1:]
    if level is None:
        level = n
    S = sp.zeros(n, n)
    nodes = get_nodes(G, "state")
    for i in range(n):
        for j in range(n):
            if A[i, j] != 0:
                sG = nx.DiGraph(G)
                sG[nodes[j]][nodes[i]]["sign"] = 0
                scp = system_feedback(sG)[1:]
                if level <= len(fcp) and level <= len(scp):
                    N = fcp[level - 1] - scp[level - 1]
                    S[i, j] = N
    return S

@cache
def net_structural_sensitivity(G: nx.DiGraph, level: Optional[int] = None) -> sp.Matrix:
    """Calculate net contribution of direct effects to system feedback.

    Args:
        G: NetworkX DiGraph representing signed digraph model
        level: Feedback level (None for highest level)
        
    Returns:
        sp.Matrix: Net feedback terms containing each direct effect
    """
    A = create_matrix(G, "signed")
    n = A.shape[0]
    fcp = net_feedback(G)[1:]
    if level is None:
        level = n
    S = sp.zeros(n, n)
    nodes = get_nodes(G, "state")
    for i in range(n):
        for j in range(n):
            if A[i, j] != 0:
                sG = nx.DiGraph(G)
                sG[nodes[j]][nodes[i]]["sign"] = 0
                scp = net_feedback(sG)[1:]
                if level <= len(fcp) and level <= len(scp):
                    N = fcp[level - 1] - scp[level - 1]
                    S[i, j] = N
    return S

@cache
def absolute_structural_sensitivity(G: nx.DiGraph, level: Optional[int] = None) -> sp.Matrix:
    """Calculate total contribution of direct effects to system feedback.

    Args:
        G: NetworkX DiGraph representing signed digraph model
        level: Feedback level (None for highest level)
        
    Returns:
        sp.Matrix: Total feedback terms containing each direct effect
    """
    A = create_matrix(G, "signed")
    n = A.shape[0]
    fcp = absolute_feedback(G)[1:]
    if level is None:
        level = n
    S = sp.zeros(n, n)
    nodes = get_nodes(G, "state")
    for i in range(n):
        for j in range(n):
            if A[i, j] != 0:
                sG = nx.DiGraph(G)
                sG[nodes[j]][nodes[i]]["sign"] = 0
                scp = absolute_feedback(sG)[1:]
                if level <= len(fcp) and level <= len(scp):
                    N = fcp[level - 1] - scp[level - 1]
                    S[i, j] = N
    return S

@cache
def weighted_structural_sensitivity(G: nx.DiGraph, level: Optional[int] = None) -> sp.Matrix:
    """Calculate weighted structual sensitvity for each direct effect.

    Args:
        G: NetworkX DiGraph representing signed digraph model
        level: Feedback level (None for highest level)
        
    Returns:
        sp.Matrix: Weighted structural sensitivity of each direct effect
    """
    A = create_matrix(G, "signed")
    n = A.shape[0]
    if level is None:
        level = n
    net = sp.Matrix(net_structural_sensitivity(G, level))
    absolute = sp.Matrix(absolute_structural_sensitivity(G, level))
    return get_weight(net, absolute)
