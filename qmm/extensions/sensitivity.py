import sympy as sp
import networkx as nx
from functools import cache
from typing import Optional
from ..core.structure import create_matrix
from ..core.stability import system_feedback, net_feedback, absolute_feedback
from ..core.helper import get_nodes, get_weight

@cache
def structural_sensitivity(
    G: nx.DiGraph, 
    level: Optional[int] = None
) -> sp.Matrix:
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
def net_structural_sensitivity(
    G: nx.DiGraph, 
    level: Optional[int] = None
) -> sp.Matrix:
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
def absolute_structural_sensitivity(
    G: nx.DiGraph, 
    level: Optional[int] = None
) -> sp.Matrix:
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
def weighted_structural_sensitivity(
    G: nx.DiGraph, 
    level: Optional[int] = None
) -> sp.Matrix:
    A = create_matrix(G, "signed")
    n = A.shape[0]
    if level is None:
        level = n
    net = sp.Matrix(net_structural_sensitivity(G, level))
    absolute = sp.Matrix(absolute_structural_sensitivity(G, level))
    return get_weight(net, absolute)
