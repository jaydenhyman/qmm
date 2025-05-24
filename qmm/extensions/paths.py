"""Analyse causal pathways, cycles and complementary feedback."""

import pandas as pd
import networkx as nx
import sympy as sp
from functools import cache
from typing import Optional
from ..core.structure import create_matrix
from ..core.stability import system_feedback, net_feedback, absolute_feedback, weighted_feedback
from ..core.helper import get_nodes, _sign_string, _arrows, get_positive, get_negative, get_weight

@cache
def get_cycles(G: nx.DiGraph) -> sp.Matrix:
    """Find all feedback cycles in the signed digraph.

    Args:
        G: NetworkX DiGraph representing signed digraph model
        
    Returns:
        sp.Matrix: Products of interactions along each cycle
    """
    A = create_matrix(G, form="symbolic")
    nodes = get_nodes(G, "state")
    node_id = {n: i for i, n in enumerate(nodes)}
    cycle_list = nx.simple_cycles(G)
    cycle_nodes = sorted([c for c in cycle_list], key=lambda x: len(x))
    C = [c + [c[0]] for c in cycle_nodes]
    cycles = sp.Matrix([sp.prod([A[node_id[c[i + 1]], node_id[c[i]]] for i in range(len(c) - 1)]) for c in C])
    return cycles

@cache
def cycles_table(G: nx.DiGraph) -> pd.DataFrame:
    """Find all feedback cycles in the signed digraph.

    Args:
        G: NetworkX DiGraph representing signed digraph model
        
    Returns:
        pd.DataFrame: Table with cycle length, path representation, and sign
    """
    cycle_nodes = sorted([path for path in nx.simple_cycles(G)], key=lambda x: (len(x), x))
    all_cycles = [cycle + [cycle[0]] for cycle in cycle_nodes]
    cycle_signs = [_sign_string(G, path) for path in all_cycles]
    cycles_df = pd.DataFrame(
        {
            "Length": [len(nodes) for nodes in cycle_nodes],
            "Cycle": [_arrows(G, path) for path in all_cycles],
            "Sign": cycle_signs,
        }
    )
    return cycles_df

@cache
def get_paths(G: nx.DiGraph, source: str, target: str, form: str = "symbolic") -> sp.Matrix:
    """Find all causal pathways between two nodes.

    Args:
        G: NetworkX DiGraph representing signed digraph model
        source: Source node
        target: Target node
        form: Type of path products ('symbolic', 'signed', or 'binary')
        
    Returns:
        sp.Matrix: Products of interactions along each path
    """
    nodes = get_nodes(G, "all")
    A = create_matrix(G, form=form)
    if source not in nodes or target not in nodes or source == target:
        raise ValueError("Invalid source or target node")
    if not nx.has_path(G, source, target):
        return sp.Matrix([sp.Integer(0)])
    path_nodes = list(nx.all_simple_paths(G, source, target))
    paths = [sp.prod(A[nodes.index(p[i + 1]), nodes.index(p[i])] for i in range(len(p) - 1)) for p in path_nodes]
    return sp.Matrix(paths)

@cache
def paths_table(G: nx.DiGraph, source: str, target: str) -> Optional[pd.DataFrame]:
    """Create table of paths between nodes.

    Args:
        G (nx.DiGraph): NetworkX DiGraph representing signed digraph model
        source (str): Source node
        target (str): Target node
        
    Returns:
        Optional[pd.DataFrame]: DataFrame containing path information or None if no paths exist
    """
    nodes = get_nodes(G, "all")
    if source not in nodes or target not in nodes or source == target:
        raise ValueError("Invalid source or target node")
    if not nx.has_path(G, source, target):
        return None
    paths = list(nx.all_simple_paths(G, source, target))
    if not paths:
        return None
    paths_df = pd.DataFrame(
        {
            "Length": [len(path) - 1 for path in paths],
            "Path": [_arrows(G, path) for path in paths],
            "Sign": [_sign_string(G, path) for path in paths],
        }
    )
    return paths_df

@cache
def complementary_feedback(G: nx.DiGraph, source: str, target: str, form: str = "symbolic") -> sp.Matrix:
    """Calculate feedback from nodes not on paths between source and target.

    Args:
        G: NetworkX DiGraph representing signed digraph model
        source: Source node
        target: Target node
        form: Type of feedback ('symbolic', 'signed', or 'binary')
        
    Returns:
        sp.Matrix: Feedback cycles in complementary subsystem
    """
    nodes = get_nodes(G, "state")
    n = len(nodes)
    if source not in nodes or target not in nodes or source == target:
        raise ValueError("Invalid source or target node")
    path_nodes = list(nx.all_simple_paths(G, source, target))
    feedback = []
    for path in path_nodes:
        path_nodes_set = set(path)
        subsystem_nodes = [node for node in nodes if node not in path_nodes_set]
        if not subsystem_nodes:
            if form == "binary":
                feedback.append(sp.Integer(1))
            else:
                feedback.append(sp.Integer(-1))
            continue
        subsystem = G.subgraph(subsystem_nodes).copy()
        level = n - len(path)
        if form == "symbolic":
            feedback.append(system_feedback(subsystem, level=level)[0])
        elif form == "signed":
            feedback.append(net_feedback(subsystem, level=level)[0])
        elif form == "binary":
            feedback.append(absolute_feedback(subsystem, level=level)[0])
        else:
            raise ValueError("Invalid form. Choose 'symbolic', 'signed', or 'binary'.")
    return sp.Matrix([sp.expand_mul(f) for f in feedback])

@cache
def system_paths(G: nx.DiGraph, source: str, target: str, form: str = "symbolic") -> sp.Matrix:
    """Calculate combined effect of paths and complementary feedback.

    Args:
        G: NetworkX DiGraph representing signed digraph model
        source: Source node
        target: Target node
        form: Type of computation ('symbolic', 'signed', or 'binary')
        
    Returns:
        sp.Matrix: Total effects including paths and feedback
    """
    if source == target:
        raise ValueError("Source and target must be different nodes")
    path = get_paths(G, source, target, form=form)
    feedback = complementary_feedback(G, source, target, form=form)
    if form == "binary":
        effect = path.multiply_elementwise(feedback)
    else:
        effect = path.multiply_elementwise(feedback) / sp.Integer(-1)
    return sp.Matrix([sp.expand_mul(e) for e in effect])

@cache
def weighted_paths(G: nx.DiGraph, source: str, target: str) -> sp.Matrix:
    """Calculate ratio of net to total path effects.

    Args:
        G: NetworkX DiGraph representing signed digraph model
        source: Source node
        target: Target node
        
    Returns:
        sp.Matrix: Net-to-total ratios for path predictions
    """
    nodes = get_nodes(G, "state")
    A_sgn = create_matrix(G, form="signed")
    if source not in nodes or target not in nodes or source == target:
        raise ValueError("Invalid source or target node")
    path_nodes = list(nx.all_simple_paths(G, source, target))
    wgt_effects = []
    for path in path_nodes:
        subsystem_nodes = [node for node in nodes if node not in path]
        if not subsystem_nodes:
            feedback = sp.Integer(-1)
        else:
            subsystem = G.subgraph(subsystem_nodes).copy()
            feedback = weighted_feedback(subsystem, level=len(nodes) - len(path))
            if feedback[0] == sp.nan:
                feedback = sp.Integer(0)
        sign = sp.prod(A_sgn[nodes.index(path[i + 1]), nodes.index(path[i])] for i in range(len(path) - 1))
        wgt_effect = sp.Integer(-1) * sign * feedback
        wgt_effects.append(wgt_effect)
    return sp.Matrix(wgt_effects)

@cache
def path_metrics(G: nx.DiGraph, source: str, target: str) -> pd.DataFrame:
    """Calculate comprehensive metrics for paths between nodes.

    Args:
        G: NetworkX DiGraph representing signed digraph model
        source: Source node
        target: Target node
        
    Returns:
        pd.DataFrame: Metrics including path length, sign, and feedback
    """
    nodes = get_nodes(G, "state")
    if source not in nodes or target not in nodes or source == target:
        raise ValueError("Invalid source or target node")
    if not nx.has_path(G, source, target):
        return pd.DataFrame()
    paths = list(nx.all_simple_paths(G, source, target))
    complementary_nodes = [[node for node in nodes if node not in set(path)] for path in paths]
    net_fb = complementary_feedback(G, source=source, target=target, form="signed")
    absolute_fb = complementary_feedback(G, source=source, target=target, form="binary")
    path_signs = get_paths(G, source=source, target=target, form="signed")
    weighted_fb = get_weight(net_fb, absolute_fb, sp.Integer(0))
    positive_fb = get_positive(net_fb, absolute_fb)
    negative_fb = get_negative(net_fb, absolute_fb)
    weighted_path = weighted_paths(G, source, target)
    n = len(paths)
    paths_df = pd.DataFrame(
        {
            "Length": [len(path) - 1 for path in paths],
            "Path": [", ".join(str(x) for x in path) for path in paths],
            "Path sign": ["+" if sign == 1 else "−" for sign in path_signs[:n]],
            "Complementary subsystem": [
                ", ".join(str(x) for x in nodes) if nodes else None for nodes in complementary_nodes
            ],
            "Net feedback": [net_fb[i] for i in range(n)],
            "Absolute feedback": [absolute_fb[i] for i in range(n)],
            "Positive feedback": [positive_fb[i] for i in range(n)],
            "Negative feedback": [negative_fb[i] for i in range(n)],
            "Weighted feedback": [weighted_fb[i] for i in range(n)],
            "Weighted path": [weighted_path[i] for i in range(n)],
        }
    )

    return paths_df
