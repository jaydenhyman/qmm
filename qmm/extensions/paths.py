"""Analyse causal pathways, cycles and complementary feedback."""

import numpy as np
import pandas as pd
import networkx as nx
import sympy as sp
from functools import cache
from typing import Optional, Literal, Tuple
from ..core.structure import create_matrix
from ..core.stability import system_feedback, net_feedback, absolute_feedback, weighted_feedback
from ..core.helper import (
    get_nodes,
    get_positive,
    get_negative,
    get_weight,
    _sign_string,
    _arrows,
    _edge_prefix,
    _check_direct_io_edges,
    _parse_observations,
)
from .effects import get_simulations


def _sorted_cycles(G: nx.DiGraph) -> list:
    """State-node cycles, each rotated to start at its minimum, sorted by (length, nodes)."""
    state = G.subgraph(get_nodes(G, "state"))
    return sorted(
        [c[c.index(min(c)):] + c[:c.index(min(c))] for c in nx.simple_cycles(state)],
        key=lambda x: (len(x), x),
    )


def _check_source_target(G: nx.DiGraph, source: str, target: str) -> None:
    """Require an effects.py pair: source is state or input, target is state or output."""
    state = get_nodes(G, "state")
    sources = state + get_nodes(G, "input")
    targets = state + get_nodes(G, "output")
    if source not in sources:
        raise ValueError(f"Invalid source node '{source}'. Valid sources are state and input nodes.")
    if target not in targets:
        raise ValueError(f"Invalid target node '{target}'. Valid targets are state and output nodes.")


@cache
def get_cycles(G: nx.DiGraph) -> pd.DataFrame:
    """Find all feedback cycles in the signed digraph.

    Args:
        G: NetworkX DiGraph representing signed digraph model

    Returns:
        pd.DataFrame: Cycle nodes and the product of interactions along each cycle

    References:
        - Mason, S.J. (1953). Feedback Theory-Some Properties of Signal Flow Graphs. Proceedings of the IRE 41, 1144–1156.
        - Levins, R. (1974). The qualitative analysis of partially specified systems. Annals of the New York Academy of Sciences 231, 123–138.
        - Puccia, C.J., Levins, R. (1985). Qualitative Modeling of Complex Systems: An Introduction to Loop Analysis and Time Averaging. Harvard University Press.

    Examples:
        ```python
        from qmm import get_cycles, load_digraph
        get_cycles(load_digraph("snowshoe_rp"))
        #        Cycle            Product
        # 0       (P,)             -a_P,P
        # 1       (R,)             -a_R,R
        # 2     (C, P)       -a_C,P*a_P,C
        # 3     (C, R)       -a_C,R*a_R,C
        # 4  (C, R, P)  a_C,P*a_P,R*a_R,C
        ```
    """
    A = create_matrix(G, form="symbolic")
    nodes = get_nodes(G, "state")
    node_id = {n: i for i, n in enumerate(nodes)}
    cycle_nodes = _sorted_cycles(G)
    products = []
    for cycle in cycle_nodes:
        closed = cycle + [cycle[0]]
        products.append(sp.prod([A[node_id[closed[i + 1]], node_id[closed[i]]] for i in range(len(closed) - 1)]))
    return pd.DataFrame({"Cycle": [tuple(c) for c in cycle_nodes], "Product": products})

@cache
def cycles_table(G: nx.DiGraph, labels: bool = False) -> pd.DataFrame:
    """Tabulate all feedback cycles in the signed digraph.

    Each cycle is written as its nodes joined by $\\rightarrow$ for a positive link and
    $\\multimap$ for a negative link, closing on the starting node. Use get_cycles for the
    cycles as tuples of node ids.

    Args:
        G: NetworkX DiGraph representing signed digraph model
        labels: If True, write nodes by their label attribute instead of their id

    Returns:
        pd.DataFrame: Cycle length, cycle, and sign

    References:
        - Mason, S.J. (1953). Feedback Theory-Some Properties of Signal Flow Graphs. Proceedings of the IRE 41, 1144–1156.
        - Levins, R. (1974). The qualitative analysis of partially specified systems. Annals of the New York Academy of Sciences 231, 123–138.
        - Puccia, C.J., Levins, R. (1985). Qualitative Modeling of Complex Systems: An Introduction to Loop Analysis and Time Averaging. Harvard University Press.

    Examples:
        ```python
        from qmm import cycles_table, load_digraph
        cycles_table(load_digraph("snowshoe_rp"))
        #    Length                                          Cycle Sign
        # 0       1                                P $\\multimap$ P    −
        # 1       1                                R $\\multimap$ R    −
        # 2       2                C $\\rightarrow$ P $\\multimap$ C    −
        # 3       2                C $\\multimap$ R $\\rightarrow$ C    −
        # 4       3  C $\\multimap$ R $\\rightarrow$ P $\\multimap$ C    +
        ```
    """
    cycle_nodes = _sorted_cycles(G)
    closed = [cycle + [cycle[0]] for cycle in cycle_nodes]
    cycles_df = pd.DataFrame(
        {
            "Length": [len(nodes) for nodes in cycle_nodes],
            "Cycle": [_arrows(G, path, labels) for path in closed],
            "Sign": [_sign_string(G, path) for path in closed],
        }
    )
    return cycles_df

@cache
def get_paths(
    G: nx.DiGraph,
    source: str,
    target: str,
    form: Literal["symbolic", "signed", "binary"] = "symbolic",
) -> pd.DataFrame:
    """Find all causal pathways between two nodes.

    Args:
        G: NetworkX DiGraph representing signed digraph model
        source: Source node (state or input)
        target: Target node (state or output)
        form: Type of path products ('symbolic', 'signed', or 'binary')

    Returns:
        pd.DataFrame: Path nodes and the product of interactions along each path

    References:
        - Mason, S.J. (1953). Feedback Theory-Some Properties of Signal Flow Graphs. Proceedings of the IRE 41, 1144–1156.
        - Levins, R. (1974). The qualitative analysis of partially specified systems. Annals of the New York Academy of Sciences 231, 123–138.
        - Puccia, C.J., Levins, R. (1985). Qualitative Modeling of Complex Systems: An Introduction to Loop Analysis and Time Averaging. Harvard University Press.

    Examples:
        ```python
        from qmm import get_paths, load_digraph
        get_paths(load_digraph("snowshoe_io"), 'Inp1', 'Out1', form='symbolic')
        #                     Path                        Product
        # 0  (Inp1, R, C, P, Out1)  a_C,R*a_P,C*b_R,Inp1*c_Out1,P
        # 1     (Inp1, R, C, Out1)       -a_C,R*b_R,Inp1*c_Out1,C
        # 2     (Inp1, C, P, Out1)       -a_P,C*b_C,Inp1*c_Out1,P
        # 3        (Inp1, C, Out1)              b_C,Inp1*c_Out1,C

        get_paths(load_digraph("snowshoe_io"), 'Inp1', 'Out1', form='signed')
        #                     Path Product
        # 0  (Inp1, R, C, P, Out1)       1
        # 1     (Inp1, R, C, Out1)      -1
        # 2     (Inp1, C, P, Out1)      -1
        # 3        (Inp1, C, Out1)       1
        ```
    """
    _check_source_target(G, source, target)
    if source == target:
        return pd.DataFrame({"Path": [(source,)], "Product": [sp.Integer(1)]})
    if not nx.has_path(G, source, target):
        return pd.DataFrame({"Path": [()], "Product": [sp.Integer(0)]})
    path_nodes = list(nx.all_simple_paths(G, source, target))
    products = []
    for p in path_nodes:
        if form == "binary":
            products.append(sp.Integer(1))
            continue
        effect = sp.Integer(1)
        for i in range(len(p) - 1):
            u, v = p[i], p[i + 1]
            s = G[u][v].get("sign", 1)
            if form == "signed":
                effect *= sp.Integer(s)
            else:
                effect *= sp.Symbol(f"{_edge_prefix(G, u, v)}_{v},{u}") * s
        products.append(effect)
    return pd.DataFrame({"Path": [tuple(p) for p in path_nodes], "Product": products})

@cache
def paths_table(G: nx.DiGraph, source: str, target: str, labels: bool = False) -> Optional[pd.DataFrame]:
    """Tabulate the causal pathways between two nodes.

    Each path is written as its nodes joined by $\\rightarrow$ for a positive link and
    $\\multimap$ for a negative link. Use get_paths for the paths as tuples of node ids.

    Args:
        G: NetworkX DiGraph representing signed digraph model
        source: Source node (state or input)
        target: Target node (state or output)
        labels: If True, write nodes by their label attribute instead of their id

    Returns:
        Optional[pd.DataFrame]: Path length, path, and sign, or None if no paths exist

    References:
        - Mason, S.J. (1953). Feedback Theory-Some Properties of Signal Flow Graphs. Proceedings of the IRE 41, 1144–1156.
        - Levins, R. (1974). The qualitative analysis of partially specified systems. Annals of the New York Academy of Sciences 231, 123–138.
        - Puccia, C.J., Levins, R. (1985). Qualitative Modeling of Complex Systems: An Introduction to Loop Analysis and Time Averaging. Harvard University Press.

    Examples:
        ```python
        from qmm import paths_table, load_digraph
        paths_table(load_digraph("snowshoe_io"), 'Inp1', 'Out1')
        #    Length                                                                     Path Sign
        # 0       4  Inp1 $\\rightarrow$ R $\\rightarrow$ C $\\rightarrow$ P $\\rightarrow$ Out1    +
        # 1       3                    Inp1 $\\rightarrow$ R $\\rightarrow$ C $\\multimap$ Out1    −
        # 2       3                    Inp1 $\\multimap$ C $\\rightarrow$ P $\\rightarrow$ Out1    −
        # 3       2                                      Inp1 $\\multimap$ C $\\multimap$ Out1    +
        ```
    """
    _check_source_target(G, source, target)
    if not nx.has_path(G, source, target):
        return None
    paths = [[source]] if source == target else list(nx.all_simple_paths(G, source, target))
    paths_df = pd.DataFrame(
        {
            "Length": [len(path) - 1 for path in paths],
            "Path": [_arrows(G, path, labels) for path in paths],
            "Sign": [_sign_string(G, path) for path in paths],
        }
    )
    return paths_df

@cache
def complementary_feedback(
    G: nx.DiGraph,
    source: str,
    target: str,
    form: Literal["symbolic", "signed", "binary"] = "symbolic",
) -> pd.DataFrame:
    """Calculate feedback from state nodes not on paths between source and target.

    Args:
        G: NetworkX DiGraph representing signed digraph model
        source: Source node (state or input)
        target: Target node (state or output)
        form: Type of feedback ('symbolic', 'signed', or 'binary')

    Returns:
        pd.DataFrame: Path nodes and complementary-subsystem feedback

    References:
        - Mason, S.J. (1953). Feedback Theory-Some Properties of Signal Flow Graphs. Proceedings of the IRE 41, 1144–1156.
        - Levins, R. (1974). The qualitative analysis of partially specified systems. Annals of the New York Academy of Sciences 231, 123–138.
        - Puccia, C.J., Levins, R. (1985). Qualitative Modeling of Complex Systems: An Introduction to Loop Analysis and Time Averaging. Harvard University Press.

    Examples:
        ```python
        from qmm import complementary_feedback, load_digraph
        complementary_feedback(load_digraph("snowshoe_io"), 'Inp1', 'Out1', form='symbolic')
        #                     Path      Feedback
        # 0  (Inp1, R, C, P, Out1)            -1
        # 1     (Inp1, R, C, Out1)        -a_P,P
        # 2     (Inp1, C, P, Out1)        -a_R,R
        # 3        (Inp1, C, Out1)  -a_P,P*a_R,R
        ```
    """
    _check_source_target(G, source, target)
    state_nodes = get_nodes(G, "state")
    feedback_funcs = {"symbolic": system_feedback, "signed": net_feedback, "binary": absolute_feedback}
    if form not in feedback_funcs:
        raise ValueError("Invalid form. Choose 'symbolic', 'signed', or 'binary'.")
    if source == target:
        paths = [[source]]
    elif not nx.has_path(G, source, target):
        return pd.DataFrame({"Path": [()], "Feedback": [sp.Integer(0)]})
    else:
        paths = list(nx.all_simple_paths(G, source, target))

    feedback = []
    for path in paths:
        subsystem_nodes = [n for n in state_nodes if n not in path]
        if not subsystem_nodes:
            feedback.append(sp.Integer(1) if form == "binary" else sp.Integer(-1))
        else:
            subsystem = G.subgraph(subsystem_nodes).copy()
            feedback.append(feedback_funcs[form](subsystem, level=len(subsystem_nodes))[0])
    return pd.DataFrame(
        {
            "Path": [tuple(p) for p in paths],
            "Feedback": [sp.expand_mul(f) for f in feedback],
        }
    )

@cache
def system_paths(
    G: nx.DiGraph,
    source: str,
    target: str,
    form: Literal["symbolic", "signed", "binary"] = "symbolic",
) -> pd.DataFrame:
    """Calculate combined effect of paths and complementary feedback.

    Args:
        G: NetworkX DiGraph representing signed digraph model
        source: Source node (state or input)
        target: Target node (state or output)
        form: Type of computation ('symbolic', 'signed', or 'binary')

    Returns:
        pd.DataFrame: Path nodes and total effect including complementary feedback

    References:
        - Mason, S.J. (1953). Feedback Theory-Some Properties of Signal Flow Graphs. Proceedings of the IRE 41, 1144–1156.
        - Levins, R. (1974). The qualitative analysis of partially specified systems. Annals of the New York Academy of Sciences 231, 123–138.
        - Puccia, C.J., Levins, R. (1985). Qualitative Modeling of Complex Systems: An Introduction to Loop Analysis and Time Averaging. Harvard University Press.

    Examples:
        ```python
        from qmm import system_paths, load_digraph
        system_paths(load_digraph("snowshoe_io"), 'Inp1', 'Out1', form='symbolic')
        #                     Path                          Effect
        # 0  (Inp1, R, C, P, Out1)   a_C,R*a_P,C*b_R,Inp1*c_Out1,P
        # 1     (Inp1, R, C, Out1)  -a_C,R*a_P,P*b_R,Inp1*c_Out1,C
        # 2     (Inp1, C, P, Out1)  -a_P,C*a_R,R*b_C,Inp1*c_Out1,P
        # 3        (Inp1, C, Out1)   a_P,P*a_R,R*b_C,Inp1*c_Out1,C
        ```
    """
    _check_direct_io_edges(G)
    path = get_paths(G, source, target, form=form)
    feedback = complementary_feedback(G, source, target, form=form)
    path_m = sp.Matrix(path["Product"].tolist())
    feedback_m = sp.Matrix(feedback["Feedback"].tolist())
    if form == "binary":
        effect = path_m.multiply_elementwise(feedback_m)
    else:
        effect = path_m.multiply_elementwise(feedback_m) / sp.Integer(-1)
    return pd.DataFrame(
        {
            "Path": list(path["Path"]),
            "Effect": [sp.expand_mul(e) for e in effect],
        }
    )

@cache
def weighted_paths(G: nx.DiGraph, source: str, target: str) -> pd.DataFrame:
    """Calculate ratio of net to total path effects.

    Args:
        G: NetworkX DiGraph representing signed digraph model
        source: Source node (state or input)
        target: Target node (state or output)

    Returns:
        pd.DataFrame: Path nodes and net-to-total ratios

    Examples:
        ```python
        from qmm import weighted_paths, load_digraph
        weighted_paths(load_digraph("snowshoe_io"), 'Inp1', 'Out1')
        #                     Path Weight
        # 0  (Inp1, R, C, P, Out1)      1
        # 1     (Inp1, R, C, Out1)     -1
        # 2     (Inp1, C, P, Out1)     -1
        # 3        (Inp1, C, Out1)      1
        ```
    """
    _check_direct_io_edges(G)
    _check_source_target(G, source, target)
    state_nodes = get_nodes(G, "state")
    path_nodes = [[source]] if source == target else list(nx.all_simple_paths(G, source, target))
    wgt_effects = []
    for path in path_nodes:
        subsystem_nodes = [n for n in state_nodes if n not in path]
        if not subsystem_nodes:
            feedback = sp.Integer(-1)
        else:
            subsystem = G.subgraph(subsystem_nodes).copy()
            feedback = weighted_feedback(subsystem, level=len(subsystem_nodes))[0]
            if feedback == sp.nan:
                feedback = sp.Integer(0)
        sign = 1
        for i in range(len(path) - 1):
            sign *= G[path[i]][path[i + 1]].get('sign', 1)
        wgt_effect = sp.Integer(-1) * sign * feedback
        wgt_effects.append(wgt_effect)
    return pd.DataFrame({"Path": [tuple(p) for p in path_nodes], "Weight": wgt_effects})

@cache
def path_metrics(G: nx.DiGraph, source: str, target: str) -> pd.DataFrame:
    """Calculate comprehensive metrics for paths between nodes.

    Args:
        G: NetworkX DiGraph representing signed digraph model
        source: Source node (state or input)
        target: Target node (state or output)

    Returns:
        pd.DataFrame: Metrics including path length, sign, and feedback

    Examples:
        ```python
        from qmm import path_metrics, load_digraph
        path_metrics(load_digraph("snowshoe_io"), 'Inp1', 'Out1')
        #    Length                   Path Sign Complementary subsystem Net feedback Absolute feedback Positive feedback Negative feedback Weighted feedback Weighted path System path
        # 0       4  (Inp1, R, C, P, Out1)    +                      ()           -1                 1                 0                 1                -1             1           1
        # 1       3     (Inp1, R, C, Out1)    −                    (P,)           -1                 1                 0                 1                -1            -1          -1
        # 2       3     (Inp1, C, P, Out1)    −                    (R,)           -1                 1                 0                 1                -1            -1          -1
        # 3       2        (Inp1, C, Out1)    +                 (R, P)           -1                 1                 0                 1                -1             1           1
        ```
    """
    _check_direct_io_edges(G)
    _check_source_target(G, source, target)
    state_nodes = get_nodes(G, "state")
    if source == target:
        paths = [[source]]
    elif not nx.has_path(G, source, target):
        return pd.DataFrame()
    else:
        paths = list(nx.all_simple_paths(G, source, target))
    subsystem_nodes = [[n for n in state_nodes if n not in set(path)] for path in paths]
    net_fb = complementary_feedback(G, source=source, target=target, form="signed")
    absolute_fb = complementary_feedback(G, source=source, target=target, form="binary")
    path_signs = get_paths(G, source=source, target=target, form="signed")
    net_m = sp.Matrix(net_fb["Feedback"].tolist())
    abs_m = sp.Matrix(absolute_fb["Feedback"].tolist())
    weighted_fb = get_weight(net_m, abs_m, sp.Integer(0))
    positive_fb = get_positive(net_m, abs_m)
    negative_fb = get_negative(net_m, abs_m)
    weighted_path = weighted_paths(G, source, target)
    system_path = system_paths(G, source, target, form="signed")
    n = len(paths)
    paths_df = pd.DataFrame(
        {
            "Length": [len(path) - 1 for path in paths],
            "Path": [tuple(path) for path in paths],
            "Sign": ["+" if sign == 1 else "−" for sign in path_signs["Product"]],
            "Complementary subsystem": [tuple(nodes) for nodes in subsystem_nodes],
            "Net feedback": [net_m[i] for i in range(n)],
            "Absolute feedback": [abs_m[i] for i in range(n)],
            "Positive feedback": [positive_fb[i] for i in range(n)],
            "Negative feedback": [negative_fb[i] for i in range(n)],
            "Weighted feedback": [weighted_fb[i] for i in range(n)],
            "Weighted path": list(weighted_path["Weight"]),
            "System path": list(system_path["Effect"]),
        }
    )
    return paths_df

def _pathway_terms(
    G: nx.DiGraph,
    source: str,
    target: str,
    n_sim: int,
    dist: str,
    seed: int,
    average_uncertain: bool,
    observe: Optional[str] = None,
) -> Tuple[list, np.ndarray, dict]:
    """Pathways from source to target and their effect in each stable simulation."""
    _check_direct_io_edges(G)
    _check_source_target(G, source, target)
    state_nodes = get_nodes(G, "state")
    path_nodes = [[source]] if source == target else list(nx.all_simple_paths(G, source, target))
    sims = get_simulations(
        G,
        n_sim=n_sim,
        dist=dist,
        seed=seed,
        perturb=(source, 1),
        observe=_parse_observations(observe) if observe else None,
        return_samples=True,
        average_uncertain=average_uncertain,
    )
    samples = sims["samples"]
    n_stable = sims["n_stable"]
    node_id = {n: i for i, n in enumerate(state_nodes)}
    A = np.zeros((n_stable, len(state_nodes), len(state_nodes)))
    for u, v in G.edges():
        if u in node_id and v in node_id:
            A[:, node_id[v], node_id[u]] = G[u][v].get("sign", 1) * samples[f"{_edge_prefix(G, u, v)}_{v},{u}"]
    det_system = np.linalg.det(-A)
    terms = np.empty((n_stable, len(path_nodes)))
    for j, path in enumerate(path_nodes):
        product = np.ones(n_stable)
        for u, v in zip(path, path[1:]):
            product = product * G[u][v].get("sign", 1) * samples[f"{_edge_prefix(G, u, v)}_{v},{u}"]
        complement = [node_id[n] for n in state_nodes if n not in path]
        det_complement = np.linalg.det(-A[:, complement][:, :, complement]) if complement else np.ones(n_stable)
        terms[:, j] = product * det_complement / det_system
    return path_nodes, terms, sims

@cache
def pathway_effects(
    G: nx.DiGraph,
    source: str,
    target: str,
    n_sim: int = 10000,
    dist: Literal["uniform", "weak", "moderate", "strong", "uniform_two_oom"] = "uniform",
    seed: int = 42,
    average_uncertain: bool = False,
    observe: str = "",
) -> pd.DataFrame:
    """Simulate the effect transmitted along each causal pathway from source to target.

    Args:
        G: NetworkX DiGraph representing signed digraph model
        source: Source node (state or input)
        target: Target node (state or output)
        n_sim: Number of stable simulations
        dist: Distribution for sampling interaction strengths
        seed: Random seed
        average_uncertain: If True, sample edges marked dashes=True in/out each draw
        observe: Observation string (node:sign, comma-separated allowed) to condition on

    Returns:
        pd.DataFrame: Pathway length, sign, sign frequencies and contribution

    References:
        - Levins, R. (1974). The qualitative analysis of partially specified systems. Annals of the New York Academy of Sciences 231, 123–138.
        - Puccia, C.J., Levins, R. (1985). Qualitative Modeling of Complex Systems: An Introduction to Loop Analysis and Time Averaging. Harvard University Press.
        - Dambacher, J.M., Li, H.W., Rossignol, P.A. (2003). Qualitative predictions in model ecosystems. Ecological Modelling 161, 79–93.

    Examples:
        ```python
        from qmm import pathway_effects, load_digraph
        pathway_effects(load_digraph("snowshoe_rp"), 'R', 'P', n_sim=1000)
        #    Length       Path Sign  Positive  Negative  Zero  Contribution
        # 0       2  (R, C, P)    +       1.0       0.0   0.0           1.0
        # 1       1     (R, P)    +       0.0       0.0   1.0           0.0
        ```
    """
    path_nodes, terms, sims = _pathway_terms(G, source, target, n_sim, dist, seed, average_uncertain, observe)
    if observe:
        mask = np.asarray(sims["valid_sims"], dtype=bool)
        if not mask.any():
            raise ValueError("No simulations matched the observations.")
        terms = terms[mask]
    size = np.abs(terms)
    tiny = np.finfo(float).tiny
    signs = np.where(size <= 1e-9 * np.maximum(size.max(axis=1, keepdims=True), tiny), 0, np.sign(terms))
    total = size.sum(axis=1, keepdims=True)
    contribution = np.where(total > 0, size / np.maximum(total, tiny), 0.0).mean(axis=0)
    paths_df = pd.DataFrame(
        {
            "Length": [len(path) - 1 for path in path_nodes],
            "Path": [tuple(path) for path in path_nodes],
            "Sign": [_sign_string(G, path) for path in path_nodes],
            "Positive": (signs > 0).mean(axis=0),
            "Negative": (signs < 0).mean(axis=0),
            "Zero": (signs == 0).mean(axis=0),
            "Contribution": contribution,
        }
    )
    return paths_df.sort_values("Contribution", ascending=False, kind="stable").reset_index(drop=True)

