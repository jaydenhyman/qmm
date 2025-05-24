"""Analyse cumulative effects from perturbation scenarios with multiple-inputs and multiple-outputs."""

import numpy as np
import sympy as sp
import networkx as nx
from scipy.stats import truncnorm
from functools import cache
from ..core.helper import get_nodes, get_weight, sign_determinacy
from ..core.structure import create_matrix
from ..core.press import adjoint_matrix, absolute_feedback_matrix
from typing import Dict, List, Optional, Any

def define_input_output(G: nx.DiGraph, remove_disconnected: bool = True) -> nx.DiGraph:
    """Define model components as state variables, inputs and outputs.

    Args:
        G: NetworkX DiGraph representing signed digraph model
        remove_disconnected: Remove disconnected components
        
    Returns:
        nx.DiGraph: Model with input, state and output classification
    """
    if not isinstance(G, nx.DiGraph):
        raise TypeError("Input must be a networkx.DiGraph.")
    G_def = G.copy()
    if remove_disconnected:
        G_undirected = G_def.to_undirected()
        connected = list(nx.connected_components(G_undirected))
        if len(connected) > 1:
            largest = max(connected, key=len)
            nodes_to_remove = [node for system in connected if system != largest for node in system]
            G_def.remove_nodes_from(nodes_to_remove)
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
    return G_def


@cache
def cumulative_effects(G: nx.DiGraph, form: str = "symbolic") -> sp.Matrix:
    """Calculate cumulative effects to multiple inputs using state-space representation.

    Args:
        G: NetworkX DiGraph representing signed digraph model
        form: Type of computation ('symbolic', 'signed', or 'binary')
        
    Returns:
        sp.Matrix: Cumulative effects on state variables and outputs
    """
    B = create_matrix(G, form=form, matrix_type="B")
    C = create_matrix(G, form=form, matrix_type="C")
    D = create_matrix(G, form=form, matrix_type="D")
    if form == "symbolic":
        effects = adjoint_matrix(G, form="symbolic")
    elif form in "signed":
        effects = adjoint_matrix(G, form="signed")
    elif form in "binary":
        effects = absolute_feedback_matrix(G)
    else:
        raise ValueError("Invalid form. Choose 'symbolic', 'signed', 'binary'.")
    cemat = sp.BlockMatrix([[effects, effects * B], [C * effects, C * effects * B + D]]).as_explicit()
    if form != "symbolic":
        cemat = cemat.subs({sym: 1 for sym in cemat.free_symbols})
    return sp.expand(cemat)


@cache
def absolute_effects(G: nx.DiGraph) -> sp.Matrix:
    """Calculate absolute effects from multiple inputs.

    Args:
        G: NetworkX DiGraph representing signed digraph model
        
    Returns:
        sp.Matrix: Total effects on state variables and outputs
    """
    return cumulative_effects(G, form="binary")


@cache
def weighted_effects(G: nx.DiGraph) -> sp.Matrix:
    """Calculate ratio of net to total terms for predicting cumulative effects.

    Args:
        G: NetworkX DiGraph representing signed digraph model
        
    Returns:
        sp.Matrix: Ratio of net to total effects
    """
    net = cumulative_effects(G, form="signed")
    absolute = cumulative_effects(G, form="binary")
    return get_weight(net, absolute)


@cache
def sign_determinacy_effects(G: nx.DiGraph, method: str = "average") -> sp.Matrix:
    """Calculate probability of correct sign prediction for cumulative effects.

    Args:
        G: NetworkX DiGraph representing signed digraph model
        method: Method for computing determinacy ('average', '95_bound')
        
    Returns:
        sp.Matrix: Sign determinacy probabilities for effects
    """
    weighted = weighted_effects(G)
    absolute = cumulative_effects(G, form="binary")
    return sign_determinacy(weighted, absolute, method=method)


@cache
def get_simulations(G: nx.DiGraph, n_sim: int = 10000, dist: str = "uniform", seed: int = 42, perturb: Optional[tuple[str, int]] = None, observe: Optional[List[tuple[str, int]]] = None) -> Dict[str, Any]:
    """Calculate average proportion of positive and negative effects from stable numerical simulations.

    Args:
        G: NetworkX DiGraph representing signed digraph model
        n_sim: Number of simulations
        dist: Distribution for sampling
        seed: Random seed
        
    Returns:
        sp.Matrix: Proportion of positive and negative effects from stable simulations
    """
    np.random.seed(seed)
    A, B, C, D = [create_matrix(G, form="symbolic", matrix_type=m) for m in "ABCD"]
    state_nodes, input_nodes, output_nodes = [get_nodes(G, t) for t in ["state", "input", "output"]]
    all_nodes = state_nodes + input_nodes + output_nodes
    node_idx = {node: i for i, node in enumerate(all_nodes)}
    tmat = sp.matrix2numpy(absolute_effects(G)).astype(int)
    dist_funcs = {
        "uniform": lambda size: np.random.uniform(0, 1, size),
        "weak": lambda size: np.random.beta(1, 3, size),
        "moderate": lambda size: np.random.beta(2, 2, size),
        "strong": lambda size: np.random.beta(3, 1, size),
        "normal_weak": lambda size: truncnorm.rvs(a=0, b=3, loc=0, scale=1/3, size=size),
        "normal_moderate": lambda size: truncnorm.rvs(a=-3, b=3, loc=0.5, scale=1/6, size=size),
        "normal_strong": lambda size: truncnorm.rvs(a=-3, b=0, loc=1, scale=1/3, size=size)
    }
    pert_idx, perturb_sign = (node_idx[perturb[0]], perturb[1]) if perturb else (None, 1)
    n_state, n_input, n_output = len(state_nodes), len(input_nodes), len(output_nodes)
    n_rows, n_cols = n_state + n_output, n_state + n_input
    symbols = list(set(A.free_symbols) | set(B.free_symbols) | set(C.free_symbols) | set(D.free_symbols))
    A_sp = sp.lambdify(symbols, A)
    B_sp = sp.lambdify(symbols, B) if n_input > 0 else None
    C_sp = sp.lambdify(symbols, C) if n_output > 0 else None
    D_sp = sp.lambdify(symbols, D) if D.shape != (0, 0) else None
    effects, valid_sims = [], []
    for _ in range(n_sim * 100):
        values = dist_funcs[dist](len(symbols))
        sim_A = A_sp(*values)
        if np.all(np.real(np.linalg.eigvals(sim_A)) < 0):
            try:
                inv_A = np.linalg.inv(-sim_A)
                B_np = B_sp(*values) if B_sp else np.array([]).reshape(n_state, 0)
                D_np = D_sp(*values) if D_sp else np.array([]).reshape(n_output, n_input)
                C_np = C_sp(*values) if C_sp else np.array([]).reshape(0, n_state)
                effect_matrix = np.zeros((n_rows, n_cols))
                if n_state > 0:
                    effect_matrix[:n_state, :n_state] = inv_A
                    if n_input > 0:
                        effect_matrix[:n_state, n_state:] = inv_A @ B_np
                if n_output > 0:
                    if n_state > 0:
                        effect_matrix[n_state:, :n_state] = C_np @ inv_A
                        if n_input > 0:
                            effect_matrix[n_state:, n_state:] = C_np @ inv_A @ B_np + D_np
                    elif n_input > 0:
                        effect_matrix[n_state:, n_state:] = D_np
                effect = effect_matrix[:, pert_idx] * perturb_sign if pert_idx is not None else effect_matrix
                effects.append(effect)
                if observe:
                    valid = all(
                        (
                            node in state_nodes
                            and (
                                (tmat[node_idx[node], pert_idx] == 0 and obs == 0)
                                or (obs != 0 and np.sign(effect[node_idx[node]]) == obs)
                            )
                        )
                        or (
                            node in output_nodes
                            and (
                                (tmat[len(state_nodes) + output_nodes.index(node), pert_idx] == 0 and obs == 0)
                                or (obs != 0 and np.sign(effect[len(state_nodes) + output_nodes.index(node)]) == obs)
                            )
                        )
                        for node, obs in observe
                    )
                    valid_sims.append(valid)
                else:
                    valid_sims.append(True)
                if len(effects) == n_sim:
                    break
            except np.linalg.LinAlgError:
                continue
    else:
        raise RuntimeError(f"Maximum iterations reached. Stable proportion: {len(effects) / (n_sim * 100):.4f}")
    return {"effects": effects, "valid_sims": valid_sims, "all_nodes": all_nodes, "tmat": tmat}


def simulation_effects(G: nx.DiGraph, n_sim: int = 10000, dist: str = "uniform", seed: int = 42, positive_only: bool = False) -> sp.Matrix:
    """Performs numerical simulations of cumulative effects using random interaction strengths.

    Args:
        G: NetworkX DiGraph representing signed digraph model
        n_sim: Number of simulations
        dist: Distribution for sampling ("uniform", "weak", "moderate", "strong")
        seed: Random seed
        positive_only: Return just the proportion of positive responses instead of sign-dominant proportions
        
    Returns:
        SymPy Matrix containing simulation results
    """
    sims = get_simulations(G, n_sim, dist, seed)
    tmat = sims["tmat"]
    state_nodes = get_nodes(G, "state")
    input_nodes = get_nodes(G, "input")
    output_nodes = get_nodes(G, "output")
    n_state, n_input, n_output = len(state_nodes), len(input_nodes), len(output_nodes)
    n_rows, n_cols = n_state + n_output, n_state + n_input
    positive = np.zeros((n_rows, n_cols), dtype=int)
    negative = np.zeros((n_rows, n_cols), dtype=int)
    for effect in sims["effects"]:
        positive += effect > 0
        negative += effect < 0
    if positive_only:
        smat = positive / n_sim
    else:
        smat = np.where(negative > positive, -negative / n_sim, positive / n_sim)
    smat = sp.Matrix(smat)
    smat = sp.Matrix([[sp.nan if not tmat[i, j] else smat[i, j] for j in range(smat.cols)] for i in range(smat.rows)])
    return smat
