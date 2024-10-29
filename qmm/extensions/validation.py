import sympy as sp
import numpy as np
import pandas as pd
from functools import cache
from typing import List, Optional, Tuple
import networkx as nx
from .effects import get_simulations
from ..core.helper import get_nodes, arrows


@cache
def marginal_likelihood(
    G: nx.DiGraph,
    perturb: Tuple[str, int],
    observe: List[Tuple[str, int]],
    n_sim: int = 10000,
    distribution: str = "uniform",
    seed: int = 42
) -> float:
    sims = get_simulations(G, n_sim, distribution, seed, perturb, observe)
    return sum(sims["valid_sims"]) / n_sim

@cache
def model_validation(G: nx.DiGraph, perturb: Tuple[str, int], observe: List[Tuple[str, int]], 
                  n_sim: int = 10000, distribution: str = "uniform", seed: int = 42) -> pd.DataFrame:
    dashed_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('dashes', False)]
    structures_data = []
    variants = []
    edge_presence = []

    for i in range(2**len(dashed_edges)):
        G_variant = G.copy()
        presence = []
        for j, (u, v) in enumerate(dashed_edges):
            if not (i & (1 << j)):
                G_variant.remove_edge(u, v)
            presence.append(bool(i & (1 << j)))
        variants.append(G_variant)
        edge_presence.append(presence)

    likelihoods = [marginal_likelihood(G, perturb, observe, n_sim, distribution, seed) for G in variants]
    
    for j, (u, v) in enumerate(dashed_edges):
        row = {"Edges": arrows(G, [u, v])}
        for i in range(len(variants)):
            row[f"Model {chr(65 + i)}"] = "\u2713" if edge_presence[i][j] else ""
        structures_data.append(row)
    
    structures_data.extend([
        {**{"Edges": "─" * 20}, **{f"Model {chr(65 + i)}": "─" * 8 for i in range(len(variants))}},
        {**{"Edges": "Marginal likelihood"}, **{f"Model {chr(65 + i)}": f"{ml:.3f}" for i, ml in enumerate(likelihoods)}}
    ])
    
    return pd.DataFrame(structures_data)


@cache
def posterior_predictions(
    G: nx.DiGraph,
    perturb: Tuple[str, int],
    observe: Optional[List[Tuple[str, int]]] = None,
    n_sim: int = 10000,
    dist: str = "uniform",
    seed: int = 42
) -> sp.Matrix:
    sims = get_simulations(G, n_sim, dist, seed, perturb, observe)
    state_nodes, output_nodes = get_nodes(G, "state"), get_nodes(G, "output")
    n, m = len(state_nodes), len(output_nodes)
    valid_count = sum(sims["valid_sims"])
    tmat = sims["tmat"]
    if valid_count == 0:
        return sp.Matrix([np.nan] * (n + m))
    effects = np.array([e[:n+m] if len(e) >= n+m else np.pad(e, (0, n+m-len(e))) for e, v in zip(sims["effects"], sims["valid_sims"]) if v])
    positive = np.sum(effects > 0, axis=0)
    negative = np.sum(effects < 0, axis=0)
    smat = positive / valid_count
    tmat_np = np.array(tmat.tolist(), dtype=bool)
    perturb_index = sims["all_nodes"].index(perturb[0])
    smat = [np.nan if not tmat_np[i, perturb_index] else smat[i] for i in range(n + m)]
    if observe:
        for node, value in observe:
            index = state_nodes.index(node) if node in state_nodes else (n + output_nodes.index(node) if node in output_nodes else None)
            if index is not None:
                smat[index] = 1 if value > 0 else (0 if value < 0 else np.nan)
    
    smat = np.where(negative > positive, -negative / valid_count, smat)
    return sp.Matrix(smat)


@cache
def diagnose_observations(
    G: nx.DiGraph,
    observe: List[Tuple[str, int]],
    n_sim: int = 10000,
    distribution: str = "uniform",
    seed: int = 42
) -> pd.DataFrame:
    perturb_nodes = get_nodes(G, "state") + get_nodes(G, "input")
    results = []
    for node in perturb_nodes:
        for sign in [1, -1]:
            try:
                likelihood = marginal_likelihood(G, (node, sign), observe, n_sim, distribution, seed)
                results.append({"Input": node, "Sign": sign, "Marginal likelihood": likelihood})
            except Exception as e:
                print(f"Error for node {node} with sign {sign}: {str(e)}")
    
    return pd.DataFrame(results).sort_values("Marginal likelihood", ascending=False).reset_index(drop=True)

