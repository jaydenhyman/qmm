"""Validate qualitative predictions of system response to press perturbations from observations."""

import sympy as sp
import numpy as np
import pandas as pd
from functools import cache
from .effects import get_simulations
from ..core.helper import (
    get_nodes,
    _arrows,
    _parse_perturbations,
    _parse_observations,
)
import networkx as nx
from typing import Any, Callable, Dict, List, Optional, Tuple, Union



@cache
def marginal_likelihood(G: nx.DiGraph, perturb: str, observe: str, n_sim: int = 10000, dist: str = "uniform", seed: int = 42, distribution: str = None) -> float:
    """Calculate proportion of simulations matching qualitative observations.

    Args:
        G: NetworkX DiGraph representing signed digraph model
        perturb: Perturbation string (node:sign, comma-separated allowed)
        observe: Observation string (node:sign, comma-separated allowed)
        n_sim: Number of simulations
        dist: Distribution for sampling
        seed: Random seed

    Returns:
        float: Marginal likelihood
    """
    graph, pert = _parse_perturbations(G, perturb)
    use_dist = distribution if distribution is not None else dist
    sims = get_simulations(graph, n_sim=n_sim, dist=use_dist, seed=seed,
                          perturb=pert,
                          observe=_parse_observations(observe) if observe else None)
    return sum(sims["valid_sims"]) / n_sim

@cache
def model_validation(G: nx.DiGraph, perturb: str, observe: str, n_sim: int = 10000, dist: str = "uniform", seed: int = 42, combinations: bool = True) -> pd.DataFrame:
    """Compare marginal likelihoods from alternative model structures.

    Args:
        G: NetworkX DiGraph representing signed digraph model
        perturb: Perturbation string (node:sign, comma-separated allowed)
        observe: Observation string (node:sign, comma-separated allowed)
        n_sim: Number of simulations
        dist: Distribution for sampling
        seed: Random seed
        combinations: If True, evaluate every combination of dashed edges. If False, only compare the full model vs. all dashed edges removed.

    Returns:
        pd.DataFrame: Marginal likelihood comparison for requested dashed-edge configurations.
    """
    dashed_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("dashes", False)]
    if not dashed_edges:
        mask_values = [0]
    elif combinations:
        mask_values = range(2 ** len(dashed_edges))
    else:
        mask_values = [(1 << len(dashed_edges)) - 1, 0]

    variants, edge_presence = [], []
    for mask in mask_values:
        G_variant = G.copy()
        presence = [bool(mask & (1 << j)) for j in range(len(dashed_edges))]
        for j, (u, v) in enumerate(dashed_edges):
            if not presence[j]:
                G_variant.remove_edge(u, v)
        variants.append(G_variant)
        edge_presence.append(presence)

    likelihoods = [marginal_likelihood(g, perturb, observe, n_sim, dist, seed) for g in variants]
    edge_cols = [_arrows(G, [u, v]) for u, v in dashed_edges]
    rows = [
        {"Marginal likelihood": likelihoods[i], **{edge_cols[j]: "\u2713" if edge_presence[i][j] else "" for j in range(len(dashed_edges))}}
        for i in range(len(variants))
    ]
    df = pd.DataFrame(rows, columns=["Marginal likelihood"] + edge_cols)
    df = df.sort_values("Marginal likelihood", ascending=False, kind="mergesort").reset_index(drop=True)
    df["Marginal likelihood"] = df["Marginal likelihood"].apply(lambda x: f"{x:.3f}")
    return df

@cache
def posterior_predictions(
    G: nx.DiGraph,
    perturb: str,
    observe: str = "",
    n_sim: int = 10000,
    dist: str = "uniform",
    seed: int = 42,
    positive_only: bool = False,
    presample: Optional[Callable[[Tuple[sp.Symbol, ...]], Dict[sp.Symbol, Any]]] = None,
) -> sp.Matrix:
    """Calculate model predictions conditioned on observations.

    Args:
        G: NetworkX DiGraph representing signed digraph model
        perturb: Perturbation string (node:sign, comma-separated allowed)
        observe: Observation string (node:sign, comma-separated allowed)
        n_sim: Number of simulations
        dist: Distribution for sampling
        seed: Random seed
        positive_only: Return just the proportion of positive responses instead of sign-dominant proportions
        presample: Optional callable passed through to get_simulations

    Returns:
        sp.Matrix: Predictions conditioned on observations
    """
    graph, pert = _parse_perturbations(G, perturb)
    observations = _parse_observations(observe) if observe else None
    sims = get_simulations(graph, n_sim=n_sim, dist=dist, seed=seed,
                          perturb=pert, observe=observations, presample=presample)

    state, outputs = get_nodes(G, "state"), get_nodes(G, "output")
    n_total = len(state) + len(outputs)
    valid_indices = [i for i, v in enumerate(sims["valid_sims"]) if v]
    valid_count = len(valid_indices)

    if valid_count == 0:
        return sp.Matrix([np.nan] * n_total)

    effects = np.array([sims["effects"][i][:n_total] for i in valid_indices])
    positive = np.sum(effects > 0, axis=0)
    negative = np.sum(effects < 0, axis=0)

    smat = positive / valid_count if positive_only else np.where(
        negative > positive, -negative / valid_count, positive / valid_count
    )

    p_idx = sims["all_nodes"].index(pert[0])
    tmat = sims["tmat"]
    smat = [sp.nan if not tmat[i, p_idx] else smat[i] for i in range(n_total)]

    return sp.Matrix(smat)

def diagnose_observations(G: nx.DiGraph, observe: str, perturb_nodes: Union[str, List[str]] = None, n_sim: int = 10000, dist: str = "uniform", seed: int = 42) -> pd.DataFrame:
    """Identify possible perturbations from marginal likelihoods.

    Args:
        G: NetworkX DiGraph representing signed digraph model
        observe: Observation string (node:sign, comma-separated allowed)
        perturb_nodes: Node subset to test - comma-separated string, 'state', 'input', or list of nodes
        n_sim: Number of simulations
        dist: Distribution for sampling
        seed: Random seed

    Returns:
        pd.DataFrame: Ranked perturbations matching observations
    """
    if perturb_nodes is None:
        perturb_nodes = get_nodes(G, "state") + get_nodes(G, "input")
    elif isinstance(perturb_nodes, str):
        if perturb_nodes == "state":
            perturb_nodes = get_nodes(G, "state")
        elif perturb_nodes == "input":
            perturb_nodes = get_nodes(G, "input")
        else:
            perturb_nodes = [node.strip() for node in perturb_nodes.split(",")]

    results = []
    for node in perturb_nodes:
        for sign in ["+", "-"]:
            try:
                likelihood = marginal_likelihood(G, f"{node}:{sign}", observe, n_sim, dist, seed)
                results.append({"Input": node, "Sign": sign, "Marginal likelihood": likelihood})
            except Exception as e:
                print(f"Error for node {node} with sign {sign}: {str(e)}")

    if not results:
        return pd.DataFrame(columns=["Input", "Sign", "Marginal likelihood"])
    return pd.DataFrame(results).sort_values("Marginal likelihood", ascending=False).reset_index(drop=True)


def bayes_factors(G_list: Union[List[nx.DiGraph], Tuple[nx.DiGraph, ...]], perturb: str, observe: str,
                 n_sim: int = 10000, dist: str = "uniform",
                 seed: int = 42, names: Optional[List[str]] = None, distribution: str = None) -> pd.DataFrame:
    """Calculate Bayes factors from the ratio of marginal likelihoods of alternative models.

    Args:
        G_list: List or tuple of NetworkX DiGraphs representing alternative models
        perturb: Perturbation string (node:sign, comma-separated allowed)
        observe: Observation string (node:sign, comma-separated allowed)
        n_sim: Number of simulations
        dist: Distribution for sampling
        seed: Random seed
        names: Optional list of model names

    Returns:
        pd.DataFrame: DataFrame containing Bayes factors
    """
    graphs = list(G_list) if isinstance(G_list, tuple) else G_list
    use_dist = distribution if distribution is not None else dist
    likelihoods = [marginal_likelihood(g, perturb, observe, n_sim, use_dist, seed) for g in graphs]
    model_names = names if names and len(names) == len(graphs) else [f"Model {chr(65+i)}" for i in range(len(graphs))]

    comparisons = [(i, j) for i in range(len(graphs)) for j in range(i + 1, len(graphs))]
    factors = {
        f"{model_names[i]}/{model_names[j]}": (
            float("inf") if likelihoods[j] == 0 and likelihoods[i] > 0 else
            0 if likelihoods[j] == 0 else likelihoods[i] / likelihoods[j]
        ) for i, j in comparisons
    }

    return pd.DataFrame({
        "Model comparison": list(factors.keys()),
        "Likelihood 1": [likelihoods[i] for i, _ in comparisons],
        "Likelihood 2": [likelihoods[j] for _, j in comparisons],
        "Bayes factor": list(factors.values()),
    })
