"""Validate qualitative predictions of system response to press perturbations from observations."""

import sympy as sp
import numpy as np
import pandas as pd
from functools import cache
from .effects import get_simulations
from ..core.helper import get_nodes, _arrows, _NodeSign
import networkx as nx
from typing import List, Optional, Tuple, Union


def _parse_observations(s: str) -> Tuple[Tuple[str, int], ...]:
    if not s:
        return tuple()
    return tuple(_NodeSign.from_str(obs.strip()).to_tuple() 
                for obs in s.split(","))

@cache
def marginal_likelihood(G: nx.DiGraph, perturb: str, observe: str, n_sim: int = 10000, distribution: str = "uniform", seed: int = 42) -> float:
    """Calculate proportion of simulations matching qualitative observations.

    Args:
        G: NetworkX DiGraph representing signed digraph model
        perturb: Node and sign to perturb
        observe: String of observations
        n_sim: Number of simulations
        distribution: Distribution for sampling 
        seed: Random seed

    Returns:
        float: Marginal likelihood
    """
    sims = get_simulations(G, n_sim=n_sim, dist=distribution, seed=seed,
                          perturb=_NodeSign.from_str(perturb).to_tuple(),
                          observe=_parse_observations(observe) if observe else None)
    return sum(sims["valid_sims"]) / n_sim

@cache
def model_validation(G: nx.DiGraph, perturb: str, observe: str, n_sim: int = 10000, distribution: str = "uniform", seed: int = 42) -> pd.DataFrame:
    """Compare marginal likelihoods from alternative model structures.

    Args:
        G: NetworkX DiGraph representing signed digraph model
        perturb: Node and sign to perturb
        observe: String of observations
        
    Returns:
        pd.DataFrame: Marginal likelihood comparison for model variants
    """
    dashed_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("dashes", False)]
    variants = []
    edge_presence = []
    for i in range(2 ** len(dashed_edges)):
        G_variant = G.copy()
        presence = []
        for j, (u, v) in enumerate(dashed_edges):
            if not (i & (1 << j)):
                G_variant.remove_edge(u, v)
            presence.append(bool(i & (1 << j)))
        variants.append(G_variant)
        edge_presence.append(presence)
    
    likelihoods = [marginal_likelihood(G, perturb, observe, n_sim, distribution, seed) for G in variants]
    columns = ["Marginal likelihood"] + [_arrows(G, [u, v]) for u, v in dashed_edges]
    rows = []
    for i in range(len(variants)):
        row = {
            "Marginal likelihood": likelihoods[i]
        }
        for j, (u, v) in enumerate(dashed_edges):
            row[_arrows(G, [u, v])] = "\u2713" if edge_presence[i][j] else ""
        rows.append(row)
    df = pd.DataFrame(rows, columns=columns).sort_values("Marginal likelihood", ascending=False).reset_index(drop=True)    
    df["Marginal likelihood"] = df["Marginal likelihood"].apply(lambda x: f"{x:.3f}")
    return df

@cache
def posterior_predictions(G: nx.DiGraph, perturb: str, observe: str = "", n_sim: int = 10000, dist: str = "uniform", seed: int = 42) -> sp.Matrix:
    """Calculate model predictions conditioned on observations.

    Args:
        G: NetworkX DiGraph representing signed digraph model
        perturb: Node and sign to perturb
        observe: String of observations
        n_sim: Number of simulations
        dist: Distribution for sampling
        seed: Random seed
        
    Returns:
        sp.Matrix: Predictions conditioned on observations
    """
    sims = get_simulations(G, n_sim=n_sim, dist=dist, seed=seed,
                          perturb=_NodeSign.from_str(perturb).to_tuple(),
                          observe=_parse_observations(observe) if observe else None)
    state_nodes, output_nodes = get_nodes(G, "state"), get_nodes(G, "output")
    n, m = len(state_nodes), len(output_nodes)
    valid_count = sum(sims["valid_sims"])
    tmat = sims["tmat"]
    if valid_count == 0:
        return sp.Matrix([np.nan] * (n + m))
    effects = np.array(
        [
            e[: n + m] if len(e) >= n + m else np.pad(e, (0, n + m - len(e)))
            for e, v in zip(sims["effects"], sims["valid_sims"])
            if v
        ]
    )
    positive = np.sum(effects > 0, axis=0)
    negative = np.sum(effects < 0, axis=0)
    smat = positive / valid_count
    tmat_np = np.array(tmat.tolist(), dtype=bool)
    perturb_node = _NodeSign.from_str(perturb).node
    perturb_index = sims["all_nodes"].index(perturb_node)
    smat = [np.nan if not tmat_np[i, perturb_index] else smat[i] for i in range(n + m)]
    
    if observe:
        observations = _parse_observations(observe)
        for node, value in observations:
            index = (
                state_nodes.index(node)
                if node in state_nodes
                else (n + output_nodes.index(node) if node in output_nodes else None)
            )
            if index is not None:
                smat[index] = 1 if value > 0 else (0 if value < 0 else np.nan)

    smat = np.where(negative > positive, -negative / valid_count, smat)
    return sp.Matrix(smat)

@cache
def diagnose_observations(G: nx.DiGraph, observe: str, n_sim: int = 10000, distribution: str = "uniform", seed: int = 42) -> pd.DataFrame:
    """Identify possible perturbations from marginal likelihoods.

    Args:
        G: NetworkX DiGraph representing signed digraph model
        observe: String of observations
        
    Returns:
        pd.DataFrame: Ranked perturbations matching observations
    """
    perturb_nodes = get_nodes(G, "state") + get_nodes(G, "input")
    results = []
    for node in perturb_nodes:
        for sign in ["+", "-"]:
            try:
                likelihood = marginal_likelihood(G, f"{node}:{sign}", observe, n_sim, distribution, seed)
                results.append({"Input": node, "Sign": sign, "Marginal likelihood": likelihood})
            except Exception as e:
                print(f"Error for node {node} with sign {sign}: {str(e)}")
    return pd.DataFrame(results).sort_values("Marginal likelihood", ascending=False).reset_index(drop=True)


def bayes_factors(G_list: Union[List[nx.DiGraph], Tuple[nx.DiGraph, ...]], perturb: str, observe: str,
                 n_sim: int = 10000, distribution: str = "uniform", 
                 seed: int = 42, names: Optional[List[str]] = None) -> pd.DataFrame:
    """Calculate Bayes factors from the ratio of marginal likelihoods of alternative models.

    Args:
        G_list: List or tuple of NetworkX DiGraphs representing alternative models
        perturb: Node and sign to perturb
        observe: String of observations
        n_sim: Number of simulations
        distribution: Distribution for sampling
        seed: Random seed
        names: Optional list of model names
        
    Returns:
        pd.DataFrame: DataFrame containing Bayes factors
    """
    # Convert tuple to list if needed
    G_list = list(G_list) if isinstance(G_list, tuple) else G_list
    likelihoods = [marginal_likelihood(G, perturb, observe, n_sim, distribution, seed) for G in G_list]
    model_names = names if names and len(names) == len(G_list) else [f"Model {chr(65+i)}" for i in range(len(G_list))]
    bayes_factors = {f"{model_names[i]}/{model_names[j]}": (
        float("inf") if likelihoods[j] == 0 and likelihoods[i] > 0 else
        0 if likelihoods[j] == 0 else likelihoods[i] / likelihoods[j]
    ) for i in range(len(G_list)) for j in range(i + 1, len(G_list))}

    return pd.DataFrame({
        "Model comparison": list(bayes_factors.keys()),
        "Likelihood 1": [likelihoods[i] for i in range(len(G_list)) for j in range(i + 1, len(G_list))],
        "Likelihood 2": [likelihoods[j] for i in range(len(G_list)) for j in range(i + 1, len(G_list))],
        "Bayes factor": list(bayes_factors.values()),
    })