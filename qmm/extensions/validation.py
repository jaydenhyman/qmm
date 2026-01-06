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
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union



@cache
def marginal_likelihood(
    G: nx.DiGraph,
    perturb: str,
    observe: str,
    n_sim: int = 10000,
    dist: Literal["uniform", "weak", "moderate", "strong", "uniform_two_oom"] = "uniform",
    seed: int = 42,
) -> float:
    """Calculate proportion of simulations matching qualitative observations.

    Args:
        G: NetworkX DiGraph representing signed digraph model
        perturb: Perturbation string (node:sign, comma-separated allowed)
        observe: Observation string (node:sign, comma-separated allowed)
        n_sim: Number of simulations
        dist: Distribution for sampling ('uniform', 'weak', 'moderate', 'strong', 'uniform_two_oom')
        seed: Random seed

    Returns:
        float: Marginal likelihood

    References:
        - Raymond, B., McInnes, J., Dambacher, J.M., Way, S., Bergstrom, D.M. (2011). Qualitative modelling of invasive species eradication on subantarctic Macquarie Island. Journal of Applied Ecology 48, 181–191.
        - Melbourne-Thomas, J., Wotherspoon, S., Raymond, B., Constable, A. (2012). Comprehensive evaluation of model uncertainty in qualitative network analyses. Ecological Monographs 82, 505–519.

    Examples:
        ```python
        from qmm import marginal_likelihood, load_digraph
        marginal_likelihood(load_digraph("snowshoe_io"), perturb='Inp1:+', observe='Out1:+', n_sim=1000)
        # 0.513
        ```
    """
    graph, pert = _parse_perturbations(G, perturb)
    sims = get_simulations(graph, n_sim=n_sim, dist=dist, seed=seed,
                          perturb=pert,
                          observe=_parse_observations(observe) if observe else None)
    return sum(sims["valid_sims"]) / n_sim

@cache
def model_validation(
    G: nx.DiGraph,
    perturb: str,
    observe: str,
    n_sim: int = 10000,
    dist: Literal["uniform", "weak", "moderate", "strong", "uniform_two_oom"] = "uniform",
    seed: int = 42,
    combinations: bool = True,
) -> pd.DataFrame:
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

    References:
        - Raymond, B., McInnes, J., Dambacher, J.M., Way, S., Bergstrom, D.M. (2011). Qualitative modelling of invasive species eradication on subantarctic Macquarie Island. Journal of Applied Ecology 48, 181–191.
        - Melbourne-Thomas, J., Wotherspoon, S., Raymond, B., Constable, A. (2012). Comprehensive evaluation of model uncertainty in qualitative network analyses. Ecological Monographs 82, 505–519.

    Examples:
        ```python
        from qmm import model_validation, load_digraph
        import networkx as nx
        G = nx.DiGraph(load_digraph("snowshoe_io"))
        G.add_edge('R', 'P', sign=1, dashes=True)
        model_validation(G, perturb='Inp1:+', observe='Out1:+', n_sim=1000, combinations=False)
        #   Marginal likelihood R $\\rightarrow$ P
        # 0               0.829                 ✓
        # 1               0.513
        ```
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
    dist: Literal["uniform", "weak", "moderate", "strong", "uniform_two_oom"] = "uniform",
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

    References:
        - Raymond, B., McInnes, J., Dambacher, J.M., Way, S., Bergstrom, D.M. (2011). Qualitative modelling of invasive species eradication on subantarctic Macquarie Island. Journal of Applied Ecology 48, 181–191.
        - Melbourne-Thomas, J., Wotherspoon, S., Raymond, B., Constable, A. (2012). Comprehensive evaluation of model uncertainty in qualitative network analyses. Ecological Monographs 82, 505–519.

    Examples:
        ```python
        from qmm import posterior_predictions, load_digraph
        posterior_predictions(load_digraph("snowshoe_io"), perturb='Inp1:+', observe='Out1:+', n_sim=1000)
        # Matrix([
        # [               1.0],
        # [-0.512670565302144],
        # [-0.512670565302144],
        # [               1.0],
        # [-0.512670565302144]])
        ```
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

def diagnose_observations(
    G: nx.DiGraph,
    observe: str,
    perturb_nodes: Union[str, List[str]] = None,
    n_sim: int = 10000,
    dist: Literal["uniform", "weak", "moderate", "strong", "uniform_two_oom"] = "uniform",
    seed: int = 42,
) -> pd.DataFrame:
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

    Examples:
        ```python
        from qmm import diagnose_observations, load_digraph
        diagnose_observations(load_digraph("snowshoe_io"), observe='Out1:+', perturb_nodes='input', n_sim=1000)
        #   Input Sign  Marginal likelihood
        # 0  Inp2    -                1.000
        # 1  Inp1    +                0.513
        # 2  Inp1    -                0.487
        # 3  Inp2    +                0.000
        ```
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


def bayes_factors(
    G_list: Union[List[nx.DiGraph], Tuple[nx.DiGraph, ...]],
    perturb: str,
    observe: str,
    n_sim: int = 10000,
    dist: Literal["uniform", "weak", "moderate", "strong", "uniform_two_oom"] = "uniform",
    seed: int = 42,
    names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Calculate Bayes factors from the ratio of marginal likelihoods of alternative models.

    Args:
        G_list: List or tuple of NetworkX DiGraphs representing alternative models
        perturb: Perturbation string (node:sign, comma-separated allowed)
        observe: Observation string (node:sign, comma-separated allowed)
        n_sim: Number of simulations
        dist: Distribution for sampling ('uniform', 'weak', 'moderate', 'strong', 'uniform_two_oom')
        seed: Random seed
        names: Optional list of model names

    Returns:
        pd.DataFrame: DataFrame containing Bayes factors

    References:
        - Raymond, B., McInnes, J., Dambacher, J.M., Way, S., Bergstrom, D.M. (2011). Qualitative modelling of invasive species eradication on subantarctic Macquarie Island. Journal of Applied Ecology 48, 181–191.
        - Melbourne-Thomas, J., Wotherspoon, S., Raymond, B., Constable, A. (2012). Comprehensive evaluation of model uncertainty in qualitative network analyses. Ecological Monographs 82, 505–519.

    Examples:
        ```python
        from qmm import bayes_factors, load_digraph
        G1 = load_digraph("snowshoe_io")
        G2 = G1.copy()
        G2.remove_edge('C', 'P')
        bayes_factors([G1, G2], perturb='Inp1:+', observe='Out1:+', n_sim=1000)
        #   Model comparison  Likelihood 1  Likelihood 2  Bayes factor
        # 0  Model A/Model B         0.513         0.489       1.04908
        ```
    """
    graphs = list(G_list) if isinstance(G_list, tuple) else G_list
    likelihoods = [marginal_likelihood(g, perturb, observe, n_sim, dist, seed) for g in graphs]
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
