"""Validate qualitative predictions of system response to press perturbations from observations."""

import sympy as sp
import numpy as np
import pandas as pd
from .effects import _simulate
from ..core.structure import define_input_output
from ..core.helper import (
    _group_uncertain_edges,
    get_dashed_alternatives,
    get_nodes,
    _parse_perturbations,
    _parse_observations,
)
import networkx as nx
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union



def marginal_likelihood(
    G: nx.DiGraph,
    perturb: str,
    observe: str,
    n_sim: int = 10000,
    dist: Literal["uniform", "weak", "moderate", "strong", "uniform_two_oom"] = "uniform",
    seed: int = 42,
    uncertain_interactions: Literal["sample", "enumerate"] = "sample",
    pair_reciprocal: bool = True,
) -> float:
    """Calculate proportion of simulations matching qualitative observations.

    Args:
        G: NetworkX DiGraph representing signed digraph model
        perturb: Comma-separated node:sign pairs applied simultaneously with equal unit magnitudes
        observe: Observation string (node:sign, comma-separated allowed)
        n_sim: Stable draws per structure, or pooled stable draws when uncertain_interactions='sample'.
        dist: Distribution for sampling ('uniform', 'weak', 'moderate', 'strong', 'uniform_two_oom')
        seed: Random seed
        uncertain_interactions: Sample uncertain interactions, or average every structure equally.
        pair_reciprocal: Keep or drop reciprocal dashed edges together.

    Returns:
        float: Marginal likelihood

    References:
        - Raymond, B., McInnes, J., Dambacher, J.M., Way, S., Bergstrom, D.M. (2011). Qualitative modelling of invasive species eradication on subantarctic Macquarie Island. Journal of Applied Ecology 48, 181–191.
        - Melbourne-Thomas, J., Wotherspoon, S., Raymond, B., Constable, A. (2012). Comprehensive evaluation of model uncertainty in qualitative network analyses. Ecological Monographs 82, 505–519.

    Examples:
        ```python
        from qmm import marginal_likelihood, load_digraph
        marginal_likelihood(load_digraph("snowshoe_io"), perturb='Inp1:+', observe='Out1:+', n_sim=1000)
        # 0.526
        ```
    """
    pert = _parse_perturbations(G, perturb)
    likelihood, count = 0.0, 0
    for sims in _simulate(G, n_sim=n_sim, dist=dist, seed=seed,
                          perturb=pert,
                                 observe=_parse_observations(observe) if observe else None,
                                 uncertain_interactions=uncertain_interactions, pair_reciprocal=pair_reciprocal,
                                 condition=False):
        likelihood += sum(sims["valid_sims"]) / sims["n_stable"]
        count += 1
    return likelihood / count


def compare_model_alternatives(
    G: nx.DiGraph,
    perturb: str,
    observe: str,
    n_sim: int = 10000,
    dist: Literal["uniform", "weak", "moderate", "strong", "uniform_two_oom"] = "uniform",
    seed: int = 42,
    combinations: bool = True,
    pair_reciprocal: bool = True,
) -> pd.DataFrame:
    """Compare marginal likelihoods from alternative model structures.

    Args:
        G: NetworkX DiGraph representing signed digraph model
        perturb: Comma-separated node:sign pairs applied simultaneously with equal unit magnitudes
        observe: Observation string (node:sign, comma-separated allowed)
        n_sim: Stable draws per structure.
        dist: Distribution for sampling
        seed: Random seed
        combinations: If True, evaluate every combination of uncertain interactions. If False, compare the model without uncertain interactions against each single interaction added.
        pair_reciprocal: If True, a reciprocal pair of dashed edges is one uncertain interaction kept or dropped together.

    Returns:
        pd.DataFrame: Marginal likelihood comparison for requested dashed-edge configurations.

    References:
        - Raymond, B., McInnes, J., Dambacher, J.M., Way, S., Bergstrom, D.M. (2011). Qualitative modelling of invasive species eradication on subantarctic Macquarie Island. Journal of Applied Ecology 48, 181–191.
        - Melbourne-Thomas, J., Wotherspoon, S., Raymond, B., Constable, A. (2012). Comprehensive evaluation of model uncertainty in qualitative network analyses. Ecological Monographs 82, 505–519.

    Examples:
        ```python
        from qmm import compare_model_alternatives, load_digraph
        import networkx as nx
        G = nx.DiGraph(load_digraph("snowshoe_io"))
        G.add_edge('R', 'P', sign=1, dashes=True)
        compare_model_alternatives(G, perturb='Inp1:+', observe='Out1:+', n_sim=1000, combinations=False)
        #    Marginal likelihood (R, P)
        # 0                0.815      ✓
        # 1                0.526
        ```
    """
    G = define_input_output(G)
    dashed_edges = [edge for group in _group_uncertain_edges(G, pair_reciprocal) for edge in group]
    variants = get_dashed_alternatives(G, combinations=combinations, pair_reciprocal=pair_reciprocal)
    edge_presence = [[g.has_edge(u, v) for u, v in dashed_edges] for g in variants]

    variants = [define_input_output(g) for g in variants]
    categories = dict(G.nodes(data="category"))
    for g in variants:
        changed = [n for n, c in g.nodes(data="category") if c != categories[n]]
        if changed:
            raise ValueError(f"Nodes change category: {', '.join(map(str, changed))}")
    likelihoods = [marginal_likelihood(g, perturb, observe, n_sim, dist, seed) for g in variants]
    edge_cols = dashed_edges
    rows = [
        {"Marginal likelihood": likelihoods[i], **{edge_cols[j]: "\u2713" if edge_presence[i][j] else "" for j in range(len(dashed_edges))}}
        for i in range(len(variants))
    ]
    df = pd.DataFrame(rows, columns=["Marginal likelihood"] + edge_cols)
    return df.sort_values("Marginal likelihood", ascending=False, kind="mergesort").reset_index(drop=True)

def posterior_predictions(
    G: nx.DiGraph,
    perturb: str,
    observe: str = "",
    n_sim: int = 10000,
    dist: Literal["uniform", "weak", "moderate", "strong", "uniform_two_oom"] = "uniform",
    seed: int = 42,
    mode: Literal["dominant", "positive"] = "dominant",
    presample: Optional[Callable[[Tuple[sp.Symbol, ...]], Dict[sp.Symbol, Any]]] = None,
    uncertain_interactions: Literal["sample", "enumerate"] = "sample",
    pair_reciprocal: bool = True,
    max_attempts: Optional[int] = None,
) -> sp.Matrix:
    """Calculate model predictions conditioned on observations.

    Args:
        G: NetworkX DiGraph representing signed digraph model
        perturb: Comma-separated node:sign pairs applied simultaneously with equal unit magnitudes
        observe: Observation string (node:sign, comma-separated allowed)
        n_sim: Stable draws matching observe, per structure when uncertain_interactions='enumerate'.
        dist: Distribution for sampling
        seed: Random seed
        mode: 'dominant' for the signed proportion of the dominant sign,
            or 'positive' for the proportion of positive responses
        presample: Optional callable passed through to get_simulations
        uncertain_interactions: Sample uncertain interactions, or average every structure equally.
        pair_reciprocal: Keep or drop reciprocal dashed edges together.
        max_attempts: Maximum draws attempted per batch; defaults to 100 * n_sim.

    Returns:
        sp.Matrix: Predictions conditioned on observations

    Raises:
        ValueError: If mode is invalid.
        RuntimeError: If a batch cannot collect enough matching stable draws.

    References:
        - Raymond, B., McInnes, J., Dambacher, J.M., Way, S., Bergstrom, D.M. (2011). Qualitative modelling of invasive species eradication on subantarctic Macquarie Island. Journal of Applied Ecology 48, 181–191.
        - Melbourne-Thomas, J., Wotherspoon, S., Raymond, B., Constable, A. (2012). Comprehensive evaluation of model uncertainty in qualitative network analyses. Ecological Monographs 82, 505–519.

    Examples:
        ```python
        from qmm import posterior_predictions, load_digraph
        posterior_predictions(load_digraph("snowshoe_io"), perturb='Inp1:+', observe='Out1:+', n_sim=1000)
        # Matrix([
        # [   1.0],
        # [-0.511],
        # [-0.511],
        # [   1.0],
        # [-0.511]])
        ```
    """
    if mode not in ("dominant", "positive"):
        raise ValueError("Invalid mode. Choose 'dominant' or 'positive'.")
    pert = _parse_perturbations(G, perturb)
    observations = _parse_observations(observe) if observe else None
    n_total = len(get_nodes(G, "state")) + len(get_nodes(G, "output"))
    positive, negative, count = np.zeros(n_total), np.zeros(n_total), 0
    for sims in _simulate(G, n_sim=n_sim, dist=dist, seed=seed,
                          perturb=pert, observe=observations, presample=presample,
                                 uncertain_interactions=uncertain_interactions, pair_reciprocal=pair_reciprocal,
                                 max_attempts=max_attempts):
        effects = np.asarray(sims["effects"])[sims["valid_sims"], :n_total]
        positive += np.mean(effects > 0, axis=0)
        negative += np.mean(effects < 0, axis=0)
        count += 1
    positive /= count
    negative /= count

    smat = positive if mode == "positive" else np.where(negative > positive, -negative, positive)

    p_cols = [sims["all_nodes"].index(node) for node, _ in pert]
    tmat = sims["tmat"]
    smat = [sp.nan if not tmat[i, p_cols].any() else smat[i] for i in range(n_total)]

    return sp.Matrix(smat)

def diagnose_observations(
    G: nx.DiGraph,
    observe: str,
    perturb_nodes: Union[str, List[str]] = None,
    n_sim: int = 10000,
    dist: Literal["uniform", "weak", "moderate", "strong", "uniform_two_oom"] = "uniform",
    seed: int = 42,
    uncertain_interactions: Literal["sample", "enumerate"] = "sample",
    pair_reciprocal: bool = True,
) -> pd.DataFrame:
    """Identify possible perturbations from marginal likelihoods.

    Args:
        G: NetworkX DiGraph representing signed digraph model
        observe: Observation string (node:sign, comma-separated allowed)
        perturb_nodes: Node subset to test - comma-separated string, 'state', 'input', or list of nodes
        n_sim: Stable draws per structure, or pooled stable draws when uncertain_interactions='sample'.
        dist: Distribution for sampling
        seed: Random seed
        uncertain_interactions: Sample uncertain interactions, or average every structure equally.
        pair_reciprocal: Keep or drop reciprocal dashed edges together.

    Returns:
        pd.DataFrame: Ranked perturbations matching observations

    Examples:
        ```python
        from qmm import diagnose_observations, load_digraph
        diagnose_observations(load_digraph("snowshoe_io"), observe='Out1:+', perturb_nodes='input', n_sim=1000)
        #   Input Sign  Marginal likelihood
        # 0  Inp2    -                1.000
        # 1  Inp1    +                0.526
        # 2  Inp1    -                0.474
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
                likelihood = marginal_likelihood(G, f"{node}:{sign}", observe, n_sim, dist, seed, uncertain_interactions, pair_reciprocal)
            except RuntimeError:
                likelihood = np.nan
            results.append({"Input": node, "Sign": sign, "Marginal likelihood": likelihood})

    if not results:
        return pd.DataFrame(columns=["Input", "Sign", "Marginal likelihood"])
    return pd.DataFrame(results).sort_values(
        "Marginal likelihood", ascending=False, na_position="last"
    ).reset_index(drop=True)


def bayes_factors(
    G_list: Union[List[nx.DiGraph], Tuple[nx.DiGraph, ...]],
    perturb: str,
    observe: str,
    n_sim: int = 10000,
    dist: Literal["uniform", "weak", "moderate", "strong", "uniform_two_oom"] = "uniform",
    seed: int = 42,
    names: Optional[List[str]] = None,
    uncertain_interactions: Literal["sample", "enumerate"] = "sample",
    pair_reciprocal: bool = True,
) -> pd.DataFrame:
    """Calculate Bayes factors from the ratio of marginal likelihoods of alternative models.

    Args:
        G_list: List or tuple of NetworkX DiGraphs representing alternative models
        perturb: Comma-separated node:sign pairs applied simultaneously with equal unit magnitudes
        observe: Observation string (node:sign, comma-separated allowed)
        n_sim: Stable draws per structure, or pooled stable draws when uncertain_interactions='sample'.
        dist: Distribution for sampling ('uniform', 'weak', 'moderate', 'strong', 'uniform_two_oom')
        seed: Random seed
        names: Optional list of model names
        uncertain_interactions: Sample uncertain interactions, or average every structure equally.
        pair_reciprocal: Keep or drop reciprocal dashed edges together.

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
        # 0  Model A/Model B         0.526         0.474      1.109705
        ```
    """
    graphs = [define_input_output(g) for g in G_list]
    model_names = names if names and len(names) == len(graphs) else [f"Model {chr(65+i)}" for i in range(len(graphs))]
    categories = dict(graphs[0].nodes(data="category"))
    for name, g in zip(model_names[1:], graphs[1:]):
        fresh = dict(g.nodes(data="category"))
        if fresh.keys() != categories.keys():
            raise ValueError(f"{name} has different nodes: {sorted(fresh.keys() ^ categories.keys())}")
        changed = [n for n in categories if fresh[n] != categories[n]]
        if changed:
            raise ValueError(f"{name}: Nodes change category: {', '.join(map(str, changed))}")
    likelihoods = [marginal_likelihood(g, perturb, observe, n_sim, dist, seed, uncertain_interactions, pair_reciprocal) for g in graphs]

    comparisons = [(i, j) for i in range(len(graphs)) for j in range(i + 1, len(graphs))]
    factors = {
        f"{model_names[i]}/{model_names[j]}": (
            float("inf") if likelihoods[j] == 0 and likelihoods[i] > 0 else
            np.nan if likelihoods[j] == 0 else likelihoods[i] / likelihoods[j]
        ) for i, j in comparisons
    }

    return pd.DataFrame({
        "Model comparison": list(factors.keys()),
        "Likelihood 1": [likelihoods[i] for i, _ in comparisons],
        "Likelihood 2": [likelihoods[j] for _, j in comparisons],
        "Bayes factor": list(factors.values()),
    })
