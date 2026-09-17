"""Identify informative indicators from press perturbations."""

import pandas as pd
import numpy as np
import networkx as nx
from ..core.helper import get_nodes, _parse_perturbations
from ..core.structure import define_input_output
from .effects import _simulate
from typing import Union, List, Literal

def mutual_information(models: Union[nx.DiGraph, List[nx.DiGraph]], perturb: str, n_sim: int = 10000, seed: int = 42, include_null: bool = False, uncertain_interactions: Literal["sample", "enumerate"] = "sample", pair_reciprocal: bool = True, weights: Literal["equal", "posterior"] = "equal", base: float = np.e) -> pd.DataFrame:
    """Calculate mutual information of variables for alternative models.

    Args:
        models: One or more NetworkX DiGraphs representing alternative models
        perturb: Comma-separated node:sign pairs applied simultaneously with equal unit magnitudes
        n_sim: Stable simulations per model, or per structure when uncertain_interactions='enumerate'.
        seed: Random seed
        include_null: If True, include a null model with equal probability (1/3)
            of positive, negative, or zero response across simulations
        uncertain_interactions: 'sample' pools sampled structures; 'enumerate' averages every structure equally.
        pair_reciprocal: Keep or drop reciprocal dashed edges together.
        weights: 'equal' gives each model equal weight; 'posterior' weights models
            by their stable proportion, averaged across structures for 'enumerate'.
            The null model, if included, has acceptance probability one.
        base: Logarithm base; the default gives nats, and 2 gives bits.

    Returns:
        pd.DataFrame: Mutual information for indicator selection

    Raises:
        ValueError: If weights is not 'equal' or 'posterior', or base is not
            finite and greater than one.

    References:
        - Hosack, G.R., Hayes, K.R., Dambacher, J.M. (2008). Assessing Model Structure Uncertainty Through an Analysis of System Feedback and Bayesian Networks. Ecological Applications 18, 1070–1082.
        - Melbourne-Thomas, J., Wotherspoon, S., Raymond, B., Constable, A. (2012). Comprehensive evaluation of model uncertainty in qualitative network analyses. Ecological Monographs 82, 505–519.

    Examples:
        ```python
        from qmm import mutual_information, load_digraph
        G1 = load_digraph("snowshoe_rp")
        G2 = G1.copy()
        G2.remove_edge('C', 'P')
        mutual_information((G1, G2), perturb='R:+', n_sim=1000)
        #   Node  Mutual Information
        # 0    P            0.693147
        # 1    R            0.693147
        # 2    C            0.153711
        ```
    """
    if weights not in ("equal", "posterior"):
        raise ValueError("weights must be 'equal' or 'posterior'.")
    if not np.isfinite(base) or base <= 1:
        raise ValueError("base must be finite and greater than 1.")
    models = [models] if not isinstance(models, (list, tuple)) else list(models)
    models = [define_input_output(G if isinstance(G, nx.DiGraph) else nx.DiGraph(G)) for G in models]
    categories = dict(models[0].nodes(data="category"))
    for i, G in enumerate(models[1:], start=1):
        fresh = dict(G.nodes(data="category"))
        if fresh.keys() != categories.keys():
            raise ValueError(f"Model {chr(65 + i)} has different nodes: {sorted(fresh.keys() ^ categories.keys())}")
        changed = [n for n in categories if fresh[n] != categories[n]]
        if changed:
            raise ValueError(f"Model {chr(65 + i)}: Nodes change category: {', '.join(map(str, changed))}")
    nodes = sorted(get_nodes(models[0], "state") + get_nodes(models[0], "output"))
    probabilities, model_weights = [], []
    for G in models:
        perturb_tuple = _parse_perturbations(G, perturb)
        response_nodes = get_nodes(G, "state") + get_nodes(G, "output")
        node_indices = [response_nodes.index(node) for node in nodes]
        sign_probabilities = np.zeros((len(nodes), 3))
        stability, batches = 0.0, 0
        for sims in _simulate(G, n_sim=n_sim, seed=seed, perturb=perturb_tuple,
                              uncertain_interactions=uncertain_interactions, pair_reciprocal=pair_reciprocal):
            effects = np.sign(np.asarray(sims["effects"])[:, node_indices])
            sign_probabilities += (effects[..., None] == (-1, 0, 1)).mean(axis=0)
            stability += sims["prop_stable"]
            batches += 1
        probabilities.append(sign_probabilities / batches)
        model_weights.append(stability / batches)
    if include_null:
        rng = np.random.RandomState(seed)
        choices = rng.choice([1.0, -1.0, 0.0], size=(n_sim, len(nodes)))
        probabilities.append((choices[..., None] == (-1, 0, 1)).mean(axis=0))
        model_weights.append(1.0)
    probabilities = np.array(probabilities)
    n_models = len(probabilities)
    model_weights = np.array(model_weights) if weights == "posterior" else np.ones(n_models)
    model_weights /= model_weights.sum()
    mi_vals = []
    for i, node in enumerate(nodes):
        conditional = probabilities[:, i, :]
        joint = conditional * model_weights[:, None]
        marginal = joint.sum(axis=0)
        mi = sum(joint[k, j] * np.log(conditional[k, j] / marginal[j])
                 for k in range(n_models)
                 for j in range(3)
                 if joint[k, j] > 0) / np.log(base)
        mi_vals.append(max(0, mi))
    return pd.DataFrame({"Node": nodes, "Mutual Information": mi_vals}).sort_values("Mutual Information", ascending=False).reset_index(drop=True)
