"""Identify informative indicators from press perturbations."""

import pandas as pd
import numpy as np
import networkx as nx
from functools import cache
from ..core.helper import get_nodes, _NodeSign
from .effects import get_simulations
from typing import Union, List

@cache
def mutual_information(models: Union[nx.DiGraph, List[nx.DiGraph]], perturb: str, n_sim: int = 10000, seed: int = 42) -> pd.DataFrame:
    """Calculate mutual information of variables for alternative models.

    Args:
        models: One or more NetworkX DiGraphs representing alternative models
        perturb: Node and sign to perturb
        n_sim: Number of simulations
        seed: Random seed
        
    Returns:
        pd.DataFrame: Mutual information for indicator selection
    """
    models = [models] if not isinstance(models, (list, tuple)) else list(models)
    models = [nx.DiGraph(G) if not isinstance(G, nx.DiGraph) else G for G in models]
    nodes = sorted(set(node for G in models for node in get_nodes(G, "state") + get_nodes(G, "output")))
    all_effects = []
    for G in models:
        sims = get_simulations(G, n_sim=n_sim, seed=seed, perturb=_NodeSign.from_str(perturb).to_tuple())
        response_nodes = get_nodes(G, "state") + get_nodes(G, "output")
        node_map = {node: i for i, node in enumerate(response_nodes)}
        sim_effects = []
        for effect in sims["effects"]:
            sim_effects.append([effect[node_map[n]] if n in node_map and node_map[n] < len(effect) else np.nan for n in nodes])
        all_effects.append(np.array(sim_effects))
    mi_vals = []
    for i, node in enumerate(nodes):
        node_effects = [effects[:, i] for effects in all_effects]
        if any(np.all(np.isnan(effects)) for effects in node_effects):
            mi_vals.append(0)
            continue
        node_effects = np.concatenate(node_effects)
        labels = np.concatenate([np.full(effects.shape[0], i) for i, effects in enumerate(all_effects)])
        valid = ~np.isnan(node_effects)
        node_effects, labels = node_effects[valid], labels[valid]
        if len(node_effects) == 0:
            mi_vals.append(0)
            continue
        joint, _, _ = np.histogram2d(labels, node_effects > 0, bins=(len(models), 2))
        joint_p = joint / joint.sum()
        l_p, e_p = joint_p.sum(axis=1), joint_p.sum(axis=0)
        mi = sum(joint_p[i, j] * np.log2(joint_p[i, j] / (l_p[i] * e_p[j] + 1e-10))
                 for i in range(len(models))
                 for j in range(2)
                 if joint_p[i, j] > 0)
        mi_vals.append(max(0, mi))
    return pd.DataFrame({"Node": nodes, "Mutual Information": mi_vals}).sort_values("Mutual Information", ascending=False).reset_index(drop=True)
