import pandas as pd
import numpy as np
import networkx as nx
from functools import cache
from ..core.helper import get_nodes
from .effects import get_simulations

@cache
def mutual_information(
    G: nx.DiGraph,
    n_sim: int = 10000,
    seed: int = 42
) -> pd.DataFrame:
    
    nodes = get_nodes(G, "state") + get_nodes(G, "output")
    inputs = get_nodes(G, "input") + get_nodes(G, "state")
    all_effects = []
    
    for node in inputs:
        sims = get_simulations(G, n_sim=n_sim, seed=seed, perturb=(node, 1))
        node_map = {node: i for i, node in enumerate(nodes)}
        sim_effects = []
        for effect in sims["effects"]:
            sim_effects.append([
                effect[node_map[n]] if n in node_map and node_map[n] < len(effect)
                else np.nan for n in nodes
            ])
        all_effects.append(np.array(sim_effects))
    
    mi_vals = []
    for i, node in enumerate(nodes):
        node_effects = np.concatenate([effects[:, i] for effects in all_effects])
        labels = np.concatenate([np.full(effects.shape[0], j) for j, effects in enumerate(all_effects)])
        valid = ~np.isnan(node_effects)
        node_effects, labels = node_effects[valid], labels[valid]
        
        if len(node_effects) == 0:
            mi_vals.append(0)
            continue
        joint, _, _ = np.histogram2d(labels, node_effects > 0, bins=(len(inputs), 2))
        joint_p = joint / joint.sum()
        l_p, e_p = joint_p.sum(axis=1), joint_p.sum(axis=0)
        mi = sum(joint_p[i, j] * np.log2(joint_p[i, j] / (l_p[i] * e_p[j] + 1e-10))
                 for i in range(len(inputs))
                 for j in range(2)
                 if joint_p[i, j] > 0)
        mi_vals.append(max(0, mi))
    return pd.DataFrame({"Node": nodes, "Mutual Information": mi_vals}).sort_values("Mutual Information", ascending=False).reset_index(drop=True)
