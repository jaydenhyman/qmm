import sympy as sp
import numpy as np
import pandas as pd
from functools import cache
from .effects import get_simulations
from ..core.helper import get_nodes, arrows, NodeSign


def parse_observations(s: str) -> tuple[tuple[str, int], ...]:
    """Parse comma-separated observations into tuple of tuples"""
    if not s:
        return tuple()
    return tuple(NodeSign.from_str(obs.strip()).to_tuple() 
                for obs in s.split(","))

@cache
def marginal_likelihood(G, perturb: str, observe: str, n_sim=10000, distribution="uniform", seed=42) -> float:
    sims = get_simulations(G, n_sim, distribution, seed,
                          NodeSign.from_str(perturb).to_tuple(),
                          parse_observations(observe))
    return sum(sims["valid_sims"]) / n_sim

@cache
def model_validation(G, perturb: str, observe: str, n_sim=10000, distribution="uniform", seed=42) -> pd.DataFrame:
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
    columns = ["Marginal likelihood"] + [arrows(G, [u, v]) for u, v in dashed_edges]
    rows = []
    for i in range(len(variants)):
        row = {
            "Marginal likelihood": likelihoods[i]
        }
        for j, (u, v) in enumerate(dashed_edges):
            row[arrows(G, [u, v])] = "\u2713" if edge_presence[i][j] else ""
        rows.append(row)
    df = pd.DataFrame(rows, columns=columns).sort_values("Marginal likelihood", ascending=False).reset_index(drop=True)    
    df["Marginal likelihood"] = df["Marginal likelihood"].apply(lambda x: f"{x:.3f}")
    return df

@cache
def posterior_predictions(G, perturb: str, observe: str = "", n_sim=10000, dist="uniform", seed=42) -> sp.Matrix:
    sims = get_simulations(G, n_sim, dist, seed,
                          NodeSign.from_str(perturb).to_tuple(),
                          parse_observations(observe))
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
    perturb_node = NodeSign.from_str(perturb).node
    perturb_index = sims["all_nodes"].index(perturb_node)
    smat = [np.nan if not tmat_np[i, perturb_index] else smat[i] for i in range(n + m)]
    
    if observe:
        observations = parse_observations(observe)
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
def diagnose_observations(G, observe: str, n_sim=10000, distribution="uniform", seed=42) -> pd.DataFrame:
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


def bayes_factors(G_list, perturb: str, observe: str, n_sim=10000, distribution="uniform", seed=42, names=None):
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