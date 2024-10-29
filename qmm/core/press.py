import numpy as np
import sympy as sp
import networkx as nx
from functools import cache
from typing import Optional, Tuple
from .structure import create_matrix
from .helper import perm, get_weight, get_nodes, sign_determinacy


@cache
def adjoint_matrix(
    G: nx.DiGraph, 
    form: str = "symbolic", 
    perturb: Optional[str] = None
) -> sp.Matrix:
    A = create_matrix(G, form=form)
    A = sp.Matrix(-A)
    nodes = get_nodes(G, "state")
    n = len(nodes)
    if perturb is not None:
        src_id = nodes.index(perturb)
        return sp.Matrix(
            [sp.Integer(-1) ** (src_id + j) * A.minor(src_id, j) 
             for j in range(n)]
        )
    adjoint_matrix = sp.expand(A.adjugate())
    return sp.Matrix(adjoint_matrix)


@cache
def absolute_feedback_matrix(
    G: nx.DiGraph, 
    perturb: Optional[str] = None
) -> sp.Matrix:
    A = create_matrix(G, form="binary")
    A_np = np.array(sp.matrix2numpy(A), dtype=int)
    nodes = get_nodes(G, "state")
    n = A_np.shape[0]
    if perturb is not None:
        perturb_index = nodes.index(perturb)
        result = np.zeros(n, dtype=int)
        for j in range(n):
            minor = np.delete(np.delete(A_np, perturb_index, 0), j, 1)
            result[j] = int(perm(minor.astype(float)))
        return sp.Matrix(result)
    tmat = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            minor = np.delete(np.delete(A_np, j, 0), i, 1)
            tmat[i, j] = int(perm(minor.astype(float)))
    return sp.Matrix(tmat)


@cache
def weighted_predictions_matrix(
    G: nx.DiGraph,
    as_nan: bool = True,
    as_abs: bool = False,
    perturb: Optional[str] = None
) -> sp.Matrix:
    amat = adjoint_matrix(G, perturb=perturb, form="signed")
    if as_abs:
        amat = sp.Abs(amat)
    tmat = absolute_feedback_matrix(G, perturb=perturb)
    if as_nan:
        wmat = get_weight(amat, tmat)
    else:
        wmat = get_weight(amat, tmat, sp.Integer(1))
    return sp.Matrix(wmat)


@cache
def sign_determinacy_matrix(
    G: nx.DiGraph,
    method: str = "average",
    as_nan: bool = True,
    as_abs: bool = False,
    perturb: Optional[str] = None
) -> sp.Matrix:
    wmat = weighted_predictions_matrix(
        G, perturb=perturb, as_nan=as_nan, as_abs=as_abs
    )
    tmat = sp.Matrix(absolute_feedback_matrix(G, perturb=perturb))
    pmat = sign_determinacy(wmat, tmat, method)
    return sp.Matrix(pmat)


@cache
def numerical_simulations(
    G: nx.DiGraph,
    n_sim: int = 10000,
    dist: str = "uniform",
    seed: int = 42,
    as_nan: bool = True,
    as_abs: bool = False,
    perturb: Optional[Tuple[str, int]] = None
) -> sp.Matrix:
    np.random.seed(seed)
    A = create_matrix(G, form="symbolic", matrix_type="A")
    state_nodes = get_nodes(G, "state")
    node_idx = {node: i for i, node in enumerate(state_nodes)}
    n = len(state_nodes)
    symbols = list(A.free_symbols)
    A_sp = sp.lambdify(symbols, A)
    pert_idx, sign = (
        (node_idx[perturb[0]], perturb[1]) 
        if perturb else (None, 1)
    )
    dist_funcs = {
        "uniform": lambda size: np.random.uniform(0, 1, size),
        "weak": lambda size: np.random.beta(1, 3, size),
        "moderate": lambda size: np.random.beta(2, 2, size),
        "strong": lambda size: np.random.beta(3, 1, size),
    }
    positive = np.zeros((n, n), dtype=int)
    negative = np.zeros((n, n), dtype=int)
    total_simulations = 0
    while total_simulations < n_sim:
        values = dist_funcs[dist](len(symbols))
        sim_A = A_sp(*values)
        if np.all(np.real(np.linalg.eigvals(sim_A)) < 0):
            try:
                inv_A = np.linalg.inv(-sim_A)
                effect = (
                    inv_A[:, pert_idx] * sign 
                    if pert_idx is not None else inv_A
                )
                positive += effect > 0
                negative += effect < 0
                total_simulations += 1
            except np.linalg.LinAlgError:
                continue
    smat = np.where(negative > positive, -negative / n_sim, positive / n_sim)
    smat = sp.Matrix(smat.astype(float).tolist())
    tmat = absolute_feedback_matrix(G)
    tmat_np = np.array(tmat.tolist(), dtype=bool)
    smat = sp.Matrix(
        [
            [sp.nan if not tmat_np[i, j] else smat[i, j] for j in range(n)]
            for i in range(n)
        ]
    )
    if not as_nan:
        smat = sp.Matrix(
            [
                [0 if sp.nan == x else x for x in row]
                for row in smat.tolist()
            ]
        )
    if as_abs:
        smat = sp.Abs(smat)
    return sp.Matrix(smat)
