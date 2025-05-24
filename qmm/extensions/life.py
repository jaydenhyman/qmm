"""Analyse change in life expectancy from press perturbations."""

import sympy as sp
from functools import cache
from ..core.structure import create_matrix
from ..core.press import adjoint_matrix
from ..core.helper import get_nodes, get_weight
from typing import Optional
import networkx as nx

@cache
def birth_matrix(G: nx.DiGraph, form: str = "symbolic", perturb: Optional[str] = None) -> sp.Matrix:
    """Create matrix of direct effects on birth rate from press perturbations.

    Args:
        G: NetworkX DiGraph representing signed digraph model
        form: Type of computation ('symbolic', 'signed')
        
    Returns:
        sp.Matrix: Positive direct effects on birth rate
    """
    A_sgn = create_matrix(G, form="signed")
    A_sym = create_matrix(G, form="symbolic")
    nodes = get_nodes(G, "state")
    n = len(nodes)
    def birth_element(i, j):
        if form == "symbolic":
            return A_sym[i, j] if A_sgn[i, j] > 0 else 0
        else:  # form == 'signed'
            return sp.Integer(1) if A_sgn[i, j] > 0 else 0
    if perturb is not None:
        src_id = nodes.index(perturb)
        return sp.Matrix(n, 1, lambda i, j: birth_element(i, src_id))
    else:
        return sp.Matrix(n, n, lambda i, j: birth_element(i, j))

@cache
def death_matrix(G: nx.DiGraph, form: str = "symbolic", perturb: Optional[str] = None) -> sp.Matrix:
    """Create matrix of direct effects on death rate from press perturbations.

    Args:
        G: NetworkX DiGraph representing signed digraph model
        form: Type of computation ('symbolic', 'signed')
        
    Returns:
        sp.Matrix: Positive direct effects on death rate
    """
    A_sgn = create_matrix(G, form="signed")
    A_sym = create_matrix(G, form="symbolic")
    nodes = get_nodes(G, "state")
    n = len(nodes)
    def death_element(i, j):
        if form == "symbolic":
            return A_sym[i, j] * sp.Integer(-1) if A_sgn[i, j] < 0 else 0
        else:  # form == 'signed'
            return sp.Integer(1) if A_sgn[i, j] < 0 else 0
    if perturb is not None:
        src_id = nodes.index(perturb)
        return sp.Matrix(n, 1, lambda i, j: death_element(i, src_id))
    else:
        return sp.Matrix(n, n, lambda i, j: death_element(i, j))

@cache
def life_expectancy_change(G: nx.DiGraph, form: str = "symbolic", type: str = "birth", perturb: Optional[str] = None) -> sp.Matrix:
    """Calculate change in life expectancy from press perturbations.

    Args:
        G: NetworkX DiGraph representing signed digraph model
        type: Change in birth or death rate ('birth' or 'death')
        form: Type of computation ('symbolic', 'signed')
        perturb: Node to perturb (None for full matrix)
        
    Returns:
        sp.Matrix: Change in life expectancy for each component
    """
    amat = adjoint_matrix(G, form=form)
    if type == "birth":
        matrix = death_matrix(G, form=form)
    else:  # type == 'death'
        matrix = birth_matrix(G, form=form)
    result = sp.expand(sp.Integer(-1) * matrix * amat)
    if perturb is not None:
        nodes = get_nodes(G, "state")
        perturb_index = nodes.index(perturb)
        return result.col(perturb_index)
    return result

@cache
def net_life_expectancy_change(G: nx.DiGraph, type: str = "birth") -> sp.Matrix:
    """Calculate net terms in life expectancy change from press perturbations.

    Args:
        G: NetworkX DiGraph representing signed digraph model
        type: Change in birth or death rate ('birth' or 'death')
        
    Returns:
        sp.Matrix: Net life expectancy change for each component
    """
    amat = adjoint_matrix(G, form="signed")
    birth = birth_matrix(G, form="signed")
    death = death_matrix(G, form="signed")
    delta_birth = death * amat * sp.Integer(-1)
    delta_death = birth * amat * sp.Integer(-1)
    if type == "birth":
        return delta_birth
    else:
        return delta_death

@cache
def absolute_life_expectancy_change(G: nx.DiGraph, type: str = "birth") -> sp.Matrix:
    """Calculate absolute terms in life expectancy change from press perturbations.

    Args:
        G: NetworkX DiGraph representing signed digraph model
        type: Change in birth or death rate ('birth' or 'death')
        
    Returns:
        sp.Matrix: Absolute life expectancy change for each component
    """
    sym_amat = adjoint_matrix(G, form="symbolic")
    n = sym_amat.shape[0]
    sym_birth = birth_matrix(G, form="symbolic")
    sym_death = death_matrix(G, form="symbolic")
    sym_delta_birth = sp.expand(sp.Integer(-1) * sym_death * sym_amat)
    sym_delta_death = sp.expand(sp.Integer(-1) * sym_birth * sym_amat)

    def count_symbols(matrix_element):
        return sum(matrix_element.count(sym) for sym in matrix_element.free_symbols)

    def create_abs_matrix(sym_delta_matrix, n):
        return sp.Matrix(n, n, lambda i, j: count_symbols(sym_delta_matrix[i, j]) // n)

    abs_birth = create_abs_matrix(sym_delta_birth, n)
    abs_death = create_abs_matrix(sym_delta_death, n)
    if type == "birth":
        return abs_birth
    else:
        return abs_death

@cache
def weighted_predictions_life_expectancy(G: nx.DiGraph, type: str = "birth", 
                                      as_nan: bool = True, as_abs: bool = False) -> sp.Matrix:
    """Calculate ratio of net to total change in life expectancy.

    Args:
        G: NetworkX DiGraph representing signed digraph model
        type: Change in birth or death rate ('birth' or 'death')
        as_nan: Return NaN for undefined ratios
        as_abs: Return absolute values
        
    Returns:
        sp.Matrix: Net-to-total ratios for life expectancy predictions
    """
    if type == "birth":
        net = net_life_expectancy_change(G, type="birth")
        absolute = absolute_life_expectancy_change(G, type="birth")
    elif type == "death":
        net = net_life_expectancy_change(G, type="death")
        absolute = absolute_life_expectancy_change(G, type="death")
    else:
        raise ValueError("type must be either 'birth' or 'death'")
    if as_nan:
        weighted = get_weight(net, absolute)
    else:
        weighted = get_weight(net, absolute, sp.Integer(1))
    if as_abs:
        weighted = sp.Abs(weighted)
    return weighted

