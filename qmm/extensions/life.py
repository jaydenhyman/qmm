"""Analyse change in life expectancy from press perturbations."""

import sympy as sp
from functools import cache
from ..core.structure import create_matrix
from ..core.press import adjoint_matrix
from ..core.helper import get_nodes, get_weight
from typing import Optional, Literal
import networkx as nx

@cache
def birth_matrix(
    G: nx.DiGraph,
    form: Literal["symbolic", "signed"] = "symbolic",
    perturb: Optional[str] = None,
) -> sp.Matrix:
    """Create matrix of direct effects on birth rate from press perturbations.

    Args:
        G: NetworkX DiGraph representing signed digraph model
        form: Type of computation ('symbolic', 'signed')

    Returns:
        sp.Matrix: Positive direct effects on birth rate

    References:
        - Dambacher, J.M., Levins, R., Rossignol, P.A. (2005). Life expectancy change in perturbed communities: Derivation and qualitative analysis. Mathematical Biosciences 197, 1–14.

    Examples:
        ```python
        from qmm import birth_matrix, load_digraph
        birth_matrix(load_digraph("snowshoe"), form='symbolic')
        # Matrix([
        # [    0,     0, 0],
        # [a_H,V,     0, 0],
        # [a_P,V, a_P,H, 0]])

        birth_matrix(load_digraph("snowshoe"), form='signed')
        # Matrix([
        # [0, 0, 0],
        # [1, 0, 0],
        # [1, 1, 0]])
        ```
    """
    A_sgn = create_matrix(G, form="signed")
    A_sym = create_matrix(G, form="symbolic")
    nodes = get_nodes(G, "state")
    if perturb is not None and perturb not in nodes:
        raise ValueError(f"Perturbation node must be one of: {nodes}")
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
def death_matrix(
    G: nx.DiGraph,
    form: Literal["symbolic", "signed"] = "symbolic",
    perturb: Optional[str] = None,
) -> sp.Matrix:
    """Create matrix of direct effects on death rate from press perturbations.

    Args:
        G: NetworkX DiGraph representing signed digraph model
        form: Type of computation ('symbolic', 'signed')

    Returns:
        sp.Matrix: Positive direct effects on death rate

    References:
        - Dambacher, J.M., Levins, R., Rossignol, P.A. (2005). Life expectancy change in perturbed communities: Derivation and qualitative analysis. Mathematical Biosciences 197, 1–14.

    Examples:
        ```python
        from qmm import death_matrix, load_digraph
        death_matrix(load_digraph("snowshoe"), form='symbolic')
        # Matrix([
        # [a_V,V, a_V,H,     0],
        # [    0,     0, a_H,P],
        # [    0,     0, a_P,P]])

        death_matrix(load_digraph("snowshoe"), form='signed')
        # Matrix([
        # [1, 1, 0],
        # [0, 0, 1],
        # [0, 0, 1]])
        ```
    """
    A_sgn = create_matrix(G, form="signed")
    A_sym = create_matrix(G, form="symbolic")
    nodes = get_nodes(G, "state")
    if perturb is not None and perturb not in nodes:
        raise ValueError(f"Perturbation node must be one of: {nodes}")
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
def life_expectancy_change(
    G: nx.DiGraph,
    form: Literal["symbolic", "signed"] = "symbolic",
    type: Literal["birth", "death"] = "birth",
    perturb: Optional[str] = None,
) -> sp.Matrix:
    """Calculate change in life expectancy from press perturbations.

    Args:
        G: NetworkX DiGraph representing signed digraph model
        type: Change in birth or death rate ('birth' or 'death')
        form: Type of computation ('symbolic', 'signed')
        perturb: Node to perturb (None for full matrix)

    Returns:
        sp.Matrix: Change in life expectancy for each component

    References:
        - Dambacher, J.M., Levins, R., Rossignol, P.A. (2005). Life expectancy change in perturbed communities: Derivation and qualitative analysis. Mathematical Biosciences 197, 1–14.

    Examples:
        ```python
        from qmm import life_expectancy_change, load_digraph
        life_expectancy_change(load_digraph("snowshoe"), form='symbolic', type='birth')
        # Matrix([
        # [-a_H,P*a_P,H*a_V,V + a_H,P*a_P,V*a_V,H - a_H,V*a_P,P*a_V,H,                                      0,                  0],
        # [                                        -a_H,P*a_H,V*a_P,H, -a_H,P*a_P,H*a_V,V + a_H,P*a_P,V*a_V,H, -a_H,P*a_H,V*a_V,H],
        # [                                        -a_H,V*a_P,H*a_P,P, -a_P,H*a_P,P*a_V,V + a_P,P*a_P,V*a_V,H, -a_H,V*a_P,P*a_V,H]])

        life_expectancy_change(load_digraph("snowshoe"), form='symbolic', type='death')
        # Matrix([
        # [                 0,                                      0,                                     0],
        # [-a_H,P*a_H,V*a_P,H,                      a_H,V*a_P,P*a_V,H,                    -a_H,P*a_H,V*a_V,H],
        # [-a_H,V*a_P,H*a_P,P, -a_P,H*a_P,P*a_V,V + a_P,P*a_P,V*a_V,H, a_H,P*a_P,H*a_V,V - a_H,P*a_P,V*a_V,H]])
        ```
    """
    amat = adjoint_matrix(G, form=form)
    if type == "birth":
        matrix = death_matrix(G, form=form)
    elif type == "death":
        matrix = birth_matrix(G, form=form)
    else:
        raise ValueError("type must be either 'birth' or 'death'")
    result = sp.expand(sp.Integer(-1) * matrix * amat)
    if perturb is not None:
        nodes = get_nodes(G, "state")
        if perturb not in nodes:
            raise ValueError(f"Perturbation node must be one of: {nodes}")
        perturb_index = nodes.index(perturb)
        return result.col(perturb_index)
    return result

@cache
def net_life_expectancy_change(
    G: nx.DiGraph,
    type: Literal["birth", "death"] = "birth",
) -> sp.Matrix:
    """Calculate net terms in life expectancy change from press perturbations.

    Args:
        G: NetworkX DiGraph representing signed digraph model
        type: Change in birth or death rate ('birth' or 'death')

    Returns:
        sp.Matrix: Net life expectancy change for each component

    References:
        - Dambacher, J.M., Levins, R., Rossignol, P.A. (2005). Life expectancy change in perturbed communities: Derivation and qualitative analysis. Mathematical Biosciences 197, 1–14.

    Examples:
        ```python
        from qmm import net_life_expectancy_change, load_digraph
        net_life_expectancy_change(load_digraph("snowshoe"), type='birth')
        # Matrix([
        # [-1, 0,  0],
        # [-1, 0, -1],
        # [-1, 0, -1]])

        net_life_expectancy_change(load_digraph("snowshoe"), type='death')
        # Matrix([
        # [ 0, 0,  0],
        # [-1, 1, -1],
        # [-1, 0,  0]])
        ```
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
def absolute_life_expectancy_change(
    G: nx.DiGraph,
    type: Literal["birth", "death"] = "birth",
) -> sp.Matrix:
    """Calculate absolute terms in life expectancy change from press perturbations.

    Args:
        G: NetworkX DiGraph representing signed digraph model
        type: Change in birth or death rate ('birth' or 'death')

    Returns:
        sp.Matrix: Absolute life expectancy change for each component

    References:
        - Dambacher, J.M., Levins, R., Rossignol, P.A. (2005). Life expectancy change in perturbed communities: Derivation and qualitative analysis. Mathematical Biosciences 197, 1–14.

    Examples:
        ```python
        from qmm import absolute_life_expectancy_change, load_digraph
        absolute_life_expectancy_change(load_digraph("snowshoe"), type='birth')
        # Matrix([
        # [3, 0, 0],
        # [1, 2, 1],
        # [1, 2, 1]])

        absolute_life_expectancy_change(load_digraph("snowshoe"), type='death')
        # Matrix([
        # [0, 0, 0],
        # [1, 1, 1],
        # [1, 2, 2]])
        ```
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
def weighted_predictions_life_expectancy(
    G: nx.DiGraph,
    type: Literal["birth", "death"] = "birth",
    as_nan: bool = True,
    as_abs: bool = False,
) -> sp.Matrix:
    """Calculate ratio of net to total change in life expectancy.

    Args:
        G: NetworkX DiGraph representing signed digraph model
        type: Change in birth or death rate ('birth' or 'death')
        as_nan: Return NaN for undefined ratios
        as_abs: Return absolute values

    Returns:
        sp.Matrix: Net-to-total ratios for life expectancy predictions

    References:
        - Dambacher, J.M., Levins, R., Rossignol, P.A. (2005). Life expectancy change in perturbed communities: Derivation and qualitative analysis. Mathematical Biosciences 197, 1–14.

    Examples:
        ```python
        from qmm import weighted_predictions_life_expectancy, load_digraph
        weighted_predictions_life_expectancy(load_digraph("snowshoe"), type='birth', as_abs=False, as_nan=True)
        # Matrix([
        # [-1/3, nan, nan],
        # [  -1,   0,  -1],
        # [  -1,   0,  -1]])

        weighted_predictions_life_expectancy(load_digraph("snowshoe"), type='birth', as_abs=True, as_nan=False)
        # Matrix([
        # [1/3, 1, 1],
        # [  1, 0, 1],
        # [  1, 0, 1]])

        weighted_predictions_life_expectancy(load_digraph("snowshoe"), type='death', as_abs=False, as_nan=True)
        # Matrix([
        # [nan, nan, nan],
        # [ -1,   1,  -1],
        # [ -1,   0,   0]])

        weighted_predictions_life_expectancy(load_digraph("snowshoe"), type='death', as_abs=True, as_nan=False)
        # Matrix([
        # [1, 1, 1],
        # [1, 1, 1],
        # [1, 0, 0]])
        ```
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
