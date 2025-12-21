"""Generate qualitative predictions of system response to press perturbations with thresholds for ambiguity."""

import numpy as np
import sympy as sp
import pandas as pd
import networkx as nx
from typing import Union, List, Optional, Callable
from .press import numerical_simulations

def _apply_thresholds(
    M: Union[sp.Matrix, np.ndarray],
    t1: float,
    t2: float,
) -> list[list[str]]:
    if isinstance(M, sp.Matrix):
        M = sp.matrix2numpy(M, dtype=float)

    def label(value: float) -> str:
        try:
            is_nan = bool(np.isnan(value))
        except TypeError:
            is_nan = (value == sp.nan)
        if is_nan:
            return "0"
        if value >= t1:
            return "+" if value >= t2 else "(+)"
        if value <= -t1:
            return "−" if value <= -t2 else "(−)"
        return "?"

    rows, cols = M.shape
    return [[label(M[i, j]) for j in range(cols)] for i in range(rows)]

def qualitative_predictions(
    G: nx.DiGraph,
    generator: Callable[..., Union[sp.Matrix, np.ndarray]] = numerical_simulations,
    t1: float = 0.8,
    t2: float = 0.95,
) -> sp.Matrix:
    """Create a sympy matrix of qualitative predictions with thresholds for ambiguity.

    Args:
        G (nx.DiGraph): Graph input for the matrix generator
        generator (Callable): Matrix generator function
        t1 (float): Lower threshold for likely predictions
        t2 (float): Upper threshold for determined predictions

    Returns:
        sp.Matrix: Qualitative predictions
    """
    matrix = generator(G)
    predictions = _apply_thresholds(matrix, t1, t2)
    rows = [[sp.Integer(0) if vals == "0" else sp.Symbol(vals) for vals in row] for row in predictions]
    return sp.Matrix(rows)

def table_of_predictions(M: Union[sp.Matrix, np.ndarray], t1: float = 0.8, t2: float = 0.95,
                        index: Optional[List[str]] = None,
                        columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Create a table of qualitative predictions with thresholds for ambiguity.

    Args:
        M (Union[sp.Matrix, np.ndarray]): Matrix of predictions from press perturbation analysis
        t1 (float): Lower threshold for likely predictions
        t2 (float): Upper threshold for determined predictions
        index (Optional[List[str]]): Row labels
        columns (Optional[List[str]]): Column labels

    Returns:
        pd.DataFrame: Qualitative predictions
    """
    if isinstance(M, sp.Matrix) and any(isinstance(v, (sp.Symbol, str)) for v in M):
        return pd.DataFrame(M.tolist(), index=index, columns=columns)
    predictions = _apply_thresholds(M, t1, t2)
    return pd.DataFrame(predictions, index=index, columns=columns)

def compare_predictions(M1: pd.DataFrame, M2: pd.DataFrame) -> pd.DataFrame:
    """Compare predictions between alternative models or prediction methods.

    Args:
        M1 (pd.DataFrame): First matrix of predictions
        M2 (pd.DataFrame): Second matrix of predictions

    Returns:
        pd.DataFrame: Combined predictions showing differences and agreements
    """
    if not M1.index.equals(M2.index) or not M1.columns.equals(M2.columns):
        raise ValueError("M1 and M2 must have the same index and columns")
    M1_str = M1.astype(str)
    M2_str = M2.astype(str)
    combined = pd.DataFrame(
        index=M1.index,
        columns=M1.columns,
        data=np.where(M1_str.values == M2_str.values, M1_str.values, M1_str.values + ", " + M2_str.values),
    )
    return combined
