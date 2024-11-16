import numpy as np
import sympy as sp
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple, Union, List, Optional

def table_of_predictions(M: Union[sp.Matrix, np.ndarray], t1: float = 0.8, t2: float = 1.0,
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
    if isinstance(M, sp.Matrix):
        M = sp.matrix2numpy(M, dtype=float)
    conditions = [
        (lambda x: x == sp.nan, "0"),
        (lambda x: x >= t2, "+"),
        (lambda x: x >= t1 and x < t2, "(+)"),
        (lambda x: x > -t1 and x < t1, "?"),
        (lambda x: x > -t2 and x <= -t1, "(\u2212)"),
        (lambda x: x <= -t2, "\u2212"),
    ]
    predictions = np.empty(M.shape, dtype=object)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            value = M[i, j]
            predictions[i, j] = next((val for cond, val in conditions if cond(value)), "0")
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

def create_plot(data: pd.DataFrame, **kwargs: Any) -> Tuple[plt.Figure, plt.Axes]:
    """Create heatmap visualization of qualitative predictions matrix.

    Args:
        data: Matrix of qualitative predictions
        **kwargs: Additional arguments passed to seaborn.heatmap
        
    Returns:
        tuple[Figure, Axes]: Matplotlib objects for customizing visualization
    """
    plt.rcParams.update(
        {
            "xtick.top": True,
            "xtick.bottom": False,
            "xtick.labeltop": True,
            "xtick.labelbottom": False,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.minor.size": 1.5,
            "ytick.minor.size": 1.5,
        }
    )
    args: Dict[str, Any] = {
        "annot": True,
        "linewidths": 0.75,
        "linecolor": "white",
        "cbar": False,
        "cmap": None,
    }
    args.update(kwargs)
    figsize = args.pop("figsize", None)
    if figsize:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = plt.subplots()
    sns.heatmap(data, ax=ax, **args)
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    plt.setp(ax.get_yticklabels(), rotation=0, ha="right")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("white")
        spine.set_linewidth(0.5)
    return fig, ax
