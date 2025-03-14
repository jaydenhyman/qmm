from .core.structure import (
    import_digraph,
    create_matrix,
    create_equations,
)

from .core.stability import (
    sign_stability,
    system_feedback,
    net_feedback,
    absolute_feedback,
    weighted_feedback,
    feedback_metrics,
    hurwitz_determinants,
    net_determinants,
    absolute_determinants,
    weighted_determinants,
    determinants_metrics,
    conditional_stability,
    simulation_stability,
)

from .core.press import (
    adjoint_matrix,
    absolute_feedback_matrix,
    weighted_predictions_matrix,
    sign_determinacy_matrix,
    numerical_simulations,
)

from .core.prediction import (
    table_of_predictions,
    compare_predictions,
)

from .core.helper import (
    list_to_digraph,
    digraph_to_list,
    get_nodes,
    get_positive,
    get_negative,
    get_weight,
    sign_determinacy,
)

from .extensions.senstability import (
    structural_sensitivity,
    net_structural_sensitivity,
    absolute_structural_sensitivity,
    weighted_structural_sensitivity,
)

from .extensions.life import (
    birth_matrix,
    death_matrix,
    life_expectancy_change,
    net_life_expectancy_change,
    absolute_life_expectancy_change,
    weighted_predictions_life_expectancy,
)

from .extensions.paths import (
    get_paths,
    paths_table,
    get_cycles,
    cycles_table,
    complementary_feedback,
    system_paths,
    weighted_paths,
    path_metrics,
)

from .extensions.effects import (
    define_input_output,
    cumulative_effects,
    absolute_effects,
    weighted_effects,
    sign_determinacy_effects,
    get_simulations,
    simulation_effects,
)

from .extensions.indicators import (
    mutual_information,
)

from .extensions.validation import (
    marginal_likelihood,
    model_validation,
    posterior_predictions,
    diagnose_observations,
    bayes_factors,
)

import pandas as pd

def configure_pandas_display(max_columns=None, max_rows=None, max_colwidth=None, display_width=None):
    pd.set_option('display.max_columns', max_columns)
    pd.set_option('display.max_rows', max_rows)
    pd.set_option('display.max_colwidth', max_colwidth)
    pd.set_option('display.width', display_width)
    pd.set_option('display.html.use_mathjax', True)

configure_pandas_display()

__all__ = [
    # structure.py
    "import_digraph",
    "create_matrix",
    "create_equations",
    # stability.py
    "sign_stability",
    "system_feedback",
    "net_feedback",
    "absolute_feedback",
    "weighted_feedback",
    "feedback_metrics",
    "hurwitz_determinants",
    "net_determinants",
    "absolute_determinants",
    "weighted_determinants",
    "determinants_metrics",
    "conditional_stability",
    "simulation_stability",
    # press.py
    "adjoint_matrix",
    "absolute_feedback_matrix",
    "weighted_predictions_matrix",
    "sign_determinacy_matrix",
    "numerical_simulations",
    # prediction.py
    "table_of_predictions",
    "compare_predictions",
    # helper.py
    "list_to_digraph",
    "digraph_to_list",
    "get_nodes",
    "get_positive",
    "get_negative",
    "get_weight",
    "sign_determinacy",
    # senstability.py
    "structural_sensitivity",
    "net_structural_sensitivity",
    "absolute_structural_sensitivity",
    "weighted_structural_sensitivity",
    # life.py
    "birth_matrix",
    "death_matrix",
    "life_expectancy_change",
    "net_life_expectancy_change",
    "absolute_life_expectancy_change",
    "weighted_predictions_life_expectancy",
    # paths.py
    "get_paths",
    "paths_table",
    "get_cycles",
    "cycles_table",
    "complementary_feedback",
    "system_paths",
    "weighted_paths",
    "path_metrics",
    # effects.py
    "define_input_output",
    "cumulative_effects",
    "absolute_effects",
    "weighted_effects",
    "sign_determinacy_effects",
    "get_simulations",
    "simulation_effects",
    # indicators.py
    "mutual_information",
    # validation.py
    "marginal_likelihood",
    "model_validation",
    "posterior_predictions",
    "diagnose_observations",
    "bayes_factors",
]
