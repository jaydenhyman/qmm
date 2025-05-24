"""Extensions for specialised analyses."""

from .senstability import (
    structural_sensitivity,
    net_structural_sensitivity,
    absolute_structural_sensitivity,
    weighted_structural_sensitivity,
)

from .life import (
    birth_matrix,
    death_matrix,
    life_expectancy_change,
    net_life_expectancy_change,
    absolute_life_expectancy_change,
    weighted_predictions_life_expectancy,
)

from .paths import (
    get_paths,
    paths_table,
    get_cycles,
    cycles_table,
    complementary_feedback,
    system_paths,
    weighted_paths,
    path_metrics,
)

from .effects import (
    define_input_output,
    cumulative_effects,
    absolute_effects,
    weighted_effects,
    sign_determinacy_effects,
    get_simulations,
    simulation_effects,
)

from .indicators import (
    mutual_information,
)

from .validation import (
    marginal_likelihood,
    model_validation,
    posterior_predictions,
    diagnose_observations,
    bayes_factors,
)

__all__ = [
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