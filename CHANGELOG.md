# Changelog

## 0.7.0

Changes since 0.6.0:

- Direct links from an input node to an output node are supported. `create_matrix(..., matrix_type="D")` is built from those links, including chains through other input and output nodes, instead of returning zeros or raising on a direct link.
- `cumulative_effects` adds the feedthrough term `D·det(−A)` to the input-to-output block; in the binary form the term is `D` times the permanent of `A`.
- `system_paths`, `weighted_paths`, `path_metrics` and `pathway_effects` accept models with direct input-to-output links instead of raising.
- Simulations raise `ValueError` for a model with no state nodes.
- `pathway_effects` reports each path's signed share of the pathway effect in place of the mean `Contribution`. A path's share in a draw is its effect divided by the sum of the absolute effects of every path in that draw. The table gives `Mean`, `Standard deviation`, `5th percentile`, `Lower quartile`, `Median`, `Upper quartile` and `95th percentile` of the share, `Dominant` (the share of draws in which the path has the largest absolute effect) and `Undefined` (the share of draws in which every path has zero effect, which have no shares), and is sorted by mean absolute share.
- `pathway_effects(..., draws=True)` returns one row per draw and path with `Draw`, `Structure`, `Path`, `Present`, `Effect` and `Share`.
- `pathway_effects` returns an empty table with the new columns when no path exists.
- `qmm.__version__` gives the installed version.

### Migration

The `Contribution` column of `pathway_effects` is removed. `Mean` is not a rename: `Contribution` was the mean absolute share with zero-effect draws counted as zero, whereas `Mean` is signed and excludes those draws, which `Undefined` counts. Callers that need the absolute share can take it from the `Share` column with `draws=True`. Callers that relied on a `ValueError` from `create_matrix(..., matrix_type="D")`, `cumulative_effects` or the path functions to detect direct input-to-output links must check for those links themselves. For models with such links, `D` is no longer zero and the input-to-output block of `cumulative_effects` includes the feedthrough term. Simulating a model without state nodes now raises `ValueError`.

## 0.6.0

Changes since 0.5.0:

- `define_input_output` accepts `remove_disconnected` again; it defaults to False, and True keeps the largest subnetwork with a warning naming the dropped nodes.
- Import stores accepted float signs as integers and reports missing `nodes`, `edges`, node `id` or edge `from`/`to` plainly.
- `import_digraph` infers a mapping or a file path; the `file_path` argument is removed.
- `numerical_simulations` takes `mode` ('dominant', 'absolute', 'positive', 'match_adjoint') instead of `as_abs`, `positive_only` and `match_adjoint`, and no longer takes `as_nan`; cells with no effect are always NaN. `simulation_effects` and `posterior_predictions` take `mode` ('dominant', 'positive') instead of `positive_only`.
- Press and observation signs passed as tuples must be -1, 0 or 1. A zero observation matches only an exact zero response; cancellation under tied strengths is not treated as no effect.
- `iter_simulations` is private; `get_simulations` is the public entry point for simulations.
- Enumerating uncertain interactions drops a structure with no matching draw, with a warning naming it, instead of failing the whole run; if no structure matches, the run fails.
- Path functions return an empty table with their columns when no path exists, instead of a sentinel zero row or None.
- `compare_model_alternatives(combinations=False)` compares the model without uncertain interactions against each single interaction added, matching `get_dashed_alternatives`, and returns numeric likelihoods.
- `get_dashed_alternatives` is exported.
- Structural sensitivity results follow the current graph; the result caches are removed.
- Path metrics reuse computed feedback values, pathway simulations reuse the cycle cover check, and stability metric tables reuse computed net and absolute values.
- Every table uses sentence case with spaces. `simulations_table` columns are `Model`, `Effect on`, `Negative`, `Zero`, `Positive`, `Valid draws`, `Stable draws` and `Attempts`; `mutual_information` returns `Mutual information`. The zero-response count is `Zero` wherever it appears.
- `mutual_information` takes `dist`, in the position the other simulation entry points use.
- `stability_analysis` is removed; `sign_stability`, `conditional_stability` and `simulation_stability` are unchanged.
- Requires Python 3.10 or newer; NumPy 2 is supported.

### Migration

Replace `as_abs=True` with `mode='absolute'`, `positive_only=True` with `mode='positive'` and `match_adjoint=True` with `mode='match_adjoint'`; drop `as_nan` from `numerical_simulations` calls. Replace `iter_simulations` with `get_simulations`, which concatenates enumerated structures into one result rather than yielding a batch per structure. Path functions (`get_paths`, `complementary_feedback`, `system_paths`, `weighted_paths`, `paths_table`, `path_metrics`) return an empty table when no path exists, so callers that tested for one sentinel row or for None must test for an empty table. Callers of `compare_model_alternatives` no longer need to parse the likelihood column. Calls with `combinations=False` now receive one row per uncertain interaction plus the base model. Rename column lookups on `simulations_table` and `mutual_information` results. Concatenate the three stability tables where `stability_analysis` was used. Callers passing `seed` positionally to `mutual_information` must move it after `dist`.

## 0.5.0

Changes since 0.4.17:

- Validate imported node identities, directed edges and interaction signs; preserve model metadata and derive node roles automatically.
- Reject imported models with multiple weakly connected components. `define_input_output` no longer accepts `remove_disconnected`; clean up the model explicitly before analysis. Directly constructed graphs are trusted.
- Sample or enumerate uncertain interactions, retain structural zeros, and reject alternatives that change node roles. Reciprocal uncertain interactions can be grouped.
- Support simultaneous press perturbations and observation conditioning. `get_simulations(condition=False)` collects a fixed number of stable draws per batch (per structure when enumerating) before observation filtering.
- Reuse saved simulation strengths for pathway decomposition and report pathway presence separately from its response sign and contribution.
- Correct the feedforward term in cumulative effects and retain available sections of stability tables when conditional stability is undefined.
- Restrict input/output path searches to permitted intermediate nodes, avoiding enumeration of irrelevant state pathways in large models.
- Improve numerical term counting, consistency between modules, mathematical regression tests and the getting-started examples.

### Migration

Update callers that pass `remove_disconnected`. Import now refuses ambiguous identities, invalid signs and disconnected models instead of silently repairing them. Alternative comparisons require consistent node roles, and enumerated sampling can fail when a requested structure cannot supply enough accepted draws within its attempt limit.

`average_uncertain` has been replaced by `uncertain_interactions="sample"` or `"enumerate"`; sampling is now the default. These modes have different structure-weighting and stopping rules; this is not a boolean rename. Reciprocal uncertain edges are grouped by default; pass `pair_reciprocal=False` to sample directed edges independently. To analyse a fixed topology, first select the desired graph and clear its uncertain-edge flags (`dashes`).

`model_validation` is now named `compare_model_alternatives`.

Simulation results now include perturbation and structure metadata. Pathway summaries include a `Present` column. Cached simulations for pathway analysis must come from the current API with `return_samples=True`.
