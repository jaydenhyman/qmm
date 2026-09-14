# Changelog

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
