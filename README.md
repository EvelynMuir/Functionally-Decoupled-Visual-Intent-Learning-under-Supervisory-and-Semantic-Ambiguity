# Functionally Decoupled Visual Intent Learning under Supervisory and Semantic Ambiguity

Supplementary data for revision 4.

- `manifest.json`: file inventory, model identifiers, generation parameters, source defaults, training seeds, Holm families, and regeneration boundaries.
- `prompts/`, `semantic/`: Intentonomy and EMOTIC scenario templates and class-wise requests, LDL prompt templates, cached priors, and gzip-compressed rationale records.
- `inputs/`: per-run measurements, five-seed unified-versus-decoupled controls (`unified_seed_results.json`), local prediction-margin overlap audits (`margin_proxy_overlap.json`), bootstrap outputs, and compact five-fold cross-fitted out-of-fold arrays (`intentonomy_oof_predictions.npz`, `oof_protocol.json`).
- `reports/`: paired benchmark differences, five-seed unified-versus-decoupled statistics (`unified_paired.csv`), fold-level standard and Nadeau-Bengio within-cell tests (`quadrant_paired.csv`), between-group difference-in-differences contrasts (`quadrant_did.csv`), agreement-group gate alignment (`gate_profile.csv`), stratified errors, confusion pairs, and diagnostic cases.
- `sources/`: semantic-resource generation/encoding scripts and an excerpt of the recorded Intentonomy generation commands. These scripts require the original project, datasets, and model checkpoints; machine-specific paths use placeholders.

## Scenario priors and prompts

Intentonomy, EMOTIC, and LDL priors (Flickr-LDL and Emotion6) use Gemini 3.1 Pro Preview with temperature 1 and thinking level high. These settings apply to scenario-prior generation across all four datasets.

The EMOTIC template follows the same archetype-generation structure as Intentonomy. Each request substitutes an emotion name and its exact definition from `semantic/emotic_prior.json`, preserving the order of all 26 classes. The template requests 3-5 distinct visual archetypes per class, each with a core difference, visual elements, and a 20-40-word text query. The shared template and all 26 expanded requests are provided in `prompts/emotic_scenario_template.txt` and `prompts/emotic_scenario_requests.json`.

## Reproduce reports

```bash
python -m pip install -r requirements.txt
python build_reports.py
```

The script recomputes base paired benchmark reports, stratified error summaries, confusion pairs, and diagnostic cases from `inputs/` and writes LaTeX tables in `tables/` without needing a GPU or original images. Revision 4 also includes pre-exported reports for `unified_paired.csv`, fold-level standard and Nadeau-Bengio `quadrant_paired.csv`, `quadrant_did.csv`, and `gate_profile.csv` matching Supplementary Tables S4 and S10-S12.

Paired-seed intervals are pointwise 95% Student-t intervals (`df=4`). Intentonomy comparisons use Holm correction across five comparators per metric, unified-versus-decoupled comparisons use Holm correction across three comparators per metric, and ambiguity-cell tests correct twelve within-cell contrasts evaluated at the fold level (`K=5`) alongside Nadeau-Bengio overlap correction (`0.45 * s^2`).

## Reproducibility boundaries

Raw images, model weights, and large feature caches are excluded. Cached artifacts, seeds, and compact analysis inputs support deterministic retraining and numerical reanalysis, whereas regenerating external text resources from scratch remains subject to closed-source model versioning for Gemini 3.1 Pro Preview and non-zero sampling temperature.
