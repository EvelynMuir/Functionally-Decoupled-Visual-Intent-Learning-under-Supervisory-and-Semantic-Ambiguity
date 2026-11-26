# Functionally Decoupled Visual Intent Learning under Supervisory and Semantic Ambiguity

Supplementary data and reanalysis inputs for the paper.

- `manifest.json`: file inventory, model identifiers, generation parameters, source defaults, training seeds, Holm families, and regeneration boundaries.
- `prompts/`, `semantic/`: Intentonomy and EMOTIC scenario templates and class-wise requests, LDL prompt templates, cached priors, and gzip-compressed rationale records.
- `inputs/`: per-run measurements, five-seed unified-versus-decoupled controls (`unified_seed_results.json`), bootstrap outputs, and compact five-fold cross-fitted out-of-fold arrays (`intentonomy_oof_predictions.npz`, `oof_protocol.json`).
- `reports/`: paired benchmark differences, five-seed unified-versus-decoupled statistics (`unified_paired.csv`), fold-level standard and Nadeau-Bengio within-cell tests (`quadrant_paired.csv`), between-group difference-in-differences contrasts (`quadrant_did.csv`), agreement-group gate alignment (`gate_profile.csv`), stratified errors, confusion pairs, and diagnostic cases.
- `sources/`: semantic-resource generation and encoding scripts together with the Intentonomy generation command summary. Running these scripts from scratch requires the datasets and pretrained checkpoints; machine-specific paths use placeholders.

## Scenario priors and prompts

Intentonomy, EMOTIC, and LDL priors (Flickr-LDL and Emotion6) use Gemini 3.1 Pro Preview with temperature 1 and thinking level high. The EMOTIC template mirrors the Intentonomy archetype structure across all 26 emotion categories (3-5 archetypes per class and 20-40 words per text query).

## Reproduce reports

```bash
python -m pip install -r requirements.txt
python build_reports.py
```

The script recomputes base paired benchmark reports, stratified error summaries, confusion pairs, and diagnostic cases from `inputs/` and writes LaTeX tables in `tables/` without needing a GPU or original images. Updated CSV reports for `unified_paired.csv`, fold-level standard and Nadeau-Bengio `quadrant_paired.csv`, `quadrant_did.csv`, and `gate_profile.csv` are provided under `reports/`.

## Reproducibility boundaries

Raw images, model weights, and large feature caches are excluded. Cached artifacts, seeds, and compact analysis inputs support deterministic retraining and numerical reanalysis, whereas regenerating external text resources from scratch remains subject to closed-source model versioning for Gemini 3.1 Pro Preview and non-zero sampling temperature.
