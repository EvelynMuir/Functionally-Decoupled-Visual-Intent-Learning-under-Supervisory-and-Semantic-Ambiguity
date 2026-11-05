# Functionally Decoupled Visual Intent Learning under Supervisory and Semantic Ambiguity

Supplementary data for revision 3.

- `manifest.json`: file provenance, seeds, generation settings and missing metadata.
- `prompts/`, `semantic/`: Intentonomy/EMOTIC scenario templates and class-wise requests, LDL prompt templates, cached priors and compressed rationale records.
- `inputs/`, `reports/`: per-run measurements, compact out-of-fold predictions, analysis settings and complete statistical/error-analysis results.
- `sources/`: semantic-resource generation/encoding scripts and an excerpt of the recorded Intentonomy generation commands. These scripts require the original project, datasets and model checkpoints; machine-specific paths are replaced with placeholders.

## Scenario priors and prompts

Intentonomy, EMOTIC and LDL priors (Flickr-LDL and Emotion6) use Gemini 3.1 Pro Preview with temperature 1 and thinking level high. These author-confirmed settings apply to scenario-prior generation across all four datasets.

The EMOTIC template follows the author-specified Intentonomy archetype-generation structure. Each request substitutes an emotion name and its exact definition from `semantic/emotic_prior.json`, preserving the order of all 26 classes. The template requests 3–5 distinct visual archetypes per class, each with a core difference, visual elements and a 20–40-word text query. The shared template and all 26 expanded requests are provided in `prompts/emotic_scenario_template.txt` and `prompts/emotic_scenario_requests.json`. Historical request/response session logs remain unavailable; the manifest distinguishes this author-specified documentation from recorded generation settings and source defaults.

## Reproduce reports

```bash
python -m pip install -r requirements.txt
python build_reports.py
```

The script recomputes `reports/` and generates LaTeX tables in `tables/` using only bundled inputs. No GPU or original images are needed.

Paired-seed intervals are pointwise 95% Student-t intervals. Intentonomy tests use Holm correction across five comparators per metric; ambiguity-cell tests use a separate family of 12 contrasts. Transfer comparisons retain their original paired tests across runs. Bootstrap results are saved 5,000-replicate estimates, conditional on selected checkpoints. Error analysis uses one five-fold training-set partition; conditional contrasts are descriptive because cross-fitted models share training images.

Raw images, model weights and large feature caches are excluded. This package supports reanalysis; unresolved generation metadata is recorded in the manifest.
