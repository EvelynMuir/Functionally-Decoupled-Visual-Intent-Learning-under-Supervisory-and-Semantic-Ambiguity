# Functionally Decoupled Visual Intent Learning under Supervisory and Semantic Ambiguity

Supplementary data for revision 3. Approximately 29.2 MiB.

- `manifest.json`: file provenance, seeds, generation settings and missing metadata.
- `prompts/`, `semantic/`: prompt templates, cached priors and compressed rationale records.
- `inputs/`, `reports/`: saved numeric inputs and statistical/error-analysis results.
- `sources/`: source snapshots for provenance; full training requires the original project and datasets. Machine-specific paths are replaced with placeholders.

## Reproduce reports

```bash
python -m pip install -r requirements.txt
python build_reports.py
```

The script recomputes `reports/` and generates LaTeX tables in `tables/` using only bundled inputs. No GPU or original images are needed.

Paired-seed intervals are pointwise 95% Student-t intervals. Intentonomy tests use Holm correction across five comparators per metric; ambiguity-cell tests use a separate family of 12 contrasts. Transfer tests retain their paired-seed protocols. Bootstrap results are saved 5,000-replicate estimates, conditional on selected checkpoints. Error analysis uses one five-fold training-set partition; conditional contrasts are descriptive because cross-fitted models share training images.

Raw images, model weights and large feature caches are excluded. This package supports reanalysis; unresolved generation metadata is recorded in the manifest.
