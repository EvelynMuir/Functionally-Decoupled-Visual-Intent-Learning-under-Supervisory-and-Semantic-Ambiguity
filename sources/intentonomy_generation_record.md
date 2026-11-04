# Intentonomy rationale-generation record

Historical generation and text-encoding commands from the 2026-03-16 experiment record. Paths are placeholders or original project-relative paths. These commands require the original project, datasets and model checkpoints. Source defaults are distinguished from recorded command-line settings in the manifest.

## Training rationales

```bash
python -u scripts/generate_vlm_rationales.py \
  --output-jsonl logs/analysis/vlm_full_20260316/rationale_full.jsonl \
  --mode rationale \
  --temperature 0.0 \
  --max-tokens 768 \
  --max-image-size 336 \
  --request-batch-size 256
```

## Validation/test rationales

```bash
python -u scripts/generate_vlm_rationales.py \
  --base-cache logs/analysis/min_agent_evidence_verification_v2_comparative_add_20260312/_cache/val_base.npz \
  --annotation-file ${DATA_ROOT}/Intentonomy/data/annotation/intentonomy_val2020.json \
  --image-dir ${DATA_ROOT}/Intentonomy/data/images/low \
  --label-source baseline_pred \
  --threshold-source-cache logs/analysis/min_agent_evidence_verification_v2_comparative_add_20260312/_cache/val_base.npz \
  --output-jsonl logs/analysis/vlm_full_20260316/val_rationale_baseline_pred.jsonl \
  --mode rationale
```

```bash
python -u scripts/generate_vlm_rationales.py \
  --base-cache logs/analysis/min_agent_evidence_verification_v2_comparative_add_20260312/_cache/test_base.npz \
  --annotation-file ${DATA_ROOT}/Intentonomy/data/annotation/intentonomy_test2020.json \
  --image-dir ${DATA_ROOT}/Intentonomy/data/images/low \
  --label-source baseline_pred \
  --threshold-source-cache logs/analysis/min_agent_evidence_verification_v2_comparative_add_20260312/_cache/val_base.npz \
  --output-jsonl logs/analysis/vlm_full_20260316/test_rationale_baseline_pred.jsonl \
  --mode rationale
```

## BGE text encoding

```bash
python -u scripts/extract_vlm_rationale_features.py \
  --input-jsonl logs/analysis/vlm_full_20260316/rationale_full.jsonl \
  --output-npz logs/analysis/vlm_full_20260316/rationale_full_bge_features.npz \
  --text-encoder bge \
  --hf-model-name BAAI/bge-large-en-v1.5 \
  --hf-cache-dir /path/to/hf-models
```
