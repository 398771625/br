# Independent FusionVLAD baseline for NCLT

This repository now contains an isolated FusionVLAD reproduction pipeline. Existing BEVPlace2 and BEV–Range files remain untouched. The pipeline reads only ordering metadata from Scan Context manifests, decodes original NCLT LiDAR through the configured CVTNet `get_velo`, builds temporal local maps and two paper-style views, trains independent VGG-style/NetVLAD branches with FC parallel fusion, extracts 2048-D descriptors, and performs guarded NCLT Standard evaluation.

## Protocol and isolation guarantees

The only database is `2012-01-15`; the six queries are `2012-02-04`, `2012-03-17`, `2012-06-15`, `2012-09-28`, `2012-11-16`, and `2013-02-23`. No `2012-01-08` data is required. Query traversals are not opened by training, normalization, validation, clustering, checkpoint selection, or tuning. `scan_contexts` and `ring_keys` are never loaded as model inputs. All paths and expected protocol values live in `config/nclt_fusionvlad.yaml`.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Edit only the path entries in `config/nclt_fusionvlad.yaml` if local roots differ.

## Complete local run

```bash
# 1. Mandatory ordering/file and verified-decoder checks
python -m tools.check_nclt_manifest_alignment --config config/nclt_fusionvlad.yaml
python -m tools.check_nclt_decoder --config config/nclt_fusionvlad.yaml --sequence 2012-01-15 --num-samples 10

# 2. Resumable view generation for all seven sequences (add --overwrite to rebuild)
python -m tools.generate_fusionvlad_views --config config/nclt_fusionvlad.yaml

# 3. Mandatory visual review before training
python -m tools.preview_fusionvlad_views --config config/nclt_fusionvlad.yaml --sequence 2012-01-15 --num-samples 10

# 4. Train only after manually approving all previews
python -m tools.train_fusionvlad --config config/nclt_fusionvlad.yaml --views-checked
# For limited memory only: append --batch-queries 1 (recorded in config_used.yaml).

# 5. Extract all seven manifest-ordered descriptor sets
python -m tools.extract_fusionvlad_descriptors --config config/nclt_fusionvlad.yaml

# 6. Strict denominator-checked NCLT Standard evaluation
python -m tools.evaluate_fusionvlad_nclt --config config/nclt_fusionvlad.yaml
```

Evaluation writes JSON and CSV only after all six denominators exactly match the configured expected values. A mismatch raises `RuntimeError` before any final Recall is printed or saved, preventing misleading results.

## Manual gates

1. Confirm manifest/file alignment and inspect representative decoder output ranges/counts.
2. Inspect every PNG under `outputs/preview_views`; verify orientation, occupancy, height/intensity, range slices, and normal-angle layers. Training deliberately requires `--views-checked`.
3. Review `runs/fusionvlad_nclt/config_used.yaml`, `environment.json`, normalization statistics, and source-validation training curve before descriptor extraction.
4. Never select a checkpoint using any of the six query results.

See `REPRODUCTION_NOTES.md` for the required separation between paper-confirmed facts, later CVTNet evidence, and project-specific reimplementation choices.
