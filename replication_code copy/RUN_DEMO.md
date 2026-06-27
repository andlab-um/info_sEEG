# Run The Demo

Run all commands from the repository root.

## Step 1: Run The Lightweight Demo

```bash
python3 replication_code/scripts/run_demo.py --config replication_code/config/demo_config.yaml
```

Expected runtime: under one minute.

## Step 2: Inspect Outputs

Generated files are written under:

```text
replication_code/outputs/demo/
```

Expected files:

- `input_inventory.csv`
- `behavior_summary.csv`
- `subject_parameter_summary.csv`
- `neural_feature_summary.csv`
- `belief_reconfiguration_by_reveal.csv`
- `rsa_input_phase_summary.csv`
- `manifest.json`

## Scope Note

The demo outputs are structural summaries. They do not perform full group-level TFR cluster inference or full PLV permutation inference. The final reviewer-facing parameter definitions are documented in `docs/final_parameter_lock.md`.

## Full PCA-Family Workflow

The PCA/state-space/RSA analyses have full reviewer scripts. After installing the full dependencies in `INSTALL.md`, run:

```bash
python3 replication_code/scripts/full_analysis/run_pca_full_pipeline.py --config replication_code/config/full_analysis_config.yaml
```

This writes outputs under `replication_code/outputs/full_analysis/`. It is separate from the lightweight demo above.

## Full Manuscript Workflow

All ported reviewer-facing scripts can be run with:

```bash
python3 replication_code/scripts/full_analysis/run_full_manuscript_pipeline.py --config replication_code/config/full_analysis_config.yaml
```

Useful options:

```bash
python3 replication_code/scripts/full_analysis/run_full_manuscript_pipeline.py --config replication_code/config/full_analysis_config.yaml --skip-heavy-model
python3 replication_code/scripts/full_analysis/run_full_manuscript_pipeline.py --config replication_code/config/full_analysis_config.yaml --skip-heavy-model --skip-seeg
```

## Troubleshooting

If the command reports a missing input file, check that it was launched from the repository root and that `replication_code/config/demo_config.yaml` still points to relative paths that exist in this repository.

If a CSV column validation error appears, the sample data layout differs from the expected manuscript workflow. Do not silently change the analysis logic; update the consistency audit and make the smallest path or schema adaptation needed for the demo.

If you want to run the original notebooks at full scale, install the full dependencies listed in `INSTALL.md` and replace the sample-data paths with a complete private dataset using relative paths controlled by a config file.
