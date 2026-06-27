# Information sEEG Replication Code Demo

This directory contains a clean, shareable replication-code package for the manuscript:

`Prefrontal coordination of belief monitoring and value integration in human exploration`

The package is intentionally separate from the original manuscript files, original notebooks, and sample data. It uses relative paths only and writes outputs under `replication_code/outputs/`.

## What This Demo Can Reproduce

The repository includes one-participant sample data and several already derived analysis tables. The demo can:

- Validate the expected sample-data layout.
- Summarize model-derived gkRL quantities for the sample participant.
- Summarize available reveal-level neural feature tables.
- Summarize belief-reconfiguration and RSA input variables from the available state-metric table.
- Generate output files with the same kinds of tabular structures expected from the full workflow.

## What This Demo Cannot Reproduce

The public sample is not the full cohort. It cannot support:

- Group-level behavioral model comparison across 25 participants.
- Cluster-based permutation inference across channels and time.
- Full ACC-vmPFC PSI or PLV statistics across all valid channel pairs and participants.
- Full PCA, decoding, RSA, and mixed-effects statistics at manuscript scale.
- Electrode localization or raw clinical SEEG preprocessing.

Any full statistical reproduction requires the complete private cohort dataset and the full scientific Python/MATLAB environment documented in `INSTALL.md`.

## Repository Structure

- `config/demo_config.yaml`: central relative-path configuration.
- `scripts/run_demo.py`: command-line entry point.
- `src/`: standard-library Python modules used by the one-participant demo.
- `docs/analysis_mapping_to_manuscript.md`: mapping from manuscript analyses to code.
- `docs/code_consistency_audit.md`: manuscript-to-code audit table.
- `docs/changes_from_original_code.md`: changes made in this package relative to the original notebooks.
- `expected_outputs/`: descriptions of expected output files.
- `outputs/`: generated demo outputs.

The original notebooks remain in `code/scipts/` and the original sample data remain in `code/sample_data/`.

## Notebook Fidelity Note

The lightweight Python demo is not a full line-by-line port of the original notebooks. It validates relative paths, expected inputs, output structures, and final parameter definitions. See `docs/notebook_fidelity_audit.md` for a module-by-module comparison.

For the manuscript analysis modules, this package now includes full reviewer scripts under `scripts/full_analysis/`. See `docs/full_manuscript_reviewer_guide.md`.

## Required Software

For the lightweight demo:

- Python 3.9 or newer.
- No required third-party Python packages.

For full manuscript-scale analyses, see `INSTALL.md`.

## Quick Start

Run from the repository root:

```bash
python3 replication_code/scripts/run_demo.py --config replication_code/config/demo_config.yaml
```

Expected outputs are written to:

```text
replication_code/outputs/demo/
```

## Full Reviewer-Facing Replication

After installing the full scientific Python environment, run:

```bash
python3 replication_code/scripts/full_analysis/run_full_manuscript_pipeline.py --config replication_code/config/full_analysis_config.yaml
```

Outputs are written under:

```text
replication_code/outputs/full_analysis/
```

For a faster check that skips PyMC refitting and sEEG epoch analyses:

```bash
python3 replication_code/scripts/full_analysis/run_full_manuscript_pipeline.py --config replication_code/config/full_analysis_config.yaml --skip-heavy-model --skip-seeg
```

## Expected Outputs

The demo writes:

- `input_inventory.csv`
- `behavior_summary.csv`
- `subject_parameter_summary.csv`
- `neural_feature_summary.csv`
- `belief_reconfiguration_by_reveal.csv`
- `rsa_input_phase_summary.csv`
- `manifest.json`


The one-participant demo validates workflow and input/output structure only. Full statistical reproduction, including group-level TFR cluster inference and PLV permutation inference, requires the complete cohort.

## Data Privacy

The public demo uses only the sample files already present in this repository. It does not copy, modify, or expose raw private clinical data. The manuscript states that full SEEG data cannot be publicly released because of patient privacy protections.

## Citation Placeholder

Please cite the associated manuscript when using this package:

Hu K. et al. `Prefrontal coordination of belief monitoring and value integration in human exploration`.
