# HACO Stability Engine

**HACO Stability Engine** is research software for auditing coordinate-frame stability in supervised machine-learning procedures.

It asks a narrow, testable question:

> If the same Euclidean learning problem is expressed in alternative proper-orthogonal coordinate frames, does the declared learning procedure preserve its predictive behavior after matched refitting?

HACO does **not** claim that every model should be rotation-invariant, and it is **not** a universal robustness score.

## Core workflow

1. Load a numeric tabular classification problem.
2. Fit a declared preprocessing pipeline using training data only.
3. Generate proper orthogonal transformations and numerically audit them.
4. Refit a fresh estimator in the identity frame and every transformed frame.
5. Compare predictions using Prediction Flip Rate (PFR).
6. Apply a predeclared policy:
   - `invariant`
   - `sensitive`
   - `informational`
7. Emit provenance-rich JSON/raw artifacts.
8. Verify schema and cryptographic integrity with `verify-report`.

## Quick start

From the repository root:

```bash
python -m pip install -e .
haco doctor
haco test --config examples/iris_knn_l2.yml --output-dir out-l2
haco verify-report out-l2/haco_report.json
haco explain out-l2/haco_report.json
```

## Frozen RC3 controls

### Euclidean-invariant control

`examples/iris_knn_l2.yml`

Expected under the frozen configuration:

- max PFR = 0
- transformation audit = PASS
- decision = PASS

### Coordinate-sensitive control

`examples/iris_knn_l1.yml`

Frozen recurrence criteria include:

- max PFR >= 1e-12
- detection rate >= 0.10
- positive folds >= 2
- transformation audit = PASS

Reference outputs are available under `reference_outputs/`.

## Commands

```text
haco version
haco doctor
haco test --config <config.yml> --output-dir <dir>
haco validate-report <haco_report.json>
haco verify-report <haco_report.json>
haco explain <haco_report.json>
```

## Input modes

- built-in scikit-learn datasets used for controls;
- numeric CSV classification data;
- built-in estimator families;
- trusted Python estimator factories implementing `fit` and `predict`.

See `examples/CSV_MODE.md` and `THREAT_MODEL.md`.

## Integrity

`haco verify-report` checks:

- JSON Schema conformance;
- canonical report SHA-256;
- hashes of referenced raw artifacts.

Modifying a report or referenced raw artifact without recomputing the protected evidence should produce `INTEGRITY FAIL`.

## Reproducibility

The repository includes:

- automated tests;
- frozen example configurations;
- reference output artifacts;
- tested environment metadata;
- cloud-sandbox execution instructions;
- offline-install guidance.

The associated validation program also includes matched invariance tests, rotation-budget experiments, high-dimensional stress tests, independent implementation checks, and external cloud-sandbox replication.

## Claim ceiling

The current evidence supports reproducible **coordinate-frame stability auditing in the tested settings**.

It does not establish:

- universal validity across all ML models/domains;
- semantic invariance for arbitrary real-world features;
- superiority over every robustness/TEVV method;
- production security certification.

## Version

Current public-release candidate: `1.0.0rc3`.

## Citation

A SoftwareX manuscript is in preparation. A `CITATION.cff` file is included and will be finalized with repository and DOI metadata before the public release.

## License

**Not yet selected. Do not publish this repository until `LICENSE` has been deliberately chosen.**

The license decision matters for both open-source reuse and any future commercial/patent strategy.
