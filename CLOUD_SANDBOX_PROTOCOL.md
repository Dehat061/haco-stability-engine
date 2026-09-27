# HACO v1.0-RC3 — Canonical External Cloud-Sandbox Protocol

For ChatGPT/cloud sandboxes with scientific dependencies already installed but no package-index access:

1. Extract the release.
2. Verify `SHA256SUMS.txt`.
3. Record Python and dependency versions.
4. Confirm required dependencies are importable.
5. Execute from the frozen release source tree by adding `src/` to `PYTHONPATH`.
6. Do not create an empty venv merely to force network dependency resolution.
7. Run package tests and the two packaged controls.
8. Run `verify-report` for every generated report.
9. Preserve failures exactly as observed.

This validates **cloud-sandbox execution reproducibility**, not clean offline wheel installation.

## Separate offline-install claim

A clean offline install requires a platform/Python-specific wheelhouse prepared in a connected environment before the evaluator starts.

## Frozen controls

Invariant:
- Iris, kNN-L2, 3 folds, 20 proper rotations, max PFR tolerance 0.
- Expected: PASS.

Sensitive:
- Iris, kNN-L1, 3 folds, 20 proper rotations.
- max PFR >= 1e-12
- detection rate >= 0.10
- positive folds >= 2
- Expected: PASS.

## Integrity falsification

After preserving an original valid report:
1. copy report/raw artifact;
2. change one PFR or decision value in the copy;
3. run `verify-report`;
4. expected: `INTEGRITY FAIL`.
