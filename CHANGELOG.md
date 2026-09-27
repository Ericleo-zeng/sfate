# Changelog

## v1.1.0 — 2026-09-27

### Added
- **Reachability guard**: `src/sfate/reachability.py` enforces the directed-reachability precondition that guarantees `ρ(Q) < 1` before the absorption solve, with a closed-transient-class counterexample in the test suite.
- **Layer B2 audit suite**: measured krylov–Schur control on the real 74,984-cell production kernel (0.84–0.89 GiB, 15–40 s, five runs) and cross-tool concordance audit supporting manuscript Results 6. Scripts are archived under `scripts/audit_b2_*.py`.
- **Metadata fixes**: B2 audit workflow records and figure-source artifacts archived for Zenodo.

### Changed
- Bumped version to 1.1.0.
- Test suite now covers 35 unit tests (including 3 reachability-precondition tests).
