# validation-protocol.md Addendum: E1 Equivalence Proposition and Solver Consistency Validation

> Suggested insertion point: after the invariant tests (I1-I10) and before the benchmark section (Sec. 5 three tiers).
> All thresholds in this document marked `[initial, to be calibrated]` will be tightened after the first round of backtesting using the measured distribution.

## E1 Equivalence Proposition (Verified)

### E1.1 Proposition

Given the **same row-stochastic transition matrix P** and the **same terminal state set A**, the block-sparse solution `(I-Q)X = R` of sfate is mathematically identical to CellRank's fate probabilities:

```
B = (I-Q)^{-1}R,   where Q = P_TT, R = P_TA; the solution is unique when absorption occurs with probability 1 (rho(Q)<1)
```

### E1.2 Justification (Three Independent Sources)

| # | Source | Key Content |
|---|---|---|
| 1 | CellRank 1 paper (Lange et al.) Eq. 18-19 | Closed-form absorption probability A=(I-Q)^{-1}S; the official implementation reformulates it as column-by-column linear-system solves; aggregation within a class equals summing absorption probabilities of all cells in that class |
| 2 | CellRank 2 paper | Representative cells are merged into a single pseudo-state before solving; the original text states "the corresponding results are mathematically equivalent", yielding an n_f-fold speedup |
| 3 | CellRank official documentation | "transform the matrix inversion problem into a set of linear systems...GMRES"; Schur decomposition is used only for macrostate inference (pyGPCCA/SLEPc) and does not participate in probability calculation |

Corollary: By dropping Schur in v1, what is lost is the **automatic identification of terminal/macrostates**, not solver accuracy.
When conditions (P, A) are identical, B is elementwise identical.

### E1.3 Risk Notes (Mapped to the design.md Risk Register)

The spectral gap 1-rho(Q) simultaneously determines: the separability of macrostates by Schur decomposition and the ill-conditioning of (I-Q) (kappa grows with metastability) => GMRES iteration count increases. Removing Schur does not eliminate this difficulty; it merely shifts it from the decomposition phase to the solver phase. Expected behavior is already covered by design.md P0 (GMRES convergence at 500k cells); solver-side fallback order: GMRES+ILU(degree=1, drop_tol sweep) -> direct or block-direct solve -> Palantir-style two-level (mathematically the same Schur-complement rearrangement of the same equation, still the same solution).

## E2 Validation Hierarchy: Tier A (Solver Equivalence) / Tier B (Pipeline Consistency)

The two tiers **must be tested separately**. Mixing them conflates implementation bugs with kernel/terminal-state differences, making failures impossible to attribute.

| Tier | Question Answered | Common Prerequisite | Comparison Pair | Pass Threshold | Meaning of Failure |
|---|---|---|---|---|---|
| **Tier A** | Is the solver implementation correct? | Same P, same A | sfate(f32+GMRES) vs spsolve(f64) gold standard | L_inf <= 1e-5 `[initial, to be calibrated]` | Implementation bug, **fix code** |
| **Tier B** | Is the kernel plus terminal-state choice reasonable? | Each complete pipeline | sfate(latent kNN plus user terminal states) vs CellRank(ConnectivityKernel plus GPCCA) | Spearman > 0.95 | Biological/design difference, **enter design review, do not modify solver** |

### Tier B Attribution Workflow (Mandatory)

```
Tier B fails
  |- Run Tier A first (same P, same A)
  |    |- Tier A fails => solver bug; fix implementation and retest Tier B
  |    \- Tier A passes => difference attributable to kernel/terminal states; write attribution report for design review
  \- Skipping Tier A and directly modifying solver tolerances or thresholds to "fix" Tier B is prohibited
```

## E3 Tier A Test Cases (All Fixtures <=5k, Seconds to Minutes, Included in PR CI)

Fixture and scale constraints: all Tier A fixtures <= 5k cells--GPCCA is not involved, so there is no Schur OOM risk (T7 measured 5k as safe; 10k OOM affects only the upper limit of the CellRank control in Tier B, see E4).

| Case | Content | Parameters | Threshold |
|---|---|---|---|
| `test_a1_lumping_equivalence` | Cross-validate two absorption-imposition methods under the same P: (1) cell-by-cell absorption (CR1 style, summed within class); (2) pseudo-state lumping (CR2 style) | n=2k synthetic three-branch chain; terminal states 3x100 cells | Per-cell L_inf <= 1e-6 across both variants (f64 full chain) -- directly verifies the CellRank 2 paper's "mathematically equivalent" |
| `test_a2_direct_vs_gmres_f64` | GMRES vs direct solve (same precision) | n=2k; Tier A fixture P (from M2 onward, remove absorbing-state-free island connected components, n=1600); tol=1e-12 | **Dual-metric (M2 calibrated)**: hard backward-error threshold -- true residual max ||(I-Q)B-R||_inf <= 1e-12; and forward-error bound L_inf <= 10 * kappa_inf(I-Q) * backward_max, where kappa_inf uses the exact M-matrix formula ||A||_inf * max(A^{-1} * 1) (obtained by one back-substitution; onenormest underestimates on this class of matrices, as measured by M2) |
| `test_a3_f32_vs_f64` | **Primary metric**: sfate production path (f32+GMRES, tol=1e-6) against f64 gold standard | n=2k plus n=5k two tiers; block_size in {1, 16, 256, full} sweep | L_inf <= 1e-5 `[initial, to be calibrated]`; inter-block_size difference <= 1e-6 (RHS blocking does not change the solution in exact arithmetic; differences may only come from BLAS paths) |
| `test_a4_invariants_on_fixture` | Rerun I1/I2/I5/I8/I9/I10 on Tier A fixtures | Same as Tier A fixtures | Use the protocol I-series thresholds |
| `test_a5_determinism` | Fixed-seed double run | Test each knn_backend candidate separately | Adjacency lists identical elementwise; columns that kNN libraries cannot seed `[to be calibrated]` and documented |

Tier A CI budget: total for all cases < 2 min; each case records runtime and peak RSS (tracemalloc), reported with results but no hard memory threshold for Tier A (scale is too small).

### E3-a2 Metric Calibration Record (M2 Measured, 2026-09-09)

The original L_inf<=1e-9 metric was rejected by measurement (2.97e-8); root cause was attributed layer by layer as follows, and after adjudication it was changed to the dual-metric table above:

1. **Fixture defect**: the 2k synthetic-chain kNN graph had ~190 absorbing-state-free island connected components -> (I-Q) numerically singular (min|diag(U)|=1.8e-14); both the gold standard and iterative solutions were meaningless on island rows. Fix: remove islands (2000->1600), and add a reachability guard in fate.py (absorbing-state-free connected component -> ValueError).
2. **Output quantization masking**: fate output was previously cast to float32, and the f64 solution's quantization residual of ~3e-8 completely masked the solver error (true difference between f64 solution and gold standard <= 3.4e-11; classical bound 250*(1.35e-13+1e-15) held strictly). Fix: output dtype follows P.
3. **eps-level forward bound unattainable**: GMRES backward error at rtol=1e-12 stopping is on the order of tol (approximately 600*eps), not eps; therefore the "10*kappa*eps" threshold cannot pass under either kappa_1 or kappa_inf norms. The classical numerical-analysis bound is forward error <= kappa * backward error; use this as the standard.

## E4 Tier B Test Cases (Record-Only, Not a Hard PR CI Gate)

| Item | Parameters |
|---|---|
| Data | 5k real-cell subset (same subsetting rule as T7/existing fixtures, fixed random seed written to disk) |
| sfate side | latent kNN kernel: k=30, gaussian, symmetrize=max, f32, knn_backend subject to M1 selection experiment; terminal states = user annotation (fixed obs column) |
| CellRank side | ConnectivityKernel (T7 exp2 verified all-CSR with no OOM); GPCCA brandts (safe at 5k); automatic macrostate with highest stability taken as terminal state |
| Terminal-set mapping | GPCCA automatic terminal states vs user annotation cannot be assumed identical: use Hungarian maximum-overlap matching on the confusion matrix; unmappable terminal states are explicitly listed in the report, excluded from Spearman, and discussed separately |
| Metrics | Per-cell fate probability vector Spearman; fate assignment (argmax) consistency (ARI plus per-class Jaccard) |
| Threshold | Spearman > 0.95 pass; 0.90-0.95 yellow card (mandatory E2 attribution workflow); < 0.90 red card (design review) |
| Protocol | Single process, swap off, `/usr/bin/time -v` logging; >=3 runs take max; results written to benchmark/report.md, annotated `[measured]` |

Note: the Tier B 5k upper limit is determined by GPCCA (T7: brandts was killed by OOM at 10k). The sfate side is not a constraint at this scale. If larger-scale Tier B comparisons are needed in the future, the CellRank-side dense Schur fallback must be addressed first (install SLEPc and use krylov -- this claim has zero empirical validation during the investigation phase and is marked `[unconfirmed]` in the protocol; do not cite it as a premise).

## E5 Relationship to Existing Sections

- I1-I10 invariants: continue to apply to all Tier A/B fixtures; not redefined here.
- Sec. 5 three-tier benchmark: unchanged; Tier A tests correctness, Tier B tests consistency, Sec. 5 tests scale; the three are orthogonal.
- design.md risk register P0/P1: A3 covers P1 (f32 precision); Tier B does not cover P0 (500k-cell GMRES convergence remains the responsibility of Sec. 5 overnight; Tier B is not extrapolatable).
