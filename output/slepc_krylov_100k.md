#  #1 SLEPc krylov-Schur 100k benchmark_slepc_krylov.py

>  paper/PLAN.md  #1  +
> `output/slepc_krylov_100k.json`+
> `output/slepc_krylov_100k_stdout.log`+
> `output/slepc_krylov_100k_time_v.txt`/usr/bin/time -v +
> `output/slepc_krylov_5k.json`
> **claim C19 Results 1 / Discussion
>  a**

## 1. PETSc/SLEPc  ——

- `~/sfate_env`overlay venv rfd3_env
  **petsc4py 3.25.5 / slepc4py 3.25.1**PETSc lib 3.25.5
   `<env>/lib/python3.12/site-packages/`
   2026-08-31 / 2026-05-06
- **rfd3_env **wheel
  ****——"" 7+ issue
  @docs/cellrank-bottleneck.md §3.2
- petsc4py/slepc4py import OK
  `cellrank ... _is_petsc_slepc_available()` = **True**krylov
- Python 3.12.9 / glibc 2.39 / cellrank 2.1.0 2.3.2
  ** B **/ pygpcca 1.0.4 / scanpy 1.11.5 / scipy 1.15.3 /
  numpy 2.4.2BLAS=1 20260909
- swap /swapfile 15.6G + nvme0n1p7 30.5G 1.53GiB
   30GBtime -v **Swaps: 0**

## 2.

-  latenttests/conftest.py `make_branching_latent`
  n=100,000d=30n_terminal=6seed=20260909——GPCCA
  M1  5  `_synthetic` macrostates/fate
  k=30 kNN  nnz  ≈1.6·n·k
- scanpy neighbors(k=30) → ConnectivityKernel(density_normalize) →
  **GPCCA.compute_schur(n_components=20, method='krylov')** →
  compute_macrostates(n_states=6) → predict_terminal_states(stability≥0.96) →
  compute_fate_probabilities(gmres+ilu, use_petsc=False C /step12 )
-  RSS 250ms+ tracemalloc
   = /usr/bin/time -v

## 3.  []

**100k exit=0**

### /usr/bin/time -v

|  |  |
|---|---|
| Elapsed wall time | **2:02.98** |
| Maximum resident set size | **1,607,860 KB ≈ 1.53 GiB** |
| Swaps | 0 |

###  RSS  + tracemalloc

|  | (s) | RSS (MB) | (MB) | tracemalloc(MB) |
|---|---|---|---|---|
| latent_gen | 0.19 | 541.2 | +0.0 | 66.6 |
| neighborsscanpy CellRank | 78.78 | 1324.9 | +727.0 | 448.6 |
| kernelConnectivityKernel | 0.31 | 1408.2 | +133.6 | 232.4 |
| **schur_krylov** | **5.22** | **1619.8** | **+247.7** | 296.0 |
| macrostates | 15.37 | 1566.0 | +134.8 | 86.7 |
| terminal_states | 0.03 | 1486.3 | +0.0 | 17.2 |
| fate_probsgmres+ilu, scipy | 11.96 | 1585.6 | +99.3 | 151.0 |

- nnz(connectivities) = nnz(transition) = **4,817,028**nnz/(n·k) = 1.606
   M1  1.5–1.7
- Schur  (100000, 20)6  30 6
  fate 2.1.0  None

###  brandts

-  n=100k  brandts 72·n² = 72×100,000² B ≈ **720 GB**
  [ @docs/cellrank-bottleneck.md §2.2]—— 30GB
- krylov-Schur  1.53 GiBSchur  +247.7MB——
  ** brandts  3 ** 5.22s

## 4.

1. **PETSC ERRORsignal 13 Broken Pipe×6 **stderr time_v.txt
   PETSc  fate  exit=0
   ——**** pyGPCCA issue #48 ""
    PETSc /
2. cellrank 2.1.0 2.3.2 brandts720GB

3. neighbors 78.78sscanpyBLAS=1 64% CellRank
   GPCCA schur+macrostates+terminal **20.6s**
4. reps=15k output/slepc_krylov_5k.json

## 5. 500k 2026-09-09 SLEPC_N=500000

`output/slepc_krylov_500k.json` / `_stdout.log` / `_time_v.txt`
**exit=0**

### /usr/bin/time -v

|  |  |
|---|---|
| Elapsed wall time | **7:30.64** |
| Maximum resident set size | **3,849,808 KB ≈ 3.67 GiB** |
| Swaps | 0 |

###

|  | (s) | RSS (MB) | (MB) | tracemalloc(MB) |
|---|---|---|---|---|
| latent_gen | 0.32 | 666.2 | +124.6 | 311.6 |
| neighborsscanpy | 145.46 | 2980.5 | +2315.4 | 1686.8 |
| kernel | 1.04 | 2685.7 | +815.1 | 1180.1 |
| **schur_krylov** | **25.91** | **3707.3** | **+1242.8** | 1492.2 |
| macrostates | 60.57 | 3601.3 | +674.6 | 439.4 |
| terminal_states | 0.08 | 2952.4 | +0.0 | 85.6 |
| fate_probsgmres+ilu, scipy | 210.03 | 3723.6 | +711.2 | 770.1 |

- nnz = **24,460,248**nnz/(n·k) = 1.631 100k  1.606 6
  PETSC ERROR signal-13  ×6  100k
-  brandts  n 72×500,000² B ≈ **18 TB** [/
  @docs/cellrank-bottleneck.md §2.2]——krylov  3.67 GiB
- **Fig1 **brandts 720GB@100k / 18TB@500kvs
  krylov 1.53GiB@100k / 3.67GiB@500kvs sfate M175k 1.41GB /
  500k 4.33GB@output/benchmark_m1_smoke.md

## 6. rfd3_env  PETSc/SLEPc  step12 postmortem "OOM  brandts"

> postmortem  step12  OOM  brandts 405GB
>  rfd3_env  petsc4py/slepc4py  krylov
> **——OOM  petsc **
>  Results 4 read-only

### 6.1

|  |  |  |
|---|---|---|
| 2026-06-07 | pygpcca 1.0.4  rfd3_env | pygpcca-1.0.4.dist-info  |
| 2026-09-02 14:22 | cellrank 2.1.0  rfd3_env | cellrank-2.1.0.dist-info  |
| 2026-09-08 02:01 | `step12_cellrank_fate_modified.py.bak.OOM` OOM  layers  OOM |  mtimepipeline  |
| 2026-09-08 02:15 | step12_cellrank_fate_v2.py |  mtime |
| **2026-09-08 02:56** | **`conda install -c conda-forge petsc4py slepc4py --freeze-installed -y`** | **rfd3_env conda-meta/history  regvelo ** |
| 2026-09-08 03:20–03:21 | step12_cellrank_fate.py  75k .done 03:21:35n_terminal=5 |  mtime + <mnt>/.../11_cellrank/.step12_cellrank_fate.done |
| 2026-09-08 03:53 |  ChatGPT  | cellrank/chatgptgmd |
| 2026-09-08 05:17 / 05:31 | ULTIMATE / INTEGRATED_FINAL  | 11_cellrank/  |

### 6.2

1. **OOM  = 2026-09-02 cellrank 09-08 02:01OOM  PETSc
   ** `compute_macrostates`  `compute_schur`  method='krylov'
    brandts → 75k  405GB → OOM killedpostmortem §3 ****
   "** pip  PETSc/SLEPc**"
2. ** = 02:56 petsc **75k brandts  405GB30GB
   → 03:21/05:17/05:31 ** krylov-Schur**①
    PETSC ERROR signal-13 PETSc
   ②**** rfd3_env  petsc4py 3.25.5/slepc4py 3.25.1
    krylov
3. ** Results 4 **step12  GPCCA DAM/PU1low unsupported
   IFN  3 Proliferating  fate  **krylov-Schur Schur
   ** brandts —— GPCCA /
   ** Schur **C sfate vs step12
    krylov  GPCCA
4. **postmortem **conda  PETSc/SLEPc §4
   gene_subset  use_petsc=False ——" sfate formalize
   " 405GB
    docs/step12_postmortem.md
5. ****conda-forge --freeze-installed——
    conda  7+  issue  pip/
   ""pip wheel

## 7.  a

- **** wheel
- ****krylov-Schur  100k ****——5.22s /  1.53GiB /
   2:03 OOM
- SLEPc  GPCCA  n²  100k ****
  @docs/cellrank-bottleneck.md §5-X2
-  PLAN.md  #1  **"krylov "** vs
  ——** 2026-09-09 **PLAN.md C19
  500k §5 krylov  50 3.67GiB/7:31Schur 25.9s
