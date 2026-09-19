# M1 smoke benchmark benchmark_m1_smoke.py

##
- env: python 3.12.9 / numpy 2.4.2 / scipy 1.15.3 / anndata 0.12.16
-  RSS `/usr/bin/time -v`  Maximum resident set size output/benchmark_time_v.txt
- tracemalloc Python numpy/scipy  pymalloc I7 ——validation-protocol §4
- BLAS =1 ≥3  max
- swap  `swapon --show`

## tracemalloc +

|  | n | k | build (s) | tracemalloc (MB) | RSS (MB) | nnz | nnz/(n·k) | CSR (MB) |
|---|---|---|---|---|---|---|---|---|
| fixture_5k_real | 5,000 | 30 | 1.9 | 285.4 | 762.1 | 251,140 | 1.6743 | 2.05 |
| synthetic_10k | 10,000 | 30 | 3.907 | 290.0 | 859.0 | 480,054 | 1.6002 | 3.92 |
| real_75k_full | 74,984 | 30 | 18.444 | 426.9 | 1413.1 | 3,807,850 | 1.6927 | 31.06 |
| synthetic_500k | 500,000 | 30 | 66.162 | 2728.6 | 4327.3 | 25,395,928 | 1.6931 | 207.17 |

###

- **fixture_5k_real**1 <2GB/<1min PASS PASS
- **synthetic_10k**1 <2GB/<1min PASS PASS
- **real_75k_full**10 <6GB/<10min PASS PASS
- **synthetic_500k**50 <16GB/<30min PASS PASS

## kNN P1 ≥3  max

| backend | n | build (s) |
|---|---|---|
| pynndescent | 5,000 | 1.556 |
| sklearn | 5,000 | 0.179 |
| pynndescent | 10,000 | 1.803 |
| sklearn | 10,000 | 0.433 |
| pynndescent | 74,984 | 6.362 |
| sklearn | 74,984 | 14.893 |

##

-  nnz/(n·k)  CSR protocol §3.4
  RSS  ≈ CSR  + kNN  scratch +  latent
  RSS  tracemalloc  = numpy/scipy  pymallocprotocol §4
   RSS  >30%
- 100  overnight design 100 <6GB  []
- 50[]/  SKIP_500K=1  [/overnight]

##

- tracemalloc  numpy/scipy  time -v
- wall time
-  ~30GB50 latent  ~60MB

## protocol §3.4 RSS

`mem(n) = 794.7 + 7.090 MB/`R² = 0.99854

- **pred(1M) = 7.88 GB×  1.25 = 9.62 GB < 32 GB → ** [1M / overnight]
- 50 4.33 GB 16GB  → 50 <16GB ** [] ** 100 ~11GB
- CSR  0.36 MB/k=30, f32, nnz≈1.5nk 19.7×——
   pynndescent  + kNN  + int64  CSR
  ""kNN scratchP1 int32 //
   overnight **hub ** nnz/(n·k)=1.69 T7  1.5–1.67

## /usr/bin/time -v

- Maximum resident set size**4,250,640 KB ≈ 4.05 GiB**Elapsed 7:50BLAS =1
-  RSS  4.33GB50+/ ~0.3GB
- **swap  swap /swapfile 15.6G + nvme0n1p7 30.5Gswapon --show **
  " swap "—— 0.76–4.33GB 25GB
  swap time -v Swaps: 050 16GB-4.33GB
   swapoff

## M1  data_recon.md  AGENTS.md

1. **anndata 0.12.16 backed  layers**read_h5ad(backed='r')  adata_annotated.h5ad
    +6.9GBlayers counts/spliced/unspliced  src/sfate/io.py h5py
2. ** latent  hub **(max) 40max 389k=30nnz/(n·k)=1.69——
   protocol I9  ≤k  min≥k + ≤2k  []
3. **kNN P1 **sklearn  kd-tree  5k/1 75k
   pynndescent 6.4s vs 14.9s sklearn ——** pynndescent **
   sklearn 50 pynndescent  66s

##

- [x] 5k  smoke175k 50—— PASS +  + nnz  <5%
- [x] tracemallocPython + RSS  + time -v
- [x] 100 []pred 9.62GB×1.25 overnight  design <6GB ——
   design  <6GB  pred 7.9GB  kNN  scratch overnight
- [x] 50 [] PASS pynndescent
