# `torch_npu.npu_fused_infer_attention_score` — kernel dispatch & load balancing

**Target:** Ascend 910B2 / Atlas A2 (DAV_C220), CANN 8.5.0.

## File map

**Kernel side (local CANN install, `…/ascendc/`):**
- Dispatch entry: `fused_infer_attention_score/fused_infer_attention_score.cpp`
- Tiling keys: `…/fused_infer_attention_score_template_tiling_key.h`, `…/arch32/fused_infer_attention_score_tilingkey.h`
- IFA: `incre_flash_attention/incre_flash_attention.cpp`, `…/arch32/incre_flash_attention_split_Bbn2s2_Us2.h`
- PFA: `prompt_flash_attention/…/arch32/prompt_flash_attention_s1s2_bns1_x910.h`, `…/prompt_flash_attention_base_api.h`
- FIA v3: `fused_infer_attention_score/fused_infer_attention_score_v3.cpp`, `…/common/arch32/fia_kernel_nonquant*.h`, `fia_block_cube_nonquant_gqa.h`
- SplitFuse: `fused_infer_attention_score/flash_attention_regular.h`, `flash_attention_interface.cpp`

**Host tiling (public: `github.com/hicann/ops-transformer`, `arch22` = 910B):**
- Path selection: `attention/fused_infer_attention_score/op_host/fused_infer_attention_score_tiling.cpp`, `…_tiling_info_parser.cpp`
- FIA v3 tiling entry: `attention/fused_infer_attention_score/op_host/arch22/fused_infer_attention_score_tiling_v3.cpp`
- **Load balancer:** `attention/common/op_host/split_core_v2.cpp` (`CalcSplitPlan`, `CalcCost`, `CalcS2Range`), `split_core.cpp`
- FIA compute tiling: `attention/common/op_host/arch22/fia_tiling_nonquant.cpp`, `fia_tiling_nonquant_mla.cpp`
- FAI/SplitFuse tiling: `attention/fused_infer_attention_score/op_host/flash_attention_infer_tiling.h`

---

## 1. One operator, one mega-kernel, four code paths

`FusedInferAttentionScore` compiles to a single `__global__` entry
(`fused_infer_attention_score.cpp`). It is **JIT/dynamically compiled per tiling key** (only 13
prebuilt `.o` on disk). The entry does nothing but branch on the integer `TILING_KEY_VAR` against
three magnitude boundaries (`arch32/fused_infer_attention_score_tilingkey.h`):

| Tiling-key range | Constant | Path dispatched | Role |
|---|---|---|---|
| `≥ 5e18` | `FAI_FLAG_TILING` | `SplitFuse::FAInfer` | new cube flash-attn, **TND only**, chunked-prefill / split-fuse |
| `≥ 1e18` | `PFA_FlAG_TILING` | `prompt_flash_attention_FIAS_OBP` (**PFA**) | prefill / prompt |
| `≥ 1e17` | `FIA_FLAG_TILING` | `fused_infer_attention` v3 (**FIA**) | MLA + paged-KV + flash-decoding specialist |
| `< 1e17` | *(else)* | `incre_flash_attention_FIAS_OBP` (**IFA**) | decode / incremental (small `S_q`) |

The **host tiling picks which path** primarily from the query sequence length (small/`S_q≈1` →
IFA decode; larger → PFA/FIA/FAI prefill), plus layout, MLA, paged-KV and dtype. *The exact
selection thresholds are in the compiled host `.so`, not in source.* Everything downstream of the
chosen tiling key **is** in source and is described below.

---

## 2. Tiling-key encoding (what the input shape turns into)

Two encodings coexist (the newer one is mid-migration — a source comment says the mechanism
"will be revised to a template parameter in October 2025"):

**(A) Bitfield** (`fused_infer_attention_score_template_tiling_key.h`, ~37 bits). Dtype is *not*
in the bitfield — it is a compile-time `#if (ORIG_DTYPE_QUERY==… && ORIG_DTYPE_KEY==… && ORIG_DTYPE_ATTENTION_OUT==…)` guard. Fields:

| Bits | Field | Values |
|---|---|---|
| 8-1 | `InOutLayoutType` | `BNSD=0`, `BSH=1`, `TND=2` |
| 18-9 | `Config` | 16 S1/S2/D/DV alignment presets |
| 22-19 | `PseMode` | outer/inner mul-add variants, `NONE=9` |
| 27-23 | `QuantMode` | per-channel / per-token / mixed / page-attention |
| 28 | `HasAttenMask` | 0/1 |
| 29 | `HasRope` | 0/1 |
| 30 | `IsPa` | paged-KV attention 0/1 |
| 31 | `IsFd` | **flash-decoding** 0/1 |
| 32 | `EmptyTensor` | 0/1 |
| 34-33 | `PFAMask` | `DISABLE=0`, `MASK_NO_BAND=1`, `MASK_BAND=2` |
| 37-35 | `PFAMatMulType` | `MM_PFA`, `MM_PA`, `MM_IFA_MLA`, `MM_IFA_MLA_PA`, `MM_PA_D512`, `MM_DN` |

Dtype triple ∈ {fp16, bf16, int8, int4, hifloat8, fp8_e5m2, fp8_e4m3, fp4}: symmetric fp16/bf16,
antiquant KV-cache (fp16-Q × int8/int4/fp8 KV), and output-quant combos are the enumerated ones.

**(B) Concatenated decimal digits** (`arch32/…_tilingkey.h`) — the legacy literals the entry
actually compares. Leading magnitude = path flag; individual digit positions encode
layout / dtype / cache+decode-mode / KV-format / cube:vector-ratio.

**FAI / SplitFuse keys are exactly 20 symbols** (all TND):
`Q{F16,BF16}_KV{…}_OUT{…}` × `{NOLSEOUT, LSEOUT}` × `{NOCACHE, PAGEDCACHE}` × `{NOMASK, CAUSALMASK}`,
plus a `LOW_PREC` variant only on fp16+NOMASK. e.g.
`QF16_..._NOCACHE_CAUSALMASK_SPLITFUSE_TILING = 5000000000000200103`.

**Causal is not a boolean** at the API. It is `sparseMode` + `preTokens`/`nextTokens`
(`aclnnFusedInferAttentionScoreV*`): `sparseMode` selects none / leftUpCausal / rightDownCausal /
band, and the host folds it into `PFAMask`/`CAUSALMASK`/the FIA sparse range.

---

## 3. Load balancing per path

Across all paths the machine is used in **MIX mode**: each AI core is an AIC (cube/mmad) paired
with 2 AIV (vector/softmax). Kernels call `GetBlockIdx()` per vector core and fold the 2 AIV onto
their shared logical work unit, e.g. IFA:

```cpp
tmpBlockIdx = GetBlockIdx();
if (tmpBlockIdx & 0x1) tmpBlockIdx = (tmpBlockIdx + GetBlockNum()*GetTaskRation())/2;
else                   tmpBlockIdx = tmpBlockIdx/2;
```

### 3a. IFA — decode path (`S_q` small)
Kernel class `IncreFlashAttentionAttenSplitBbn2s2Us2` (name = the split axes B, b·n2, s2, U·s2).

- **Unit of work = one `(batch b, kv-head n2)` pair** (flattened `bn2`). GQA query-group `G` is
  *not* split — all `G` query heads of a KV head run on one core (shared K/V).
- **Non-flash-decode:** host prefix-sum `coreSidxEnd[]` hands each core a contiguous run of `bn2`.
- **Flash-decode (`splitKVParams.s2 > 0`):** turned on when `B·N` is too small to fill all cores
  and `S_kv` is long (the decode case). Grid becomes `B · kvHeadNum · splitKVNum`, one core per
  `(b, n2, KV-chunk)`. **Two passes in one launch:**
  1. each core does partial flash-attention over its KV chunk → writes partial `accumOut` (fp32)
     and per-chunk softmax `max`/`sum` (the LSE state) to **workspace GM**.
  2. `SyncAll()` barrier, then `FlashDecodeCompute()` re-partitions by `(b,n2)` and one core
     **combines** all `splitKVNum` chunks via log-sum-exp rescale
     (`exp(lseMax_i − globalMax)`, weight = `chunkSum/totalSum`, weighted sum of partials) →
     final `attentionOut` (+ combined LSE if `softmaxLseFlag`).
- **Variable `S_kv` imbalance:** former/tail core split (`formerCoreNum` cores get
  `blockSplitBn2Range`, rest get `tailSplitedBatchRange`); per-batch `actualSeqLengthsKV` recomputes
  the inner KV loop at runtime; chunks past a batch's real length go idle (`remainSinnerSize≤0`);
  `S_kv==0` batches short-circuit to a zero-output.
- **Causal:** **no effect on the split** and no triangular path. With `S_q=1` a query attends to
  all past keys, so a causal mask is a no-op for scoring. (Only sliding-window `windowSize` clamps
  `curActualSeqLen`.)
- **Driver fields:** `increFlashAttentionSingleCoreParams.{usedCoreNum, formerCoreNum,
  blockSplitBn2Range, tailSplitedBatchRange}`, `coreSidxEnd[]`,
  `splitKVParams.{s2, sInnerLoopSize, accumOutSize, logSumExpSize}`.

### 3b. PFA — prefill path
- **Unit of work = one query-row tile (`singleProcessSOuterSize` ≤128 rows) × one `(batch, head)`.**
  Total list = `(B·N) × ceil(S_q / tile)`. KV (`sInner`) is the reduction loop *inside* a task.
- **Non-causal:** uniform work per tile → simple contiguous former/tail split from host
  (`ComputeEachCore` reads per-core `[start,end]` range arrays).
- **Causal — in-kernel serpentine** (`ComputeEachCoreBalance`, used when `batchSize==1 &&
  headNumRatio==1`). Three combined tricks so each core gets ≈ equal *triangular area*:
  1. **reverse tile order** so index 0 = bottom (heaviest) tile:
     `sOuterLoopIdx = sOuterBlockNum-1 - tilingIdx/(…)`;
  2. **reflect odd cores** `coreIdx = blockNum - coreIdx`;
  3. **serpentine stride** `tilingIdx += (blockNum - tilingIdx%blockNum)*2 - 1` — pairs a light
     tile in one band with a heavy tile in the mirrored position of the next band.
  Fully-masked tiles are skipped (`if (sInnerLastToken ≤ sInnerFirstToken) continue;`).
- The newer **base_api** PFA variant does the same via an `isTriuMask` flag: even bands forward,
  odd bands reversed (`(currIter+2)*blk - 1 - GetBlockIdx()`); otherwise plain cyclic
  `process += GetBlockNum()`.
- **Driver fields:** `promptAttentionSingleCoreParams.{singleProcessSOuterSize, actualCoreNums}`,
  `promptAttentionBaseParams.{dimNumOfseq, headNumSize, preTokens, nextTokens, splitS2}`,
  per-core range arrays in `promptAttentionSeqParams`, `accumSOuterTilingNums[]`.

### 3c. FIA v3 — MLA / paged-KV / GQA specialist
Cube/Vec disaggregated (`FiaBlockCubeNonQuantGqa` for mmad, `FiaBlockVecNonQuant` for softmax,
`FiaBlockVecFlashDecode` for long-KV). 48 compiled variants, almost all with **1:1 AIC:AIV ratio**
(`KERNEL_TYPE_MIX_AIC_1_1`), keyed on layout (BSH/BNSD/BSND/TND) × KV-format (ND/NZ/BNSD) ×
PagedCache × MLA × FlashDecoding.

- **Balancing is decided on the host** (see §4) — the kernel just reads its slice from per-core
  **cumulative-work boundary arrays** `bN2End[]`, `gS1End[]`/`mEnd[]`, `s2End[]`:
  ```cpp
  constInfo.bN2Start = aiCoreIdx==0 ? 0 : bN2End[aiCoreIdx-1];
  constInfo.s2Start  = s2End[aiCoreIdx-1];  // nonzero ⇒ a query tile is split across cores at a KV boundary
  ```
- Because boundaries carry a KV offset, a single query tile can be **split across cores along the
  KV axis** (`headS2Split`/`tailS2Split`), i.e. flash-decoding for prefill with few batch·head.
- **Driver fields:** `baseParams.usedCoreNum`, `outerSplitParams.{bN2End[], mEnd[]/gS1End[], s2End[]}`,
  `fdParams.s2SplitStartIdxOfCore[]`, `sparseMode`.

### 3d. SplitFuse `FAInfer` — chunked-prefill / mixed batch (TND)
Purpose-built for concatenated prefill+decode batches (chunked prefill). Cube-mmad-tiled
flash-attention (`BlockMmadQK` + `BlockMmadPV`) with a `PRE_LAUNCH=2` software-pipelined cube↔vec
handshake and KV processed in `MAX_KV_STACK_LEN=512` stacks; paged KV via block tables.

- **Parallelization = plain round-robin** over a flattened task list
  `for (taskIdx = coreIdx; taskIdx < totalTaskNum; taskIdx += coreNum)`, where
  `totalTaskNum = Σ_batch (curQNBlockNum × curQSBlockNum)`.
- **Causal does *not* rebalance** here — it only shrinks per-task KV work
  (`noSkipKvS = min(kvSeqlen, (qSBlockIdx+1)*tile + max(0, kvSeqlen−qSeqlen))`). Balance relies on
  tasks ≫ cores so the cyclic stride averages the triangular skew.
- **Driver fields:** `FAInferTilingData.{totalTaskNum, firstBatchTaskNum, batch, numHeads,
  kvHeads, blockSize, maskType}`; core count comes from runtime `GetBlockNum()`, not tiling.

---

## 4. Host tiling (from `hicann/ops-transformer`, arch22 = 910B)

This section is sourced from the open host tiling, which the local install ships only as a `.so`.

### 4.1 Which kernel is selected
Selection is in `fused_infer_attention_score_tiling*.cpp` + `…_info_parser.cpp`:

- **Legacy IFA (decode)** is taken when
  `querySize == 1 && qkHeadDim == vHeadDim && no queryRope && no keyRope` (`isLegacyIfa_`).
  I.e. **single query token, non-MLA** → the incremental decode kernel.
- **FAI / SplitFuse** (`IsUsingFAI()`): **`TND` layout** and FAI-eligible — `D ≤ 256`,
  paged `blockSize` 16-aligned, KV tensor contiguous, supported `sparseMode`. This is the
  chunked-prefill / mixed-batch path.
- **FIA v3** takes the MLA / paged-KV / GQA cases (RoPE-split Q/K, `qkHeadDim ≠ vHeadDim`,
  block table present on non-TND, etc.), routed through `TilingFusedInferAttentionScoreV3` →
  `FiaTilingRegistry::DoTilingImpl`.
- **PFA** is the default multi-token prefill path (`TilingProcess4PFA` →
  `PromptFlashAttentionTiling::RunBigKernelTilingWithParams`; base key `1e18 + config`).

The tiling key's leading magnitude then encodes the path (5e18 FAI / 1e18 PFA / 1e17 FIA / else
IFA) and its lower digits encode layout, dtype, quant, sparse — exactly the fields decoded in §2.

### 4.2 Load balancing — the real algorithm (`common/op_host/split_core_v2.cpp`)
The FIA core split is **cost-model greedy bin-packing**, not a fixed formula, in `CalcSplitPlan`:

- **Work space** = 4-D `(batch b, kv-head n2, query-tile M, KV-tile S2)`. Per-block cost models the
  cube+vector time:
  ```cpp
  int64_t CalcCost(basicM, basicS2) {                 // per (M,S2) block
      return 6 * ceil(basicM/16) + 10 * ceil(basicS2/64);
  }
  ```
- **Greedy assign** hierarchically — whole batch → row(M) → block(S2) → forced — packing blocks
  onto core `c` until its accumulated cost reaches
  `avgCost = unassignedCost / (coreNum − curCoreIdx)`, then advancing to the next core. Output =
  prefix boundary arrays `bN2End[c]`, `mEnd[c]`, `s2End[c]` (what the kernel reads in §3c).
- **usedCoreNum** is chosen by a **sweep**: try every core count from
  `minCore = round(sqrt(totalBlockNum))` to `maxCore = min(coreNum, totalBlockNum)` and keep the
  one that minimizes peak per-core cost `maxCost`.
- **KV-split across cores (flash-decode)** falls out naturally: when a single `(b,n2,M)` row's cost
  exceeds a core's budget, the greedy cuts it at an `S2` boundary (`curKvSplitPart++`), and
  `IsNeedRecordFDInfo`/`RecordFDInfo` mark that row for a reduction (combine) pass. So few
  batch·head + long KV ⇒ automatic KV splitting; no explicit `B·N < coreNum` threshold needed.
- **Causal balancing is exact and implicit.** There is **no** separate triangular formula: causal
  simply changes how many `S2` blocks are *valid* per query tile via `CalcS2Range` /
  `CalcPreTokenLeftUp` / `CalcNextTokenLeftUp`, driven by `sparseMode`:
  ```cpp
  s2FirstToken = s1FirstToken - preTokenLeftUp;   // LEFT_UP_CAUSAL / RIGHT_DOWN / BAND
  s2LastToken  = s1LastToken  + nextTokenLeftUp;   // BAND: s2Size - s1Size + preToken/nextToken
  ```
  Top-of-triangle query tiles have fewer valid `S2` blocks → lower `CalcCost` → the greedy packs
  more of them per core. Because the split equalizes *summed real cost*, the triangular imbalance
  is absorbed automatically.
- **Non-causal** (`!attenMaskFlag`): `s2Start=0`, `s2End=ceil(s2Size/s2Base)` for every tile →
  uniform cost per M tile → the greedy degenerates to an even contiguous split.

> Note the **two different causal-balancing mechanisms** in the tree: the **FIA v3 / PFA-base_api**
> path balances on the *host* (cost-greedy `CalcSplitPlan` above), while the **older PFA `x910`
> kernel** balances *in-kernel* with a serpentine tile walk (§3b). Which one runs depends on which
> sub-kernel the host selects for the shape.

---

## 5. Decision summary (how input shape → kernel)

```
                          FusedInferAttentionScore  (host tiling → tiling key → JIT kernel)
                                        │
  small S_q (decode) ───────────────────┼──────────────────── larger S_q (prefill)
       │                                                              │
     IFA                                     ┌────────────┬───────────┴───────────┐
  (§3a)                                    PFA          FIA v3                 SplitFuse
  split over (b, n2);                     (§3b)         (§3c)                   (§3d)
  flash-decode KV-split                 Sq-tile ×      MLA / paged-KV /        TND chunked-prefill;
  + 2-pass LSE combine                  (B·N);         GQA; host cumulative-    cyclic task list;
  when B·N small & S_kv long.           in-kernel      work KV-split;          causal shrinks work,
  Causal irrelevant (Sq≈1).             serpentine     exact triangular         doesn't rebalance.
                                        for causal.    balance on host.
```

- **Causal handling differs sharply by path:** irrelevant on IFA decode; in-kernel *serpentine*
  reordering on PFA; host *cumulative-work KV-split* on FIA v3; *no rebalancing* (work-shrink only)
  on SplitFuse. Non-causal everywhere reduces to a uniform even split.
- **KV-axis splitting across cores** (flash-decoding) appears on IFA (decode) and FIA v3 (prefill
  with few batch·head but long KV); PFA and SplitFuse keep KV as an in-task reduction.
- To confirm which path/key a *specific* shape actually hits (the one thing not in source), read
  the tiling struct returned by `aclnnFusedInferAttentionScoreGetWorkspaceSize`, or profile.


> **Where the source lives.** The kernel side is shipped as source in the local CANN install, so
> the *set of kernels*, the *tiling-key → kernel* mapping, and each kernel's *core-split strategy*
> are read directly from it (§1–§3). The **host tiling** (the function that picks IFA/PFA/FIA/FAI
> and computes the exact core counts / split boundaries) is shipped **only as a compiled `.so`**
> in the local install (`op_host/lib/linux`) — but its **full source is public** at
> [`hicann/ops-transformer`](https://github.com/hicann/ops-transformer) under the "CANN Open
> Software" license, where **`arch22` = DaVinci V220 = Ascend 910B/910B2** (our chip; `arch35`/`arch38`
> are newer parts). §4 below is taken from that host source — so the exact selection conditions
> and the load-balancing algorithm are now fully sourced, not inferred.

