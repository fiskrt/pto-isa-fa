# `fa2.cpp`: source-level kernel walkthrough

This note describes what the current source actually does. It does not assume behavior from a FlashAttention paper or from another implementation. The relevant code is `fa2.cpp`, `fa2.h`, and the three macros called by the four compute stages.

## The short version

We have four stages:

1. **QK, on Cube:** compute one score subtile `Q_block @ K_subtile^T` in fp32.
2. **P, on Vec:** assemble those score subtiles into a logical `CUBE_S0 x TILE_S1` tile, apply the causal mask if requested, and perform the online softmax update. The unnormalized exponentials are cast to fp16.
3. **PV, on Cube:** compute the fp32 partial numerator `P_tile @ V_tile`, accumulating across all `CUBE_S1` subtiles in the logical tile.
4. **GU, on Vec:** update the running fp32 output numerator, rescale the old numerator when the softmax maximum changes, and divide by the final running sum on the last tile.

The Cube and Vec parts do not hand these large tiles directly through local memory. They use three ring FIFOs in **GM**:

```text
Q,K -> [Cube QK] -> qk_tile_fifo (fp32) -> [Vec P]
     -> p_tile_fifo (fp16) -> [Cube PV] -> pv_tile_fifo (fp32) -> [Vec GU] -> O
```

Cross-core FFTS flags say when a GM FIFO entry is ready and when a complete wrap of the ring has been consumed. Local `set_flag`/`wait_flag` pairs separately coordinate MTE, Vec, Cube matrix, and fix-pipe operations inside one core.

For the default configuration:

```text
HEAD_SIZE = 128
CUBE_S0   = 128
CUBE_S1   = 128
TILE_S1   = 256
kTileFactor = TILE_S1 / CUBE_S1 = 2
VEC_CORES = 2
CV_FIFO_SIZE = 8
QK_PRELOAD = 4
```

one logical attention tile is `128 x 256`, but Cube sees two `128 x 128` score/P subtiles. The default GM payloads are:

| Handoff | One Cube subtile | One logical tile / FIFO entry | Ring of 8, per active Cube/Vec pair |
|---|---:|---:|---:|
| QK -> P, fp32 | `128*128*4 = 64 KiB` | `128*256*4 = 128 KiB` | `1 MiB` |
| P -> PV, fp16 | `128*128*2 = 32 KiB` | `128*256*2 = 64 KiB` | `512 KiB` |
| PV -> GU, fp32 | accumulated result only | `128*128*4 = 64 KiB` | `512 KiB` |

Thus the three GM stage boundaries move `256 + 128 + 128 = 512 KiB` per logical tile when counting both the producer write and consumer read, before counting K/V input loads and the final O store.

## Execution trace: following one row block through the kernel

The easiest way to understand this kernel is to follow one `CUBE_S0`-row block through `runTFABodyRuntime`. The important point is that the source contains both Cube and Vec code, but an executing Cube core only enters the `DAV_CUBE` branches and its two associated Vec subcores only enter the `DAV_VEC` branches. The four stages therefore do not execute serially just because they appear in that order in the source. They run as a cross-core pipeline and meet at GM buffers and FFTS flags.

Assume this concrete configuration for the trace:

```text
HEAD_SIZE = 128
CUBE_S0   = 128
CUBE_S1   = 128
TILE_S1   = 1024
VEC_CORES = 2
CV_FIFO_SIZE = 8
QK_PRELOAD = 4

kTileFactor = TILE_S1 / CUBE_S1 = 8
Vec_S0      = CUBE_S0 / VEC_CORES / kTileFactor = 8
```

Here a logical attention tile is `128 x 1024`. It is too wide to be one Cube operation. The Cube-facing unit is `128 x 128`, so QK performs eight independent `128 x 128` output matmuls. PV later performs eight `128 x 128 @ 128 x 128` matmuls into one shared fp32 `128 x 128` accumulator. On Vec, each of the two Vec subcores owns 64 rows and processes those rows as eight slices of 8 rows; each Vec softmax call sees an `8 x 1024` tile.

### We enter the row-block loop

The kernel first maps the physical core to a logical `block_idx`. For this trace, suppose `block_idx=b`. The 128 query rows are:

```text
Q_block = Q[b*128 : (b+1)*128, :]
```

The code builds block-local pointers:

```text
q_block                 -> those 128 Q rows
p_tile_fifo_block       -> this physical core's fp16 P ring
qk_tile_fifo_block      -> this physical core's fp32 QK ring
pv_tile_fifo_block      -> this physical core's fp32 PV ring
o_out_block             -> final output rows for logical block b
```

The communication buffers use `comm_slot=physical_block_idx`, not `block_idx`. If this core later processes block `b+24`, it reuses the same three rings after draining the old synchronization credits.

Before any computation, the Cube side seeds its local MTE/matrix events and the Vec side seeds its MTE/Vec events. The QK and PV accumulator tiles are placed at alternating accumulator-memory addresses. K has two L1 tiles, and P and V each have two L1 tiles. The Vec side has two score tiles, two fp16 exponential tiles, and two PV input tiles. These are the ping/pong resources used while the pipeline advances.

### First we run ahead with QK and P

The warmup loop tries to produce four logical QK/P tiles before PV/GU begins. Consider logical S1 tile 0, covering K/V rows `0..1023`.

QK starts with `sub_tile_id=0`, covering K rows `0..127`. Because this is the first QK operation for the row block, it loads the complete `128 x 128` fp16 Q block into its single Q L1 tile:

```text
Q L1 load: 128 x 128 x 2 bytes = 32 KiB
```

Q remains in that L1 tile for every later S1 subtile and logical tile belonging to this row block. This is the main Q reuse in the kernel.

The first K subtile is also loaded from GM into K L1 ping buffer 0:

```text
K[0:128, :]: 128 x 128 x 2 bytes = 32 KiB
```

`pto_macro_matmul<128,128,128>` then extracts Q and K from L1 into L0A/L0B and executes the Cube matmul. For this shape the internal `Cube_K` is 128, so there is one matrix reduction segment. The fp16 inputs produce one fp32 `128 x 128` accumulator tile.

The final-store form of `TSTORE` sends that accumulator through the Cube final/fix path into the QK GM ring. The destination for subtile 0 is:

```text
qk ring entry 0 + 0 * (128*128 floats)
```

and the payload is:

```text
128 x 128 x 4 bytes = 64 KiB
```

QK then moves to `sub_tile_id=1`. It reuses Q, loads K rows `128..255` into K L1 ping buffer 1, computes the next `128 x 128` fp32 result, and stores it 64 KiB after the first subtile. The K source alternates between its two L1 tiles:

```text
subtile 0 -> kMatTile[0]
subtile 1 -> kMatTile[1]
subtile 2 -> kMatTile[0]
subtile 3 -> kMatTile[1]
...
```

The associated `QK_EVENT_ID0/1` waits prevent MTE from overwriting a K L1 buffer while the previous matrix operation still uses it. Inside each matmul, L0A and L0B additionally alternate between addresses `0` and `0x8000`; this is a second, lower-level double buffer used to overlap L1-to-L0 extraction with Cube execution when a matmul has multiple K segments.

After eight QK matmuls, the GM entry contains:

```text
8 adjacent [128,128] fp32 subtiles
= 128*1024 fp32 elements
= 512 KiB
```

It is physically a concatenation of eight Cube subtiles, not one ordinary row-major `[128,1024]` matrix. Only after the eighth final-store has been issued does QK signal `BUF0_QK_READY` from `PIPE_FIX`.

### P observes QK ready and assembles a Vec tile

The two Vec subcores have been waiting at `BUF0_QK_READY`. When the flag arrives, each Vec subcore starts on its half of the 128 rows.

For `row_slice=0`, Vec subcore 0 takes rows `0..7` of each of the eight stored QK subtiles. It performs eight GM-to-UB loads into column views of one `8 x 1024` fp32 `qkVecTile`:

```text
QK subtile 0 rows 0..7 -> qkVecTile[:,   0:128]
QK subtile 1 rows 0..7 -> qkVecTile[:, 128:256]
...
QK subtile 7 rows 0..7 -> qkVecTile[:, 896:1024]
```

That call transfers `8*1024*4 = 32 KiB` into UB. Vec subcore 1 simultaneously does the same for its first local slice, global block rows `64..71`. The following `row_slice` calls cover `8..15`, `16..23`, and so on on the first Vec subcore, and `72..79`, `80..87`, and so on on the second.

On logical tile 0, the softmax macro initializes the online state. For each row it calculates:

```text
m = rowmax(QK)
P_fp32 = exp((QK - m) / sqrt(128))
l = rowsum(P_fp32)
P_fp16 = cast(P_fp32)
```

`m2_global_max` and `l2_global_sum` remain in Vec UB for the next logical S1 tile. The fp16 result is held in `x_expT`, another `8 x 1024` Vec tile.

P now scatters `x_expT` to GM in the same eight-subtile layout that PV expects. One row-slice call writes:

```text
8 x 1024 x 2 bytes = 16 KiB
```

Across eight row slices and two Vec subcores, the complete logical P entry is:

```text
128 x 1024 fp16 = 256 KiB
```

After the final row slice has issued its GM stores, P signals `BUF1_SM_READY` from the MTE3 store pipe. PV is allowed to read the entry only after this point.

The P-side score source and GU-side PV source use a union allocation in UB. `qkVecTile[0]` occupies the same address as `pvVecTile[0]`, and index 1 is shared in the same way. `p_gu_src_pingpong_id` rotates these aliases and their event IDs. This is safe because a particular shared buffer is not used for a P score tile and a GU PV tile at the same time; the event waits enforce that lifetime boundary.

### The warmup repeats before PV starts

The same sequence is started for logical tiles 1, 2, and 3. QK writes ring entries 1, 2, and 3; P waits for each ready flag, performs its online-softmax update, and writes the corresponding fp16 P entries.

For tiles after tile 0, P no longer initializes the softmax state. Suppose the old row maximum and sum are `m_old` and `l_old`. P computes:

```text
m_new = max(m_old, rowmax(QK_tile))
alpha = exp((m_old - m_new) / sqrt(128))
l_new = alpha*l_old + rowsum(exp((QK_tile-m_new)/sqrt(128)))
```

It stores `alpha` in the local UB ring:

```text
l1_exp_max_ififo[tile_id % 8]
```

GU will need exactly this factor when the matching PV tile arrives. This local ring is necessary because P is four tiles ahead of GU during the steady state. It is unrelated to the optional GM `exp_max_ififo`, which is only written for intermediate checking.

At this point the pipeline has four P tiles queued in GM and four rescale-factor tiles queued in Vec UB. The warmup ends and the main `tile_id` loop starts consuming tile 0.

### In steady state, future QK overlaps current PV

At steady-state iteration `tile_id=0`, the code sets:

```text
next_qk_tile = tile_id + QK_PRELOAD = 4
```

The Cube loop now interleaves two different matmul streams at `CUBE_S1` granularity:

```text
QK subtile 0 of logical tile 4
PV subtile 0 of logical tile 0
QK subtile 1 of logical tile 4
PV subtile 1 of logical tile 0
...
QK subtile 7 of logical tile 4
PV subtile 7 of logical tile 0
```

This is why QK and PV have separate L1 ping/pong sets and separate accumulator assignments. QK is trying to keep the P stage supplied several tiles in the future while PV consumes the oldest ready P tile.

On the Vec side, the analogous ordering is:

```text
P row slices for logical tile 4
GU for logical tile 0
```

The exact amount of hardware overlap is controlled by dependencies and the separate Cube/Vec execution streams, but this ordering exposes QK/P work from the future at the same time as PV/GU drains the present tile.

### PV consumes the eight P subtiles

PV waits once on `BUF1_SM_READY`, then loads P subtile 0 from the P GM ring and V rows `0..127` from the input tensor. Both operands are fp16 `128 x 128` matrices in this example.

The first call uses `AccMode::InitPartialSum`:

```text
pvAcc = P[:, 0:128] @ V[0:128, :]
```

It does not store this partial result to GM. PV then alternates the P and V L1 ping/pong tiles while accumulating:

```text
pvAcc += P[:, 128:256] @ V[128:256, :]
pvAcc += P[:, 256:384] @ V[256:384, :]
...
pvAcc += P[:, 896:1024] @ V[896:1024, :]
```

Subtiles 1 through 6 use partial accumulation. Subtile 7 uses `AccFinalSum`, telling the Cube/fix path that the fp32 accumulator is complete. Therefore the eight Cube matmuls produce only one GM transfer:

```text
PV logical result: 128 x 128 fp32 = 64 KiB
```

Before this final store, PV checks that the selected `pv_tile_fifo` ring entry is no longer owned by GU. It then writes the result and signals `UPDATE_READY` from `PIPE_FIX`.

The P and V input matrices have independent two-entry L1 ping/pong arrays:

```text
subtile 0 -> pMatTile[0], vMatTile[0]
subtile 1 -> pMatTile[1], vMatTile[1]
subtile 2 -> pMatTile[0], vMatTile[0]
...
```

`PV_EVENT_ID0/1` protect those pairs. The fp32 PV accumulator is separate and remains live across all eight subtile calls.

### GU receives the completed PV tile

After `UPDATE_READY`, both Vec subcores load their half of the PV result:

```text
Vec subcore 0: PV rows  0..63, 64*128*4 = 32 KiB
Vec subcore 1: PV rows 64..127, 64*128*4 = 32 KiB
```

For logical tile 0, GU simply loads this PV tile into `runningOTile`. There is no old numerator to rescale.

On steady iteration 1, PV produces logical tile 1 and GU uses the factor that P saved earlier for tile 1:

```text
runningOTile = runningOTile * l1_exp_max_ififo[1] + PV_1
```

The factor is one fp32 value per row and is broadcast over all 128 output columns. The running fp32 numerator never goes back to GM between tiles; it remains in `runningOTile` in each Vec subcore's UB. This avoids a read and write of the accumulated output on every S1 tile.

GU repeats this update for every logical tile. On the final tile it also divides every row by `l2_global_sum`. Only then does each Vec subcore store its half to `o_out`. For this row block, the only running-output GM write is the final:

```text
128 x 128 fp32 = 64 KiB
```

### How the ring buffers prevent overwrite

Every logical tile uses `tile_id % 8` as its ring entry. Ready flags are per tile, but free-space synchronization is aggregated per complete eight-entry wrap.

For example, QK can write QK entries 0 through 7. Before logical tile 8 reuses entry 0, it waits for `BUF0_SM_CONSUMED`. P emits that flag only after it has consumed the last row slice of tile 7. P and PV use the same pattern with `BUF1_SV_CONSUMED`; PV and GU use `UPDATE_CONSUMED`.

Thus the two forms of buffering serve different levels:

```text
L0 ping/pong:
    overlaps L1 -> L0 extraction with Cube matrix execution

L1 ping/pong:
    allows K, P, and V for the next Cube subtile to reuse alternating storage

Vec UB ping/pong:
    alternates score/exponential/PV working tiles and their local events

GM ring of 8:
    absorbs whole logical-tile skew between Cube and Vec stages

QK_PRELOAD of 4:
    deliberately establishes four tiles of distance between QK/P and PV/GU
```

When there are fewer than four future tiles left, `next_qk_tile` becomes `-1`. QK and P stop producing, while PV and GU continue draining the already queued tiles. At the end of the row block, the code waits for all local ping/pong events, drains any unmatched ring-wrap flags, executes an all-pipe barrier, and only then lets that physical core reuse its buffers for its next assigned row block.

## Mathematical operation

The kernel handles one 2-D attention problem. There is no batch or head index in this kernel:

```text
Q: [s0, HEAD_SIZE], fp16
K: [s1, HEAD_SIZE], fp16
V: [s1, HEAD_SIZE], fp16
O: [s0, HEAD_SIZE], fp32
```

For every query row, the intended streamed calculation is:

```text
S_t     = Q @ K_t^T                                      # fp32
m_t     = max(m_{t-1}, rowmax(S_t))                       # stored unscaled
alpha_t = exp((m_{t-1} - m_t) / sqrt(HEAD_SIZE))
P_t     = exp((S_t - m_t) / sqrt(HEAD_SIZE))              # cast to fp16
l_t     = alpha_t * l_{t-1} + rowsum(P_t)                 # fp32
U_t     = alpha_t * U_{t-1} + P_t @ V_t                   # fp32
O       = U_last / l_last                                 # fp32
```

The first tile initializes `m`, `l`, and `U`, so it does not need `alpha_0`. `P_t` is deliberately **not normalized** before PV. Normalization happens once, in GU, on the final logical S1 tile.

An important implementation detail is that the max is kept in the unscaled QK domain. The code subtracts the max first and then multiplies by `1/sqrt(HEAD_SIZE)`. The rescale factor uses the same ordering, so the mathematics is consistent.

## Parameters and derived dimensions

### Compile-time template parameters

`LaunchTFA` and the device kernel are specialized by these values:

| Name | Meaning |
|---|---|
| `HEAD_SIZE` | Head dimension `D`; the reduction dimension of QK and the output width of PV/GU. |
| `CUBE_S0` | Number of query rows owned by one logical row block. It is Cube matmul `M`. |
| `CUBE_S1` | Width of one Cube key/value subtile. It is QK matmul `N` and PV matmul `K`. Default 128. |
| `TILE_S1` | Width of one logical streamed softmax tile. Default 256. It must be divisible by `CUBE_S1`. |
| `QK_PRELOAD` | Number of QK/P logical tiles produced before PV/GU starts. Default 4. |
| `CV_FIFO_SIZE` | Entry count of all three GM ring FIFOs. Default 8. |
| `INTERMEDIATE_CHECK` | Enables a GM store of per-tile `exp_max`; the normal JIT instantiation passes `false`. |
| `CAUSAL_MASK` | Compiles in causal pruning and the exact triangular mask. |
| `CV_FIFO_CONS_SYNC_PERIOD` | Required to be at least 1, but otherwise unused by the current implementation. |

The source derives:

```text
block_rows   = s0 / CUBE_S0
kTileFactor  = TILE_S1 / CUBE_S1
Vec_S0       = CUBE_S0 / VEC_CORES / kTileFactor
VecGuRows    = CUBE_S0 / VEC_CORES
total_tiles  = s1 / TILE_S1
```

`kTileFactor` has two meanings in the scheduling:

- Cube performs that many column subtiles per logical S1 tile.
- Each Vec subcore performs that many row slices so that its `CUBE_S0 / VEC_CORES` rows are covered without making the Vec tile larger.

The compile-time checks require:

```text
TILE_S1 % CUBE_S1 == 0
CUBE_S0 % (VEC_CORES * kTileFactor) == 0
QK_PRELOAD >= 1
QK_PRELOAD > 1, unless kTileFactor == 1
```

The host additionally asserts:

```text
s0 > 0, s1 > 0
s0 % CUBE_S0 == 0
s1 % CUBE_S1 == 0
s1 % TILE_S1 == 0
```

There is no tail path. Every runtime dimension must tile exactly.

### Runtime pointer parameters

| Parameter | Source-level role |
|---|---|
| `ffts` / `ffts_addr` | Address returned by `rtGetC2cCtrlAddr`; installed with `set_ffts_base_addr` and used by cross Cube/Vec flag synchronization. |
| `q` | Contiguous fp16 `[s0, HEAD_SIZE]`. |
| `k` | Contiguous fp16 `[s1, HEAD_SIZE]`; exposed to QK as a transposed/DN view. |
| `v` | Contiguous fp16 `[s1, HEAD_SIZE]`. |
| `p_tile_fifo` | GM ring, fp16, P producer to PV consumer. |
| `exp_max_ififo` | GM fp32 ring used only when `INTERMEDIATE_CHECK=true`. Normal GU gets its rescale factors from a local UB ring instead. |
| `global_sum_out` | Passed into `compute_p` but not read or written by the current source. |
| `exp_max_out` | Passed into `compute_p` but not read or written by the current source. |
| `o_out` | Actual fp32 `[s0, HEAD_SIZE]` result. |
| `o_parts_out` | Passed into GU but unused by the current source. |
| `qk_tile_fifo` | GM ring, fp32, QK producer to P consumer. |
| `pv_tile_fifo` | GM ring, fp32, PV producer to GU consumer. |
| `profile_data` | Optional profiling storage. The normal JIT wrapper passes null. |
| `cv_comm_buf` | Allocated and passed by the JIT host, but unused in `fa2.cpp`. Synchronization uses `ffts_addr`, not this buffer. |

Before launch, the host calls `PTO_PREFETCH` for the full Q, K, and V tensors. The byte arguments are `s0*HEAD_SIZE*2` for Q and `s1*HEAD_SIZE*2` each for K and V. The device launch always requests 24 AICores.

## Row-block ownership and core scheduling

The work unit assigned to a Cube/Vec core group is a row block:

```text
Q_block = Q[block_idx*CUBE_S0 : (block_idx+1)*CUBE_S0, :]
```

For dense attention, physical core `c` handles:

```text
block_idx = c, c + 24, c + 48, ...
```

For causal attention, if `block_rows` is divisible by 48, the source uses mirrored pairs. Each physical core receives top blocks and their bottom mirrors. That balances cheap upper blocks, which have few legal key tiles, against expensive lower blocks, which have many. Otherwise it falls back to the same stride-24 schedule as dense attention.

`comm_slot = physical_block_idx`. Therefore a physical core reuses the same GM FIFO region for every row block it processes sequentially. The end-of-block flag drains are necessary so a credit left by one block is not mistaken for a credit belonging to the next block.

The actual tensor outputs use `block_idx`, not `physical_block_idx`, so mirrored scheduling does not change output order.

## GM FIFO layout

All offsets below are in elements. For one physical communication slot:

```text
p_fifo_block_stride  = CV_FIFO_SIZE * CUBE_S0 * TILE_S1       # fp16
qk_fifo_block_stride = CV_FIFO_SIZE * CUBE_S0 * TILE_S1       # fp32
pv_fifo_block_stride = CV_FIFO_SIZE * CUBE_S0 * HEAD_SIZE     # fp32
```

The logical ring index is always:

```text
buf_idx = tile_id % CV_FIFO_SIZE
```

QK and P do **not** store one ordinary row-major `[CUBE_S0, TILE_S1]` matrix. Each FIFO entry is a concatenation of `kTileFactor` independent Cube subtiles:

```text
entry base
  + 0 * CUBE_S0*CUBE_S1 -> subtile 0, [CUBE_S0, CUBE_S1]
  + 1 * CUBE_S0*CUBE_S1 -> subtile 1, [CUBE_S0, CUBE_S1]
  + ...
```

This is why `compute_p` adds `sub_col*CUBE_S0*CUBE_S1` when gathering columns, and why `compute_pv` uses the same offset when reading P. At the GM boundary the units are Cube-shaped, even though Vec treats the gathered result as one `Vec_S0 x TILE_S1` row-major tile.

PV is different. All S1 subtiles have already been reduced into one accumulator, so an entry is an ordinary fp32 `[CUBE_S0, HEAD_SIZE]` matrix.

The host workspace sizes are formulas over `block_rows`, while the device indexes them by physical `comm_slot`. In the usual dense case only cores whose initial `physical_block_idx < block_rows` enter the loop; in the paired causal case `block_rows >= 48`, so all 24 slots exist.

## Stage 1: `compute_qk`

For logical S1 tile `tile_id` and Cube column subtile `sub_tile_id`:

```text
s0_index = block_idx * CUBE_S0
s1_index = tile_id*TILE_S1 + sub_tile_id*CUBE_S1
```

The Cube operands are:

```text
Q: [CUBE_S0, HEAD_SIZE], fp16
K: [HEAD_SIZE, CUBE_S1], fp16 transposed view
C: [CUBE_S0, CUBE_S1], fp32 accumulator
```

The Q tile is loaded from GM to L1 only when `tile_id==0 && sub_tile_id==0`, then reused for every K subtile belonging to this row block. K is loaded for every subtile. The call is:

```cpp
pto_macro_matmul<CUBE_S0, HEAD_SIZE, CUBE_S1>(..., AccMode::InitFinalSum)
```

Within that macro, the HEAD_SIZE reduction can be split again into `Cube_K` panels. `Cube_K` is the largest of 256, 128, 64, or 32 for which both the `M x Cube_K` L0A panel and `Cube_K x N` L0B panel fit in a 32 KiB ping/pong buffer. L1-to-L0 extraction alternates addresses `0` and `0x8000`, overlapping the next extraction with matrix execution. For the default `M=N=K=128`, `Cube_K=128`, so one QK subtile is one macro matmul call with no K segmentation.

The fp32 accumulator result is stored to the QK GM FIFO. For default dimensions, one store is 64 KiB and two stores complete the 128 KiB logical FIFO entry.

On the first subtile of a ring wrap, QK waits for `BUF0_SM_CONSUMED`. After the final logical subtile it signals `BUF0_QK_READY`, allowing P to consume that tile.

### Causal pruning in QK

The last query row in the row block is:

```text
block_last_s0 = s0_index + CUBE_S0 - 1
```

If a K subtile starts strictly after that row, every score in that subtile is future-masked, so Cube skips the matmul and store. The final skipped subtile still emits `BUF0_QK_READY`. P pre-fills its local score tile with a large negative value before selectively loading active QK subtiles, so skipped GM storage is never consumed as valid data.

## Stage 2: `compute_p`

P runs on two Vec subcores. `get_subblockid()` selects which half of the query block a Vec core owns:

```text
subblock_base_rows = (CUBE_S0 / 2) * get_subblockid()
local_row_offset   = row_slice * Vec_S0
row_offset         = subblock_base_rows + local_row_offset
```

With defaults, `Vec_S0=32`. The calls cover:

```text
Vec subcore 0: rows  0..31,  32..63
Vec subcore 1: rows 64..95, 96..127
```

Each call gathers `kTileFactor` QK blocks from GM into one fp32:

```text
qkVecTile: [Vec_S0, TILE_S1]
```

With defaults, each Vec call reads `32*256*4 = 32 KiB`; four calls across two Vec subcores and two row slices read the complete 128 KiB QK FIFO entry.

P waits for `BUF0_QK_READY` once, on `row_slice==0`. After the final row slice it releases the QK ring at wrap boundaries through `BUF0_SM_CONSUMED`.

### Online softmax state

The following fp32 reduction tiles contain one value per row owned by a Vec subcore:

| Variable | Meaning in the current tile |
|---|---|
| `m1_local_max` | Current-tile row max; in the non-init path its slice is then overwritten with the new global max. |
| `l1_local_sum` | Sum of the current unnormalized exponential tile. |
| `m2_global_max` | Running global max, kept resident in UB across S1 tiles. |
| `l2_global_sum` | Running global denominator, kept resident in UB across S1 tiles. |
| `l1_exp_max_ififo[buf]` | UB ring of `alpha_t = exp((m_{t-1}-m_t)/sqrt(D))`; P produces it ahead of GU. |
| `input_reduce_tmp` | Scratch used by row reductions. Only `float_tile_bytes/8` bytes are reserved at its assigned address. |
| `triu` | fp32 temporary used to build the causal triangular mask. |

The first logical tile calls the `init=true` softmax specialization. Later tiles calculate a new max, update the global sum, and save `alpha_t` into `l1_exp_max_ififo[tile_id % CV_FIFO_SIZE]` for GU.

The output `x_expT` is:

```text
[Vec_S0, TILE_S1], fp16
```

It contains the unnormalized `P_t`. P scatters it back to GM as `kTileFactor` `[Vec_S0, CUBE_S1]` pieces. Across both Vec subcores and all row slices this produces one fp16 `[CUBE_S0, TILE_S1]` logical FIFO entry: 64 KiB with defaults.

P waits for P-FIFO room via `BUF1_SV_CONSUMED` at ring wrap boundaries, and signals `BUF1_SM_READY` after the last row slice has been stored.

### Exact causal mask in P

For a row slice, QK subtiles starting after its last query row are left at `-3.40282e38`. The softmax macro then builds a triangular tile with:

```text
diagonal = absolute_query_start - absolute_key_tile_start + 1
```

multiplies the masked positions by the same large negative value, and adds it to QK. This gives the row-level `key_index <= query_index` mask. Coarse QK pruning avoids useless Cube work; this Vec mask handles the diagonal boundary exactly.

## Stage 3: `compute_pv`

For every logical tile, PV loops over the same `kTileFactor` S1 subtiles. Its Cube operands are:

```text
P_subtile: [CUBE_S0, CUBE_S1], fp16, from p_tile_fifo
V_subtile: [CUBE_S1, HEAD_SIZE], fp16, from input V
accumulator: [CUBE_S0, HEAD_SIZE], fp32
```

The first subtile initializes the accumulator, middle subtiles accumulate partial sums, and the final active subtile marks the accumulator final:

```text
subtile 0: InitPartialSum, or InitFinalSum if it is also last
middle:    AccPartialSum
last:      AccFinalSum
```

The resulting logical partial numerator is:

```text
PV_t = P_t @ V_t: [CUBE_S0, HEAD_SIZE], fp32
```

Only that fully accumulated result is written to `pv_tile_fifo`. With default dimensions it is 64 KiB.

PV waits for `BUF1_SM_READY` on its first subtile. Once it has loaded the final active P subtile, it returns P-FIFO credit at ring wrap boundaries with `BUF1_SV_CONSUMED`. Before overwriting the PV ring it waits for `UPDATE_CONSUMED`, then signals `UPDATE_READY` after the fp32 store.

In causal mode, PV stops once the next Cube subtile starts after the row-block start. For the intended/default tilings this is the last subtile that can contain legal positions for the block; masked P values make the future elements within that subtile zero.

## Stage 4: `compute_gu`

Each Vec subcore loads its half of the fp32 PV entry:

```text
pvGlobal: [CUBE_S0 / 2, HEAD_SIZE]
```

GU waits for `UPDATE_READY`. Tile 0 is loaded directly into `runningOTile`. For later tiles:

```text
runningOTile = runningOTile * l1_exp_max_ififo[tile] + pvVecTile
```

The row-wise multiplication broadcasts the per-row `alpha_t` over `HEAD_SIZE`. On the last tile the macro also performs:

```text
runningOTile /= l2_global_sum
```

The running numerator stays in Vec UB for the complete S1 loop. Only the final tile is stored to `o_out`, one fp32 `[CUBE_S0/2, HEAD_SIZE]` store per Vec subcore.

After loading PV, GU returns ring credit at wrap boundaries using `UPDATE_CONSUMED`.

## Warmup and steady-state pipeline

The source first produces `min(QK_PRELOAD, num_tiles_s1)` QK and P tiles. With the default preload of four, the conceptual schedule is:

```text
warmup:
    QK(0..3) -> P(0..3)

steady iteration t:
    Cube: QK(t+4) interleaved with PV(t), one CUBE_S1 subtile at a time
    Vec:  P (t+4) row slices, then GU(t)
```

The code interleaves future QK/P work and current PV/GU work in the same loop. The two compiled core roles progress independently and block only at the cross-core flags or local memory events. Near the end, `next_qk_tile=-1`, so only PV/GU drains remain.

The three GM FIFOs all have depth 8 by default, which is larger than the preload of 4. The FIFO depth controls overwrite safety; preload controls the Cube/Vec phase distance. They are independent parameters, although the code gives QK/P and PV the same `CV_FIFO_SIZE`.

## Synchronization flags

Cross-core flags are:

| Flag | Producer -> consumer | Meaning |
|---|---|---|
| `BUF0_QK_READY` | QK Cube -> P Vec | One complete logical QK tile is in GM. |
| `BUF0_SM_CONSUMED` | P Vec -> QK Cube | A full QK ring wrap has been consumed; producer may reuse it. |
| `BUF1_SM_READY` | P Vec -> PV Cube | One complete logical fp16 P tile is in GM. |
| `BUF1_SV_CONSUMED` | PV Cube -> P Vec | A full P ring wrap has been consumed. |
| `UPDATE_READY` | PV Cube -> GU Vec | One complete fp32 PV tile is in GM. |
| `UPDATE_CONSUMED` | GU Vec -> PV Cube | A full PV ring wrap has been consumed. |

Ready flags are sent once per logical tile. Consumption flags are aggregated: the consumer signals when `(tile_id+1) % FIFO_DEPTH == 0`, and the producer waits before tile IDs that are nonzero multiples of the depth. This makes the ring operate in full-ring credits rather than one credit per entry.

`cv_drain_fifo_free_flags` consumes any unmatched end-of-block credit. This matters because the same physical core's FFTS flags and communication slot are reused for its next row block.

`QK_EVENT_ID0/1` and `PV_EVENT_ID0/1` are local ping/pong event IDs. They protect L1 matrix tiles from being overwritten while MTE/matrix operations still use them. `EVENT_ID0..3` elsewhere initialize and drain local pipeline dependencies. These events are not the same as the six cross-core FIFO flags above.

## On-chip tile allocation

### Cube L1

The Cube source tiles are assigned consecutively in L1:

```text
1 * Q tile: [CUBE_S0, HEAD_SIZE], fp16
2 * K tile: [HEAD_SIZE, CUBE_S1], fp16
2 * P tile: [CUBE_S0, CUBE_S1], fp16
2 * V tile: [CUBE_S1, HEAD_SIZE], fp16
```

The allocation formula is:

```text
2*CUBE_S0*HEAD_SIZE
+ 4*HEAD_SIZE*CUBE_S1
+ 4*CUBE_S0*CUBE_S1
+ 4*CUBE_S1*HEAD_SIZE bytes
```

For all-default 128 dimensions this is 229,376 bytes, or 224 KiB, below the source's 512 KiB check.

QK and PV fp32 accumulator tiles are assigned to alternating accumulator-memory addresses `0x0` and `0x10000`. `assign_running_acc_tile` keeps a static toggle per accumulator tile type. This separates consecutive accumulator lifetimes and permits overlap with final stores. Independently, the matmul macro ping-pongs L0A/L0B extraction buffers at `0x0` and `0x8000`.

### Vec UB

Important Vec tiles are:

```text
2 * qkVecTile: [Vec_S0, TILE_S1], fp32
2 * x_expT:    [Vec_S0, TILE_S1], fp16
runningOTile:  [CUBE_S0/2, HEAD_SIZE], fp32
2 * pvVecTile: [CUBE_S0/2, HEAD_SIZE], fp32
8 * exp-max reduction tiles: [CUBE_S0/2, 1], fp32
softmax reduction and scratch tiles
guScratch: [CUBE_S0/2, 32], fp32
```

`qkVecTile[i]` and `pvVecTile[i]` are deliberately a union: they receive the same UB address with a stride equal to the larger tile. Their lifetimes are scheduled through the shared `p_gu_src_pingpong_id`. This saves UB because P's score source and GU's PV source do not need distinct storage at the same ping/pong index.

With default dimensions, the assigned Vec UB tail reaches 179,200 bytes, or 175 KiB, under the 192 KiB limit. The static size expression in `allocate_vec_tile_buffers` undercounts the actual assignment by one 256-byte reduction tile, but the default still fits.

The ping/pong counters are:

| Variable | What it rotates |
|---|---|
| `p_gu_src_pingpong_id` | Two QK Vec tiles, two fp16 exponential tiles, and two GU PV tiles/events. |
| `k_src_pingpong_id` | Two K L1 tiles and QK local event IDs. |
| `pv_src_pingpong_id` | Two P L1 tiles, two V L1 tiles, and PV local event IDs. |

## Default per-tile traffic, by stage

For `CUBE_S0=CUBE_S1=HEAD_SIZE=128` and `TILE_S1=256`, per query row block and logical S1 tile:

| Stage | GM reads | GM writes |
|---|---:|---:|
| QK | K: 64 KiB; Q: 32 KiB only once for the entire row block | QK: 128 KiB |
| P | QK: 128 KiB | P: 64 KiB |
| PV | P: 64 KiB; V: 64 KiB | PV: 64 KiB |
| GU | PV: 64 KiB | O: 64 KiB only on the final logical tile |

K and V are reloaded for every query row block. Q is reused across all S1 tiles within its row block. The table describes explicit payloads in the source; cache behavior and transaction granularity are not inferable from `fa2.cpp`.

## Causal tile count and balancing

Dense mode processes:

```text
num_tiles_s1 = s1 / TILE_S1
```

Causal mode limits each row block to:

```text
num_tiles_s1 = min(total_tiles_s1,
                   1 + floor(block_idx*CUBE_S0 / TILE_S1))
```

This is a coarse logical-tile cutoff. QK has a finer Cube-subtile cutoff, and P applies the exact elementwise triangle. The combination is why upper causal blocks do much less Cube work and why mirrored row-block scheduling improves load balance.

The causal shortcuts assume compatible relationships among `CUBE_S0`, `CUBE_S1`, and `TILE_S1`. The source checks divisibility needed by its loops, but it does not statically assert every alignment implied by the PV early-stop condition. The default `CUBE_S0=CUBE_S1=128`, `TILE_S1=256` satisfies those relationships.

## Source-level caveats

These are properties of the current code, not inferred intent:

- `global_sum_out`, `exp_max_out`, and `o_parts_out` are vestigial parameters in the current path.
- `cv_comm_buf` and `CV_FIFO_CONS_SYNC_PERIOD` currently have no operational use.
- `INTERMEDIATE_CHECK=true` stores `l1_exp_max_ififo` to GM, but the normal JIT instantiation is `false`.
- The profiling buffer reserves 3 KiB per logical row block: the first KiB for Cube and the next two KiB for the two Vec subcores. Each participant writes start/end counters in its own region.
- In `compute_gu`, a one-tile run divides by `l2_global_sum` only inside the compile-time `CAUSAL_MASK` branch. Therefore a dense configuration with exactly one logical S1 tile returns the unnormalized PV numerator in the current source. Multi-tile dense runs normalize in the normal last-tile path.
- The Vec UB compile-time accounting expression is 256 bytes smaller than the sequence of default address assignments, as noted above.

## Practical performance interpretation

This is a Cube/Vec producer-consumer pipeline whose main cost is not just the two matmuls. A complete logical tile crosses GM three times because the producer and consumer are different core types. The key tuning tradeoffs are therefore:

- `CUBE_S0`: more query rows improve Q reuse and Cube work per handoff, but increase accumulator, L1, UB, and every FIFO entry.
- `CUBE_S1`: controls the actual Cube operand width and the P/V transfer unit. It also affects L0 fitting and causal pruning granularity.
- `TILE_S1`: controls softmax/GU streaming frequency. A larger value amortizes ready flags and GU updates, but increases QK/P FIFO entries and Vec score tiles.
- `QK_PRELOAD`: changes how far QK/P runs ahead of PV/GU. It hides stage imbalance but does not increase ring capacity.
- `CV_FIFO_SIZE`: determines how much producer/consumer skew GM can absorb and directly multiplies workspace size.

For the default shape, the Vec UB is already 175 KiB of a checked 192 KiB and each active core pair owns 2 MiB of the three primary GM rings. That makes `TILE_S1`, Vec ping/pong depth, and FIFO depth memory-sensitive knobs. The Cube L1 allocation is less tight at 224 KiB of a checked 512 KiB.
