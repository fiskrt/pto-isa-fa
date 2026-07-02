/*
 * C-ABI launcher so a torch_npu tensor can drive the FA kernel directly (no .bin golden files).
 *
 * Q/K/V come in as device pointers from torch tensors; O is written into a torch fp32 tensor.
 * All the intermediate FIFO / scratch buffers the kernel needs are sliced out of a single
 * caller-provided workspace block. Only the tiling (HEAD_SIZE / CUBE_S0 / CUBE_S1 / TILE_S1 /
 * QK_PRELOAD) is fixed at compile time; the sequence lengths S0 (rows) and S1 (cols) are chosen
 * at runtime by the caller, so a single instantiated kernel serves arbitrary shapes as long as
 * S0 % CUBE_S0 == 0 and S1 % TILE_S1 == 0.
 *
 * Expected tensor layouts (row-major, matching the kernel):
 *   q : [S0, HEAD]  float16
 *   k : [HEAD, S1]  float16   (i.e. K transposed: k[h, j] == K[j, h])
 *   v : [S1, HEAD]  float16
 *   o : [S0, HEAD]  float32   (output)
 */

#include <acl/acl.h>
#include <cstdint>
#include <cstdio>

#include "runtime/rt.h"
#include "fa_performance_kernel.h"

namespace {

// ---- Fixed tiling. Must match the INSTANTIATE_TFA(...) case in libfa_performance_kernel.so. ----
constexpr int HEAD_SIZE = 128;
constexpr int CUBE_S0 = 128;
constexpr int CUBE_S1 = kFaCubeS1;       // 128
constexpr int TILE_S1 = kFaTileS1;       // 256
constexpr int QK_PRELOAD = kFaQkPreload; // 4

constexpr int tile_factor = TILE_S1 / CUBE_S1;

// Per-block FIFO strides depend only on the fixed tiling (independent of S0/S1).
constexpr size_t qk_fifo_stride = static_cast<size_t>(kFaCvFifoSize) * CUBE_S0 * tile_factor * CUBE_S1;
constexpr size_t p_max_fifo_stride = static_cast<size_t>(kFaCvFifoSize) * CUBE_S0;
constexpr size_t pv_fifo_stride = static_cast<size_t>(kFaCvFifoSize) * CUBE_S0 * HEAD_SIZE;

constexpr size_t kWsAlign = 512; // each sub-buffer is 512-byte aligned

constexpr size_t align_up(size_t x)
{
    return (x + kWsAlign - 1) & ~(kWsAlign - 1);
}

// The nine scratch buffers, in the order they are carved out of the single caller-provided
// workspace block. Sizes scale with block_rows = S0/CUBE_S0 and num_tiles = S1/TILE_S1, so they
// are computed at runtime from (s0, s1). Keep this in sync with Scratch's member order.
struct Sizes {
    size_t bytes[9];
    static constexpr size_t kCount = 9;
};

Sizes compute_sizes(int s0, int s1)
{
    const size_t block_rows = static_cast<size_t>(s0) / CUBE_S0;
    const size_t num_tiles = static_cast<size_t>(s1) / TILE_S1;

    Sizes sz{};
    sz.bytes[0] = qk_fifo_stride * block_rows * sizeof(aclFloat16);              // p_tile_fifo (half)
    sz.bytes[1] = p_max_fifo_stride * block_rows * sizeof(float);               // exp_max_ififo
    sz.bytes[2] = static_cast<size_t>(s0) * num_tiles * sizeof(float);          // global_sum
    sz.bytes[3] = static_cast<size_t>(s0) * num_tiles * sizeof(float);          // exp_max
    sz.bytes[4] = static_cast<size_t>(s0) * HEAD_SIZE * sizeof(float) * num_tiles; // o_parts
    sz.bytes[5] = qk_fifo_stride * block_rows * sizeof(float);                  // qk_tile_fifo
    sz.bytes[6] = pv_fifo_stride * block_rows * sizeof(float);                  // pv_tile_fifo
    sz.bytes[7] = kFaProfileBytesPerBlock * block_rows;                         // profile
    sz.bytes[8] = block_rows * kFaCvCommSlotBytes;                             // cv_comm
    return sz;
}

struct Scratch {
    void *p_tile_fifo = nullptr;   // half
    void *exp_max_ififo = nullptr; // float
    void *global_sum = nullptr;    // float
    void *exp_max = nullptr;       // float
    void *o_parts = nullptr;       // float
    void *qk_tile_fifo = nullptr;  // float
    void *pv_tile_fifo = nullptr;  // float
    void *profile = nullptr;       // uint8
    void *cv_comm = nullptr;       // uint8
};

// Total bytes the workspace must hold for (s0, s1), assuming a 512-byte-aligned base.
size_t workspace_bytes(int s0, int s1)
{
    const Sizes sz = compute_sizes(s0, s1);
    size_t off = 0;
    for (size_t i = 0; i < Sizes::kCount; ++i) {
        off = align_up(off);
        off += sz.bytes[i];
    }
    return off;
}

// Slice the nine sub-buffers out of `base` at the same offsets workspace_bytes() accounts for.
// `base` must be 512-byte aligned (torch NPU allocations are).
void carve_workspace(void *base, int s0, int s1, Scratch &out)
{
    const Sizes sz = compute_sizes(s0, s1);
    void **dst[Sizes::kCount] = {
        &out.p_tile_fifo, &out.exp_max_ififo, &out.global_sum,  &out.exp_max, &out.o_parts,
        &out.qk_tile_fifo, &out.pv_tile_fifo, &out.profile,     &out.cv_comm,
    };
    uintptr_t b = reinterpret_cast<uintptr_t>(base);
    size_t off = 0;
    for (size_t i = 0; i < Sizes::kCount; ++i) {
        off = align_up(off);
        *dst[i] = reinterpret_cast<void *>(b + off);
        off += sz.bytes[i];
    }
}

} // namespace

extern "C" {

// Report the fixed tiling this library was built for. HEAD is the required head size; the caller
// must pick S0 as a multiple of `s0_multiple` (CUBE_S0) and S1 as a multiple of `s1_multiple`
// (TILE_S1). Any of the out-pointers may be null.
void tfa_config(int *head, int *s0_multiple, int *s1_multiple)
{
    if (head) {
        *head = HEAD_SIZE;
    }
    if (s0_multiple) {
        *s0_multiple = CUBE_S0;
    }
    if (s1_multiple) {
        *s1_multiple = TILE_S1;
    }
}

// Bytes the caller must allocate for the workspace passed to tfa_run(), for a given shape.
size_t tfa_workspace_size(int s0, int s1)
{
    return workspace_bytes(s0, s1);
}

// Launch the FA kernel on device pointers taken from torch tensors, for the given S0/S1.
// `workspace` is a single caller-allocated device block of at least tfa_workspace_size(s0, s1)
// bytes (512-byte aligned); tfa_run slices the kernel's scratch buffers out of it. The caller owns
// the workspace lifetime and must keep it alive until the launch completes.
// If stream_handle is null, a temporary stream is created/synced/destroyed internally (and the
// aclrtSynchronizeStream error code is returned). If the caller passes its own stream (e.g. torch's
// current stream), the kernel is only *enqueued* — the caller owns ordering/synchronization/timing —
// so tfa_run returns 0 without blocking. This keeps the launch async, which is required for correct
// benchmarking via torch events (do_bench).
int tfa_run(void *q, void *k, void *v, void *o, void *workspace, void *stream_handle, int s0, int s1)
{
    if (workspace == nullptr) {
        fprintf(stderr, "[tfa] workspace pointer is null\n");
        return -1;
    }
    if (s0 <= 0 || s1 <= 0 || (s0 % CUBE_S0) != 0 || (s1 % TILE_S1) != 0) {
        fprintf(stderr, "[tfa] invalid shape S0=%d S1=%d (need S0%%%d==0, S1%%%d==0)\n", s0, s1, CUBE_S0, TILE_S1);
        return -3;
    }
    Scratch scratch;
    carve_workspace(workspace, s0, s1, scratch);

    uint64_t ffts = 0;
    uint32_t ffts_len = 0;
    rtGetC2cCtrlAddr(&ffts, &ffts_len);

    aclrtStream stream = reinterpret_cast<aclrtStream>(stream_handle);
    bool own_stream = false;
    if (stream == nullptr) {
        if (aclrtCreateStream(&stream) != ACL_SUCCESS) {
            fprintf(stderr, "[tfa] aclrtCreateStream failed\n");
            return -2;
        }
        own_stream = true;
    }

    LaunchTFA<HEAD_SIZE, CUBE_S0, CUBE_S1, TILE_S1, QK_PRELOAD, kFaCvFifoSize, false, false,
              kFaCvFifoConsSyncPeriod>(
        static_cast<uint32_t>(s0), static_cast<uint32_t>(s1), reinterpret_cast<uint16_t *>(ffts),
        reinterpret_cast<aclFloat16 *>(q), reinterpret_cast<aclFloat16 *>(k), reinterpret_cast<aclFloat16 *>(v),
        reinterpret_cast<aclFloat16 *>(scratch.p_tile_fifo), reinterpret_cast<float *>(scratch.exp_max_ififo),
        reinterpret_cast<float *>(scratch.global_sum), reinterpret_cast<float *>(scratch.exp_max),
        reinterpret_cast<float *>(o), reinterpret_cast<float *>(scratch.o_parts),
        reinterpret_cast<float *>(scratch.qk_tile_fifo), reinterpret_cast<float *>(scratch.pv_tile_fifo),
        reinterpret_cast<uint8_t *>(scratch.profile), stream, reinterpret_cast<uint8_t *>(scratch.cv_comm));

    if (own_stream) {
        // We created this stream, so we must sync before destroying it; the sync
        // also surfaces any aicore execution error as the return code.
        const int rc = aclrtSynchronizeStream(stream);
        aclrtDestroyStream(stream);
        return rc;
    }
    // Caller-provided stream: enqueue only, let the caller synchronize/time.
    return 0;
}

} // extern "C"
