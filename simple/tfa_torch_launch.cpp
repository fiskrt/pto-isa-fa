/*
 * C-ABI launcher so a torch_npu tensor can drive the FA kernel directly (no .bin golden files).
 *
 * Q/K/V come in as device pointers from torch tensors; O is written into a torch fp32 tensor.
 * All the intermediate FIFO / scratch buffers the kernel needs are allocated here (lazily, once)
 * and reused across calls. The shape is fixed at compile time and MUST match a case that was
 * instantiated into libfa_performance_kernel.so (see generated_cases.h / run.sh cases).
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

// ---- Fixed PoC shape. Must be an instantiated case in libfa_performance_kernel.so. ----
constexpr int S0 = 128;
constexpr int HEAD_SIZE = 128;
constexpr int S1 = 1024;
constexpr int CUBE_S0 = 128;
constexpr int CUBE_S1 = kFaCubeS1;       // 128
constexpr int TILE_S1 = kFaTileS1;       // 256
constexpr int QK_PRELOAD = kFaQkPreload; // 4

constexpr int tile_factor = TILE_S1 / CUBE_S1;
constexpr int block_rows = S0 / CUBE_S0;
constexpr int num_tiles = S1 / TILE_S1;

// Scratch buffer sizes (mirror run_tfa() in main.cpp for this shape).
constexpr size_t qk_fifo_stride = static_cast<size_t>(kFaCvFifoSize) * CUBE_S0 * tile_factor * CUBE_S1;
constexpr size_t p_max_fifo_stride = static_cast<size_t>(kFaCvFifoSize) * CUBE_S0;
constexpr size_t pv_fifo_stride = static_cast<size_t>(kFaCvFifoSize) * CUBE_S0 * HEAD_SIZE;

constexpr size_t qk_fifo_bytes = qk_fifo_stride * block_rows * sizeof(float);
constexpr size_t p_fifo_bytes_half = qk_fifo_stride * block_rows * sizeof(aclFloat16);
constexpr size_t exp_max_ififo_bytes = p_max_fifo_stride * block_rows * sizeof(float);
constexpr size_t pv_fifo_bytes = pv_fifo_stride * block_rows * sizeof(float);
constexpr size_t gsum_bytes = static_cast<size_t>(S0) * num_tiles * sizeof(float);
constexpr size_t o_parts_bytes = static_cast<size_t>(S0) * HEAD_SIZE * sizeof(float) * num_tiles;
constexpr size_t profile_bytes = kFaProfileBytesPerBlock * block_rows;
constexpr size_t cv_comm_bytes = static_cast<size_t>(S0 / CUBE_S0) * kFaCvCommSlotBytes;

struct Scratch {
    bool init = false;
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
Scratch g_scratch;

bool malloc_dev(void **p, size_t bytes)
{
    return aclrtMalloc(p, bytes, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS;
}

bool alloc_scratch_once()
{
    if (g_scratch.init) {
        return true;
    }
    if (!malloc_dev(&g_scratch.p_tile_fifo, p_fifo_bytes_half) ||
        !malloc_dev(&g_scratch.exp_max_ififo, exp_max_ififo_bytes) ||
        !malloc_dev(&g_scratch.global_sum, gsum_bytes) || !malloc_dev(&g_scratch.exp_max, gsum_bytes) ||
        !malloc_dev(&g_scratch.o_parts, o_parts_bytes) || !malloc_dev(&g_scratch.qk_tile_fifo, qk_fifo_bytes) ||
        !malloc_dev(&g_scratch.pv_tile_fifo, pv_fifo_bytes) || !malloc_dev(&g_scratch.profile, profile_bytes) ||
        !malloc_dev(&g_scratch.cv_comm, cv_comm_bytes)) {
        return false;
    }
    g_scratch.init = true;
    return true;
}

} // namespace

extern "C" {

// Report the fixed shape this library was built for.
void tfa_shape(int *s0, int *head, int *s1)
{
    if (s0) {
        *s0 = S0;
    }
    if (head) {
        *head = HEAD_SIZE;
    }
    if (s1) {
        *s1 = S1;
    }
}

// Launch the FA kernel on device pointers taken from torch tensors.
// If stream_handle is null, a temporary stream is created/synced/destroyed internally
// (and the aclrtSynchronizeStream error code is returned). If the caller passes its own
// stream (e.g. torch's current stream), the kernel is only *enqueued* — the caller owns
// ordering/synchronization/timing — so tfa_run returns 0 without blocking. This keeps the
// launch async, which is required for correct benchmarking via torch events (do_bench).
int tfa_run(void *q, void *k, void *v, void *o, void *stream_handle)
{
    if (!alloc_scratch_once()) {
        fprintf(stderr, "[tfa] scratch allocation failed\n");
        return -1;
    }

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

    LaunchTFA<S0, HEAD_SIZE, S1, CUBE_S0, CUBE_S1, TILE_S1, QK_PRELOAD, kFaCvFifoSize, false, false,
              kFaCvFifoConsSyncPeriod>(
        reinterpret_cast<uint16_t *>(ffts), reinterpret_cast<aclFloat16 *>(q), reinterpret_cast<aclFloat16 *>(k),
        reinterpret_cast<aclFloat16 *>(v), reinterpret_cast<aclFloat16 *>(g_scratch.p_tile_fifo),
        reinterpret_cast<float *>(g_scratch.exp_max_ififo), reinterpret_cast<float *>(g_scratch.global_sum),
        reinterpret_cast<float *>(g_scratch.exp_max), reinterpret_cast<float *>(o),
        reinterpret_cast<float *>(g_scratch.o_parts), reinterpret_cast<float *>(g_scratch.qk_tile_fifo),
        reinterpret_cast<float *>(g_scratch.pv_tile_fifo), reinterpret_cast<uint8_t *>(g_scratch.profile), stream,
        reinterpret_cast<uint8_t *>(g_scratch.cv_comm));

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
