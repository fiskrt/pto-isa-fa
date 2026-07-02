/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef FA_PERFORMANCE_KERNEL_H
#define FA_PERFORMANCE_KERNEL_H

#include <acl/acl.h>
#include <cstddef>
#include <cstdint>

// Shared defaults for FA performance kernels and host driver
constexpr int kFaCvFifoSize = 8;
constexpr int kFaCvFifoConsSyncPeriod = kFaCvFifoSize / 2;
constexpr int kFaCubeS1 = 128;
constexpr int kFaTileS1 = 256;
constexpr int kFaQkPreload = 4;
constexpr std::size_t kFaProfileBytesPerBlock = 1024 * 3; // cube + two vec subblocks
constexpr std::size_t kFaCvCommSlotBytes = 512U;
constexpr int VEC_CORES = 2; // Default to 2 vector cores per cube

// Persistent-kernel core cap. The host launches min(S0/CUBE_S0, kFaMaxCores) cores and each core loops
// over the row-blocks assigned to it (LPT-balanced for the causal mask). Must stay < pto::kCvMaxCores
// (25) so the kernel keeps the direct comm_slot == get_block_idx() path. 24 = physical AI-core count.
constexpr int kFaMaxCores = 24;

// S0 (rows) and S1 (cols) are runtime arguments so a single instantiation serves arbitrary
// shapes; only the tiling below (HEAD_SIZE/CUBE_S0/CUBE_S1/TILE_S1/...) is fixed at compile time.
template <int HEAD_SIZE, int CUBE_S0, int CUBE_S1 = kFaCubeS1, int TILE_S1 = kFaTileS1,
          int QK_PRELOAD = kFaQkPreload, int CV_FIFO_SIZE = kFaCvFifoSize, bool INTERMEDIATE_CHECK = false,
          bool CAUSAL_MASK = false, int CV_FIFO_CONS_SYNC_PERIOD = kFaCvFifoConsSyncPeriod>
void LaunchTFA(uint32_t S0, uint32_t S1, uint16_t *ffts, aclFloat16 *q, aclFloat16 *k, aclFloat16 *v,
               aclFloat16 *p_tile_fifo, float *exp_max_ififo, float *global_sum_out, float *exp_max_out, float *o_out,
               float *o_parts_out, float *qk_tile_fifo, float *pv_tile_fifo, uint8_t *profile_data, aclrtStream stream,
               uint8_t *cv_comm_buf = nullptr);

// Overload without profiling buffer.
template <int HEAD_SIZE, int CUBE_S0, int CUBE_S1, int TILE_S1, int QK_PRELOAD, int CV_FIFO_SIZE,
          bool INTERMEDIATE_CHECK, bool CAUSAL_MASK, int CV_FIFO_CONS_SYNC_PERIOD>
void LaunchTFA(uint32_t S0, uint32_t S1, uint16_t *ffts, aclFloat16 *q, aclFloat16 *k, aclFloat16 *v,
               aclFloat16 *p_tile_fifo, float *exp_max_ififo, float *global_sum_out, float *exp_max_out, float *o_out,
               float *o_parts_out, float *qk_tile_fifo, float *pv_tile_fifo, aclrtStream stream,
               uint8_t *cv_comm_buf = nullptr);

#endif // FA_PERFORMANCE_KERNEL_H