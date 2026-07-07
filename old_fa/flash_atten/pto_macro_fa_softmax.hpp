/*
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_MACRO_FA_SOFTMAX_HPP
#define PTO_MACRO_FA_SOFTMAX_HPP

#include <pto/pto-inst.hpp>

namespace pto {

// -----------------------------------------------------------------------------
// FlashAttention streaming softmax (tile-level)
//
// Given one QK tile X (fp32), compute x_exp = exp(scale * (X - new_global_max)).
// This function maintains per-row running state (global_max, global_sum) so that we can
// stream over S1 tiles without materializing the full attention matrix.
//
// Performance notes:
// - Keep intermediate computations in fp32 for numerical stability.
// - The `init` specialization initializes running state for the first S1 tile.
// - The 2D->1D reshape for TCVT is used to avoid layout constraints and keep the cast fast.
// -----------------------------------------------------------------------------

constexpr PTO_INTERNAL float constexpr_sqrt(float x)
{
    if (x <= 0.0f)
        return 0.0f;
    float guess = x;
    for (int i = 0; i < 8; ++i) {
        guess = 0.5f * (guess + x / guess);
    }
    return guess;
}

constexpr AICORE inline float constexpr_inv_sqrt(float x)
{
    return 1.0f / constexpr_sqrt(x);
}

// Apply the causal mask directly to the score tile. Unlike TTRI + TMULS + TADD,
// this only writes the invalid suffix of each row. Vector addresses remain
// 256-byte aligned for fp32; a masked first repeat handles an unaligned suffix.
template <typename TileData>
AICORE inline void causal_suffix_fill(TileData &dst, int first_invalid_col_row0)
{
    using T = typename TileData::DType;
    static_assert(std::is_same_v<T, float>, "causal_suffix_fill currently supports fp32 scores only");
    static_assert((TileData::Cols % 64) == 0, "causal_suffix_fill requires a 64-element-aligned row width");
    static_assert((TileData::RowStride % 64) == 0,
                  "causal_suffix_fill requires a 256-byte-aligned fp32 row stride");

    constexpr int kElementsPerRepeat = 64;
    float negInf = -3.40282e+38f;
    __ubuf__ T *dst_ptr = reinterpret_cast<__ubuf__ T *>(dst.data());

    set_mask_norm();
    for (int row = 0; row < static_cast<int>(TileData::Rows); ++row) {
        int first_invalid = first_invalid_col_row0 + row;
        if (first_invalid < 0) {
            first_invalid = 0;
        } else if (first_invalid >= static_cast<int>(TileData::Cols)) {
            continue;
        }

        __ubuf__ T *row_ptr = dst_ptr + row * TileData::RowStride;
        const int first_repeat = first_invalid / kElementsPerRepeat;
        const int first_lane = first_invalid % kElementsPerRepeat;
        const int aligned_start = first_repeat * kElementsPerRepeat;

        if (first_lane != 0) {
            const uint64_t suffix_mask = ~((uint64_t{1} << first_lane) - 1);
            set_vector_mask(0, suffix_mask);
            vector_dup(row_ptr + aligned_start, negInf, 1, 1, 1, 8, 0);
            pipe_barrier(PIPE_V);
        }

        const int full_start = aligned_start + (first_lane == 0 ? 0 : kElementsPerRepeat);
        const int full_repeats = (static_cast<int>(TileData::Cols) - full_start) / kElementsPerRepeat;
        if (full_repeats > 0) {
            set_vector_mask(-1, -1);
            vector_dup(row_ptr + full_start, negInf, static_cast<uint8_t>(full_repeats), 1, 1, 8, 0);
            pipe_barrier(PIPE_V);
        }
    }
    set_vector_mask(-1, -1);
}

template <int HEAD_SIZE, bool CAUSAL_MASK, typename ReduceTileD1, typename TileDataD2, typename TileDataS1>
AICORE inline void softmax_opt_fa_init_impl(TileDataD2 __out__ x_exp, TileDataS1 __in__ input_x,
                                            ReduceTileD1 __out__ local_max, ReduceTileD1 __out__ local_sum,
                                            ReduceTileD1 __out__ new_global_max, ReduceTileD1 __out__ new_global_sum,
                                            ReduceTileD1 __out__ exp_max, TileDataS1 __out__ tmp_float,
                                            TileDataS1 __out__ p_tile_f32, TileDataS1 triu, int s0_index, int s1_index)
{
    (void)local_max;
    (void)exp_max;
    (void)local_sum;

    constexpr float scale = constexpr_inv_sqrt(HEAD_SIZE);
    using Tile1D_fp32 = Tile<TileType::Vec, float, 1, TileDataS1::Rows * TileDataS1::Cols, BLayout::RowMajor, 1,
                             TileDataS1::Rows * TileDataS1::Cols>;
    using Tile1D_out = Tile<TileType::Vec, typename TileDataD2::DType, 1, TileDataS1::Rows * TileDataS1::Cols,
                            BLayout::RowMajor, 1, TileDataS1::Rows * TileDataS1::Cols>;
    Tile1D_fp32 p_tile_f32_1d;
    Tile1D_out x_exp_1d;
    if constexpr (CAUSAL_MASK) {
        const int diagonal = s0_index - s1_index + 1;
        causal_suffix_fill(input_x, diagonal);
        (void)triu;
    }
    // FA2.0 init mode
    TROWMAX(new_global_max, input_x, tmp_float);
#if defined(__DAV_C220_VEC__)
    pipe_barrier(PIPE_V);
#endif
    TROWEXPANDSUB(p_tile_f32, input_x, new_global_max);
    TMULS(p_tile_f32, p_tile_f32, scale);
    TEXP(p_tile_f32, p_tile_f32);
#if defined(__DAV_C220_VEC__)
    pipe_barrier(PIPE_V);
#endif
    TROWSUM(new_global_sum, p_tile_f32, tmp_float);

    TRESHAPE(p_tile_f32_1d, p_tile_f32);
    TRESHAPE(x_exp_1d, x_exp);
    TCVT(x_exp_1d, p_tile_f32_1d, RoundMode::CAST_ROUND);
}

template <int HEAD_SIZE, bool CAUSAL_MASK, typename ReduceTileD1, typename TileDataD2, typename TileDataS1>
AICORE inline void softmax_opt_fa_not_init_impl(TileDataD2 __out__ x_exp, TileDataS1 __in__ input_x,
                                                ReduceTileD1 __out__ local_max, ReduceTileD1 __out__ local_sum,
                                                ReduceTileD1 __out__ new_global_max,
                                                ReduceTileD1 __out__ new_global_sum, ReduceTileD1 __out__ exp_max,
                                                TileDataS1 __out__ tmp_float, TileDataS1 __out__ p_tile_f32,
                                                TileDataS1 triu, int s0_index, int s1_index)
{
    constexpr float scale = constexpr_inv_sqrt(HEAD_SIZE);

    using ReduceTileD2 = Tile<TileType::Vec, float, 1, ReduceTileD1::Rows, BLayout::RowMajor, 1, ReduceTileD1::Rows>;
    using Tile1D_fp32 = Tile<TileType::Vec, float, 1, TileDataS1::Rows * TileDataS1::Cols, BLayout::RowMajor, 1,
                             TileDataS1::Rows * TileDataS1::Cols>;
    using Tile1D_out = Tile<TileType::Vec, typename TileDataD2::DType, 1, TileDataS1::Rows * TileDataS1::Cols,
                            BLayout::RowMajor, 1, TileDataS1::Rows * TileDataS1::Cols>;

    ReduceTileD2 tmp_shw_local_max;
    ReduceTileD2 tmp_shw_new_global_max;
    ReduceTileD2 tmp_shw_exp_max;
    ReduceTileD2 tmp_shw_new_global_sum;
    ReduceTileD2 tmp_shw_local_sum;
    Tile1D_fp32 p_tile_f32_1d;
    Tile1D_out x_exp_1d;

    if constexpr (CAUSAL_MASK) {
        const int diagonal = s0_index - s1_index + 1;
        causal_suffix_fill(input_x, diagonal);
        (void)triu;
    }
    // FA2.0 streaming mode (not first tile): update (global_max, global_sum) and rescale old sums.
    TROWMAX(local_max, input_x, tmp_float);
#if defined(__DAV_C220_VEC__)
    pipe_barrier(PIPE_V);
#endif
    TRESHAPE(tmp_shw_local_max, local_max);
    TRESHAPE(tmp_shw_new_global_max, new_global_max);
    TMAX(tmp_shw_local_max, tmp_shw_local_max, tmp_shw_new_global_max);
#if defined(__DAV_C220_VEC__)
    pipe_barrier(PIPE_V);
#endif
    TRESHAPE(tmp_shw_exp_max, exp_max);
    TSUB(tmp_shw_exp_max, tmp_shw_new_global_max, tmp_shw_local_max);
#if defined(__DAV_C220_VEC__)
    pipe_barrier(PIPE_V);
#endif

    TMULS(tmp_shw_new_global_max, tmp_shw_local_max, 1.0f); // just copy
#if defined(__DAV_C220_VEC__)
    pipe_barrier(PIPE_V);
#endif
    TROWEXPANDSUB(p_tile_f32, input_x, local_max);
    TMULS(tmp_shw_exp_max, tmp_shw_exp_max, scale);
    TMULS(p_tile_f32, p_tile_f32, scale);
    TEXP(tmp_shw_exp_max, tmp_shw_exp_max);
    TRESHAPE(tmp_shw_exp_max, exp_max);
    TEXP(p_tile_f32, p_tile_f32);
    TRESHAPE(tmp_shw_exp_max, exp_max);

    TRESHAPE(p_tile_f32_1d, p_tile_f32);
    TRESHAPE(x_exp_1d, x_exp);
    TCVT(x_exp_1d, p_tile_f32_1d, RoundMode::CAST_ROUND);
#if defined(__DAV_C220_VEC__)
    pipe_barrier(PIPE_V);
#endif
    TRESHAPE(tmp_shw_new_global_sum, new_global_sum);
    TMUL(tmp_shw_new_global_sum, tmp_shw_exp_max, tmp_shw_new_global_sum);
    TROWSUM(local_sum, p_tile_f32, tmp_float);
    TRESHAPE(tmp_shw_local_sum, local_sum);
#if defined(__DAV_C220_VEC__)
    pipe_barrier(PIPE_V);
#endif
    TADD(tmp_shw_new_global_sum, tmp_shw_new_global_sum, tmp_shw_local_sum);
}

template <bool init = false, int HEAD_SIZE, bool CAUSAL_MASK, typename ReduceTileD1, typename TileDataD2,
          typename TileDataS1>
AICORE inline void pto_macro_fa_softmax(TileDataD2 __out__ x_exp, TileDataS1 __in__ input_x,
                                        ReduceTileD1 __out__ local_max, ReduceTileD1 __out__ local_sum,
                                        ReduceTileD1 __in__ new_global_max, ReduceTileD1 __out__ new_global_sum,
                                        ReduceTileD1 __out__ exp_max, TileDataS1 __out__ input_reduce_tmp,
                                        TileDataS1 __out__ p_tile_fp32, TileDataS1 triu, int s0_index, int s1_index)
{
    if (s1_index <= s0_index || !CAUSAL_MASK) {
        if constexpr (init) {
            softmax_opt_fa_init_impl<HEAD_SIZE, CAUSAL_MASK, ReduceTileD1, TileDataD2, TileDataS1>(
                x_exp, input_x, local_max, local_sum, new_global_max, new_global_sum, exp_max, input_reduce_tmp,
                p_tile_fp32, triu, s0_index, s1_index);
        } else {
            softmax_opt_fa_not_init_impl<HEAD_SIZE, CAUSAL_MASK, ReduceTileD1, TileDataD2, TileDataS1>(
                x_exp, input_x, local_max, local_sum, new_global_max, new_global_sum, exp_max, input_reduce_tmp,
                p_tile_fp32, triu, s0_index, s1_index);
        }
    } else if constexpr (CAUSAL_MASK) {
        TEXPANDS(x_exp, static_cast<typename TileDataD2::DType>(0));
        TEXPANDS(exp_max, 1.0f);
#if defined(__DAV_C220_VEC__)
        pipe_barrier(PIPE_V);
#endif
    }
}

} // namespace pto

#endif
