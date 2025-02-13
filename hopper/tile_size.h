/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include <tuple>

#include "flash.h"

// Return {kBlockM, kBlockN, MmaPV_is_RS, IntraWGOverlap}
constexpr std::tuple<int, int, bool, bool> tile_size_fwd_sm90(
        int headdim, int headdim_v, bool is_causal, bool is_local, int element_size=2,
        bool v_colmajor=false, bool paged_kv=false, bool softcap=false, 
        ScalingRecipe scaling_recipe=ScalingRecipe::PerKVHead) {
    // int kOffset = (scaling_recipe == ScalingRecipe::PerQKVToken ? (headdim % 64 == 0 ? 32 : 64) : 0);
    // int kOffset = (headdim % 64 == 0 ? 32 : 64);
    int kOffset = (headdim % 64 == 0 ? 64: 64);
    if (element_size == 2) {
        if (headdim <= 64) {
            bool same_hdim = (headdim == headdim_v);  // if not same hdim, we're targeting hdimv=512
            return {same_hdim ? 192 : 64, (same_hdim ? 128 : 64) - kOffset, same_hdim, true};
            // Good for long seqlen (>= 4k) but suffers from tile quantization at short seqlen
            // return {192, is_causal || is_local ? 192 : 176, true, false};
        } else if (headdim <= 96) {
            return {192, (is_local || paged_kv ? 128 : 144) - kOffset, false, true};
        } else if (headdim <= 128) {
            return {128, (is_causal || is_local || paged_kv ? 128 : 176) - kOffset, true, true};
            // {128, 192, false, false} and {192, 128, false, true} are quite good too
            // 128 x 192 hits the limit of smem if MmaPV_is_RS, 128 x 144 hits the limit if !MmaPV_is_RS
        } else if (headdim <= 192) {
            return {128, (paged_kv || is_local ? 96 : (headdim_v <= 128 ? 128 : 112)) - kOffset, true, true};  // 128 x 112 hits the limit of smem
        } else {
            return {128, (is_local ? 64 : 80) - kOffset, true, true};  // 128 x 80 hits the limit of smem
        }
    } else {
        if (headdim <= 64) {
            return {192, 160 - kOffset, true, true};
        } else if (headdim <= 96) {
            return {192, 128 - kOffset, true, true};
        } else if (headdim <= 128) {
            return {128, (paged_kv ? 160 : (v_colmajor || (softcap && is_local) ? 192 : 224)) - kOffset, true, true};
        } else if (headdim <= 192) {
            return {128, ((paged_kv || softcap) && is_local ? 128 : 160) - kOffset, true, true};
        } else {
            return {128, (is_local ? 64 : 128) - kOffset, true, !paged_kv};  // PagedKV uses more registers so we disabled IntraWGOverlap
        }
    }
}

// Return {kBlockM, kBlockN, kNWarps, kStages, Q_in_regs}
constexpr std::tuple<int, int, int, int, bool> tile_size_fwd_sm8x(
        bool sm86_or_89, int headdim, int headdim_v, bool is_causal, bool is_local, int element_size=2,
        bool paged_kv=false, bool varlen_and_split=false,
        bool softcap=false, bool append_kv=false) {
    if (element_size == 2) {
        if (headdim <= 64) {
            return {128, varlen_and_split ? 80 : (is_local ? 96 : 112), 4, 1, false};
        } else if (headdim <= 96) {
            return {128, varlen_and_split || is_local ? 48 : 64, 4, 1, false};
        } else if (headdim <= 128) {
            bool const use_8_warps = sm86_or_89 | varlen_and_split;
            return {128, use_8_warps ? (varlen_and_split ? (is_local ? 96 : 112) : (is_local ? 96 : 128)) : (is_local ? 48 : 64), use_8_warps ? 8 : 4, 1, use_8_warps};
        } else if (headdim <= 192) {
            bool const kBlockN_64 = append_kv || is_local || varlen_and_split || paged_kv;
            return {128, kBlockN_64 ? 64 : 96, 8, sm86_or_89 ? 1 : 2, !kBlockN_64};
        } else {
            return {128, sm86_or_89 ? (append_kv ? 32 : (varlen_and_split || is_local ? 48 : 64)) : (append_kv ? 48 : (varlen_and_split || is_local ? 64 : 96)), 8, 1, sm86_or_89 && !append_kv};
        }
    } else {
        // Placeholder for now
        return {128, 64, 8, 2, false};
    }
}
