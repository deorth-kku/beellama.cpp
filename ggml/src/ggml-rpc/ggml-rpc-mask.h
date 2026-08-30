#pragma once

// Shared client/worker helper for the RPC FILL_CAUSAL_MASK fast path
// (see ggml-rpc.cpp). Kept in a header so tests can exercise the exact
// verification the client runs before sending boundaries over the wire.

#include "ggml.h"

#include <cstdint>
#include <vector>

// Single-pass check that `data` is a row-major 2-D mask of pure causal
// prefixes: every element is keep (0.0) or drop (-inf), and once a row turns
// to drop it stays drop. On success fills `boundaries[i]` with the index of
// the last keep element of row i (-1 if the row is all drop).
// `data` must hold n_rows * n_kv elements of `type`, row-major, with row i at
// offset i * n_kv (the caller guarantees the contiguous layout).
inline bool is_pure_causal_mask(const void * data, ggml_type type, int64_t n_kv, int64_t n_rows, std::vector<int32_t> & boundaries) {
    boundaries.assign(n_rows, -1);
    if (type == GGML_TYPE_F16) {
        const uint16_t * p = (const uint16_t *) data;
        for (int64_t i = 0; i < n_rows; i++) {
            const uint16_t * row = p + i * n_kv;
            int64_t b = -1;
            bool dropped = false;
            for (int64_t j = 0; j < n_kv; j++) {
                if (row[j] == 0x0000) { if (dropped) return false; b = j; }
                else if (row[j] == 0xFC00) { dropped = true; }
                else return false;
            }
            boundaries[i] = (int32_t) b;
        }
    } else if (type == GGML_TYPE_F32) {
        const uint32_t * p = (const uint32_t *) data;
        for (int64_t i = 0; i < n_rows; i++) {
            const uint32_t * row = p + i * n_kv;
            int64_t b = -1;
            bool dropped = false;
            for (int64_t j = 0; j < n_kv; j++) {
                if (row[j] == 0x00000000u) { if (dropped) return false; b = j; }
                else if (row[j] == 0xFF800000u) { dropped = true; }
                else return false;
            }
            boundaries[i] = (int32_t) b;
        }
    } else {
        return false;
    }
    return true;
}
