#pragma once

// Shared client/worker helper for the RPC mask fast paths (see ggml-rpc.cpp).
// Kept in a header so tests can exercise the exact verification the client
// runs before sending compressed mask data over the wire.

#include "ggml.h"

#include <cstdint>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

// Result of scanning a 2-D mask.
enum class rpc_mask_kind {
    NOT_BINARY,   // some element is neither 0.0 nor -inf -> use the data path
    BINARY,       // every element is 0.0 or -inf, but some row is not a keep-prefix
    PURE_CAUSAL,  // every row is a keep-prefix then drop (a subset of BINARY)
};

// Per-row state, updated while the data is scanned so the boundary and the
// pure-causal flag come out of the same pass that packs `bits`.
struct rpc_mask_row_state {
    int64_t last_keep = -1;  // index of the last keep element seen in the row
    bool    seen_drop = false;
    bool    pure = true;     // false once a keep appears after a drop
};

// Fold one chunk of a row into the row state in O(1). `col` is the column of
// the first element of the chunk; bit k of `drop_mask` set = element (col + k)
// is drop; `n` is the number of elements in the chunk (n <= 32).
inline void rpc_mask_note_chunk(rpc_mask_row_state & st, int64_t col, uint32_t drop_mask, int n) {
    const uint32_t mask = (n >= 32) ? 0xFFFFFFFFu : ((1u << n) - 1u);
    const uint32_t keep_mask = (~drop_mask) & mask;
    if (keep_mask == mask) {
        // whole chunk is keep
        st.last_keep = col + n - 1;
        return;
    }
    if (keep_mask != 0) {
        // index of the highest set bit, without compiler builtins
        int hi = 0;
        uint32_t x = keep_mask;
        if (x >> 16) { hi = 16; x >>= 16; }
        if (x >> 8)  { hi += 8;  x >>= 8; }
        if (x >> 4)  { hi += 4;  x >>= 4; }
        if (x >> 2)  { hi += 2;  x >>= 2; }
        if (x >> 1)  { hi += 1; }
        st.last_keep = col + hi;
        // a keep after a drop: earlier chunks already had a drop, or this
        // chunk's keeps are not a low-bit prefix (a 0 followed by a 1)
        if (st.seen_drop || (keep_mask & (keep_mask + 1u)) != 0) {
            st.pure = false;
        }
    }
    if ((drop_mask & mask) != 0) {
        st.seen_drop = true;
    }
}

// Pack the drop bits of a binary mask (1 bit/element, 1 = drop). The data is
// known to be binary, so only the drop values are compared.
inline void rpc_mask_pack_bits(const void * data, ggml_type type, int64_t n_elements, std::vector<uint8_t> & bits) {
    bits.assign((n_elements + 7) / 8, 0);
    uint8_t * out = bits.data();
    if (type == GGML_TYPE_F16) {
        const uint16_t * p = (const uint16_t *) data;
        int64_t i = 0;
#if defined(__AVX2__)
        for (; i + 16 <= n_elements; i += 16) {
            const __m256i v = _mm256_loadu_si256((const __m256i *) (p + i));
            const __m256i eq_drop = _mm256_cmpeq_epi16(v, _mm256_set1_epi16((int16_t) 0xFC00));
            // bit 2k of m is set iff lane k is drop -> pack to bit k
            const uint32_t m = _mm256_movemask_epi8(eq_drop);
            out[i / 8]     = (uint8_t) (((m >> 0) & 1u) << 0 | ((m >> 2) & 1u) << 1 | ((m >> 4) & 1u) << 2 | ((m >> 6) & 1u) << 3 |
                                       ((m >> 8) & 1u) << 4 | ((m >> 10) & 1u) << 5 | ((m >> 12) & 1u) << 6 | ((m >> 14) & 1u) << 7);
            out[i / 8 + 1] = (uint8_t) (((m >> 16) & 1u) << 0 | ((m >> 18) & 1u) << 1 | ((m >> 20) & 1u) << 2 | ((m >> 22) & 1u) << 3 |
                                       ((m >> 24) & 1u) << 4 | ((m >> 26) & 1u) << 5 | ((m >> 28) & 1u) << 6 | ((m >> 30) & 1u) << 7);
        }
#endif
        for (; i < n_elements; i++) {
            if (p[i] == 0xFC00) {
                out[i / 8] |= (uint8_t) (1u << (i & 7));
            }
        }
    } else if (type == GGML_TYPE_F32) {
        const uint32_t * p = (const uint32_t *) data;
        int64_t i = 0;
#if defined(__AVX2__)
        for (; i + 8 <= n_elements; i += 8) {
            const __m256i v = _mm256_loadu_si256((const __m256i *) (p + i));
            const __m256i eq_drop = _mm256_cmpeq_epi32(v, _mm256_set1_epi32(0xFF800000));
            // bit 4k of m is set iff lane k is drop -> pack to bit k
            const uint32_t m = _mm256_movemask_epi8(eq_drop);
            out[i / 8] = (uint8_t) (((m >> 0) & 1u) << 0 | ((m >> 4) & 1u) << 1 | ((m >> 8) & 1u) << 2 | ((m >> 12) & 1u) << 3 |
                                   ((m >> 16) & 1u) << 4 | ((m >> 20) & 1u) << 5 | ((m >> 24) & 1u) << 6 | ((m >> 28) & 1u) << 7);
        }
#endif
        for (; i < n_elements; i++) {
            if (p[i] == 0xFF800000u) {
                out[i / 8] |= (uint8_t) (1u << (i & 7));
            }
        }
    }
}

// Scan a row-major contiguous mask (n_kv * n_rows elements). The boundaries
// and the kind come out of a single pass over the data (the per-chunk drop
// masks are folded into the row state, so there is no pass over the packed
// bits); `bits` (1 bit/element, 1 = drop) is packed afterwards, only when
// the mask is binary.
// If every element is keep (0.0) or drop (-inf), fills `bits` and
// `boundaries[i]` (index of the last keep element of row i, -1 if the row is
// all drop), and reports whether every row is a keep-prefix.
// Returns the kind; on NOT_BINARY, `bits`/`boundaries` are undefined.
inline rpc_mask_kind scan_mask(const void * data, ggml_type type, int64_t n_kv, int64_t n_rows,
                               std::vector<uint8_t> & bits, std::vector<int32_t> & boundaries) {
    const int64_t n_elements = n_kv * n_rows;
    boundaries.assign(n_rows, -1);
    if (n_elements == 0) {
        return rpc_mask_kind::PURE_CAUSAL;
    }
    bool pure_causal = true;

    // The scan only folds the per-chunk drop masks into the row state; the
    // bits are packed in one flat pass afterwards (only when binary).
    if (type == GGML_TYPE_F16) {
        const uint16_t * p = (const uint16_t *) data;
        for (int64_t row = 0; row < n_rows; row++) {
            rpc_mask_row_state st;
            const int64_t row_base = row * n_kv;
            int64_t j = 0;
#if defined(__AVX2__)
            for (; j + 16 <= n_kv; j += 16) {
                const __m256i v = _mm256_loadu_si256((const __m256i *) (p + row_base + j));
                const __m256i eq_drop = _mm256_cmpeq_epi16(v, _mm256_set1_epi16((int16_t) 0xFC00));
                const __m256i eq_keep = _mm256_cmpeq_epi16(v, _mm256_setzero_si256());
                const __m256i bad = _mm256_andnot_si256(_mm256_or_si256(eq_drop, eq_keep), _mm256_set1_epi16((int16_t) 0xFFFF));
                if (_mm256_movemask_epi8(bad) != 0) {
                    return rpc_mask_kind::NOT_BINARY;
                }
                // bit 2k of m is set iff lane k is drop -> pack to bit k
                const uint32_t m = _mm256_movemask_epi8(eq_drop);
                const uint32_t drop_mask =
                    ((m >> 0) & 1u) << 0 | ((m >> 2) & 1u) << 1 | ((m >> 4) & 1u) << 2 | ((m >> 6) & 1u) << 3 |
                    ((m >> 8) & 1u) << 4 | ((m >> 10) & 1u) << 5 | ((m >> 12) & 1u) << 6 | ((m >> 14) & 1u) << 7 |
                    ((m >> 16) & 1u) << 8 | ((m >> 18) & 1u) << 9 | ((m >> 20) & 1u) << 10 | ((m >> 22) & 1u) << 11 |
                    ((m >> 24) & 1u) << 12 | ((m >> 26) & 1u) << 13 | ((m >> 28) & 1u) << 14 | ((m >> 30) & 1u) << 15;
                rpc_mask_note_chunk(st, j, drop_mask, 16);
            }
#endif
            for (; j < n_kv; j += 32) {
                uint32_t drop_mask = 0;
                const int64_t n = (n_kv - j) < 32 ? (n_kv - j) : 32;
                for (int64_t k = 0; k < n; k++) {
                    const uint16_t v = p[row_base + j + k];
                    if (v == 0xFC00) {
                        drop_mask |= 1u << k;
                    } else if (v != 0x0000) {
                        return rpc_mask_kind::NOT_BINARY;
                    }
                }
                rpc_mask_note_chunk(st, j, drop_mask, (int) n);
            }
            boundaries[row] = (int32_t) st.last_keep;
            pure_causal = pure_causal && st.pure;
        }
    } else if (type == GGML_TYPE_F32) {
        const uint32_t * p = (const uint32_t *) data;
        for (int64_t row = 0; row < n_rows; row++) {
            rpc_mask_row_state st;
            const int64_t row_base = row * n_kv;
            int64_t j = 0;
#if defined(__AVX2__)
            for (; j + 8 <= n_kv; j += 8) {
                const __m256i v = _mm256_loadu_si256((const __m256i *) (p + row_base + j));
                const __m256i eq_drop = _mm256_cmpeq_epi32(v, _mm256_set1_epi32(0xFF800000));
                const __m256i eq_keep = _mm256_cmpeq_epi32(v, _mm256_setzero_si256());
                const __m256i bad = _mm256_andnot_si256(_mm256_or_si256(eq_drop, eq_keep), _mm256_set1_epi32(0xFFFFFFFF));
                if (_mm256_movemask_epi8(bad) != 0) {
                    return rpc_mask_kind::NOT_BINARY;
                }
                // bit 4k of m is set iff lane k is drop -> pack to bit k
                const uint32_t m = _mm256_movemask_epi8(eq_drop);
                const uint32_t drop_mask =
                    ((m >> 0) & 1u) << 0 | ((m >> 4) & 1u) << 1 | ((m >> 8) & 1u) << 2 | ((m >> 12) & 1u) << 3 |
                    ((m >> 16) & 1u) << 4 | ((m >> 20) & 1u) << 5 | ((m >> 24) & 1u) << 6 | ((m >> 28) & 1u) << 7;
                rpc_mask_note_chunk(st, j, drop_mask, 8);
            }
#endif
            for (; j < n_kv; j += 32) {
                uint32_t drop_mask = 0;
                const int64_t n = (n_kv - j) < 32 ? (n_kv - j) : 32;
                for (int64_t k = 0; k < n; k++) {
                    const uint32_t v = p[row_base + j + k];
                    if (v == 0xFF800000u) {
                        drop_mask |= 1u << k;
                    } else if (v != 0x00000000u) {
                        return rpc_mask_kind::NOT_BINARY;
                    }
                }
                rpc_mask_note_chunk(st, j, drop_mask, (int) n);
            }
            boundaries[row] = (int32_t) st.last_keep;
            pure_causal = pure_causal && st.pure;
        }
    } else {
        return rpc_mask_kind::NOT_BINARY;
    }
    const rpc_mask_kind kind = pure_causal ? rpc_mask_kind::PURE_CAUSAL : rpc_mask_kind::BINARY;
    // bits are only consumed by the BINARY wire path (and by oversized
    // PURE_CAUSAL masks that fall back to it), so pack them only now
    rpc_mask_pack_bits(data, type, n_elements, bits);
    return kind;
}

// Check that every element in `data` is either keep (0.0) or drop (-inf),
// and pack the result into `bits` (1 bit per element: 0 = keep, 1 = drop).
// `data` must hold `n_elements` elements of `type`, contiguous.
// On success, `bits` has ceil(n_elements / 8) bytes.
inline bool is_binary_mask(const void * data, ggml_type type, int64_t n_elements, std::vector<uint8_t> & bits) {
    std::vector<int32_t> boundaries;
    // Binary-ness does not depend on the row layout, so treat the flat array
    // as a single row (n_kv = n_elements, n_rows = 1).
    const rpc_mask_kind kind = scan_mask(data, type, n_elements, 1, bits, boundaries);
    return kind != rpc_mask_kind::NOT_BINARY;
}

// Check that `data` is a row-major 2-D mask of pure causal prefixes: every
// element is keep (0.0) or drop (-inf), and once a row turns to drop it stays
// drop. On success fills `boundaries[i]` with the index of the last keep
// element of row i (-1 if the row is all drop).
// `data` must hold n_rows * n_kv elements of `type`, row-major, with row i at
// offset i * n_kv (the caller guarantees the contiguous layout).
inline bool is_pure_causal_mask(const void * data, ggml_type type, int64_t n_kv, int64_t n_rows, std::vector<int32_t> & boundaries) {
    std::vector<uint8_t> bits;
    const rpc_mask_kind kind = scan_mask(data, type, n_kv, n_rows, bits, boundaries);
    return kind == rpc_mask_kind::PURE_CAUSAL;
}
// Worker-side refill helpers (inverse of scan_mask), kept here so the tests
// can exercise the exact expansion the worker runs.

// Expand packed bits (1 bit/element, 1 = drop) to f16 elements:
// 0x0000 = keep (+0.0), 0xFC00 = drop (-inf).
inline void expand_mask_bits_f16(const uint8_t * bits, uint16_t * out, int64_t n_elements) {
    int64_t i = 0;
#if defined(__AVX512BW__)
    // lane k of `shifts` holds k, so shifting the byte right by lane k
    // leaves bit k of the byte in lane k
    const __m256i shifts = _mm256_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    const __m256i one = _mm256_set1_epi16(1);
    const __m256i drop = _mm256_set1_epi16((int16_t) 0xFC00);
    for (; i + 16 <= n_elements; i += 16) {
        // low byte covers elements i..i+7, high byte covers i+8..i+15
        const uint16_t b = (uint16_t) (bits[i / 8] | (uint16_t (bits[i / 8 + 1]) << 8));
        const __m256i v = _mm256_set1_epi16(b);
        __m256i x = _mm256_srlv_epi16(v, shifts);  // lane k = b >> k
        x = _mm256_and_si256(x, one);              // lane k = bit k of b
        x = _mm256_mullo_epi16(x, drop);           // 0 -> 0x0000, 1 -> 0xFC00
        _mm256_storeu_si256((__m256i *) (out + i), x);
    }
#elif defined(__AVX2__)
    const __m256i drop = _mm256_set1_epi16((int16_t) 0xFC00);
    for (; i + 16 <= n_elements; i += 16) {
        // low byte covers elements i..i+7, high byte covers i+8..i+15
        const uint16_t b = (uint16_t) (bits[i / 8] | (uint16_t (bits[i / 8 + 1]) << 8));
        // AVX2 has no _mm256_srlv_epi16 (AVX-512BW), so unpack the bits
        // per lane; lane k = bit k of b
        alignas(32) uint16_t tmp[16];
        for (int k = 0; k < 16; ++k) {
            tmp[k] = (b >> k) & 1u;
        }
        __m256i x = _mm256_load_si256((const __m256i *) tmp);
        x = _mm256_mullo_epi16(x, drop);  // 0 -> 0x0000, 1 -> 0xFC00
        _mm256_storeu_si256((__m256i *) (out + i), x);
    }
#endif
    for (; i < n_elements; i++) {
        out[i] = (bits[i / 8] & (1u << (i & 7))) ? uint16_t(0xFC00) : uint16_t(0x0000);
    }
}

// Expand packed bits (1 bit/element, 1 = drop) to f32 elements:
// 0x00000000 = keep (+0.0), 0xFF800000 = drop (-inf).
inline void expand_mask_bits_f32(const uint8_t * bits, uint32_t * out, int64_t n_elements) {
    int64_t i = 0;
#if defined(__AVX512BW__)
    // _mm256_srlv_epi32 is AVX-512BW, not AVX2
    const __m256i shifts = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    const __m256i one = _mm256_set1_epi32(1);
    const __m256i drop = _mm256_set1_epi32(0xFF800000);
    for (; i + 8 <= n_elements; i += 8) {
        const __m256i v = _mm256_set1_epi32((int32_t) bits[i / 8]);
        __m256i x = _mm256_srlv_epi32(v, shifts);  // lane k = b >> k
        x = _mm256_and_si256(x, one);              // lane k = bit k of b
        x = _mm256_mullo_epi32(x, drop);           // 0 -> 0x00000000, 1 -> 0xFF800000
        _mm256_storeu_si256((__m256i *) (out + i), x);
    }
#endif
    for (; i < n_elements; i++) {
        out[i] = (bits[i / 8] & (1u << (i & 7))) ? uint32_t(0xFF800000u) : uint32_t(0x00000000u);
    }
}

// Refill a pure causal mask from per-row boundaries: row i is keep for
// j <= boundaries[i] and drop otherwise (-1 -> whole row drop).
inline void fill_causal_mask_f16(uint16_t * out, int64_t n_kv, int64_t n_rows, const int32_t * boundaries) {
#if defined(__AVX2__)
    const __m256i keep = _mm256_setzero_si256();
    const __m256i drop = _mm256_set1_epi16((int16_t) 0xFC00);
    for (int64_t i = 0; i < n_rows; i++) {
        uint16_t * row = out + i * n_kv;
        const int64_t keep_n = boundaries[i] + 1;  // -1 -> 0
        int64_t j = 0;
        for (; j + 16 <= keep_n; j += 16) {
            _mm256_storeu_si256((__m256i *) (row + j), keep);
        }
        for (; j < keep_n; j++) {
            row[j] = uint16_t(0x0000);
        }
        for (; j + 16 <= n_kv; j += 16) {
            _mm256_storeu_si256((__m256i *) (row + j), drop);
        }
        for (; j < n_kv; j++) {
            row[j] = uint16_t(0xFC00);
        }
    }
    return;
#else
    for (int64_t i = 0; i < n_rows; i++) {
        uint16_t * row = out + i * n_kv;
        const int64_t keep_n = boundaries[i] + 1;  // -1 -> 0
        for (int64_t j = 0; j < keep_n; j++) {
            row[j] = uint16_t(0x0000);
        }
        for (int64_t j = keep_n; j < n_kv; j++) {
            row[j] = uint16_t(0xFC00);
        }
    }
#endif
}

// Refill a pure causal mask from per-row boundaries (f32).
inline void fill_causal_mask_f32(uint32_t * out, int64_t n_kv, int64_t n_rows, const int32_t * boundaries) {
#if defined(__AVX2__)
    const __m256i keep = _mm256_setzero_si256();
    const __m256i drop = _mm256_set1_epi32(0xFF800000);
    for (int64_t i = 0; i < n_rows; i++) {
        uint32_t * row = out + i * n_kv;
        const int64_t keep_n = boundaries[i] + 1;  // -1 -> 0
        int64_t j = 0;
        for (; j + 8 <= keep_n; j += 8) {
            _mm256_storeu_si256((__m256i *) (row + j), keep);
        }
        for (; j < keep_n; j++) {
            row[j] = uint32_t(0x00000000u);
        }
        for (; j + 8 <= n_kv; j += 8) {
            _mm256_storeu_si256((__m256i *) (row + j), drop);
        }
        for (; j < n_kv; j++) {
            row[j] = uint32_t(0xFF800000u);
        }
    }
    return;
#else
    for (int64_t i = 0; i < n_rows; i++) {
        uint32_t * row = out + i * n_kv;
        const int64_t keep_n = boundaries[i] + 1;  // -1 -> 0
        for (int64_t j = 0; j < keep_n; j++) {
            row[j] = uint32_t(0x00000000u);
        }
        for (int64_t j = keep_n; j < n_kv; j++) {
            row[j] = uint32_t(0xFF800000u);
        }
    }
#endif
}