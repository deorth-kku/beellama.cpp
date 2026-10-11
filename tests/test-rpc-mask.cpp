// Unit tests for the FILL_CAUSAL_MASK fast-path verification
// (ggml/src/ggml-rpc/ggml-rpc-mask.h). The client must only send boundaries
// over the wire when the mask is exactly a per-row +0.0/-inf causal prefix;
// anything else (SWA holes, alibi values, NaN, ...) must be rejected so the
// normal data path is used instead.

#include "ggml-rpc-mask.h"

#include <cstdint>
#include <cstdio>
#include <vector>

#undef NDEBUG
#include <cassert>

namespace {

const int64_t N_KV = 8;

// row i keeps j <= b[i] (b[i] == -1 -> whole row drop)
std::vector<uint16_t> f16_rows(const std::vector<int64_t> & b) {
    std::vector<uint16_t> data(b.size() * N_KV);
    for (size_t i = 0; i < b.size(); i++) {
        for (int64_t j = 0; j < N_KV; j++) {
            data[i * N_KV + j] = (j <= b[i]) ? uint16_t(0x0000) : uint16_t(0xFC00);
        }
    }
    return data;
}

std::vector<uint32_t> f32_rows(const std::vector<int64_t> & b) {
    std::vector<uint32_t> data(b.size() * N_KV);
    for (size_t i = 0; i < b.size(); i++) {
        for (int64_t j = 0; j < N_KV; j++) {
            data[i * N_KV + j] = (j <= b[i]) ? uint32_t(0x00000000u) : uint32_t(0xFF800000u);
        }
    }
    return data;
}

} // namespace

int main() {
    // ---- F16: keep = 0x0000 (+0.0), drop = 0xFC00 (-inf) ----
    {
        // mixed rows: all-keep, partial prefix, all-drop
        std::vector<uint16_t> data = f16_rows({ N_KV - 1, 2, -1 });
        std::vector<int32_t> boundaries;
        assert(is_pure_causal_mask(data.data(), GGML_TYPE_F16, N_KV, 3, boundaries));
        assert(boundaries.size() == 3);
        assert(boundaries[0] == 7);
        assert(boundaries[1] == 2);
        assert(boundaries[2] == -1);
    }
    {
        // SWA hole / empty cell: a drop followed by a keep must be rejected
        std::vector<uint16_t> data = f16_rows({ 5, 5 });
        data[0 * N_KV + 3] = uint16_t(0xFC00); // hole inside row 0
        std::vector<int32_t> boundaries;
        assert(!is_pure_causal_mask(data.data(), GGML_TYPE_F16, N_KV, 2, boundaries));
    }
    {
        // alibi-style non-zero keep value (0x3C00 = +1.0 in F16) must be rejected
        std::vector<uint16_t> data = f16_rows({ 5, 5 });
        data[0 * N_KV + 1] = uint16_t(0x3C00);
        std::vector<int32_t> boundaries;
        assert(!is_pure_causal_mask(data.data(), GGML_TYPE_F16, N_KV, 2, boundaries));
    }
    {
        // NaN (0x7E00) must be rejected
        std::vector<uint16_t> data = f16_rows({ 5, 5 });
        data[0 * N_KV + 1] = uint16_t(0x7E00);
        std::vector<int32_t> boundaries;
        assert(!is_pure_causal_mask(data.data(), GGML_TYPE_F16, N_KV, 2, boundaries));
    }

    // ---- F32: keep = 0x00000000 (+0.0), drop = 0xFF800000 (-inf) ----
    {
        std::vector<uint32_t> data = f32_rows({ N_KV - 1, 0, -1 });
        std::vector<int32_t> boundaries;
        assert(is_pure_causal_mask(data.data(), GGML_TYPE_F32, N_KV, 3, boundaries));
        assert(boundaries.size() == 3);
        assert(boundaries[0] == 7);
        assert(boundaries[1] == 0);
        assert(boundaries[2] == -1);
    }
    {
        // SWA hole
        std::vector<uint32_t> data = f32_rows({ 5, 5 });
        data[1 * N_KV + 2] = uint32_t(0xFF800000u);
        std::vector<int32_t> boundaries;
        assert(!is_pure_causal_mask(data.data(), GGML_TYPE_F32, N_KV, 2, boundaries));
    }
    {
        // -1.0 (0xBE800000) keep value must be rejected
        std::vector<uint32_t> data = f32_rows({ 5, 5 });
        data[0 * N_KV + 1] = uint32_t(0xBE800000u);
        std::vector<int32_t> boundaries;
        assert(!is_pure_causal_mask(data.data(), GGML_TYPE_F32, N_KV, 2, boundaries));
    }
    {
        // NaN (0xFFFFFFFF) must be rejected
        std::vector<uint32_t> data = f32_rows({ 5, 5 });
        data[0 * N_KV + 1] = uint32_t(0xFFFFFFFFu);
        std::vector<int32_t> boundaries;
        assert(!is_pure_causal_mask(data.data(), GGML_TYPE_F32, N_KV, 2, boundaries));
    }

    // ---- unsupported element type ----
    {
        std::vector<int32_t> data(N_KV, 0);
        std::vector<int32_t> boundaries;
        assert(!is_pure_causal_mask(data.data(), GGML_TYPE_I32, N_KV, 1, boundaries));
    }

    // ---- is_binary_mask: F16 ----
    {
        // all keep
        std::vector<uint16_t> data(N_KV, 0x0000);
        std::vector<uint8_t> bits;
        assert(is_binary_mask(data.data(), GGML_TYPE_F16, N_KV, bits));
        assert(bits.size() == 1);
        assert(bits[0] == 0x00);
    }
    {
        // all drop
        std::vector<uint16_t> data(N_KV, 0xFC00);
        std::vector<uint8_t> bits;
        assert(is_binary_mask(data.data(), GGML_TYPE_F16, N_KV, bits));
        assert(bits[0] == 0xFF);
    }
    {
        // mixed: first 4 keep, last 4 drop
        std::vector<uint16_t> data = { 0x0000, 0x0000, 0x0000, 0x0000, 0xFC00, 0xFC00, 0xFC00, 0xFC00 };
        std::vector<uint8_t> bits;
        assert(is_binary_mask(data.data(), GGML_TYPE_F16, N_KV, bits));
        assert(bits[0] == 0xF0);
    }
    {
        // non-binary value (0x3C00 = +1.0) must be rejected
        std::vector<uint16_t> data = { 0x0000, 0x3C00, 0x0000, 0x0000, 0xFC00, 0xFC00, 0xFC00, 0xFC00 };
        std::vector<uint8_t> bits;
        assert(!is_binary_mask(data.data(), GGML_TYPE_F16, N_KV, bits));
    }
    {
        // NaN (0x7E00) must be rejected
        std::vector<uint16_t> data = { 0x0000, 0x7E00, 0x0000, 0x0000, 0xFC00, 0xFC00, 0xFC00, 0xFC00 };
        std::vector<uint8_t> bits;
        assert(!is_binary_mask(data.data(), GGML_TYPE_F16, N_KV, bits));
    }
    {
        // -1.0 (0xFC00 is -inf, 0xBF80 is -1.0) must be rejected
        std::vector<uint16_t> data = { 0x0000, 0xBF80, 0x0000, 0x0000, 0xFC00, 0xFC00, 0xFC00, 0xFC00 };
        std::vector<uint8_t> bits;
        assert(!is_binary_mask(data.data(), GGML_TYPE_F16, N_KV, bits));
    }

    // ---- is_binary_mask: F32 ----
    {
        // mixed: first 4 keep, last 4 drop
        std::vector<uint32_t> data = { 0, 0, 0, 0, 0xFF800000u, 0xFF800000u, 0xFF800000u, 0xFF800000u };
        std::vector<uint8_t> bits;
        assert(is_binary_mask(data.data(), GGML_TYPE_F32, N_KV, bits));
        assert(bits[0] == 0xF0);
    }
    {
        // non-binary: +1.0 (0x3F800000) must be rejected
        std::vector<uint32_t> data = { 0, 0x3F800000u, 0, 0, 0xFF800000u, 0xFF800000u, 0xFF800000u, 0xFF800000u };
        std::vector<uint8_t> bits;
        assert(!is_binary_mask(data.data(), GGML_TYPE_F32, N_KV, bits));
    }
    {
        // NaN (0xFFFFFFFF) must be rejected
        std::vector<uint32_t> data = { 0, 0xFFFFFFFFu, 0, 0, 0xFF800000u, 0xFF800000u, 0xFF800000u, 0xFF800000u };
        std::vector<uint8_t> bits;
        assert(!is_binary_mask(data.data(), GGML_TYPE_F32, N_KV, bits));
    }

    // ---- is_binary_mask: unsupported type ----
    {
        std::vector<int32_t> data(N_KV, 0);
        std::vector<uint8_t> bits;
        assert(!is_binary_mask(data.data(), GGML_TYPE_I32, N_KV, bits));
    }

    // ---- is_binary_mask: multi-byte span (n_elements > 8) ----
    {
        // 16 elements: 8 keep + 8 drop
        std::vector<uint16_t> data(16);
        for (int i = 0; i < 8; i++) data[i] = 0x0000;
        for (int i = 8; i < 16; i++) data[i] = 0xFC00;
        std::vector<uint8_t> bits;
        assert(is_binary_mask(data.data(), GGML_TYPE_F16, 16, bits));
        assert(bits.size() == 2);
        assert(bits[0] == 0x00);
        assert(bits[1] == 0xFF);
    }

    std::printf("test-rpc-mask: all tests passed\n");
    return 0;
}
