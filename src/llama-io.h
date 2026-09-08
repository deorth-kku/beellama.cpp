#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct ggml_tensor;

// a single data block of a state stream, at an absolute offset in stream coordinates
// tensor != nullptr: tensor data, transferred via ggml_backend_tensor_get/set
// tensor == nullptr: raw data, `data` holds the source pointer (writes only)
struct llama_io_block {
    size_t offset = 0;
    size_t size = 0;

    ggml_tensor * tensor = nullptr;
    size_t tensor_offset = 0;

    const void * data = nullptr;
};

class llama_io_write_i {
public:
    llama_io_write_i() = default;
    virtual ~llama_io_write_i() = default;

    virtual void write(const void * src, size_t size) = 0;
    virtual void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) = 0;

    // current stream position
    virtual size_t tell() const { return 0; }

    // write the blocks, which must be in ascending offset order
    // after the call, the stream position is at the end of the last block
    virtual void write_blocks(const std::vector<llama_io_block> & blocks) {
        for (const auto & block : blocks) {
            if (block.tensor) {
                write_tensor(block.tensor, block.tensor_offset, block.size);
            } else {
                write(block.data, block.size);
            }
        }
    }

    // bytes written so far
    virtual size_t n_bytes() = 0;

    void write_string(const std::string & str);
};

class llama_io_read_i {
public:
    llama_io_read_i() = default;
    virtual ~llama_io_read_i() = default;

    virtual void read(void * dst, size_t size) = 0;
    virtual void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) = 0;

    // current stream position
    virtual size_t tell() const { return 0; }

    // read the blocks, which must be in ascending offset order
    // the stream position must be at the offset of the first block
    // after the call, the stream position is at the end of the last block
    virtual void read_blocks(const std::vector<llama_io_block> & blocks) {
        for (const auto & block : blocks) {
            read_tensor(block.tensor, block.tensor_offset, block.size);
        }
    }

    // bytes read so far
    virtual size_t n_bytes() = 0;

    void read_string(std::string & str);
};
