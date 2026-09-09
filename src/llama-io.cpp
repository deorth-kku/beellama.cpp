#include "llama-io.h"

#include "ggml-backend.h"
#include "ggml.h"
#include "llama-mmap.h"

#include <algorithm>
#include <condition_variable>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

void llama_io_write_i::write_string(const std::string & str) {
    uint32_t str_size = str.size();

    write(&str_size,  sizeof(str_size));
    write(str.data(), str_size);
}

void llama_io_read_i::read_string(std::string & str) {
    uint32_t str_size;
    read(&str_size, sizeof(str_size));

    std::vector<char> buf(str_size);
    read(buf.data(), str_size);

    str.assign(buf.data(), str_size);
}

// split a list of blocks into contiguous groups, balancing by total bytes
// contiguous groups keep the file locality per worker thread
static std::vector<std::pair<size_t, size_t>> io_file_block_groups(const std::vector<llama_io_block> & blocks, size_t n_threads) {
    size_t total = 0;
    for (const auto & block : blocks) {
        total += block.size;
    }

    const size_t n_groups = std::min<size_t>(n_threads, blocks.size());
    const size_t target = total / n_groups;

    std::vector<std::pair<size_t, size_t>> groups;
    size_t i = 0;
    while (i < blocks.size()) {
        const size_t i0 = i;
        size_t bytes = 0;
        while (i < blocks.size() && bytes < target) {
            bytes += blocks[i].size;
            ++i;
        }
        if (i == i0) {
            ++i;
        }
        groups.emplace_back(i0, i);
    }

    return groups;
}

// a fixed-size pool of worker threads, reused across read_blocks/write_blocks calls
struct io_file_workers {
    std::vector<std::thread> threads;
    std::mutex mutex;
    std::condition_variable cv;
    std::queue<std::function<void()>> tasks;
    size_t n_running = 0;
    std::exception_ptr eptr;
    bool shutdown = false;

    explicit io_file_workers(size_t n) {
        for (size_t i = 0; i < n; ++i) {
            threads.emplace_back([this]() {
                for (;;) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(mutex);
                        cv.wait(lock, [this]() { return shutdown || !tasks.empty(); });
                        if (shutdown && tasks.empty()) {
                            return;
                        }
                        task = std::move(tasks.front());
                        tasks.pop();
                        ++n_running;
                    }

                    try {
                        task();
                    } catch (...) {
                        std::lock_guard<std::mutex> lock(mutex);
                        if (!eptr) {
                            eptr = std::current_exception();
                        }
                    }

                    {
                        std::lock_guard<std::mutex> lock(mutex);
                        --n_running;
                    }
                    cv.notify_all();
                }
            });
        }
    }

    ~io_file_workers() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            shutdown = true;
        }
        cv.notify_all();
        for (auto & t : threads) {
            t.join();
        }
    }

    // run all tasks, rethrowing the first error on the caller
    void run(std::vector<std::function<void()>> && task_list) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            for (auto & task : task_list) {
                tasks.push(std::move(task));
            }
        }
        cv.notify_all();

        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [this]() { return tasks.empty() && n_running == 0; });

        auto eptr = this->eptr;
        this->eptr = nullptr;
        if (eptr) {
            std::rethrow_exception(eptr);
        }
    }
};

llama_io_write_file::llama_io_write_file(llama_file * f, size_t n_threads)
    : file(f), n_threads(std::max<size_t>(1, std::min<size_t>(8, n_threads))) {}

llama_io_write_file::~llama_io_write_file() = default;

void llama_io_write_file::write(const void * src, size_t size) {
    file->write_raw(src, size);
    size_written += size;
}

void llama_io_write_file::write_tensor(ggml_tensor * tensor, size_t offset, size_t size) {
    temp_buffer.resize(size);
    ggml_backend_tensor_get(tensor, temp_buffer.data(), offset, size);
    write(temp_buffer.data(), temp_buffer.size());
}

size_t llama_io_write_file::tell() const {
    return file->tell();
}

void llama_io_write_file::write_blocks(const std::vector<llama_io_block> & blocks) {
    if (blocks.empty()) {
        return;
    }

    // flush the buffered prefix before the absolute-offset writes
    file->flush();

    size_t total = 0;
    for (const auto & block : blocks) {
        total += block.size;
    }

    // transfer the blocks in parallel, each worker owns its staging buffer
    auto process = [&](size_t i0, size_t i1) {
        std::vector<uint8_t> buf;

        for (size_t i = i0; i < i1; ++i) {
            const auto & block = blocks[i];

            size_t done = 0;
            while (done < block.size) {
                const size_t chunk = std::min(IO_CHUNK, block.size - done);

                if (block.tensor) {
                    buf.resize(chunk);
                    ggml_backend_tensor_get(block.tensor, buf.data(), block.tensor_offset + done, chunk);
                }

                const uint8_t * src = block.tensor ? buf.data() : (const uint8_t *) block.data + done;
                file->write_at(src, chunk, block.offset + done);

                done += chunk;
            }
        }
    };

    const auto groups = io_file_block_groups(blocks, n_threads);
    std::vector<std::function<void()>> tasks;
    tasks.reserve(groups.size());
    for (const auto & g : groups) {
        tasks.emplace_back([process, g]() { process(g.first, g.second); });
    }
    get_workers().run(std::move(tasks));

    // leave the stream at the end of the last block
    file->seek(blocks.back().offset + blocks.back().size, SEEK_SET);
    size_written += total;
}

size_t llama_io_write_file::n_bytes() {
    return size_written;
}

io_file_workers & llama_io_write_file::get_workers() {
    if (!workers) {
        workers.reset(new io_file_workers(n_threads));
    }
    return *workers;
}

llama_io_read_file::llama_io_read_file(llama_file * f, size_t n_threads)
    : file(f), n_threads(std::max<size_t>(1, std::min<size_t>(8, n_threads))) {}

llama_io_read_file::~llama_io_read_file() = default;

void llama_io_read_file::read(void * dst, size_t size) {
    file->read_raw(dst, size);
    size_read += size;
}

void llama_io_read_file::read_tensor(ggml_tensor * tensor, size_t offset, size_t size) {
    temp_buffer.resize(size);
    read(temp_buffer.data(), size);
    ggml_backend_tensor_set(tensor, temp_buffer.data(), offset, size);
}

size_t llama_io_read_file::tell() const {
    return file->tell();
}

void llama_io_read_file::read_blocks(const std::vector<llama_io_block> & blocks) {
    if (blocks.empty()) {
        return;
    }

    size_t total = 0;
    for (const auto & block : blocks) {
        total += block.size;
    }

    // transfer the blocks in parallel, each worker owns its staging buffer
    auto process = [&](size_t i0, size_t i1) {
        std::vector<uint8_t> buf;

        for (size_t i = i0; i < i1; ++i) {
            const auto & block = blocks[i];

            size_t done = 0;
            while (done < block.size) {
                const size_t chunk = std::min(IO_CHUNK, block.size - done);

                buf.resize(chunk);
                file->read_at(buf.data(), chunk, block.offset + done);
                ggml_backend_tensor_set(block.tensor, buf.data(), block.tensor_offset + done, chunk);

                done += chunk;
            }
        }
    };

    const auto groups = io_file_block_groups(blocks, n_threads);
    std::vector<std::function<void()>> tasks;
    tasks.reserve(groups.size());
    for (const auto & g : groups) {
        tasks.emplace_back([process, g]() { process(g.first, g.second); });
    }
    get_workers().run(std::move(tasks));

    // leave the stream at the end of the last block
    file->seek(blocks.back().offset + blocks.back().size, SEEK_SET);
    size_read += total;
}

size_t llama_io_read_file::n_bytes() {
    return size_read;
}

io_file_workers & llama_io_read_file::get_workers() {
    if (!workers) {
        workers.reset(new io_file_workers(n_threads));
    }
    return *workers;
}
