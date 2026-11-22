// Unit test for RPC_CMD_SET_TENSOR_HASH_BATCH.
//
// A raw-socket client that speaks the wire protocol directly against a
// running rpc-server started with -c (hash cache enabled). It verifies:
// - the server advertises the TENSOR_HASH_BATCH flag in the HELLO response
// - hit shards are reported as hit and their data lands in the server buffer
// - miss entries are reported as miss
// - results are returned in request order
//
// The packed structs and command numbers below must stay in sync with
// ggml/src/ggml-rpc/ggml-rpc.cpp.

#include "ggml.h"
#include "ggml-rpc.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#ifdef _WIN32
#  include <winsock2.h>
#  include <ws2tcpip.h>
#else
#  include <sys/socket.h>
#  include <netinet/in.h>
#  include <arpa/inet.h>
#  include <unistd.h>
#endif

// ---- wire-format mirrors of the server structs ----

#pragma pack(push, 1)
struct test_rpc_tensor {
    uint64_t id;
    uint32_t type;
    uint64_t buffer;
    uint32_t ne[GGML_MAX_DIMS];
    uint32_t nb[GGML_MAX_DIMS];
    uint32_t op;
    int32_t  op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];
    int32_t  flags;
    uint64_t src[GGML_MAX_SRC];
    uint64_t view_src;
    uint64_t view_offs;
    uint64_t data;
    char name[GGML_MAX_NAME];
    int32_t use_count;
};

struct test_rpc_msg_hello_req {
    uint8_t conn_caps[24];
};

struct test_rpc_msg_hello_rsp {
    uint8_t major;
    uint8_t minor;
    uint8_t patch;
    uint8_t flags;
    uint8_t conn_caps[24];
};

struct test_rpc_msg_alloc_buffer_req {
    uint32_t device;
    uint64_t size;
};

struct test_rpc_msg_alloc_buffer_rsp {
    uint64_t remote_ptr;
    uint64_t remote_size;
};

struct test_rpc_msg_buffer_get_base_req {
    uint64_t remote_ptr;
};

struct test_rpc_msg_buffer_get_base_rsp {
    uint64_t base_ptr;
};

struct test_rpc_msg_set_tensor_hash_req {
    test_rpc_tensor tensor;
    uint64_t offset;
    uint64_t hash;
};

struct test_rpc_msg_get_tensor_req {
    test_rpc_tensor tensor;
    uint64_t offset;
    uint64_t size;
};
#pragma pack(pop)

// command numbers, must match the rpc_cmd enum in ggml-rpc.cpp
enum : uint8_t {
    TEST_RPC_CMD_ALLOC_BUFFER = 0,
    TEST_RPC_CMD_GET_ALIGNMENT,
    TEST_RPC_CMD_GET_MAX_SIZE,
    TEST_RPC_CMD_BUFFER_GET_BASE,
    TEST_RPC_CMD_FREE_BUFFER,
    TEST_RPC_CMD_BUFFER_CLEAR,
    TEST_RPC_CMD_SET_TENSOR,
    TEST_RPC_CMD_SET_TENSOR_HASH,
    TEST_RPC_CMD_GET_TENSOR,
    TEST_RPC_CMD_COPY_TENSOR,
    TEST_RPC_CMD_GRAPH_COMPUTE,
    TEST_RPC_CMD_GET_DEVICE_MEMORY,
    TEST_RPC_CMD_INIT_TENSOR,
    TEST_RPC_CMD_GET_ALLOC_SIZE,
    TEST_RPC_CMD_HELLO,
    TEST_RPC_CMD_DEVICE_COUNT,
    TEST_RPC_CMD_GRAPH_RECOMPUTE,
    TEST_RPC_CMD_MEMSET_TENSOR,
    TEST_RPC_CMD_SET_TENSOR_2D,
    TEST_RPC_CMD_GET_TENSOR_2D,
    TEST_RPC_CMD_COMM_INIT,
    TEST_RPC_CMD_COMM_ALLREDUCE,
    TEST_RPC_CMD_COMM_FREE,
    TEST_RPC_CMD_FILL_CAUSAL_MASK,
    TEST_RPC_CMD_SET_MASK_BITS,
    TEST_RPC_CMD_SET_TENSOR_HASH_BATCH,
};

static_assert(TEST_RPC_CMD_HELLO == 14, "rpc_cmd enum drifted, update this mirror");

// same FNV-1a the server uses to name cache files
static uint64_t fnv_hash(const uint8_t * data, size_t len) {
    uint64_t hash = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < len; i++) {
        hash ^= data[i];
        hash *= 0x100000001b3ULL;
    }
    return hash;
}

static std::string hex16(uint64_t value) {
    char buf[17];
    snprintf(buf, sizeof(buf), "%016llx", (unsigned long long) value);
    return std::string(buf);
}

class test_socket {
    int fd = -1;

public:
    ~test_socket() {
        if (fd >= 0) {
#ifdef _WIN32
            closesocket(fd);
#else
            close(fd);
#endif
        }
    }

    bool connect(const char * host, int port) {
#ifdef _WIN32
        WSADATA wsa;
        WSAStartup(MAKEWORD(2, 2), &wsa);
#endif
        fd = (int) socket(AF_INET, SOCK_STREAM, 0);
        if (fd < 0) {
            return false;
        }
        struct sockaddr_in addr;
        std::memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        addr.sin_port = htons((uint16_t) port);
        if (inet_pton(AF_INET, host, &addr.sin_addr) != 1) {
            return false;
        }
        return ::connect(fd, (struct sockaddr *) &addr, sizeof(addr)) == 0;
    }

    bool send_all(const void * data, size_t size) {
        const uint8_t * p = (const uint8_t *) data;
        while (size > 0) {
            int n = send(fd, (const char *) p, (int) size, 0);
            if (n <= 0) {
                return false;
            }
            p += n;
            size -= (size_t) n;
        }
        return true;
    }

    bool recv_all(void * data, size_t size) {
        uint8_t * p = (uint8_t *) data;
        while (size > 0) {
            int n = recv(fd, (char *) p, (int) size, 0);
            if (n <= 0) {
                return false;
            }
            p += n;
            size -= (size_t) n;
        }
        return true;
    }
};

static bool send_cmd(test_socket & s, uint8_t cmd, const void * req, size_t size) {
    uint64_t n = size;
    uint8_t c = cmd;
    return s.send_all(&c, 1) && s.send_all(&n, sizeof(n)) && (size == 0 || s.send_all(req, size));
}

static bool recv_rsp(test_socket & s, void * buf, size_t size) {
    uint64_t n = 0;
    if (!s.recv_all(&n, sizeof(n)) || n != size) {
        return false;
    }
    return s.recv_all(buf, size);
}

int main(int argc, char ** argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <host:port> <cache_dir>\n", argv[0]);
        return 1;
    }
    const std::string endpoint = argv[1];
    const std::filesystem::path cache_dir = argv[2];

    // 4 shards: entries 0 and 2 are cache hits, 1 and 3 are misses
    const size_t shard_size = 256 * 1024;
    const size_t n_shards = 4;
    const bool is_hit[4] = { true, false, true, false };
    std::vector<std::vector<uint8_t>> content(n_shards);
    std::vector<uint64_t> hash(n_shards);
    for (size_t i = 0; i < n_shards; i++) {
        content[i].resize(shard_size);
        for (size_t j = 0; j < shard_size; j++) {
            content[i][j] = (uint8_t) ((i + 1) * 31 + j);
        }
        hash[i] = fnv_hash(content[i].data(), shard_size);
        if (is_hit[i]) {
            std::ofstream ofs(cache_dir / hex16(hash[i]), std::ios::binary);
            if (!ofs || !(ofs.write((const char *) content[i].data(), shard_size))) {
                fprintf(stderr, "failed to write cache file %s\n", hex16(hash[i]).c_str());
                return 1;
            }
        }
    }

    const size_t pos = endpoint.find(':');
    const std::string host = endpoint.substr(0, pos);
    const int port = std::stoi(endpoint.substr(pos + 1));

    test_socket sock;
    if (!sock.connect(host.c_str(), port)) {
        fprintf(stderr, "failed to connect to %s\n", endpoint.c_str());
        return 1;
    }

    // HELLO
    test_rpc_msg_hello_req hello_req = {};
    test_rpc_msg_hello_rsp hello_rsp = {};
    if (!send_cmd(sock, TEST_RPC_CMD_HELLO, &hello_req, sizeof(hello_req)) ||
        !recv_rsp(sock, &hello_rsp, sizeof(hello_rsp))) {
        fprintf(stderr, "HELLO failed\n");
        return 1;
    }
    if (hello_rsp.major != RPC_PROTO_MAJOR_VERSION || hello_rsp.minor > RPC_PROTO_MINOR_VERSION) {
        fprintf(stderr, "server version mismatch: %d.%d.%d\n", hello_rsp.major, hello_rsp.minor, hello_rsp.patch);
        return 1;
    }
    if ((hello_rsp.flags & 0x2) == 0) {
        fprintf(stderr, "server did not advertise RPC_DISPATCHER_FLAGS_TENSOR_HASH_BATCH\n");
        return 1;
    }

    // ALLOC_BUFFER
    test_rpc_msg_alloc_buffer_req alloc_req = { 0, n_shards * shard_size };
    test_rpc_msg_alloc_buffer_rsp alloc_rsp = {};
    if (!send_cmd(sock, TEST_RPC_CMD_ALLOC_BUFFER, &alloc_req, sizeof(alloc_req)) ||
        !recv_rsp(sock, &alloc_rsp, sizeof(alloc_rsp))) {
        fprintf(stderr, "ALLOC_BUFFER failed\n");
        return 1;
    }

    // BUFFER_GET_BASE
    test_rpc_msg_buffer_get_base_req base_req = { alloc_rsp.remote_ptr };
    test_rpc_msg_buffer_get_base_rsp base_rsp = {};
    if (!send_cmd(sock, TEST_RPC_CMD_BUFFER_GET_BASE, &base_req, sizeof(base_req)) ||
        !recv_rsp(sock, &base_rsp, sizeof(base_rsp))) {
        fprintf(stderr, "BUFFER_GET_BASE failed\n");
        return 1;
    }

    // batch request: | n | n x rpc_msg_set_tensor_hash_req |
    const size_t req_size = sizeof(uint32_t) + n_shards * sizeof(test_rpc_msg_set_tensor_hash_req);
    std::vector<uint8_t> req(req_size);
    uint32_t n = (uint32_t) n_shards;
    std::memcpy(req.data(), &n, sizeof(n));
    for (size_t i = 0; i < n_shards; i++) {
        test_rpc_msg_set_tensor_hash_req * entry =
            (test_rpc_msg_set_tensor_hash_req *) (req.data() + sizeof(uint32_t) + i * sizeof(test_rpc_msg_set_tensor_hash_req));
        std::memset(entry, 0, sizeof(*entry));
        entry->tensor.type = GGML_TYPE_F32;
        entry->tensor.buffer = alloc_rsp.remote_ptr;
        entry->tensor.data = base_rsp.base_ptr + i * shard_size;
        entry->tensor.ne[0] = (uint32_t) (shard_size / 4);
        entry->tensor.ne[1] = 1;
        entry->tensor.ne[2] = 1;
        entry->tensor.ne[3] = 1;
        entry->tensor.nb[0] = 4;
        entry->tensor.nb[1] = (uint32_t) shard_size;
        entry->tensor.nb[2] = (uint32_t) shard_size;
        entry->tensor.nb[3] = (uint32_t) shard_size;
        entry->offset = 0;
        entry->hash = hash[i];
    }

    std::vector<uint8_t> rsp(sizeof(uint32_t) + n_shards);
    if (!send_cmd(sock, TEST_RPC_CMD_SET_TENSOR_HASH_BATCH, req.data(), req.size()) ||
        !recv_rsp(sock, rsp.data(), rsp.size())) {
        fprintf(stderr, "SET_TENSOR_HASH_BATCH failed\n");
        return 1;
    }
    uint32_t rsp_n = 0;
    std::memcpy(&rsp_n, rsp.data(), sizeof(rsp_n));
    if (rsp_n != n_shards) {
        fprintf(stderr, "unexpected result count: %u\n", rsp_n);
        return 1;
    }
    for (size_t i = 0; i < n_shards; i++) {
        if (rsp[sizeof(uint32_t) + i] != (uint8_t) (is_hit[i] ? 1 : 0)) {
            fprintf(stderr, "result[%zu] = %d, expected %d\n", i, rsp[sizeof(uint32_t) + i], is_hit[i] ? 1 : 0);
            return 1;
        }
    }

    // read the hit shards back and check the data
    for (size_t i = 0; i < n_shards; i++) {
        if (!is_hit[i]) {
            continue;
        }
        test_rpc_msg_get_tensor_req get_req = {};
        get_req.tensor.type = GGML_TYPE_F32;
        get_req.tensor.buffer = alloc_rsp.remote_ptr;
        get_req.tensor.data = base_rsp.base_ptr + i * shard_size;
        get_req.tensor.ne[0] = (uint32_t) (shard_size / 4);
        get_req.tensor.ne[1] = 1;
        get_req.tensor.ne[2] = 1;
        get_req.tensor.ne[3] = 1;
        get_req.tensor.nb[0] = 4;
        get_req.tensor.nb[1] = (uint32_t) shard_size;
        get_req.tensor.nb[2] = (uint32_t) shard_size;
        get_req.tensor.nb[3] = (uint32_t) shard_size;
        get_req.offset = 0;
        get_req.size = shard_size;
        std::vector<uint8_t> data(shard_size);
        if (!send_cmd(sock, TEST_RPC_CMD_GET_TENSOR, &get_req, sizeof(get_req)) ||
            !recv_rsp(sock, data.data(), data.size())) {
            fprintf(stderr, "GET_TENSOR failed for shard %zu\n", i);
            return 1;
        }
        if (std::memcmp(data.data(), content[i].data(), shard_size) != 0) {
            fprintf(stderr, "shard %zu content mismatch\n", i);
            return 1;
        }
    }

    printf("test-rpc-hash-batch: ok\n");
    return 0;
}
