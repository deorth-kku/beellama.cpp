#pragma once

#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#define RPC_PROTO_MAJOR_VERSION    10
#define RPC_PROTO_MINOR_VERSION    3
#define RPC_PROTO_PATCH_VERSION    0

#ifdef  __cplusplus
static_assert(GGML_OP_COUNT == 100, "GGML_OP_COUNT has changed - update RPC_PROTO_PATCH_VERSION");
#endif

#define GGML_RPC_MAX_SERVERS       16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_rpc_init(const char * endpoint, uint32_t device);

GGML_BACKEND_API bool ggml_backend_is_rpc(ggml_backend_t backend);

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_rpc_buffer_type(const char * endpoint, uint32_t device);

// bulk weight load over RPC: hash each tensor's data in parallel (n_threads),
// send one SET_TENSOR_HASH_BATCH, then send the data of the tensors the
// server does not have cached. Returns false if the buffer is not an RPC
// buffer or the server predates the batch command (nothing was sent).
GGML_BACKEND_API bool ggml_backend_buffer_set_tensors_hash_batch(ggml_backend_buffer_t buffer,
        struct ggml_tensor * const * tensors, const void * const * datas, const size_t * sizes,
        size_t n_tensors, size_t n_threads);

GGML_BACKEND_API void ggml_backend_rpc_get_device_memory(const char * endpoint, uint32_t device, size_t * free, size_t * total);

GGML_BACKEND_API void ggml_backend_rpc_start_server(
        const char * endpoint,
        const char * cache_dir,
        size_t n_threads,
        size_t n_devices,
        ggml_backend_dev_t * devices);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_rpc_reg(void);
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_rpc_add_server(const char * endpoint);

#ifdef  __cplusplus
}
#endif
