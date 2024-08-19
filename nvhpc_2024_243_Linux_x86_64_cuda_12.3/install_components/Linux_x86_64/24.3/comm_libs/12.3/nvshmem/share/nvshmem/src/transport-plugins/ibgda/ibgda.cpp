/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include <assert.h>        // for assert
#include <stdint.h>        // for uint8_t, uint64_t, uint32_t
#include <cuda.h>          // for cudaError_enum::CUDA_SUCCESS
#include <cuda_runtime.h>  // for cudaFree, cudaMemcpyAsync
#include <driver_types.h>  // for cudaError::cudaSuccess, cuda...
#include <endian.h>        // for htobe32, htobe64
#include <errno.h>         // for ENOMEM
#include <linux/types.h>   // for __be32
#include <math.h>          // for ceil, log2
#include <stdbool.h>       // for bool, true, false
#include <stddef.h>        // for NULL, size_t, offsetof
#include <stdint.h>        // IWYU pragma: keep
// IWYU pragma: no_include <bits/stdint-uintn.h>
#include <stdio.h>      // for fprintf, stderr, printf
#include <stdlib.h>     // for calloc, free, malloc
#include <string.h>     // for memset, memcpy, strcmp, strstr
#include <sys/types.h>  // for off_t
#include <algorithm>    // for for_each, remove_if
#include <cctype>       // for tolower, isspace
#include <string>       // for basic_string, string, operat...
#include <vector>       // for vector

#include "cudawrap.h"                     // for CUPFN, nvshmemi_cuda_fn_table
#include "env_defs_internal.h"            // for nvshmemi_options_s, nvshmemi...
#include "infiniband/mlx5dv.h"            // for DEVX_SET, DEVX_ST_SZ_BYTES
#include "infiniband/verbs.h"             // for ibv_ah_attr, ibv_port_attr
#include "mlx5_ifc.h"                     // for mlx5_ifc_qpc_bits, mlx5_ifc_...
#include "mlx5_prm.h"                     // for mlx5_ifc_cqc_bits, mlx5_ifc_...
#include "common/nvshmem_common_ibgda.h"  // for nvshmemi_ibgda_device_qp_t
#include "nvshmemi_bootstrap_defines.h"   // for bootstrap_handle_t
#include "nvshmemi_transport_defines.h"   // for nvshmem_mem_handle_t, NVSHME...
#include "host/nvshmemx_error.h"          // for nvshmemx_status::NVSHMEMX_ER...
#include "transport.h"                    // for nvshmem_transport, nvshmem_t...
#include "transport_common.h"             // for nvshmemt_hca_info, nvshmemt_...
#include "transport_ib_common.h"          // for nvshmemt_ib_common_mem_handle
#include "transport_mlx5_common.h"        // for nvshmemt_ib_common_query_end...

#define CUDA_RUNTIME_ERROR_STRING(result)                                         \
    do {                                                                          \
        if (unlikely(cudaSuccess != result)) {                                    \
            fprintf(stderr, "[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, \
                    cudaGetErrorString(result));                                  \
        }                                                                         \
    } while (0)

#define NVSHMEMI_IBGDA_CQE_SIZE 64
#define NVSHMEMI_IBGDA_MAX_INLINE_SIZE (8 * 32)

#define MAX_NUM_HCAS 16
#define MAX_NUM_PORTS 4
#define MAX_NUM_PES_PER_NODE 32

#define IBGDA_DC_ACCESS_KEY 0x5623CEAF

#define IBGDA_MLX5_QPC_ATOMIC_MODE_UP_TO_64BIT 0x3
#define IBGDA_DBSIZE 8
#define IBGDA_SRQ_TYPE_VALUE 0x1

#define IBGDA_LOG_MAX_MSG_SIZE 30  // 30 is max allowed on IB QPs
#define IBGDA_MIN_RNR_NAK 12

#define IBGDA_GRH_HOP_LIMIT 255

// First slot is reserved for non-fetch operations.
#define IBGDA_IBUF_RESERVED_SLOTS 1

#define IBGDA_GPAGE_BITS 16
#define IBGDA_GPAGE_SIZE (1ULL << IBGDA_GPAGE_BITS)
#define IBGDA_GPAGE_OFF (IBGDA_GPAGE_SIZE - 1)
#define IBGDA_GPAGE_MASK (~(IBGDA_GPAGE_OFF))

#define IBGDA_MIN(x, y) ((x) < (y) ? (x) : (y))
#define IBGDA_MAX(x, y) ((x) > (y) ? (x) : (y))

#define IBGDA_ROUND_UP(V, SIZE) (((V) + (SIZE)-1) / (SIZE) * (SIZE))

#define IBGDA_ROUND_UP_POW2(_n)                 \
    ({                                          \
        typeof(_n) pow2 = 0;                    \
        assert((_n) >= 1);                      \
        for (pow2 = 1; pow2 < (_n); pow2 <<= 1) \
            ;                                   \
        pow2;                                   \
    })

#define IBGDA_ROUND_UP_POW2_OR_0(_n) (((_n) == 0) ? 0 : IBGDA_ROUND_UP_POW2(_n))

#define IBGDA_ROUND_DOWN_POW2_OR_0(_n)                  \
    ({                                                  \
        typeof(_n) pow2 = IBGDA_ROUND_UP_POW2_OR_0(_n); \
        (((_n) < pow2) ? pow2 / 2 : pow2);              \
    })

template <typename T>
inline T IBGDA_ILOG2(T _n) {
    return (T)ceil(log2((double)_n));
}

#define IBGDA_ILOG2_OR0(_n) (((_n) == 0) ? 0 : IBGDA_ILOG2(_n))

enum { IBGDA_MLX5_QPC_ST_RC = 0x0, IBGDA_MLX5_QPC_ST_DCI = 0x5 };

enum {
    IBGDA_MLX5_UMEM_VALID_DISABLE = 0x0,
    IBGDA_MLX5_UMEM_VALID_ENABLE = 0x1,
};

enum { IBGDA_MLX5_NC_UAR_SIZE = 8 };

typedef enum {
    IBGDA_NIC_MAPPING_MEMTYPE_REQUEST_AUTO = 0,
    IBGDA_NIC_MAPPING_MEMTYPE_REQUEST_GPUMEM,
    IBGDA_NIC_MAPPING_MEMTYPE_REQUEST_HOSTMEM,
} ibgda_nic_mapping_memtype_reqeust_t;

typedef enum {
    IBGDA_MEM_TYPE_HOST = 0,
    IBGDA_MEM_TYPE_GPU = 1,
    IBGDA_MEM_TYPE_NIC = 2,
} ibgda_mem_type_t;

struct ibgda_mem_object {
    ibgda_mem_type_t mem_type;
    struct {
        void *cpu_ptr;
        void *gpu_ptr;
        size_t size;
    } base;
    struct {
        void *cpu_ptr;
        void *gpu_ptr;
        size_t size;
    } aligned;
    union {
        struct mlx5dv_devx_umem *umem;
        struct mlx5dv_devx_uar *uar;
    };
    bool has_cpu_mapping : 1;
    bool has_gpu_mapping : 1;
    bool has_nic_mapping : 1;
};

struct ibgda_cq {
    struct mlx5dv_devx_obj *devx_cq;
    uint32_t cqn;
    uint32_t num_cqe;
    struct ibgda_mem_object *cq_mobject;
    struct ibgda_mem_object *dbr_mobject;
    struct mlx5dv_devx_uar *uar;
};

struct ibgda_ep {
    nvshmemi_ibgda_device_qp_type_t qp_type;

    union {
        struct mlx5dv_devx_obj *devx_qp;
        struct ibv_qp *ib_qp;
    };
    uint32_t qpn;
    int portid;

    size_t sq_cnt;
    off_t sq_buf_offset;
    size_t rq_cnt;
    off_t rq_buf_offset;

    struct ibgda_mem_object *wq_mobject;
    struct ibgda_mem_object *dbr_mobject;
    struct ibgda_mem_object *uar_mobject;

    union {
        struct ibgda_cq *send_cq;  // Valid only on DCI
        struct ibv_ah *ah;         // Valid only on RC
    };

    uint32_t user_index;
};

struct ibgda_dct_handle {
    nvshmemi_ibgda_device_dct_t dev_dct;
    bool support_half_av_seg;
};

struct ibgda_rc_handle {
    uint32_t qpn;
    uint16_t lid;
    // RoCE
    uint64_t spn;
    uint64_t iid;
};

struct ibgda_internal_buffer {
    struct ibgda_mem_object *mem_object;
    struct nvshmemt_ib_common_mem_handle *mem_handle;
};

struct ibgda_device {
    struct ibv_device *dev;
    struct ibv_pd *pd; /* protection domain */
    struct ibv_context *context;
    struct ibv_device_attr device_attr;
    struct ibv_port_attr port_attr[MAX_NUM_PORTS];
    union ibv_gid gid[MAX_NUM_PORTS];
    struct {
        int num_eps;
        struct ibgda_ep **eps;
        struct ibgda_dct_handle *dct_handles;
        struct ibv_pd *pd; /* parent domain */
        struct ibv_srq *srq;
        struct ibv_cq *send_cq;
        struct ibv_cq *recv_cq;
        struct ibv_ah *ah;
        struct mlx5dv_ah dah;
        struct ibv_ah_attr ah_attr;
    } dct;
    struct {
        struct ibv_srq *srq;
        struct ibv_cq *recv_cq;
        int pdn;
        int srqn;
        int rcqn;
        struct ibgda_internal_buffer internal_buf;
    } qp_shared_object;  // For DCI and RC
    struct {
        int num_eps;
        int num_shared_eps;
        nvshmemi_ibgda_device_qp_map_type_t map_by;
        struct ibgda_ep **eps;
    } dci;
    struct {
        int num_eps_per_pe;
        nvshmemi_ibgda_device_qp_map_type_t map_by;
        struct ibgda_ep **eps;
        struct ibgda_rc_handle *peer_ep_handles;
    } rc;
    bool support_nic_buf_on_gpumem;
    bool support_nic_buf_on_hostmem;
    bool support_half_av_seg;
    bool may_skip_cst;
};

typedef struct {
    struct nvshmemi_options_s *options;
    void *devices;
    int *dev_ids;
    int *port_ids;
    int n_dev_ids;
    int selected_dev_id;
    int log_level;
    bool dmabuf_support;
    cudaStream_t my_stream;
} nvshmemt_ibgda_state_t;

struct ibgda_device_local_only_mhandle_cache {
    nvshmemi_ibgda_device_local_only_mhandle_t mhandle;
    void *
        dev_ptr;  // Ptr to GPU buffer that contains a copy of this mhandle. CPU cannot dereference.
};

// CPU cannot dereference next
static std::vector<struct ibgda_device_local_only_mhandle_cache> ibgda_device_local_only_mhandles;

static std::vector<nvshmemi_ibgda_device_key_t> ibgda_device_lkeys;
static std::vector<nvshmemi_ibgda_device_key_t> ibgda_device_rkeys;

// Ptr to GPU buffer. CPU cannot dereference.
static void *ibgda_device_lkeys_d = 0;
static void *ibgda_device_rkeys_d = 0;

/* transport constants */
static int ibgda_qp_depth;
static int ibgda_srq_depth;
static int ibgda_num_requests_in_batch;
static int ibgda_num_fetch_slots_per_dci;
static int ibgda_num_fetch_slots_per_rc;

/* ibv state */
static struct nvshmemt_ibv_function_table ftable;
static void *ibv_handle;

/* CUDA function table */
static struct nvshmemi_cuda_fn_table *ibgda_cuda_syms;

static ibgda_mem_type_t ibgda_nic_buf_location;

static int ibgda_parse_qp_map_by(nvshmemi_ibgda_device_qp_map_type_t *out_map_by, const char *str) {
    int status = 0;
    nvshmemi_ibgda_device_qp_map_type_t map_by;
    std::string req = str;

    // Trim whitespace
    req.erase(std::remove_if(req.begin(), req.end(), ::isspace), req.end());

    // To lower case
    std::for_each(req.begin(), req.end(), [](decltype(*req.begin()) &c) { c = ::tolower(c); });

    if (req == "cta") {
        map_by = NVSHMEMI_IBGDA_DEVICE_QP_MAP_TYPE_CTA;
    } else if (req == "sm") {
        map_by = NVSHMEMI_IBGDA_DEVICE_QP_MAP_TYPE_SM;
    } else if (req == "warp") {
        map_by = NVSHMEMI_IBGDA_DEVICE_QP_MAP_TYPE_WARP;
    } else if (req == "dct") {
        map_by = NVSHMEMI_IBGDA_DEVICE_QP_MAP_TYPE_DCT;
    } else {
        status = NVSHMEMX_ERROR_INVALID_VALUE;
    }

    if (status == 0) {
        *out_map_by = map_by;
    }

    return status;
}

int nvshmemt_ibgda_progress(nvshmem_transport_t t) {
    /* TODO: Implement me. Here we need to check for errors from the device */
    return 0;
}

int nvshmemt_ibgda_show_info(nvshmem_mem_handle_t *mem_handles, int transport_id,
                             int transport_count, int npes, int mype) {
    for (int i = 0; i < npes; ++i) {
        printf("[%d] mem_handle %d : %p\n", mype, transport_id,
               &mem_handles[i * transport_count + transport_id]);
        struct nvshmemt_ib_common_mem_handle *mem_handle =
            (struct nvshmemt_ib_common_mem_handle
                 *)&mem_handles[i * transport_count + transport_id];
        printf("[%d] lkey %x rkey %x mr %p\n", mype, mem_handle->lkey, mem_handle->rkey,
               mem_handle->mr);
    }
    return 0;
}

static int get_pci_path(int dev, char **pci_path, nvshmem_transport_t t) {
    int status = NVSHMEMX_SUCCESS;

    struct nvshmem_transport *transport = (struct nvshmem_transport *)t;
    nvshmemt_ibgda_state_t *ibgda_state = (nvshmemt_ibgda_state_t *)transport->state;
    int dev_id = ibgda_state->dev_ids[dev];
    const char *ib_name =
        (const char *)((struct ibgda_device *)ibgda_state->devices)[dev_id].dev->name;

    status = nvshmemt_ib_iface_get_mlx_path(ib_name, pci_path);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "nvshmemt_ib_iface_get_mlx_path failed \n");

out:
    return status;
}

int nvshmemt_ibgda_can_reach_peer(int *access, struct nvshmem_transport_pe_info *peer_info,
                                  nvshmem_transport_t t) {
    int status = 0;

    *access = NVSHMEM_TRANSPORT_CAP_GPU_WRITE | NVSHMEM_TRANSPORT_CAP_GPU_READ |
              NVSHMEM_TRANSPORT_CAP_GPU_ATOMICS;

    return status;
}

int nvshmemt_ibgda_get_mem_handle(nvshmem_mem_handle_t *mem_handle,
                                  nvshmem_mem_handle_t *mem_handle_in, void *buf, size_t length,
                                  nvshmem_transport_t t, bool local_only) {
    int status = 0;
    struct nvshmem_transport *transport = (struct nvshmem_transport *)t;
    nvshmemt_ibgda_state_t *ibgda_state = (nvshmemt_ibgda_state_t *)transport->state;
    size_t cumem_granularity = 1ULL << transport->log2_cumem_granularity;
    struct ibgda_device *device = ((struct ibgda_device *)ibgda_state->devices +
                                   ibgda_state->dev_ids[ibgda_state->selected_dev_id]);

    __be32 device_lkey;
    struct nvshmemt_ib_common_mem_handle *handle;

    nvshmemi_ibgda_device_local_only_mhandle_t *device_mhandle_d = NULL;
    bool did_emplace = false;

    nvshmemi_ibgda_device_state_t *ibgda_device_state;
    ibgda_device_state = (nvshmemi_ibgda_device_state_t *)transport->type_specific_shared_state;
    assert(ibgda_device_state != NULL);

    memset((void *)mem_handle, 0, sizeof(*mem_handle));
    status = nvshmemt_ib_common_reg_mem_handle(&ftable, device->pd, mem_handle, buf, length,
                                               local_only, ibgda_state->dmabuf_support,
                                               ibgda_cuda_syms, ibgda_state->log_level);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "Unable to register memory handle.\n");

    handle = (struct nvshmemt_ib_common_mem_handle *)mem_handle;
    device_lkey = htobe32(handle->lkey);

    if (local_only) {
        struct ibgda_device_local_only_mhandle_cache device_mhandle_cache;
        nvshmemi_ibgda_device_local_only_mhandle_t *device_mhandle_h =
            &device_mhandle_cache.mhandle;

        void *mhandle_gpu_ptr;

        cudaPointerAttributes buf_attributes;

        status = cudaPointerGetAttributes(&buf_attributes, buf);
        NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_INTERNAL, out,
                              "cudaPointerGetAttributes failed.\n");

        status = cudaMalloc((void **)&device_mhandle_d, sizeof(*device_mhandle_d));
        NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                              "cudaMalloc failed.\n");

        device_mhandle_h->lkey = device_lkey;
        device_mhandle_h->start = (uint64_t)buf;
        device_mhandle_h->end = (uint64_t)buf + length - 1;
        device_mhandle_h->is_sysmem_scope = (buf_attributes.type != cudaMemoryTypeDevice);
        device_mhandle_h->next = NULL;

        status = cudaMemcpyAsync((void *)device_mhandle_d, (const void *)device_mhandle_h,
                                 sizeof(*device_mhandle_d), cudaMemcpyHostToDevice,
                                 ibgda_state->my_stream);
        NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_INTERNAL, out,
                              "Copying device_mhandle to GPU memory failed.\n");

        device_mhandle_cache.dev_ptr = device_mhandle_d;

        if (ibgda_device_local_only_mhandles.empty()) {
            ibgda_device_state->globalmem.local_only_mhandle_head = device_mhandle_d;
        } else {
            struct ibgda_device_local_only_mhandle_cache *last_mhandle_cache =
                &ibgda_device_local_only_mhandles.back();
            mhandle_gpu_ptr = (void *)((uintptr_t)last_mhandle_cache->dev_ptr +
                                       offsetof(nvshmemi_ibgda_device_local_only_mhandle_t, next));
            last_mhandle_cache->mhandle.next = device_mhandle_d;
            status = cudaMemcpyAsync(mhandle_gpu_ptr, (const void *)&device_mhandle_d,
                                     sizeof(device_mhandle_d), cudaMemcpyHostToDevice,
                                     ibgda_state->my_stream);
            NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_INTERNAL, out,
                                  "Setting local_only_mhandle in GPU memory failed.\n");
        }

        ibgda_device_local_only_mhandles.emplace_back(device_mhandle_cache);
        did_emplace = true;
    } else {
        size_t num_lkeys;
        size_t num_elements;
        nvshmemi_ibgda_device_key_t device_key = {.key = device_lkey,
                                                  .next_addr = (uint64_t)buf + length};

        // length must be divisible by cumem_granularity, which is a power of 2.
        assert((length & (cumem_granularity - 1)) == 0);

        num_elements = length >> transport->log2_cumem_granularity;
        while (num_elements > 0) {
            ibgda_device_lkeys.emplace_back(device_key);
            --num_elements;
        }

        did_emplace = true;

        if (ibgda_device_lkeys_d) {
            status = cudaFree(ibgda_device_lkeys_d);
            NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_INTERNAL, out,
                                  "cudaFree failed.\n");
            ibgda_device_lkeys_d = 0;
        }

        num_lkeys = ibgda_device_lkeys.size();

        // Put lkeys in constant memory first for cache optimization
        memcpy(ibgda_device_state->constmem.lkeys, ibgda_device_lkeys.data(),
               IBGDA_MIN(num_lkeys, NVSHMEMI_IBGDA_MAX_CONST_LKEYS) *
                   sizeof(nvshmemi_ibgda_device_key_t));

        // If we have overflow, put the rest in global memory
        if (num_lkeys > NVSHMEMI_IBGDA_MAX_CONST_LKEYS) {
            size_t lkeys_array_size =
                sizeof(nvshmemi_ibgda_device_key_t) * (num_lkeys - NVSHMEMI_IBGDA_MAX_CONST_LKEYS);

            nvshmemi_ibgda_device_key_t *data_ptr =
                &ibgda_device_lkeys.data()[NVSHMEMI_IBGDA_MAX_CONST_LKEYS];

            status = cudaMalloc(&ibgda_device_lkeys_d, lkeys_array_size);
            NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                                  "cudaMalloc failed.\n");

            status = cudaMemcpyAsync(ibgda_device_lkeys_d, (const void *)data_ptr, lkeys_array_size,
                                     cudaMemcpyHostToDevice, ibgda_state->my_stream);
            NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_INTERNAL, out,
                                  "Copying lkeys to GPU memory failed.\n");
        }
        ibgda_device_state->globalmem.lkeys = (nvshmemi_ibgda_device_key_t *)ibgda_device_lkeys_d;
    }

    status = cudaStreamSynchronize(ibgda_state->my_stream);
    NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_INTERNAL, out,
                          "stream synchronize failed.\n");

out:
    if (status) {
        if (device_mhandle_d) cudaFree(device_mhandle_d);
        if (did_emplace) {
            if (local_only) {
                // Recoverable
                ibgda_device_local_only_mhandles.pop_back();
            } else {
                // Unrecoverable
                ibgda_device_lkeys.clear();
            }
        }
        nvshmemt_ib_common_release_mem_handle(&ftable, mem_handle, ibgda_state->log_level);
    }
    return status;
}

static int ibgda_mobject_nic_map(struct ibgda_mem_object *mobject, struct ibv_context *context,
                                 uint32_t access) {
    int status = 0;
    void *addr;
    struct mlx5dv_devx_umem *umem = NULL;

    assert(mobject);
    assert(!mobject->has_nic_mapping);
    assert(context);

    if (mobject->mem_type == IBGDA_MEM_TYPE_GPU) {
        addr = (void *)mobject->aligned.gpu_ptr;
    } else if (mobject->mem_type == IBGDA_MEM_TYPE_HOST) {
        addr = mobject->aligned.cpu_ptr;
    } else {
        assert(0);
    }

    umem = mlx5dv_devx_umem_reg(context, addr, mobject->aligned.size, access);
    if (!umem) {
        status = NVSHMEMX_ERROR_INTERNAL;
        goto out;
    }

    mobject->umem = umem;
    mobject->has_nic_mapping = true;

out:
    return status;
}

static void ibgda_mobject_nic_unmap(struct ibgda_mem_object *mobject) {
    int status = 0;

    assert(mobject);
    assert(mobject->has_nic_mapping);
    assert(mobject->mem_type != IBGDA_MEM_TYPE_NIC);
    assert(mobject->umem);

    status = mlx5dv_devx_umem_dereg(mobject->umem);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "mlx5dv_devx_umem_dereg failed.\n");

    mobject->has_nic_mapping = false;
    mobject->umem = NULL;

out:
    return;
}

static int ibgda_gpu_mem_alloc(struct ibgda_mem_object **pmobject, size_t size, size_t alignment,
                               bool host_mapping) {
    // TODO: Support host mapping through gdrcopy or dmabuf
    assert(!host_mapping);

    int status = 0;

    int attr_val;

    void *ptr = 0;
    void *aligned_ptr;
    size_t bufsize = size;

    struct ibgda_mem_object *mobject =
        (struct ibgda_mem_object *)calloc(1, sizeof(struct ibgda_mem_object));
    NVSHMEMI_NULL_ERROR_JMP(mobject, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "Unable to allocate a new mobject.\n");

    if (alignment > 0) bufsize = size + alignment - 1;

    status = cudaMalloc(&ptr, bufsize);
    NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_INTERNAL, out,
                          "cudaMalloc failed.\n");

    attr_val = 1;
    status =
        CUPFN(ibgda_cuda_syms,
              cuPointerSetAttribute(&attr_val, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, (CUdeviceptr)ptr));
    NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                          "cuPointerSetAttribute failed.\n");

    status = cudaMemset(ptr, 0, bufsize);
    NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_INTERNAL, out,
                          "cudaMemset failed.\n");

    if (alignment > 0) {
        aligned_ptr = (void *)((size_t)((char *)ptr + alignment - 1) & (~(alignment - 1)));
    } else {
        aligned_ptr = ptr;
    }

    mobject->mem_type = IBGDA_MEM_TYPE_GPU;

    mobject->base.gpu_ptr = ptr;
    mobject->base.size = bufsize;

    mobject->aligned.gpu_ptr = aligned_ptr;
    mobject->aligned.size = size;

    mobject->has_cpu_mapping = false;
    mobject->has_gpu_mapping = true;
    mobject->has_nic_mapping = false;

    *pmobject = mobject;

out:
    if (status) {
        if (ptr) {
            cudaError_t _status = cudaFree(ptr);
            CUDA_RUNTIME_ERROR_STRING(_status);
        }

        if (mobject) free(mobject);
    }
    return status;
}

static void ibgda_gpu_mem_free(struct ibgda_mem_object *mobject) {
    cudaError_t status;

    if (!mobject) return;

    assert(mobject->mem_type == IBGDA_MEM_TYPE_GPU);

    status = cudaFree(mobject->base.gpu_ptr);
    CUDA_RUNTIME_ERROR_STRING(status);

    free(mobject);
}

static int ibgda_host_mem_alloc(struct ibgda_mem_object **pmobject, size_t size, size_t alignment,
                                bool gpu_mapping) {
    int status;

    void *ptr = NULL;

    bool did_host_reg = false;
    void *gpu_ptr;

    struct ibgda_mem_object *mobject =
        (struct ibgda_mem_object *)calloc(1, sizeof(struct ibgda_mem_object));
    NVSHMEMI_NULL_ERROR_JMP(mobject, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "Unable to allocate a new mobject.\n");

    status = posix_memalign(&ptr, alignment, size);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out, "posix_memalign failed.\n");

    memset(ptr, 0, size);

    if (gpu_mapping) {
        status = cudaHostRegister(ptr, size, cudaHostRegisterPortable | cudaHostRegisterMapped);
        NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_INTERNAL, out,
                              "cudaHostRegister failed.\n");
        did_host_reg = true;

        status = cudaHostGetDevicePointer(&gpu_ptr, ptr, 0);
        NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_INTERNAL, out,
                              "cudaHostGetDevicePointer failed.\n");

        mobject->base.gpu_ptr = gpu_ptr;
        mobject->aligned.gpu_ptr = gpu_ptr;
        mobject->has_gpu_mapping = true;
    }

    mobject->base.cpu_ptr = ptr;
    mobject->base.size = size;

    mobject->aligned.cpu_ptr = ptr;
    mobject->aligned.size = size;

    mobject->has_cpu_mapping = true;

    *pmobject = mobject;

out:
    if (status) {
        if (did_host_reg) {
            cudaError_t _status = cudaHostUnregister(ptr);
            CUDA_RUNTIME_ERROR_STRING(_status);
        }
        if (ptr) free(ptr);
        if (mobject) free(mobject);
    }
    return status;
}

static void ibgda_host_mem_free(struct ibgda_mem_object *mobject) {
    cudaError_t status;

    if (!mobject) return;

    assert(mobject->mem_type == IBGDA_MEM_TYPE_HOST);

    if (mobject->has_gpu_mapping) {
        status = cudaHostUnregister(mobject->base.cpu_ptr);
        CUDA_RUNTIME_ERROR_STRING(status);
    }

    free(mobject->base.cpu_ptr);

    free(mobject);
}

static int ibgda_nic_mem_gpu_map(struct ibgda_mem_object **pmobject, struct mlx5dv_devx_uar *uar,
                                 size_t size) {
    int status = 0;
    bool did_host_reg = false;

    void *ptr = 0;

    struct ibgda_mem_object *mobject =
        (struct ibgda_mem_object *)calloc(1, sizeof(struct ibgda_mem_object));
    NVSHMEMI_NULL_ERROR_JMP(mobject, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "Unable to allocate a new mobject.\n");

    status = cudaHostRegister(
        uar->reg_addr, size,
        cudaHostRegisterPortable | cudaHostRegisterMapped | cudaHostRegisterIoMemory);
    NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_INTERNAL, out,
                          "cudaHostRegister failed.\n");
    did_host_reg = true;

    status = cudaHostGetDevicePointer(&ptr, uar->reg_addr, 0);
    NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_INTERNAL, out,
                          "cudaHostGetDevicePointer failed.\n");

    mobject->mem_type = IBGDA_MEM_TYPE_NIC;

    mobject->base.cpu_ptr = uar->reg_addr;
    mobject->base.gpu_ptr = ptr;
    mobject->base.size = size;

    mobject->aligned.cpu_ptr = uar->reg_addr;
    mobject->aligned.gpu_ptr = ptr;
    mobject->aligned.size = size;

    mobject->uar = uar;

    mobject->has_cpu_mapping = true;
    mobject->has_gpu_mapping = true;
    mobject->has_nic_mapping = true;

    *pmobject = mobject;

out:
    if (status) {
        if (did_host_reg) {
            cudaError_t _status = cudaHostUnregister(uar->reg_addr);
            CUDA_RUNTIME_ERROR_STRING(_status);
        }
        if (mobject) free(mobject);
    }
    return status;
}

static void ibgda_nic_mem_gpu_unmap(struct ibgda_mem_object *mobject) {
    cudaError_t status;

    if (!mobject) return;

    assert(mobject->mem_type == IBGDA_MEM_TYPE_NIC);

    status = cudaHostUnregister(mobject->uar->reg_addr);
    CUDA_RUNTIME_ERROR_STRING(status);

    free(mobject);
}

static inline int ibgda_nic_control_alloc(struct ibgda_mem_object **pmobject, size_t size,
                                          size_t alignment) {
    assert(ibgda_nic_buf_location == IBGDA_MEM_TYPE_GPU ||
           ibgda_nic_buf_location == IBGDA_MEM_TYPE_HOST);
    if (ibgda_nic_buf_location == IBGDA_MEM_TYPE_GPU)
        return ibgda_gpu_mem_alloc(pmobject, size, alignment, false);
    else
        return ibgda_host_mem_alloc(pmobject, size, alignment, true);
}

static inline void ibgda_nic_control_free(struct ibgda_mem_object *mobject) {
    assert(ibgda_nic_buf_location == IBGDA_MEM_TYPE_GPU ||
           ibgda_nic_buf_location == IBGDA_MEM_TYPE_HOST);
    if (ibgda_nic_buf_location == IBGDA_MEM_TYPE_GPU)
        ibgda_gpu_mem_free(mobject);
    else
        ibgda_host_mem_free(mobject);
}

static int ibgda_create_cq(struct ibgda_cq **pgcq, const struct ibgda_device *device, int ncqes) {
    int status = 0;

    struct ibgda_cq *gcq = NULL;

    struct ibv_pd *pd = device->pd;
    struct ibv_context *context = pd->context;

    void *cq_context;

    uint8_t cmd_in[DEVX_ST_SZ_BYTES(create_cq_in)] = {
        0,
    };
    uint8_t cmd_out[DEVX_ST_SZ_BYTES(create_cq_out)] = {
        0,
    };

    struct ibgda_mem_object *cq_mobject = NULL;
    struct mlx5dv_devx_umem *cq_umem = NULL;
    int num_cqe = IBGDA_ROUND_UP_POW2_OR_0(ncqes);
    size_t cq_buf_size = num_cqe * NVSHMEMI_IBGDA_CQE_SIZE;

    struct ibgda_mem_object *dbr_mobject = NULL;
    struct mlx5dv_devx_umem *dbr_umem = NULL;

    struct mlx5dv_devx_uar *uar = NULL;

    uint32_t eqn;

    gcq = (struct ibgda_cq *)calloc(1, sizeof(struct ibgda_cq));
    NVSHMEMI_NULL_ERROR_JMP(gcq, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "Unable to allocate mem for cq.\n");

    // Allocate and map CQ buffer
    status = ibgda_nic_control_alloc(&cq_mobject, cq_buf_size, IBGDA_GPAGE_SIZE);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "cannot allocate cq buf.\n");

    status = cudaMemset(cq_mobject->base.gpu_ptr, 0xff, cq_mobject->base.size);
    NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_INTERNAL, out,
                          "cudaMemset failed.\n");

    status = ibgda_mobject_nic_map(cq_mobject, context, IBV_ACCESS_LOCAL_WRITE);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "cannot register cq buf.\n");
    cq_umem = cq_mobject->umem;

    // Allocate and map DBR
    status = ibgda_nic_control_alloc(&dbr_mobject, IBGDA_DBSIZE, IBGDA_GPAGE_SIZE);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "cannot allocate dbr buf for qpair.\n");

    status = ibgda_mobject_nic_map(dbr_mobject, context, IBV_ACCESS_LOCAL_WRITE);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "cannot register dbr buf for qpair.\n");
    dbr_umem = dbr_mobject->umem;

    // Query the first EQ
    status = mlx5dv_devx_query_eqn(context, 0, &eqn);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "mlx5dv_devx_query_eqn failed.\n");

    // CQ needs UAR but IBGDA never uses it.
    // So, we don't map this UAR to GPU space.
    uar = mlx5dv_devx_alloc_uar(context, MLX5DV_UAR_ALLOC_TYPE_NC);
    NVSHMEMI_NULL_ERROR_JMP(uar, status, ENOMEM, out, "cannot allocate mlx5dv_devx_uar\n");

    DEVX_SET(create_cq_in, cmd_in, opcode, MLX5_CMD_OP_CREATE_CQ);
    DEVX_SET(create_cq_in, cmd_in, cq_umem_id, cq_umem->umem_id);  // CQ buffer
    DEVX_SET(create_cq_in, cmd_in, cq_umem_valid,
             IBGDA_MLX5_UMEM_VALID_ENABLE);  // Enable cq_umem_id
    DEVX_SET(create_cq_in, cmd_in, cq_umem_offset, 0x0);

    cq_context = DEVX_ADDR_OF(create_cq_in, cmd_in, cq_context);
    DEVX_SET(cqc, cq_context, dbr_umem_valid, IBGDA_MLX5_UMEM_VALID_ENABLE);
    DEVX_SET(cqc, cq_context, cqe_sz, MLX5_CQE_SIZE_64B);
    DEVX_SET(cqc, cq_context, cc, 0x1);  // Use collapsed CQ
    DEVX_SET(cqc, cq_context, oi, 0x1);  // Allow overrun
    DEVX_SET(cqc, cq_context, dbr_umem_id, dbr_umem->umem_id);
    DEVX_SET(cqc, cq_context, log_cq_size, IBGDA_ILOG2_OR0(num_cqe));
    DEVX_SET(cqc, cq_context, uar_page, uar->page_id);
    DEVX_SET(cqc, cq_context, c_eqn, eqn);
    DEVX_SET(cqc, cq_context, log_page_size, IBGDA_GPAGE_BITS - MLX5_ADAPTER_PAGE_SHIFT);
    DEVX_SET64(cqc, cq_context, dbr_addr, 0x0);  // DBR offset

    gcq->devx_cq =
        mlx5dv_devx_obj_create(context, cmd_in, sizeof(cmd_in), cmd_out, sizeof(cmd_out));
    NVSHMEMI_NULL_ERROR_JMP(gcq->devx_cq, status, NVSHMEMX_ERROR_INTERNAL, out,
                            "Unable to create CQ.\n");

    gcq->cqn = DEVX_GET(create_cq_out, cmd_out, cqn);
    gcq->num_cqe = num_cqe;
    gcq->cq_mobject = cq_mobject;
    gcq->dbr_mobject = dbr_mobject;
    gcq->uar = uar;

    *pgcq = gcq;

out:
    if (status) {
        if (uar) mlx5dv_devx_free_uar(uar);
        if (dbr_umem) ibgda_mobject_nic_unmap(dbr_mobject);
        if (dbr_mobject) ibgda_nic_control_free(dbr_mobject);
        if (cq_umem) ibgda_mobject_nic_unmap(cq_mobject);
        if (cq_mobject) ibgda_nic_control_free(cq_mobject);
        if (gcq) free(gcq);
    }
    return status;
}

static void ibgda_destroy_cq(struct ibgda_cq *gcq) {
    if (!gcq) return;

    if (gcq->devx_cq) {
        mlx5dv_devx_obj_destroy(gcq->devx_cq);
    }

    if (gcq->uar) {
        mlx5dv_devx_free_uar(gcq->uar);
    }

    if (gcq->dbr_mobject) {
        ibgda_mobject_nic_unmap(gcq->dbr_mobject);
        ibgda_nic_control_free(gcq->dbr_mobject);
    }

    if (gcq->cq_mobject) {
        ibgda_mobject_nic_unmap(gcq->cq_mobject);
        ibgda_nic_control_free(gcq->cq_mobject);
    }

    free(gcq);
}

static void ibgda_get_device_cq(nvshmemi_ibgda_device_cq_t *dev_cq, const struct ibgda_cq *cq) {
    dev_cq->cqn = cq->cqn;
    dev_cq->ncqes = cq->num_cqe;

    assert(cq->cq_mobject->has_gpu_mapping);
    dev_cq->cqe = (void *)cq->cq_mobject->aligned.gpu_ptr;

    assert(cq->dbr_mobject->has_gpu_mapping);
    dev_cq->dbrec = (__be32 *)cq->dbr_mobject->aligned.gpu_ptr;
}

static int ibgda_qp_rst2init(struct ibgda_ep *ep, const struct ibgda_device *device, int portid) {
    int status = 0;

    uint8_t cmd_in[DEVX_ST_SZ_BYTES(rst2init_qp_in)] = {
        0,
    };
    uint8_t cmd_out[DEVX_ST_SZ_BYTES(rst2init_qp_out)] = {
        0,
    };

    void *qpc;

    const struct ibv_port_attr *port_attr = device->port_attr + (portid - 1);

    assert(ep->qp_type == NVSHMEMI_IBGDA_DEVICE_QP_TYPE_DCI ||
           ep->qp_type == NVSHMEMI_IBGDA_DEVICE_QP_TYPE_RC);

    DEVX_SET(rst2init_qp_in, cmd_in, opcode, MLX5_CMD_OP_RST2INIT_QP);
    DEVX_SET(rst2init_qp_in, cmd_in, qpn, ep->qpn);

    qpc = DEVX_ADDR_OF(rst2init_qp_in, cmd_in, qpc);
    if (ep->qp_type == NVSHMEMI_IBGDA_DEVICE_QP_TYPE_DCI) {
        DEVX_SET64(qpc, qpc, dc_access_key, IBGDA_DC_ACCESS_KEY);
    } else if (ep->qp_type == NVSHMEMI_IBGDA_DEVICE_QP_TYPE_RC) {
        DEVX_SET(qpc, qpc, rwe, 1); /* remote write access */
        DEVX_SET(qpc, qpc, rre, 1); /* remote read access */
        DEVX_SET(qpc, qpc, rae, 1); /* remote atomic access */
        /* Currently, NVSHMEM APIs only support atomics up to 64. This field can be updated to
         * support atomics up to 256 bytes. */
        DEVX_SET(qpc, qpc, atomic_mode, IBGDA_MLX5_QPC_ATOMIC_MODE_UP_TO_64BIT);
    }

    DEVX_SET(qpc, qpc, primary_address_path.vhca_port_num, portid);

    if (port_attr->link_layer == IBV_LINK_LAYER_INFINIBAND)
        DEVX_SET(qpc, qpc, primary_address_path.pkey_index, 0);

    DEVX_SET(qpc, qpc, pm_state, MLX5_QPC_PM_STATE_MIGRATED);
    DEVX_SET(qpc, qpc, counter_set_id, 0x0);  // Not connected to a counter set

    status = mlx5dv_devx_obj_modify(ep->devx_qp, cmd_in, sizeof(cmd_in), cmd_out, sizeof(cmd_out));
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "Error in mlx5dv_devx_obj_modify for RST2INIT_QP with syndrome %x\n",
                          DEVX_GET(rst2init_qp_out, cmd_out, syndrome));

    ep->portid = portid;

out:
    return status;
}

static int ibgda_dci_init2rtr(nvshmemt_ibgda_state_t *ibgda_state, struct ibgda_ep *ep,
                              const struct ibgda_device *device, int portid) {
    int status = 0;

    uint8_t cmd_in[DEVX_ST_SZ_BYTES(init2rtr_qp_in)] = {
        0,
    };
    uint8_t cmd_out[DEVX_ST_SZ_BYTES(init2rtr_qp_out)] = {
        0,
    };

    void *qpc;

    const struct ibv_port_attr *port_attr = device->port_attr + (portid - 1);

    assert(ep->qp_type == NVSHMEMI_IBGDA_DEVICE_QP_TYPE_DCI);

    DEVX_SET(init2rtr_qp_in, cmd_in, opcode, MLX5_CMD_OP_INIT2RTR_QP);
    DEVX_SET(init2rtr_qp_in, cmd_in, qpn, ep->qpn);

    qpc = DEVX_ADDR_OF(init2rtr_qp_in, cmd_in, qpc);
    DEVX_SET(qpc, qpc, mtu, port_attr->active_mtu);
    DEVX_SET(qpc, qpc, log_msg_max, IBGDA_LOG_MAX_MSG_SIZE);

    if (port_attr->link_layer == IBV_LINK_LAYER_INFINIBAND) {
        DEVX_SET(qpc, qpc, primary_address_path.sl, ibgda_state->options->IB_SL);
    } else if (port_attr->link_layer == IBV_LINK_LAYER_ETHERNET) {
        DEVX_SET(qpc, qpc, primary_address_path.tclass, ibgda_state->options->IB_TRAFFIC_CLASS);
        DEVX_SET(qpc, qpc, primary_address_path.eth_prio, ibgda_state->options->IB_SL);
        DEVX_SET(qpc, qpc, primary_address_path.dscp, ibgda_state->options->IB_TRAFFIC_CLASS >> 2);
    }

    status = mlx5dv_devx_obj_modify(ep->devx_qp, cmd_in, sizeof(cmd_in), cmd_out, sizeof(cmd_out));
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "Error in mlx5dv_devx_obj_modify for INIT2RTR_QP with syndrome %x\n",
                          DEVX_GET(init2rtr_qp_out, cmd_out, syndrome));

out:
    return status;
}

static int ibgda_rc_init2rtr(nvshmemt_ibgda_state_t *ibgda_state, struct ibgda_ep *ep,
                             const struct ibgda_device *device, int portid,
                             struct ibgda_rc_handle *peer_ep_handle) {
    int status = 0;

    uint8_t cmd_in[DEVX_ST_SZ_BYTES(init2rtr_qp_in)] = {
        0,
    };
    uint8_t cmd_out[DEVX_ST_SZ_BYTES(init2rtr_qp_out)] = {
        0,
    };

    void *qpc;

    const struct ibv_port_attr *port_attr = device->port_attr + (portid - 1);

    assert(ep->qp_type == NVSHMEMI_IBGDA_DEVICE_QP_TYPE_RC);

    DEVX_SET(init2rtr_qp_in, cmd_in, opcode, MLX5_CMD_OP_INIT2RTR_QP);
    DEVX_SET(init2rtr_qp_in, cmd_in, qpn, ep->qpn);

    qpc = DEVX_ADDR_OF(init2rtr_qp_in, cmd_in, qpc);
    DEVX_SET(qpc, qpc, mtu, port_attr->active_mtu);
    DEVX_SET(qpc, qpc, log_msg_max, IBGDA_LOG_MAX_MSG_SIZE);
    DEVX_SET(qpc, qpc, remote_qpn, peer_ep_handle->qpn);
    DEVX_SET(qpc, qpc, min_rnr_nak, IBGDA_MIN_RNR_NAK);
    DEVX_SET(qpc, qpc, log_rra_max, IBGDA_ILOG2_OR0(device->device_attr.max_qp_rd_atom));

    if (port_attr->link_layer == IBV_LINK_LAYER_INFINIBAND) {
        DEVX_SET(qpc, qpc, primary_address_path.tclass, ibgda_state->options->IB_TRAFFIC_CLASS);
        DEVX_SET(qpc, qpc, primary_address_path.rlid, peer_ep_handle->lid);
        DEVX_SET(qpc, qpc, primary_address_path.mlid, 0);
        DEVX_SET(qpc, qpc, primary_address_path.sl, ibgda_state->options->IB_SL);
        DEVX_SET(qpc, qpc, primary_address_path.grh, false);
    } else if (port_attr->link_layer == IBV_LINK_LAYER_ETHERNET) {
        struct ibv_ah_attr ah_attr;
        struct ibv_ah *ah;
        struct mlx5dv_obj dv;
        struct mlx5dv_ah dah;

        ah_attr.is_global = 1;
        ah_attr.port_num = portid;
        ah_attr.grh.dgid.global.subnet_prefix = peer_ep_handle->spn;
        ah_attr.grh.dgid.global.interface_id = peer_ep_handle->iid;
        ah_attr.grh.sgid_index = ibgda_state->options->IB_GID_INDEX;
        ah_attr.grh.traffic_class = ibgda_state->options->IB_TRAFFIC_CLASS;
        ah_attr.sl = ibgda_state->options->IB_SL;
        ah_attr.src_path_bits = 0;

        ah = ftable.create_ah(device->pd, &ah_attr);
        NVSHMEMI_NULL_ERROR_JMP(ah, status, NVSHMEMX_ERROR_INTERNAL, out, "Unable to create ah.\n");

        dv.ah.in = ah;
        dv.ah.out = &dah;
        mlx5dv_init_obj(&dv, MLX5DV_OBJ_AH);

        memcpy(DEVX_ADDR_OF(qpc, qpc, primary_address_path.rmac_47_32), &dah.av->rmac,
               sizeof(dah.av->rmac));
        DEVX_SET(qpc, qpc, primary_address_path.hop_limit, IBGDA_GRH_HOP_LIMIT);
        DEVX_SET(qpc, qpc, primary_address_path.src_addr_index, ibgda_state->options->IB_GID_INDEX);
        DEVX_SET(qpc, qpc, primary_address_path.eth_prio, ibgda_state->options->IB_SL);
        DEVX_SET(qpc, qpc, primary_address_path.udp_sport, ah_attr.dlid);
        DEVX_SET(qpc, qpc, primary_address_path.dscp, ibgda_state->options->IB_TRAFFIC_CLASS >> 2);

        memcpy(DEVX_ADDR_OF(qpc, qpc, primary_address_path.rgid_rip), &dah.av->rgid,
               sizeof(dah.av->rgid));
        ep->ah = ah;
    }

    status = mlx5dv_devx_obj_modify(ep->devx_qp, cmd_in, sizeof(cmd_in), cmd_out, sizeof(cmd_out));
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "Error in mlx5dv_devx_obj_modify for INIT2RTR_QP with syndrome %x\n",
                          DEVX_GET(init2rtr_qp_out, cmd_out, syndrome));
out:
    return status;
}

static int ibgda_qp_rtr2rts(struct ibgda_ep *ep, const struct ibgda_device *device, int portid) {
    int status = 0;

    uint8_t cmd_in[DEVX_ST_SZ_BYTES(rtr2rts_qp_in)] = {
        0,
    };
    uint8_t cmd_out[DEVX_ST_SZ_BYTES(rtr2rts_qp_out)] = {
        0,
    };

    void *qpc;

    assert(ep->qp_type == NVSHMEMI_IBGDA_DEVICE_QP_TYPE_DCI ||
           ep->qp_type == NVSHMEMI_IBGDA_DEVICE_QP_TYPE_RC);

    DEVX_SET(rtr2rts_qp_in, cmd_in, opcode, MLX5_CMD_OP_RTR2RTS_QP);
    DEVX_SET(rtr2rts_qp_in, cmd_in, qpn, ep->qpn);

    qpc = DEVX_ADDR_OF(rtr2rts_qp_in, cmd_in, qpc);
    DEVX_SET(qpc, qpc, log_ack_req_freq, 0x0);  // Ack every packet
    DEVX_SET(qpc, qpc, log_sra_max, IBGDA_ILOG2_OR0(device->device_attr.max_qp_rd_atom));
    DEVX_SET(qpc, qpc, next_send_psn, 0x0);
    DEVX_SET(qpc, qpc, retry_count, 7);
    DEVX_SET(qpc, qpc, rnr_retry, 7);
    DEVX_SET(qpc, qpc, primary_address_path.ack_timeout, 20);

    status = mlx5dv_devx_obj_modify(ep->devx_qp, cmd_in, sizeof(cmd_in), cmd_out, sizeof(cmd_out));
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "Error in mlx5dv_devx_obj_modify for RTR2RTS_QP with syndrome %x\n",
                          DEVX_GET(rtr2rts_qp_out, cmd_out, syndrome));

out:
    return status;
}

static int ibgda_destroy_internal_buffer(nvshmemt_ibgda_state_t *ibgda_state,
                                         struct ibgda_device *device) {
    int status = 0;

    struct ibgda_mem_object *internal_buf_mobject = NULL;
    struct nvshmemt_ib_common_mem_handle *internal_buf_mhandle = NULL;

    internal_buf_mobject = device->qp_shared_object.internal_buf.mem_object;
    internal_buf_mhandle = device->qp_shared_object.internal_buf.mem_handle;

    if (internal_buf_mhandle) {
        nvshmemt_ib_common_release_mem_handle(&ftable, (nvshmem_mem_handle_t *)internal_buf_mhandle,
                                              ibgda_state->log_level);
        free(internal_buf_mhandle);
    }

    if (internal_buf_mobject) {
        ibgda_gpu_mem_free(internal_buf_mobject);
    }

    return status;
}

static int ibgda_create_internal_buffer(struct ibgda_internal_buffer *internal_buf,
                                        nvshmemt_ibgda_state_t *ibgda_state,
                                        struct ibgda_device *device, int n_pes) {
    int status = 0;

    struct ibgda_mem_object *internal_buf_mobject = NULL;
    struct nvshmemt_ib_common_mem_handle *internal_buf_mhandle = NULL;

    size_t size_per_dci =
        NVSHMEMI_IBGDA_IBUF_SLOT_SIZE * (ibgda_num_fetch_slots_per_dci + IBGDA_IBUF_RESERVED_SLOTS);
    size_t size_per_rc =
        NVSHMEMI_IBGDA_IBUF_SLOT_SIZE * (ibgda_num_fetch_slots_per_rc + IBGDA_IBUF_RESERVED_SLOTS);
    size_t buf_size =
        (size_per_dci * device->dci.num_eps) + (size_per_rc * device->rc.num_eps_per_pe * n_pes);

    status = ibgda_gpu_mem_alloc(&internal_buf_mobject, buf_size, IBGDA_GPAGE_SIZE, false);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "cannot allocate internal buffer.\n");

    internal_buf_mhandle =
        (struct nvshmemt_ib_common_mem_handle *)calloc(1, sizeof(*internal_buf_mhandle));
    NVSHMEMI_NULL_ERROR_JMP(internal_buf_mhandle, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "Unable to allocate internal_buf_mhandle.\n");

    status = nvshmemt_ib_common_reg_mem_handle(
        &ftable, device->pd, (nvshmem_mem_handle_t *)internal_buf_mhandle,
        (void *)internal_buf_mobject->aligned.gpu_ptr, internal_buf_mobject->aligned.size, false,
        ibgda_state->dmabuf_support, ibgda_cuda_syms, ibgda_state->log_level);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "Unable to register memory for IBGDA transport.\n");

    internal_buf->mem_object = internal_buf_mobject;
    internal_buf->mem_handle = internal_buf_mhandle;

out:
    if (status) {
        if (internal_buf_mhandle) {
            nvshmemt_ib_common_release_mem_handle(
                &ftable, (nvshmem_mem_handle_t *)internal_buf_mhandle, ibgda_state->log_level);
            free(internal_buf_mhandle);
        }
        if (internal_buf_mobject) ibgda_gpu_mem_free(internal_buf_mobject);
    }
    return status;
}

static int ibgda_destroy_qp_shared_objects(nvshmemt_ibgda_state_t *ibgda_state,
                                           struct ibgda_device *device) {
    int status = 0;

    status = ibgda_destroy_internal_buffer(ibgda_state, device);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "ibgda_destroy_internal_buffer failed.\n");

    if (device->qp_shared_object.recv_cq) {
        status = ftable.destroy_cq(device->qp_shared_object.recv_cq);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "ibgda_destroy_cq failed for recv_cq.\n");
    }

    if (device->qp_shared_object.srq) {
        status = ftable.destroy_srq(device->qp_shared_object.srq);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "destroy_srq failed.\n");
    }

out:
    return status;
}

static int ibgda_create_qp_shared_objects(nvshmemt_ibgda_state_t *ibgda_state,
                                          struct ibgda_device *device, int n_pes) {
    int status = 0;

    struct ibv_context *context = device->context;
    struct ibv_pd *pd = device->pd;

    struct ibv_srq *srq = NULL;
    struct ibv_srq_init_attr srq_init_attr;

    struct ibv_cq *recv_cq = NULL;

    mlx5dv_obj dv_obj;
    struct mlx5dv_pd dvpd;
    struct mlx5dv_cq dvscq;
    struct mlx5dv_cq dvrcq;
    struct mlx5dv_srq dvsrq;

    int pdn = 0;
    int srqn = 0;
    int rcqn = 0;

    // Initialization
    memset(&srq_init_attr, 0, sizeof(srq_init_attr));
    memset(&dvpd, 0, sizeof(dvpd));
    memset(&dvscq, 0, sizeof(dvscq));
    memset(&dvrcq, 0, sizeof(dvrcq));
    memset(&dvsrq, 0, sizeof(dvsrq));

    // Query pdn
    memset(&dv_obj, 0, sizeof(dv_obj));
    dv_obj.pd.in = pd;
    dv_obj.pd.out = &dvpd;

    status = mlx5dv_init_obj(&dv_obj, MLX5DV_OBJ_PD);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "mlx5dv PD initialization failed.\n");

    pdn = dvpd.pdn;

    // Create srq on host memory.
    srq_init_attr.attr.max_wr = ibgda_srq_depth;
    srq_init_attr.attr.max_sge = 1;

    srq = ftable.create_srq(pd, &srq_init_attr);
    NVSHMEMI_NULL_ERROR_JMP(srq, status, NVSHMEMX_ERROR_INTERNAL, out, "ibv_create_srq failed.\n");

    memset(&dv_obj, 0, sizeof(dv_obj));
    dvsrq.comp_mask = MLX5DV_SRQ_MASK_SRQN;
    dv_obj.srq.in = srq;
    dv_obj.srq.out = &dvsrq;

    status = mlx5dv_init_obj(&dv_obj, MLX5DV_OBJ_SRQ);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "mlx5dv SRQ initialization failed.\n");

    srqn = dvsrq.srqn;
    NVSHMEMI_EQ_ERROR_JMP(srqn, 0, NVSHMEMX_ERROR_INTERNAL, out,
                          "Unable to allocate SRQ for your device. "
                          "This may occur if your ofed is older than version 5.0.\n");

    // Create recv_cq on host memory.
    recv_cq = ftable.create_cq(context, ibgda_srq_depth, NULL, NULL, 0);
    NVSHMEMI_NULL_ERROR_JMP(recv_cq, status, NVSHMEMX_ERROR_INTERNAL, out,
                            "ibv_create_cq for recv_cq failed.\n");

    memset(&dv_obj, 0, sizeof(dv_obj));
    dv_obj.cq.in = recv_cq;
    dv_obj.cq.out = &dvrcq;

    status = mlx5dv_init_obj(&dv_obj, MLX5DV_OBJ_CQ);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "mlx5dv RCQ initialization failed.\n");

    rcqn = dvrcq.cqn;

    status = ibgda_create_internal_buffer(&device->qp_shared_object.internal_buf, ibgda_state,
                                          device, n_pes);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "ibgda_create_internal_buffer failed.\n");

    // Output
    device->qp_shared_object.srq = srq;
    device->qp_shared_object.recv_cq = recv_cq;
    device->qp_shared_object.pdn = pdn;
    device->qp_shared_object.srqn = srqn;
    device->qp_shared_object.rcqn = rcqn;

out:
    if (status) {
        if (recv_cq) ftable.destroy_cq(recv_cq);
        if (srq) ftable.destroy_srq(srq);
    }
    return status;
}

static int ibgda_alloc_and_map_qp_uar(struct ibv_context *context,
                                      struct ibgda_mem_object **out_mobject) {
    int status = 0;

    struct mlx5dv_devx_uar *uar = NULL;
    struct ibgda_mem_object *uar_mobject = NULL;
    size_t uar_reg_size = 0;
    uint8_t log_bf_reg_size = 0;

#ifdef HAVE_MLX5DV_UAR_ALLOC_TYPE_NC_DEDICATED
    uar = mlx5dv_devx_alloc_uar(context, MLX5DV_UAR_ALLOC_TYPE_NC_DEDICATED);
    if (uar)
        uar_reg_size = IBGDA_MLX5_NC_UAR_SIZE;
    else
#endif
    {
        uint8_t cmd_cap_in[DEVX_ST_SZ_BYTES(query_hca_cap_in)] = {
            0,
        };
        uint8_t cmd_cap_out[DEVX_ST_SZ_BYTES(query_hca_cap_out)] = {
            0,
        };
        void *cap;

        DEVX_SET(query_hca_cap_in, cmd_cap_in, opcode, MLX5_CMD_OP_QUERY_HCA_CAP);
        DEVX_SET(query_hca_cap_in, cmd_cap_in, op_mod,
                 MLX5_SET_HCA_CAP_OP_MOD_GENERAL_DEVICE | (MLX5_CAP_GENERAL << 1) |
                     HCA_CAP_OPMOD_GET_CUR);

        status = mlx5dv_devx_general_cmd(context, cmd_cap_in, sizeof(cmd_cap_in), cmd_cap_out,
                                         sizeof(cmd_cap_out));
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "mlx5dv_devx_general_cmd for hca cap failed.\n");

        cap = DEVX_ADDR_OF(query_hca_cap_out, cmd_cap_out, capability.cmd_hca_cap);
        log_bf_reg_size = DEVX_GET(cmd_hca_cap, cap, log_bf_reg_size);

        // The size of 1st + 2nd half (as when we use alternating DB)
        uar_reg_size = 1LLU << log_bf_reg_size;

        // Allocate UAR. This will be used as a DB/BF register.
        uar = mlx5dv_devx_alloc_uar(context, MLX5DV_UAR_ALLOC_TYPE_BF);
        NVSHMEMI_NULL_ERROR_JMP(uar, status, ENOMEM, out, "cannot allocate mlx5dv_devx_uar\n");
    }

    // Map the UAR to GPU
    status = ibgda_nic_mem_gpu_map(&uar_mobject, uar, uar_reg_size);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "ibgda_nic_mem_gpu_map failed.\n");

    *out_mobject = uar_mobject;

out:
    if (status) {
        if (uar_mobject) ibgda_nic_mem_gpu_unmap(uar_mobject);
        if (uar) mlx5dv_devx_free_uar(uar);
    }
    return status;
}

static void ibgda_unmap_and_free_qp_uar(struct ibgda_mem_object *mobject) {
    struct mlx5dv_devx_uar *uar = NULL;

    if (!mobject) return;

    uar = mobject->uar;

    ibgda_nic_mem_gpu_unmap(mobject);

    if (uar) mlx5dv_devx_free_uar(uar);
}

/**
 * Create a RC or DCI QP.
 * DCT creation is not handled by this function.
 */
static int ibgda_create_qp(struct ibgda_ep **ep_ptr, struct ibgda_device *device, int portid,
                           uint32_t qp_idx, nvshmemi_ibgda_device_qp_type_t qp_type) {
    struct ibv_pd *pd = device->pd;
    struct ibv_context *context = pd->context;
    struct ibgda_ep *ep = NULL;

    void *qp_context;

    uint8_t cmd_in[DEVX_ST_SZ_BYTES(create_qp_in)] = {
        0,
    };
    uint8_t cmd_out[DEVX_ST_SZ_BYTES(create_qp_out)] = {
        0,
    };

    uint8_t cmd_cap_in[DEVX_ST_SZ_BYTES(query_hca_cap_in)] = {
        0,
    };
    uint8_t cmd_cap_out[DEVX_ST_SZ_BYTES(query_hca_cap_out)] = {
        0,
    };
    void *cap;

    struct ibgda_mem_object *uar_mobject = NULL;

    size_t wq_buf_size;
    struct ibgda_mem_object *wq_mobject = NULL;
    struct mlx5dv_devx_umem *wq_umem = NULL;

    struct ibgda_mem_object *dbr_mobject = NULL;
    struct mlx5dv_devx_umem *dbr_umem = NULL;

    int cqe_version = 0;

    struct ibgda_cq *send_cq = NULL;

    size_t num_wqebb = IBGDA_ROUND_UP_POW2_OR_0(ibgda_qp_depth);

    int status = 0;

    assert(qp_type == NVSHMEMI_IBGDA_DEVICE_QP_TYPE_DCI ||
           qp_type == NVSHMEMI_IBGDA_DEVICE_QP_TYPE_RC);

    DEVX_SET(query_hca_cap_in, cmd_cap_in, opcode, MLX5_CMD_OP_QUERY_HCA_CAP);
    DEVX_SET(
        query_hca_cap_in, cmd_cap_in, op_mod,
        MLX5_SET_HCA_CAP_OP_MOD_GENERAL_DEVICE | (MLX5_CAP_GENERAL << 1) | HCA_CAP_OPMOD_GET_CUR);

    status = mlx5dv_devx_general_cmd(context, cmd_cap_in, sizeof(cmd_cap_in), cmd_cap_out,
                                     sizeof(cmd_cap_out));
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "mlx5dv_devx_general_cmd for hca cap failed.\n");

    cap = DEVX_ADDR_OF(query_hca_cap_out, cmd_cap_out, capability.cmd_hca_cap);
    cqe_version = DEVX_GET(cmd_hca_cap, cap, cqe_version);
    if (cqe_version != 1) {
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_NOT_SUPPORTED, out,
                           "hca_cap.cqe_version != 1 is not supported.\n");
    }

    // Create send_cq on GPU memory.
    status = ibgda_create_cq(&send_cq, device, ibgda_qp_depth);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "ibgda_create_cq failed.\n");

    ep = (struct ibgda_ep *)calloc(1, sizeof(struct ibgda_ep));
    NVSHMEMI_NULL_ERROR_JMP(ep, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "Unable to allocate mem for ep.\n");

    // Allocate and map UAR. This will be used as a DB/BF register.
    status = ibgda_alloc_and_map_qp_uar(context, &uar_mobject);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "ibgda_alloc_and_map_qp_uar failed\n");

    // Allocate WQ buffer.
    wq_buf_size = num_wqebb * MLX5_SEND_WQE_BB;  // num_wqebb is always a power of 2
    status = ibgda_nic_control_alloc(&wq_mobject, wq_buf_size, IBGDA_GPAGE_SIZE);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "cannot allocate wq buf for qpair.\n");

    status = ibgda_mobject_nic_map(wq_mobject, context, IBV_ACCESS_LOCAL_WRITE);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "cannot register wq buf for qpair.\n");
    wq_umem = wq_mobject->umem;

    // Allocate Doorbell Register buffer.
    status = ibgda_nic_control_alloc(&dbr_mobject, IBGDA_DBSIZE, IBGDA_GPAGE_SIZE);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "cannot allocate dbr buf for qpair.\n");

    status = ibgda_mobject_nic_map(dbr_mobject, context, IBV_ACCESS_LOCAL_WRITE);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "cannot register dbr buf for qpair.\n");
    dbr_umem = dbr_mobject->umem;

    DEVX_SET(create_qp_in, cmd_in, opcode, MLX5_CMD_OP_CREATE_QP);
    DEVX_SET(create_qp_in, cmd_in, wq_umem_id, wq_umem->umem_id);  // WQ buffer
    DEVX_SET(create_qp_in, cmd_in, wq_umem_valid,
             IBGDA_MLX5_UMEM_VALID_ENABLE);  // Enable wq_umem_id

    qp_context = DEVX_ADDR_OF(create_qp_in, cmd_in, qpc);
    DEVX_SET(qpc, qp_context, st,
             qp_type == NVSHMEMI_IBGDA_DEVICE_QP_TYPE_DCI ? IBGDA_MLX5_QPC_ST_DCI
                                                          : IBGDA_MLX5_QPC_ST_RC);
    DEVX_SET(qpc, qp_context, pm_state, MLX5_QPC_PM_STATE_MIGRATED);
    DEVX_SET(qpc, qp_context, pd, device->qp_shared_object.pdn);
    DEVX_SET(qpc, qp_context, uar_page, uar_mobject->uar->page_id);  // BF register
    DEVX_SET(qpc, qp_context, rq_type, IBGDA_SRQ_TYPE_VALUE);        // Shared Receive Queue
    DEVX_SET(qpc, qp_context, srqn_rmpn_xrqn, device->qp_shared_object.srqn);
    DEVX_SET(qpc, qp_context, cqn_snd, send_cq->cqn);
    DEVX_SET(qpc, qp_context, cqn_rcv, device->qp_shared_object.rcqn);
    DEVX_SET(qpc, qp_context, log_sq_size, IBGDA_ILOG2_OR0(num_wqebb));
    DEVX_SET(qpc, qp_context, log_rq_size, 0);
    DEVX_SET(qpc, qp_context, cs_req, 0);                                     // Disable CS Request
    DEVX_SET(qpc, qp_context, cs_res, 0);                                     // Disable CS Response
    DEVX_SET(qpc, qp_context, dbr_umem_valid, IBGDA_MLX5_UMEM_VALID_ENABLE);  // Enable dbr_umem_id
    DEVX_SET64(qpc, qp_context, dbr_addr,
               0);  // Offset 0 of dbr_umem_id (behavior changed because of dbr_umem_valid)
    DEVX_SET(qpc, qp_context, dbr_umem_id, dbr_umem->umem_id);  // DBR buffer
    DEVX_SET(qpc, qp_context, user_index, qp_idx);
    DEVX_SET(qpc, qp_context, page_offset, 0);

    ep->devx_qp = mlx5dv_devx_obj_create(context, cmd_in, sizeof(cmd_in), cmd_out, sizeof(cmd_out));
    NVSHMEMI_NULL_ERROR_JMP(ep->devx_qp, status, NVSHMEMX_ERROR_INTERNAL, out,
                            "Unable to create QP for EP.\n");

    ep->qpn = DEVX_GET(create_qp_out, cmd_out, qpn);
    ep->portid = portid;

    ep->sq_cnt = num_wqebb;
    ep->sq_buf_offset = 0;

    ep->rq_cnt = 0;
    ep->rq_buf_offset = 0;

    ep->wq_mobject = wq_mobject;
    ep->dbr_mobject = dbr_mobject;
    ep->uar_mobject = uar_mobject;

    ep->send_cq = send_cq;

    ep->qp_type = qp_type;

    ep->user_index = qp_idx;

    *ep_ptr = ep;

out:
    if (status) {
        if (dbr_umem) ibgda_mobject_nic_unmap(dbr_mobject);
        if (dbr_mobject) ibgda_nic_control_free(dbr_mobject);
        if (wq_umem) ibgda_mobject_nic_unmap(wq_mobject);
        if (wq_mobject) ibgda_nic_control_free(wq_mobject);
        if (uar_mobject) ibgda_unmap_and_free_qp_uar(uar_mobject);
        if (send_cq) ibgda_destroy_cq(send_cq);
        if (ep) free(ep);
    }

    return status;
}

static int ibgda_get_rc_handle(struct ibgda_rc_handle *rc_handle, const struct ibgda_ep *ep,
                               const struct ibgda_device *device) {
    const struct ibv_port_attr *port_attr = &device->port_attr[ep->portid - 1];
    const union ibv_gid *gid = &device->gid[ep->portid - 1];

    assert(ep->qp_type == NVSHMEMI_IBGDA_DEVICE_QP_TYPE_RC);

    rc_handle->qpn = ep->qpn;
    rc_handle->lid = port_attr->lid;
    if (rc_handle->lid == 0) {
        rc_handle->spn = gid->global.subnet_prefix;
        rc_handle->iid = gid->global.interface_id;
    }

    return 0;
}

static int ibgda_destroy_dct_shared_objects(nvshmemt_ibgda_state_t *ibgda_state,
                                            struct ibgda_device *device) {
    int status = 0;

    if (device->dct.ah) {
        status = ftable.destroy_ah(device->dct.ah);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "destroy_ah failed.\n");
    }
    if (device->dct.send_cq) {
        status = ftable.destroy_cq(device->dct.send_cq);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "destroy_cq failed for send_cq.\n");
    }
    if (device->dct.recv_cq) {
        status = ftable.destroy_cq(device->dct.recv_cq);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "destroy_cq failed for recv_cq.\n");
    }
    if (device->dct.pd) {
        status = ftable.dealloc_pd(device->dct.pd);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "ibv_dealloc_pd failed for dct.pd.\n");
    }

out:
    return status;
}

static int ibgda_create_dct_shared_objects(nvshmemt_ibgda_state_t *ibgda_state,
                                           struct ibgda_device *device, int portid) {
    int status = 0;

    const struct ibv_port_attr *port_attr = device->port_attr + (portid - 1);
    struct ibv_context *context = device->context;

    struct ibv_pd *pd = NULL;
    struct ibv_parent_domain_init_attr pd_init_attr;

    struct ibv_srq *srq = NULL;
    struct ibv_srq_init_attr srq_init_attr;

    struct ibv_cq *send_cq = NULL;
    struct ibv_cq *recv_cq = NULL;

    struct ibv_ah *ah = NULL;
    struct mlx5dv_ah dah;
    struct ibv_ah_attr ah_attr;
    struct mlx5dv_obj dv;

    bool support_half_av_seg;
    int hca_support_compact_address_vector;

    uint8_t cmd_cap_in[DEVX_ST_SZ_BYTES(query_hca_cap_in)] = {
        0,
    };
    uint8_t cmd_cap_out[DEVX_ST_SZ_BYTES(query_hca_cap_out)] = {
        0,
    };
    void *cap;

    DEVX_SET(query_hca_cap_in, cmd_cap_in, opcode, MLX5_CMD_OP_QUERY_HCA_CAP);
    DEVX_SET(
        query_hca_cap_in, cmd_cap_in, op_mod,
        MLX5_SET_HCA_CAP_OP_MOD_GENERAL_DEVICE | (MLX5_CAP_GENERAL << 1) | HCA_CAP_OPMOD_GET_CUR);

    status = mlx5dv_devx_general_cmd(context, cmd_cap_in, sizeof(cmd_cap_in), cmd_cap_out,
                                     sizeof(cmd_cap_out));
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "mlx5dv_devx_general_cmd for hca cap failed.\n");

    cap = DEVX_ADDR_OF(query_hca_cap_out, cmd_cap_out, capability.cmd_hca_cap);
    hca_support_compact_address_vector = DEVX_GET(cmd_hca_cap, cap, compact_address_vector);

    memset(&pd_init_attr, 0, sizeof(pd_init_attr));
    memset(&srq_init_attr, 0, sizeof(srq_init_attr));

    pd_init_attr.pd = device->pd;
    pd = ibv_alloc_parent_domain(context, &pd_init_attr);
    NVSHMEMI_NULL_ERROR_JMP(pd, status, NVSHMEMX_ERROR_INTERNAL, out,
                            "ibv_alloc_parent_domain failed.\n");

    srq_init_attr.attr.max_wr = ibgda_srq_depth;
    srq_init_attr.attr.max_sge = 1;

    srq = ftable.create_srq(pd, &srq_init_attr);
    NVSHMEMI_NULL_ERROR_JMP(srq, status, NVSHMEMX_ERROR_INTERNAL, out, "ibv_create_srq failed.\n");

    send_cq = ftable.create_cq(context, ibgda_srq_depth, NULL, NULL, 0);
    NVSHMEMI_NULL_ERROR_JMP(send_cq, status, NVSHMEMX_ERROR_INTERNAL, out,
                            "ibv_create_cq for send_cq failed.\n");

    recv_cq = ftable.create_cq(context, ibgda_srq_depth, NULL, NULL, 0);
    NVSHMEMI_NULL_ERROR_JMP(recv_cq, status, NVSHMEMX_ERROR_INTERNAL, out,
                            "ibv_create_cq for recv_cq failed.\n");

    if (port_attr->lid == 0) {
        ah_attr.is_global = 1;
        ah_attr.grh.dgid.global.subnet_prefix = device->gid[portid - 1].global.subnet_prefix;
        ah_attr.grh.dgid.global.interface_id = device->gid[portid - 1].global.interface_id;
        ah_attr.grh.flow_label = 0;
        ah_attr.grh.sgid_index = ibgda_state->options->IB_GID_INDEX;
        ah_attr.grh.traffic_class = ibgda_state->options->IB_TRAFFIC_CLASS;
        ah_attr.grh.hop_limit = IBGDA_GRH_HOP_LIMIT;
        support_half_av_seg = false;
    } else {
        // Only IB supports is_global = 0.
        assert(port_attr->link_layer == IBV_LINK_LAYER_INFINIBAND);
        ah_attr.dlid = port_attr->lid;
        ah_attr.is_global = 0;
        support_half_av_seg = hca_support_compact_address_vector;
    }
    ah_attr.sl = ibgda_state->options->IB_SL;
    ah_attr.src_path_bits = 0;
    ah_attr.port_num = portid;

    ah = ftable.create_ah(device->pd, &ah_attr);
    NVSHMEMI_NULL_ERROR_JMP(ah, status, NVSHMEMX_ERROR_INTERNAL, out, "Unable to create ah.\n");

    dv.ah.in = ah;
    dv.ah.out = &dah;
    mlx5dv_init_obj(&dv, MLX5DV_OBJ_AH);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "mlx5dv AH initialization failed.\n");

    device->dct.pd = pd;
    device->dct.srq = srq;
    device->dct.send_cq = send_cq;
    device->dct.recv_cq = recv_cq;
    device->dct.ah = ah;
    memcpy(&device->dct.dah, &dah, sizeof(dah));
    memcpy(&device->dct.ah_attr, &ah_attr, sizeof(ah_attr));
    device->support_half_av_seg = support_half_av_seg;

out:
    if (status) {
        if (ah) ftable.destroy_ah(ah);
        if (recv_cq) ftable.destroy_cq(recv_cq);
        if (send_cq) ftable.destroy_cq(send_cq);
        if (srq) ftable.destroy_srq(srq);
    }
    return status;
}

static int ibgda_create_dct(nvshmemt_ibgda_state_t *ibgda_state, struct ibgda_ep **ep_ptr,
                            const struct ibgda_device *device, int portid) {
    int status = 0;

    struct ibgda_ep *ep = NULL;
    struct ibv_qp *ib_qp = NULL;

    struct ibv_qp_init_attr_ex ib_qp_attr_ex;
    struct mlx5dv_qp_init_attr dv_init_attr;
    struct ibv_qp_attr ib_qp_attr;

    const struct ibv_port_attr *port_attr = device->port_attr + (portid - 1);

    memset(&ib_qp_attr_ex, 0, sizeof(ib_qp_attr_ex));
    memset(&dv_init_attr, 0, sizeof(dv_init_attr));

    ep = (struct ibgda_ep *)calloc(1, sizeof(struct ibgda_ep));
    NVSHMEMI_NULL_ERROR_JMP(ep, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "Unable to allocate mem for ep.\n");

    dv_init_attr.comp_mask = MLX5DV_QP_INIT_ATTR_MASK_DC;
    dv_init_attr.dc_init_attr.dc_type = MLX5DV_DCTYPE_DCT;
    dv_init_attr.dc_init_attr.dct_access_key = IBGDA_DC_ACCESS_KEY;

    ib_qp_attr_ex.pd = device->dct.pd;
    ib_qp_attr_ex.comp_mask = IBV_QP_INIT_ATTR_PD;
    ib_qp_attr_ex.qp_type = IBV_QPT_DRIVER;
    ib_qp_attr_ex.srq = device->dct.srq;
    ib_qp_attr_ex.send_cq = device->dct.send_cq;
    ib_qp_attr_ex.recv_cq = device->dct.recv_cq;

    ib_qp_attr_ex.cap.max_send_wr = ibgda_state->options->QP_DEPTH;
    ib_qp_attr_ex.cap.max_recv_wr = ibgda_state->options->QP_DEPTH;
    ib_qp_attr_ex.cap.max_send_sge = 1;
    ib_qp_attr_ex.cap.max_recv_sge = 1;
    ib_qp_attr_ex.cap.max_inline_data = NVSHMEMI_IBGDA_MAX_INLINE_SIZE;

    ib_qp = mlx5dv_create_qp(device->context, &ib_qp_attr_ex, &dv_init_attr);
    NVSHMEMI_NULL_ERROR_JMP(ib_qp, status, NVSHMEMX_ERROR_INTERNAL, out,
                            "mlx5dv_create_qp failed.\n");

    // RST2INIT
    memset(&ib_qp_attr, 0, sizeof(ib_qp_attr));
    ib_qp_attr.qp_state = IBV_QPS_INIT;
    ib_qp_attr.pkey_index = 0;
    ib_qp_attr.port_num = portid;
    ib_qp_attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ |
                                 IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_ATOMIC;

    status = ftable.modify_qp(ib_qp, &ib_qp_attr,
                              IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "ibv_modify_qp rst2init for dct failed.\n");

    // INIT2RTR
    memset(&ib_qp_attr, 0, sizeof(ib_qp_attr));
    ib_qp_attr.qp_state = IBV_QPS_RTR;
    ib_qp_attr.path_mtu = port_attr->active_mtu;
    ib_qp_attr.min_rnr_timer = 12;
    memcpy(&ib_qp_attr.ah_attr, &device->dct.ah_attr, sizeof(ib_qp_attr.ah_attr));

    status = ftable.modify_qp(ib_qp, &ib_qp_attr,
                              IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_MIN_RNR_TIMER);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "ibv_modify_qp init2rtr for dct failed.\n");

    ep->qp_type = NVSHMEMI_IBGDA_DEVICE_QP_TYPE_DCT;

    ep->ib_qp = ib_qp;
    ep->qpn = ib_qp->qp_num;
    ep->portid = portid;

    *ep_ptr = ep;

out:
    if (status) {
        if (ib_qp) {
            int _status = ftable.destroy_qp(ib_qp);
            if (_status) NVSHMEMI_ERROR_PRINT("ibv_destroy_qp for dct failed.\n");
        }
        if (ep) free(ep);
    }
    return status;
}

static int ibgda_get_dct_handle(struct ibgda_dct_handle *dct_handle, const struct ibgda_ep *ep,
                                const struct ibgda_device *device) {
    assert(ep->qp_type == NVSHMEMI_IBGDA_DEVICE_QP_TYPE_DCT);

    memcpy(&dct_handle->dev_dct, device->dct.dah.av, sizeof(dct_handle->dev_dct));
    // Don't do htobe32 here as we need to determine whether the ext field should be set or not.
    dct_handle->dev_dct.dqp_dct = ep->qpn;
    dct_handle->dev_dct.key.dc_key = htobe64(IBGDA_DC_ACCESS_KEY);
    dct_handle->support_half_av_seg = device->support_half_av_seg;

    return 0;
}

static int ibgda_destroy_ep(struct ibgda_ep *ep) {
    int status = 0;

    if (!ep) return status;

    if (ep->qp_type == NVSHMEMI_IBGDA_DEVICE_QP_TYPE_DCT) {
        if (ep->ib_qp) {
            int status = ftable.destroy_qp(ep->ib_qp);
            if (status) NVSHMEMI_ERROR_PRINT("ibv_destroy_qp failed.\n");
        }
    } else {
        if (ep->devx_qp) {
            mlx5dv_devx_obj_destroy(ep->devx_qp);
        }

        if (ep->dbr_mobject) {
            ibgda_mobject_nic_unmap(ep->dbr_mobject);
            ibgda_nic_control_free(ep->dbr_mobject);
        }

        if (ep->uar_mobject) {
            struct mlx5dv_devx_uar *uar = ep->uar_mobject->uar;
            ibgda_nic_mem_gpu_unmap(ep->uar_mobject);
            mlx5dv_devx_free_uar(uar);
        }

        if (ep->wq_mobject) {
            if (ep->wq_mobject->uar) ibgda_mobject_nic_unmap(ep->wq_mobject);
            ibgda_nic_control_free(ep->wq_mobject);
        }
        ibgda_destroy_cq(ep->send_cq);
    }
    free(ep);

    return status;
}

static void ibgda_get_device_qp_mvars(nvshmemi_ibgda_device_qp_management_t *dev_mvars,
                                      struct ibgda_device *device, const struct ibgda_ep *ep) {
    memset(dev_mvars, 0, sizeof(*dev_mvars));
}

static void ibgda_get_device_qp(nvshmemi_ibgda_device_qp_t *dev_qp, struct ibgda_device *device,
                                const struct ibgda_ep *ep) {
    uintptr_t ibuf_dci_start;
    uintptr_t ibuf_rc_start;
    void *ibuf_ptr;

    size_t size_per_dci =
        NVSHMEMI_IBGDA_IBUF_SLOT_SIZE * (ibgda_num_fetch_slots_per_dci + IBGDA_IBUF_RESERVED_SLOTS);
    size_t size_per_rc =
        NVSHMEMI_IBGDA_IBUF_SLOT_SIZE * (ibgda_num_fetch_slots_per_rc + IBGDA_IBUF_RESERVED_SLOTS);

    assert(ep->qp_type == NVSHMEMI_IBGDA_DEVICE_QP_TYPE_DCI ||
           ep->qp_type == NVSHMEMI_IBGDA_DEVICE_QP_TYPE_RC);

    dev_qp->qpn = ep->qpn;

    assert(ep->wq_mobject->has_gpu_mapping);
    dev_qp->tx_wq.wqe = (void *)ep->wq_mobject->aligned.gpu_ptr;

    assert(ep->dbr_mobject->has_gpu_mapping);
    dev_qp->tx_wq.dbrec = (__be32 *)((uintptr_t)ep->dbr_mobject->aligned.gpu_ptr + sizeof(__be32));

    assert(ep->uar_mobject->has_gpu_mapping);
    dev_qp->tx_wq.bf = (void *)ep->uar_mobject->aligned.gpu_ptr;

    dev_qp->tx_wq.nwqes = ep->sq_cnt;

    ibuf_dci_start = (uintptr_t)device->qp_shared_object.internal_buf.mem_object->aligned.gpu_ptr;
    ibuf_rc_start = ibuf_dci_start + (size_per_dci * device->dci.num_eps);

    if (ep->qp_type == NVSHMEMI_IBGDA_DEVICE_QP_TYPE_DCI) {
        ibuf_ptr = (void *)(ibuf_dci_start + (size_per_dci * ep->user_index));
        dev_qp->ibuf.nslots = ibgda_num_fetch_slots_per_dci;
    } else if (ep->qp_type == NVSHMEMI_IBGDA_DEVICE_QP_TYPE_RC) {
        ibuf_ptr = (void *)(ibuf_rc_start + (size_per_rc * ep->user_index));
        dev_qp->ibuf.nslots = ibgda_num_fetch_slots_per_rc;
    }

    dev_qp->ibuf.lkey = htobe32(device->qp_shared_object.internal_buf.mem_handle->lkey);
    dev_qp->ibuf.rkey = htobe32(device->qp_shared_object.internal_buf.mem_handle->rkey);
    dev_qp->ibuf.buf = ibuf_ptr;

    dev_qp->qp_type = ep->qp_type;

    ibgda_get_device_qp_mvars(&dev_qp->mvars, device, ep);
}

static void ibgda_get_device_dct(nvshmemi_ibgda_device_dct_t *dev_dct,
                                 const struct ibgda_dct_handle *dct_handle,
                                 const struct ibgda_device *device) {
    memcpy(dev_dct, &dct_handle->dev_dct, sizeof(*dev_dct));
    dev_dct->dqp_dct =
        htobe32(((device->support_half_av_seg ? 0ULL : 1ULL) << 31) | dev_dct->dqp_dct);
}

static int ibgda_setup_gpu_state(nvshmem_transport_t t, struct ibgda_device *device) {
    nvshmemt_ibgda_state_t *ibgda_state = (nvshmemt_ibgda_state_t *)t->state;
    int status = 0;
    int n_pes = t->n_pes;
    int mype = t->my_pe;

    int num_dct_handles = device->dct.num_eps * n_pes;
    int num_rc_handles = device->rc.num_eps_per_pe * n_pes;
    int num_cq_handles = device->dci.num_eps + (device->rc.num_eps_per_pe * (n_pes - 1));

    int cq_idx = 0;

    nvshmemi_ibgda_device_state_t *ibgda_device_state_h;

    nvshmemi_ibgda_device_qp_t *dci_d = NULL;
    nvshmemi_ibgda_device_qp_t *dci_h = NULL;

    nvshmemi_ibgda_device_qp_t *rc_d = NULL;
    nvshmemi_ibgda_device_qp_t *rc_h = NULL;

    nvshmemi_ibgda_device_dct_t *dct_d = NULL;
    nvshmemi_ibgda_device_dct_t *dct_h = NULL;

    nvshmemi_ibgda_device_cq_t *cq_d = NULL;
    nvshmemi_ibgda_device_cq_t *cq_h = NULL;

    ibgda_device_state_h = (nvshmemi_ibgda_device_state_t *)t->type_specific_shared_state;
    assert(ibgda_device_state_h != NULL);

    memset(ibgda_device_state_h, 0, sizeof(*ibgda_device_state_h));

    // Setup DCT table
    dct_h = (nvshmemi_ibgda_device_dct_t *)calloc(num_dct_handles, sizeof(*dct_h));
    NVSHMEMI_NULL_ERROR_JMP(dct_h, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "dct_h allocation failed.\n");

    for (int i = 0; i < num_dct_handles; ++i) {
        ibgda_get_device_dct(&dct_h[i], &device->dct.dct_handles[i], device);
    }

    // Add some DCTs to constant memory
    memcpy(ibgda_device_state_h->constmem.dcts, dct_h,
           sizeof(*dct_h) * IBGDA_MIN(num_dct_handles, NVSHMEMI_IBGDA_MAX_CONST_DCTS));

    // Add the rest of DCTs to global memory
    if (num_dct_handles > NVSHMEMI_IBGDA_MAX_CONST_DCTS) {
        int num_elements = num_dct_handles - NVSHMEMI_IBGDA_MAX_CONST_DCTS;
        status = cudaMalloc(&dct_d, num_elements * sizeof(*dct_d));
        NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                              "dct_d allocation failed.\n");

        status = cudaMemcpyAsync(dct_d, (const void *)&dct_h[NVSHMEMI_IBGDA_MAX_CONST_DCTS],
                                 sizeof(*dct_d) * num_elements, cudaMemcpyHostToDevice,
                                 ibgda_state->my_stream);
        NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_INTERNAL, out,
                              "Copying dct_h to dct_d failed.\n");
    }

    // Get GPU DCIs, RCs, and send CQs
    status = cudaMalloc(&dci_d, device->dci.num_eps * sizeof(*dci_d));
    NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                          "dci_d allocation failed.\n");

    dci_h = (nvshmemi_ibgda_device_qp_t *)calloc(device->dci.num_eps, sizeof(*dci_h));
    NVSHMEMI_NULL_ERROR_JMP(dci_h, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "dci_h allocation failed.\n");

    status = cudaMalloc(&cq_d, num_cq_handles * sizeof(*cq_d));
    NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                          "cq_d allocation failed.\n");

    cq_h = (nvshmemi_ibgda_device_cq_t *)calloc(num_cq_handles, sizeof(*cq_h));
    NVSHMEMI_NULL_ERROR_JMP(cq_h, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "cq_h allocation failed.\n");

    for (int i = 0; i < device->dci.num_eps; ++i) {
        uintptr_t base_mvars_d_addr =
            (uintptr_t)(&dci_d[i]) + offsetof(nvshmemi_ibgda_device_qp_t, mvars);
        ibgda_get_device_qp(&dci_h[i], device, device->dci.eps[i]);

        dci_h[i].tx_wq.cq = &cq_d[cq_idx];

        ibgda_get_device_cq(&cq_h[cq_idx], device->dci.eps[i]->send_cq);
        cq_h[cq_idx].cons_head =
            (uint64_t *)(base_mvars_d_addr +
                         offsetof(nvshmemi_ibgda_device_qp_management_t, tx_wq.cons_head));
        cq_h[cq_idx].cons_tail =
            (uint64_t *)(base_mvars_d_addr +
                         offsetof(nvshmemi_ibgda_device_qp_management_t, tx_wq.cons_tail));
        cq_h[cq_idx].wqe_head =
            (uint64_t *)(base_mvars_d_addr +
                         offsetof(nvshmemi_ibgda_device_qp_management_t, tx_wq.wqe_head));
        cq_h[cq_idx].wqe_tail =
            (uint64_t *)(base_mvars_d_addr +
                         offsetof(nvshmemi_ibgda_device_qp_management_t, tx_wq.wqe_tail));
        cq_h[cq_idx].qpn = dci_h[i].qpn;
        cq_h[cq_idx].qp_type = dci_h[i].qp_type;
        ++cq_idx;
    }

    status = cudaMemcpyAsync(dci_d, (const void *)dci_h, sizeof(*dci_h) * device->dci.num_eps,
                             cudaMemcpyHostToDevice, ibgda_state->my_stream);
    NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_INTERNAL, out,
                          "Copying dci_h to dci_d failed.\n");

    if (num_rc_handles > 0) {
        status = cudaMalloc(&rc_d, num_rc_handles * sizeof(*rc_d));
        NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                              "rc_d allocation failed.\n");

        rc_h = (nvshmemi_ibgda_device_qp_t *)calloc(num_rc_handles, sizeof(*rc_h));
        NVSHMEMI_NULL_ERROR_JMP(rc_h, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                                "rc_h allocation failed.\n");

        for (int i = 0; i < num_rc_handles; ++i) {
            uintptr_t base_mvars_d_addr =
                (uintptr_t)(&rc_d[i]) + offsetof(nvshmemi_ibgda_device_qp_t, mvars);

            // No RC QP to self
            if (i / device->rc.num_eps_per_pe == mype) {
                continue;
            }

            ibgda_get_device_qp(&rc_h[i], device, device->rc.eps[i]);

            rc_h[i].tx_wq.cq = &cq_d[cq_idx];

            ibgda_get_device_cq(&cq_h[cq_idx], device->rc.eps[i]->send_cq);
            cq_h[cq_idx].cons_head =
                (uint64_t *)(base_mvars_d_addr +
                             offsetof(nvshmemi_ibgda_device_qp_management_t, tx_wq.cons_head));
            cq_h[cq_idx].cons_tail =
                (uint64_t *)(base_mvars_d_addr +
                             offsetof(nvshmemi_ibgda_device_qp_management_t, tx_wq.cons_tail));
            cq_h[cq_idx].wqe_head =
                (uint64_t *)(base_mvars_d_addr +
                             offsetof(nvshmemi_ibgda_device_qp_management_t, tx_wq.wqe_head));
            cq_h[cq_idx].wqe_tail =
                (uint64_t *)(base_mvars_d_addr +
                             offsetof(nvshmemi_ibgda_device_qp_management_t, tx_wq.wqe_tail));
            cq_h[cq_idx].qpn = rc_h[i].qpn;
            cq_h[cq_idx].qp_type = rc_h[i].qp_type;

            ++cq_idx;
        }

        status = cudaMemcpyAsync(rc_d, (const void *)rc_h, sizeof(*rc_h) * num_rc_handles,
                                 cudaMemcpyHostToDevice, ibgda_state->my_stream);
        NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_INTERNAL, out,
                              "Copying rc_h to rc_d failed.\n");
    }

    status = cudaMemcpyAsync(cq_d, (const void *)cq_h, sizeof(*cq_h) * num_cq_handles,
                             cudaMemcpyHostToDevice, ibgda_state->my_stream);
    NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_INTERNAL, out,
                          "Copying cq_h to cq_d failed.\n");

    // Post the device state
    ibgda_device_state_h->globalmem.dcis = dci_d;
    ibgda_device_state_h->globalmem.rcs = rc_d;
    ibgda_device_state_h->globalmem.dcts = dct_d;
    ibgda_device_state_h->globalmem.cqs = cq_d;
    ibgda_device_state_h->log2_cumem_granularity = t->log2_cumem_granularity;
    ibgda_device_state_h->num_shared_dcis = device->dci.num_shared_eps;
    ibgda_device_state_h->num_exclusive_dcis = device->dci.num_eps - device->dci.num_shared_eps;
    ibgda_device_state_h->dci_map_type = device->dci.map_by;
    ibgda_device_state_h->ndcts_per_pe = device->dct.num_eps;
    ibgda_device_state_h->num_dct_groups =
        IBGDA_MAX(ibgda_device_state_h->num_exclusive_dcis / (device->dct.num_eps * n_pes), 1);
    ibgda_device_state_h->num_rc_per_pe = device->rc.num_eps_per_pe;
    ibgda_device_state_h->rc_map_type = device->rc.map_by;
    ibgda_device_state_h->num_requests_in_batch = ibgda_num_requests_in_batch;
    ibgda_device_state_h->support_half_av_seg = device->support_half_av_seg;
    ibgda_device_state_h->may_skip_cst = device->may_skip_cst;

    assert(ibgda_nic_buf_location == IBGDA_MEM_TYPE_GPU ||
           ibgda_nic_buf_location == IBGDA_MEM_TYPE_HOST);
    ibgda_device_state_h->nic_buf_on_gpumem = (ibgda_nic_buf_location == IBGDA_MEM_TYPE_GPU);

    status = cudaStreamSynchronize(ibgda_state->my_stream);
    NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_INTERNAL, out,
                          "stream synchronize failed.\n");

out:
    if (status) {
        if (dci_d) cudaFree(dci_d);
        if (dct_d) cudaFree(dct_d);
        if (cq_d) cudaFree(cq_d);
        if (rc_d) cudaFree(rc_d);
    }
    if (dci_h) free(dci_h);
    if (dct_h) free(dct_h);
    if (cq_h) free(cq_h);
    if (rc_h) free(rc_h);
    return status;
}

int nvshmemt_ibgda_connect_endpoints(nvshmem_transport_t t, int selected_dev_id) {
    int status = 0;
    int n_pes = t->n_pes;
    int mype = t->my_pe;

    struct ibgda_dct_handle *local_dct_handles = NULL;
    struct ibgda_rc_handle *local_rc_handles = NULL;

    nvshmemt_ibgda_state_t *ibgda_state = (nvshmemt_ibgda_state_t *)t->state;
    struct ibgda_device *device;
    int portid;
    int warp_size;
    int max_threads_per_block;
    int rdma_write_ordering = 0;
    bool support_half_av_seg = true;

    CUdevice gpu_device_id;

    ibgda_state->selected_dev_id = selected_dev_id;

    status = CUPFN(ibgda_cuda_syms, cuCtxGetDevice(&gpu_device_id));
    if (status != CUDA_SUCCESS) {
        status = NVSHMEMX_ERROR_INTERNAL;
        goto out;
    }

    device = ((struct ibgda_device *)ibgda_state->devices +
              ibgda_state->dev_ids[ibgda_state->selected_dev_id]);
    portid = ibgda_state->port_ids[ibgda_state->selected_dev_id];

    // Create DCT.
    device->dct.num_eps = ibgda_state->options->IBGDA_NUM_DCT;
    if (device->dct.num_eps <= 0) {
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INVALID_VALUE, out,
                           "NVSHMEM_IBGDA_NUM_DCT must be greater than 0.\n");
    } else if (device->dct.num_eps < 2) {
        NVSHMEMI_WARN_PRINT("Setting NVSHMEM_IBGDA_NUM_DCT lower than 2 may impact performance.\n");
    }

    status = ibgda_create_dct_shared_objects(ibgda_state, device, portid);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "ibgda_create_dct_shared_objects failed.\n");

    local_dct_handles =
        (struct ibgda_dct_handle *)calloc(device->dct.num_eps, sizeof(*local_dct_handles));
    NVSHMEMI_NULL_ERROR_JMP(local_dct_handles, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "allocation of local_dct_handles failed.\n");

    device->dct.dct_handles = (struct ibgda_dct_handle *)calloc(device->dct.num_eps * n_pes,
                                                                sizeof(*device->dct.dct_handles));
    NVSHMEMI_NULL_ERROR_JMP(device->dct.dct_handles, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "allocation of dct_handles failed.\n");

    device->dct.eps = (struct ibgda_ep **)calloc(device->dct.num_eps, sizeof(*device->dct.eps));
    NVSHMEMI_NULL_ERROR_JMP(device->dct.eps, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "allocation of dct.eps failed.\n");

    for (int i = 0; i < device->dct.num_eps; ++i) {
        status = ibgda_create_dct(ibgda_state, &device->dct.eps[i], device, portid);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "ibgda_create_dct failed on DCT #%d.\n", i);

        status = ibgda_get_dct_handle(&local_dct_handles[i], device->dct.eps[i], device);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "ibgda_get_dct_handle failed on DCT #%d.\n", i);
    }

    // Exchange DCT info with other PEs.
    status =
        t->boot_handle->allgather((void *)local_dct_handles, (void *)device->dct.dct_handles,
                                  sizeof(*local_dct_handles) * device->dct.num_eps, t->boot_handle);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "allgather of dct handles failed.\n");

    // All EPs in all PEs must support half av seg if we want to use it.
    // Otherwise, fallback to use full av seg.
    for (int i = 0; i < device->dct.num_eps * n_pes; ++i) {
        support_half_av_seg &= device->dct.dct_handles[i].support_half_av_seg;
    }
    device->support_half_av_seg = support_half_av_seg;

    // Get info about DCI.
    status = cudaDeviceGetAttribute(&warp_size, cudaDevAttrWarpSize, gpu_device_id);
    NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_INTERNAL, out,
                          "cudaDeviceGetAttribute querying warp size failed.\n");
    status = cudaDeviceGetAttribute(&max_threads_per_block, cudaDevAttrMaxThreadsPerBlock,
                                    gpu_device_id);
    NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_INTERNAL, out,
                          "cudaDeviceGetAttribute querying max threads per block failed.\n");

    device->dci.num_shared_eps = ibgda_state->options->IBGDA_NUM_SHARED_DCI;
    if (device->dci.num_shared_eps < 1) {
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INVALID_VALUE, out,
                           "NVSHMEM_IBGDA_NUM_SHARED_DCI must be a positive number.\n");
    }

    status = ibgda_parse_qp_map_by(&device->dci.map_by, ibgda_state->options->IBGDA_DCI_MAP_BY);
    NVSHMEMI_NZ_ERROR_JMP(status, status, out, "NVSHMEM_IBGDA_DCI_MAP_BY is not valid.\n");

    INFO(ibgda_state->log_level, "NVSHMEM_IBGDA_DCI_MAP_BY is set to %s.\n",
         ibgda_state->options->IBGDA_DCI_MAP_BY);

    device->dci.num_eps = ibgda_state->options->IBGDA_NUM_DCI;
    if (device->dci.num_eps <= 0) {
        int num_eps = 0;
        int mpc = 0;
        status = cudaDeviceGetAttribute(&mpc, cudaDevAttrMultiProcessorCount, gpu_device_id);
        NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_INTERNAL, out,
                              "cudaDeviceGetAttribute querying multiprocessor count failed.\n");

        switch (device->dci.map_by) {
            case NVSHMEMI_IBGDA_DEVICE_QP_MAP_TYPE_CTA:
            case NVSHMEMI_IBGDA_DEVICE_QP_MAP_TYPE_SM:
                num_eps = mpc;
                break;
            case NVSHMEMI_IBGDA_DEVICE_QP_MAP_TYPE_WARP:
                num_eps = mpc * warp_size;
                break;
            case NVSHMEMI_IBGDA_DEVICE_QP_MAP_TYPE_DCT:
                num_eps = device->dct.num_eps * n_pes;
                break;
            default:
                NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INVALID_VALUE, out,
                                   "NVSHMEM_IBGDA_DCI_MAP_BY=%s is not supported.\n",
                                   ibgda_state->options->IBGDA_DCI_MAP_BY);
                break;
        }

        device->dci.num_eps = num_eps + device->dci.num_shared_eps;
    }
    assert(device->dci.num_eps > 0);

    if (device->dci.num_shared_eps > device->dci.num_eps) {
        NVSHMEMI_ERROR_JMP(
            status, NVSHMEMX_ERROR_INVALID_VALUE, out,
            "NVSHMEM_IBGDA_NUM_SHARED_DCI must be less than or equal to NVSHMEM_IBGDA_NUM_DCI.\n");
    }

    INFO(ibgda_state->log_level, "Creating %d DCI QPs (shared: %d, exclusive: %d)\n",
         device->dci.num_eps, device->dci.num_shared_eps,
         device->dci.num_eps - device->dci.num_shared_eps);

    if (ibgda_num_fetch_slots_per_dci < warp_size) {
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INVALID_VALUE, out,
                           "NVSHMEM_IBGDA_NUM_FETCH_SLOTS_PER_DCI must be at least %d.\n",
                           warp_size);
    }

    // Get info about RC.
    device->rc.num_eps_per_pe = ibgda_state->options->IBGDA_NUM_RC_PER_PE;
    if (device->rc.num_eps_per_pe < 0) {
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INVALID_VALUE, out,
                           "NVSHMEM_IBGDA_NUM_RC_PER_PE must be positive or zero.\n");
    } else if (device->rc.num_eps_per_pe > 0) {
        if (ibgda_num_fetch_slots_per_rc < warp_size) {
            NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INVALID_VALUE, out,
                               "NVSHMEM_IBGDA_NUM_FETCH_SLOTS_PER_RC must be at least %d.\n",
                               warp_size);
        }

        status = ibgda_parse_qp_map_by(&device->rc.map_by, ibgda_state->options->IBGDA_RC_MAP_BY);
        NVSHMEMI_NZ_ERROR_JMP(status, status, out, "NVSHMEM_IBGDA_RC_MAP_BY is not valid.\n");

        INFO(ibgda_state->log_level, "NVSHMEM_IBGDA_RC_MAP_BY is set to %s.\n",
             ibgda_state->options->IBGDA_RC_MAP_BY);

        switch (device->rc.map_by) {
            case NVSHMEMI_IBGDA_DEVICE_QP_MAP_TYPE_CTA:
            case NVSHMEMI_IBGDA_DEVICE_QP_MAP_TYPE_SM:
            case NVSHMEMI_IBGDA_DEVICE_QP_MAP_TYPE_WARP:
                break;
            default:
                NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INVALID_VALUE, out,
                                   "NVSHMEM_IBGDA_RC_MAP_BY=%s is not supported.\n",
                                   ibgda_state->options->IBGDA_RC_MAP_BY);
                break;
        }
    }
    INFO(ibgda_state->log_level, "Creating %d RC QPs\n", device->rc.num_eps_per_pe);

    // Create qp shared objects
    status = ibgda_create_qp_shared_objects(ibgda_state, device, n_pes);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "ibgda_create_qp_shared_objects failed.\n");

    // Create DCI
    device->dci.eps = (struct ibgda_ep **)calloc(device->dci.num_eps, sizeof(*device->dci.eps));
    NVSHMEMI_NULL_ERROR_JMP(device->dci.eps, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "allocation of dci.eps failed.\n");

    for (int i = 0; i < device->dci.num_eps; ++i) {
        status = ibgda_create_qp(&device->dci.eps[i], device, portid, i,
                                 NVSHMEMI_IBGDA_DEVICE_QP_TYPE_DCI);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "ibgda_create_dci failed on DCI #%d.\n", i);
    }

    // Transition DCI to RTS.
    for (int i = 0; i < device->dci.num_eps; ++i) {
        status = ibgda_qp_rst2init(device->dci.eps[i], device, portid);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "ibgda_qp_rst2init failed on DCI #%d.\n", i);

        status = ibgda_dci_init2rtr(ibgda_state, device->dci.eps[i], device, portid);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "ibgda_dci_init2rtr failed on DCI #%d.\n", i);

        status = ibgda_qp_rtr2rts(device->dci.eps[i], device, portid);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "ibgda_qp_rtr2rts failed on DCI #%d.\n", i);
    }

    // Create RC
    if (device->rc.num_eps_per_pe > 0) {
        int num_rc_eps = device->rc.num_eps_per_pe * n_pes;

        local_rc_handles = (struct ibgda_rc_handle *)calloc(num_rc_eps, sizeof(*local_rc_handles));
        NVSHMEMI_NULL_ERROR_JMP(local_rc_handles, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                                "allocation of local_rc_handles failed.\n");

        device->rc.peer_ep_handles =
            (struct ibgda_rc_handle *)calloc(num_rc_eps, sizeof(*device->rc.peer_ep_handles));
        NVSHMEMI_NULL_ERROR_JMP(device->rc.peer_ep_handles, status, NVSHMEMX_ERROR_OUT_OF_MEMORY,
                                out, "allocation of rc.peer_ep_handles failed.\n");

        device->rc.eps = (struct ibgda_ep **)calloc(num_rc_eps, sizeof(*device->dci.eps));
        NVSHMEMI_NULL_ERROR_JMP(device->rc.eps, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                                "allocation of rc.eps failed.\n");

        for (int i = 0; i < num_rc_eps; ++i) {
            // Do not create loopback to self
            if (i / device->rc.num_eps_per_pe == mype) {
                continue;
            }
            status = ibgda_create_qp(&device->rc.eps[i], device, portid, i,
                                     NVSHMEMI_IBGDA_DEVICE_QP_TYPE_RC);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                  "ibgda_create_dci failed on RC #%d.\n", i);

            status = ibgda_get_rc_handle(&local_rc_handles[i], device->rc.eps[i], device);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                  "ibgda_get_rc_handle failed on RC #%d.\n", i);
        }

        // Exchange info
        status = t->boot_handle->alltoall(
            (void *)local_rc_handles, (void *)device->rc.peer_ep_handles,
            sizeof(*local_rc_handles) * device->rc.num_eps_per_pe, t->boot_handle);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "alltoall of rc handles failed.\n");

        // Transition to RTS
        for (int i = 0; i < num_rc_eps; ++i) {
            // No loopback to self
            if (i / device->rc.num_eps_per_pe == mype) {
                continue;
            }

            status = ibgda_qp_rst2init(device->rc.eps[i], device, portid);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                  "ibgda_qp_rst2init failed on RC #%d.\n", i);

            status = ibgda_rc_init2rtr(ibgda_state, device->rc.eps[i], device, portid,
                                       &device->rc.peer_ep_handles[i]);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                  "ibgda_rc_init2rtr failed on RC #%d.\n", i);

            status = ibgda_qp_rtr2rts(device->rc.eps[i], device, portid);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                  "ibgda_qp_rtr2rts failed on RC #%d.\n", i);
        }
    }

    // Determine if we need CST.
#if CUDA_VERSION >= 11050
    status = cudaDeviceGetAttribute(&rdma_write_ordering, cudaDevAttrGPUDirectRDMAWritesOrdering,
                                    gpu_device_id);
    if (status) {
        NVSHMEMI_WARN_PRINT(
            "Cannot query cudaDevAttrGPUDirectRDMAWritesOrdering. Assuming "
            "cudaGPUDirectRDMAWritesOrderingNone\n");
        status = 0;
    }

    // GPU guarantees incoming PCIe write ordering. No need to do CST.
    device->may_skip_cst = !!(rdma_write_ordering >= cudaGPUDirectRDMAWritesOrderingOwner);
#else
    device->may_skip_cst = 0;
#endif
    INFO(ibgda_state->log_level, "CST is %s.\n",
         (device->may_skip_cst ? "not required" : "required"));

    // Setup QPs / CQs on GPU.
    status = ibgda_setup_gpu_state(t, device);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "ibgda_setup_gpu_state failed.\n");

out:
    if (status) {
        // TODO: Implement cleanup
    }

    if (local_rc_handles) {
        free(local_rc_handles);
    }

    if (local_dct_handles) {
        free(local_dct_handles);
    }
    return status;
}

int nvshmemt_ibgda_release_mem_handle(nvshmem_mem_handle_t *mem_handle, nvshmem_transport_t t) {
    int status = 0;
    nvshmemt_ibgda_state_t *ibgda_state = (nvshmemt_ibgda_state_t *)t->state;

    nvshmemi_ibgda_device_state_t *ibgda_device_state =
        (nvshmemi_ibgda_device_state_t *)t->type_specific_shared_state;
    assert(ibgda_device_state != NULL);

    struct nvshmemt_ib_common_mem_handle *handle =
        (struct nvshmemt_ib_common_mem_handle *)mem_handle;
    if (handle->local_only) {
        uint32_t position = 0;
        struct ibgda_device_local_only_mhandle_cache *prev_mhandle_cache = NULL;
        struct ibgda_device_local_only_mhandle_cache *next_mhandle_cache = NULL;
        struct ibgda_device_local_only_mhandle_cache *curr_mhandle_cache = NULL;
        void *mhandle_gpu_ptr;

        // Find the position in the host-side cache.
        for (auto it = ibgda_device_local_only_mhandles.begin();
             it != ibgda_device_local_only_mhandles.end(); ++it) {
            if (it->mhandle.start == (uint64_t)handle->buf) {
                curr_mhandle_cache = &ibgda_device_local_only_mhandles.data()[position];
                if (position > 0)
                    prev_mhandle_cache = &ibgda_device_local_only_mhandles.data()[position - 1];
                if (position < ibgda_device_local_only_mhandles.size() - 1)
                    next_mhandle_cache = &ibgda_device_local_only_mhandles.data()[position + 1];
                break;
            }
            ++position;
        }
        NVSHMEMI_NULL_ERROR_JMP(curr_mhandle_cache, status, NVSHMEMX_ERROR_INVALID_VALUE, out,
                                "mem_handle is not registered.\n");

        // Remove this element from the linked list on both host and GPU.
        if (prev_mhandle_cache) {
            if (next_mhandle_cache)
                prev_mhandle_cache->mhandle.next =
                    (nvshmemi_ibgda_device_local_only_mhandle_t *)next_mhandle_cache->dev_ptr;
            else
                prev_mhandle_cache->mhandle.next = NULL;
            mhandle_gpu_ptr = (void *)((uintptr_t)prev_mhandle_cache->dev_ptr +
                                       offsetof(nvshmemi_ibgda_device_local_only_mhandle_t, next));
            status =
                cudaMemcpyAsync(mhandle_gpu_ptr, (const void *)&prev_mhandle_cache->mhandle.next,
                                sizeof(prev_mhandle_cache->mhandle.next), cudaMemcpyHostToDevice,
                                ibgda_state->my_stream);
            NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_INTERNAL, out,
                                  "Setting local_only_mhandle in GPU memory failed.\n");
        } else {
            // The caller will trigger device state update.
            if (next_mhandle_cache)
                ibgda_device_state->globalmem.local_only_mhandle_head =
                    (nvshmemi_ibgda_device_local_only_mhandle_t *)next_mhandle_cache->dev_ptr;
            else
                ibgda_device_state->globalmem.local_only_mhandle_head = NULL;
        }

        // Free the copy of this element on GPU.
        status = cudaFree(curr_mhandle_cache->dev_ptr);
        NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_INTERNAL, out,
                              "cudaFree failed.\n");

        ibgda_device_local_only_mhandles.erase(ibgda_device_local_only_mhandles.begin() + position);
    }

    // TODO: Clean up non-local-only mem_handle

    status = nvshmemt_ib_common_release_mem_handle(&ftable, mem_handle, ibgda_state->log_level);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "nvshmemt_ib_common_release_mem_handle failed.\n");

    status = cudaStreamSynchronize(ibgda_state->my_stream);
    NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_INTERNAL, out,
                          "stream synchronize failed.\n");

out:
    return status;
}

int nvshmemt_ibgda_finalize(nvshmem_transport_t transport) {
    struct ibgda_device *device;
    nvshmemt_ibgda_state_t *ibgda_state = (nvshmemt_ibgda_state_t *)transport->state;
    nvshmemi_ibgda_device_state_t *ibgda_device_state_h;

    int status = 0, dev_id;
    int n_pes = transport->n_pes;
    int mype = transport->my_pe;
    int num_rc_eps;

    ibgda_device_lkeys.clear();
    ibgda_device_rkeys.clear();

    if (ibgda_device_lkeys_d) {
        cudaFree(ibgda_device_lkeys_d);
        ibgda_device_lkeys_d = 0;
    }
    if (ibgda_device_rkeys_d) {
        cudaFree(ibgda_device_rkeys_d);
        ibgda_device_rkeys_d = 0;
    }

    ibgda_device_state_h = (nvshmemi_ibgda_device_state_t *)transport->type_specific_shared_state;
    if (ibgda_device_state_h) {
        if (ibgda_device_state_h->globalmem.dcts) cudaFree(ibgda_device_state_h->globalmem.dcts);
        if (ibgda_device_state_h->globalmem.dcis) cudaFree(ibgda_device_state_h->globalmem.dcis);
        if (ibgda_device_state_h->globalmem.cqs) cudaFree(ibgda_device_state_h->globalmem.cqs);
        if (ibgda_device_state_h->globalmem.rcs) cudaFree(ibgda_device_state_h->globalmem.rcs);
    }

    dev_id = ibgda_state->dev_ids[ibgda_state->selected_dev_id];
    device = ((struct ibgda_device *)ibgda_state->devices + dev_id);

    for (int i = 0; i < device->dci.num_eps; ++i) {
        status = ibgda_destroy_ep(device->dci.eps[i]);
    }

    for (int i = 0; i < device->dct.num_eps; ++i) {
        status = ibgda_destroy_ep(device->dct.eps[i]);
    }

    num_rc_eps = device->rc.num_eps_per_pe * n_pes;
    for (int i = 0; i < num_rc_eps; ++i) {
        // Do not create loopback to self
        if (i / device->rc.num_eps_per_pe == mype) {
            continue;
        }
        status = ibgda_destroy_ep(device->rc.eps[i]);
    }

    status = ibgda_destroy_qp_shared_objects(ibgda_state, device);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "ibgda_destroy_qp_shared_objects failed.\n");

    status = ibgda_destroy_dct_shared_objects(ibgda_state, device);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "ibgda_destroy_dct_shared_objects failed.\n");

    nvshmemt_ibv_ftable_fini(&ibv_handle);

    if (transport->state) {
        free(transport->state);
    }

    if (transport->device_pci_paths) {
        for (int i = 0; i < transport->n_devices; i++) {
            free(transport->device_pci_paths[i]);
        }
        free(transport->device_pci_paths);
    }

    free(transport);

out:
    return status;
}

int nvshmemt_ibgda_add_device_remote_mem_handles(nvshmem_transport_t t, int transport_stride,
                                                 nvshmem_mem_handle_t *mem_handles,
                                                 uint64_t heap_offset, size_t size) {
    nvshmemt_ibgda_state_t *ibgda_state = (nvshmemt_ibgda_state_t *)t->state;
    int status = 0;
    int n_pes = t->n_pes;

    size_t num_rkeys;

    nvshmemi_ibgda_device_state_t *ibgda_device_state;

    ibgda_device_state = (nvshmemi_ibgda_device_state_t *)t->type_specific_shared_state;
    assert(ibgda_device_state != NULL);

    static_assert(sizeof(struct nvshmemt_ib_common_mem_handle) <= NVSHMEM_MEM_HANDLE_SIZE,
                  "static_assert(sizeof(T) <= NVSHMEM_MEM_HANDLE_SIZE) failed");

    size_t cumem_granularity = 1ULL << t->log2_cumem_granularity;
    size_t num_elements;
    // size must be divisible by cumem_granularity, which is a power of 2.
    assert((size & (cumem_granularity - 1)) == 0);

    num_elements = size >> t->log2_cumem_granularity;
    while (num_elements > 0) {
        for (int i = 0; i < n_pes; ++i) {
            // sizeof(struct ibgda_mem_handle) <= sizeof(nvshmem_mem_handle_t)
            // So, we calculate the pointer with nvshmem_mem_handle_t and convert to
            // ibgda_mem_handle later.
            struct nvshmemt_ib_common_mem_handle *gmhandle =
                (struct nvshmemt_ib_common_mem_handle
                     *)&mem_handles[i * transport_stride + t->index];

            nvshmemi_ibgda_device_key_t device_key = {.key = htobe32(gmhandle->rkey),
                                                      .next_addr = heap_offset + size};

            ibgda_device_rkeys.emplace_back(device_key);
        }
        --num_elements;
    }

    if (ibgda_device_rkeys_d) {
        status = cudaFree(ibgda_device_rkeys_d);
        NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_INTERNAL, out,
                              "cudaFree failed.\n");
        ibgda_device_rkeys_d = 0;
    }

    num_rkeys = ibgda_device_rkeys.size();

    // For cache optimization, put rkeys in constant memory first.
    memcpy(
        ibgda_device_state->constmem.rkeys, ibgda_device_rkeys.data(),
        IBGDA_MIN(num_rkeys, NVSHMEMI_IBGDA_MAX_CONST_RKEYS) * sizeof(nvshmemi_ibgda_device_key_t));

    // Put the rest that don't fit in constant memory in global memory
    if (num_rkeys > NVSHMEMI_IBGDA_MAX_CONST_RKEYS) {
        size_t rkeys_array_size =
            sizeof(nvshmemi_ibgda_device_key_t) * (num_rkeys - NVSHMEMI_IBGDA_MAX_CONST_RKEYS);

        nvshmemi_ibgda_device_key_t *data_ptr =
            &ibgda_device_rkeys.data()[NVSHMEMI_IBGDA_MAX_CONST_RKEYS];

        status = cudaMalloc(&ibgda_device_rkeys_d, rkeys_array_size);
        NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                              "cudaMalloc failed.\n");

        status = cudaMemcpyAsync(ibgda_device_rkeys_d, (const void *)data_ptr, rkeys_array_size,
                                 cudaMemcpyHostToDevice, ibgda_state->my_stream);
        NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_INTERNAL, out,
                              "Copying rkeys to GPU memory failed.\n");

        status = cudaStreamSynchronize(ibgda_state->my_stream);
        NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_INTERNAL, out,
                              "stream synchronize failed.\n");
    }

    ibgda_device_state->globalmem.rkeys = (nvshmemi_ibgda_device_key_t *)ibgda_device_rkeys_d;
out:
    if (status) {
        // Unrecoverable error
        if (ibgda_device_rkeys_d) cudaFree(ibgda_device_rkeys_d);
        ibgda_device_rkeys.clear();
    }
    return status;
}

static ibgda_nic_mapping_memtype_reqeust_t ibgda_parse_nic_mapping_memtype_request(
    const char *str) {
    std::string req = str;

    // Trim whitespace
    req.erase(std::remove_if(req.begin(), req.end(), ::isspace), req.end());

    // To lower case
    std::for_each(req.begin(), req.end(), [](decltype(*req.begin()) &c) { c = ::tolower(c); });

    if (req == "gpumem")
        return IBGDA_NIC_MAPPING_MEMTYPE_REQUEST_GPUMEM;
    else if (req == "hostmem")
        return IBGDA_NIC_MAPPING_MEMTYPE_REQUEST_HOSTMEM;
    else
        return IBGDA_NIC_MAPPING_MEMTYPE_REQUEST_AUTO;
}

static int ibgda_check_nic_mapping_memtypes(struct ibgda_device *device,
                                            ibgda_nic_mapping_memtype_reqeust_t request_memtype) {
    int status = 0;

    bool try_gpumem = ((request_memtype == IBGDA_NIC_MAPPING_MEMTYPE_REQUEST_AUTO) ||
                       (request_memtype == IBGDA_NIC_MAPPING_MEMTYPE_REQUEST_GPUMEM));
    bool try_hostmem = ((request_memtype == IBGDA_NIC_MAPPING_MEMTYPE_REQUEST_AUTO) ||
                        (request_memtype == IBGDA_NIC_MAPPING_MEMTYPE_REQUEST_HOSTMEM));

    bool can_use_gpumem = false;
    bool can_use_hostmem = false;

    struct ibgda_mem_object *mobject = NULL;

    if (try_gpumem) {
        status = ibgda_gpu_mem_alloc(&mobject, IBGDA_DBSIZE, IBGDA_GPAGE_SIZE, false);
        if (status) goto out_try_gpumem;

        status = ibgda_mobject_nic_map(mobject, device->context, IBV_ACCESS_LOCAL_WRITE);
        if (status) goto out_try_gpumem;

        can_use_gpumem = true;

    out_try_gpumem:
        if (mobject) {
            if (mobject->has_nic_mapping) ibgda_mobject_nic_unmap(mobject);
            ibgda_gpu_mem_free(mobject);
        }
        mobject = NULL;
        status = 0;
    }

    if (try_hostmem) {
        status = ibgda_host_mem_alloc(&mobject, IBGDA_DBSIZE, IBGDA_GPAGE_SIZE, true);
        if (status) goto out_try_hostmem;

        status = ibgda_mobject_nic_map(mobject, device->context, IBV_ACCESS_LOCAL_WRITE);
        if (status) goto out_try_hostmem;

        can_use_hostmem = true;

    out_try_hostmem:
        if (mobject) {
            if (mobject->has_nic_mapping) ibgda_mobject_nic_unmap(mobject);
            ibgda_host_mem_free(mobject);
        }
        mobject = NULL;
        status = 0;
    }

    device->support_nic_buf_on_gpumem = can_use_gpumem;
    device->support_nic_buf_on_hostmem = can_use_hostmem;

    if (!can_use_gpumem && !can_use_hostmem) return NVSHMEMX_ERROR_NOT_SUPPORTED;

    return 0;
}

static int ibgda_check_gpu_mapping_nic_uar(struct ibgda_device *device) {
    int status = 0;

    struct ibgda_mem_object *mobject = NULL;

    status = ibgda_alloc_and_map_qp_uar(device->context, &mobject);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "ibgda_alloc_and_map_qp_uar failed.\n");

out:
    if (mobject) ibgda_unmap_and_free_qp_uar(mobject);
    return status;
}

int nvshmemt_init(nvshmem_transport_t *t, struct nvshmemi_cuda_fn_table *table, int api_version) {
    struct nvshmemt_hca_info hca_list[MAX_NUM_HCAS];
    struct nvshmemt_hca_info pe_hca_mapping[MAX_NUM_PES_PER_NODE];
    struct nvshmemi_options_s *options = NULL;

    int status = 0;
    int exclude_list = 0;
    int hca_list_count = 0;
    int pe_hca_map_count = 0;
    int user_selection = 0;
    int offset = 0;
    int num_devices = 0;
    int lowest_stream_priority;
    int highest_stream_priority;
    uint32_t atomic_host_endian_size = 0;

    struct nvshmem_transport *transport = NULL;
    nvshmemt_ibgda_state_t *ibgda_state;
    struct ibgda_device *device;
    struct ibv_device **dev_list = NULL;

    bool nic_buf_on_gpumem = true;
    bool nic_buf_on_hostmem = true;

    ibgda_nic_mapping_memtype_reqeust_t nic_mapping_memtype_request;

    if (api_version != NVSHMEM_TRANSPORT_INTERFACE_VERSION) {
        NVSHMEMI_ERROR_PRINT(
            "NVSHMEM provided an incompatible version of the transport interface. "
            "This transport supports a maximum API version of %d\n",
            NVSHMEM_TRANSPORT_INTERFACE_VERSION);
        return NVSHMEMX_ERROR_INVALID_VALUE;
    }

    ibgda_cuda_syms = table;

    options = (struct nvshmemi_options_s *)calloc(1, sizeof(struct nvshmemi_options_s));
    NVSHMEMI_NULL_ERROR_JMP(options, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "Unable to allocate options stuct for ibgda transport.\n");

    status = nvshmemi_env_options_init(options);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "Unable to initialize NVSHMEM options.\n");

    transport = (struct nvshmem_transport *)malloc(sizeof(struct nvshmem_transport));
    NVSHMEMI_NULL_ERROR_JMP(transport, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "Unable to allocate transport stuct for ibgda transport.\n");
    memset(transport, 0, sizeof(struct nvshmem_transport));

    ibgda_srq_depth = options->SRQ_DEPTH;
    if (ibgda_srq_depth <= 0) {
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INVALID_VALUE, out,
                           "NVSHMEM_SRQ_DEPTH must be a positive number.\n");
    }

    ibgda_qp_depth = options->QP_DEPTH;
    if (ibgda_qp_depth > 0) {
        ibgda_qp_depth = IBGDA_ROUND_UP_POW2_OR_0(ibgda_qp_depth);
    }
    if (ibgda_qp_depth <= 0) {
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INVALID_VALUE, out,
                           "NVSHMEM_QP_DEPTH must be a positive number.\n");
    } else if (ibgda_qp_depth < NVSHMEMI_IBGDA_MIN_QP_DEPTH) {
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INVALID_VALUE, out,
                           "NVSHMEM_QP_DEPTH must be at least %d.\n", NVSHMEMI_IBGDA_MIN_QP_DEPTH);
    } else if (ibgda_qp_depth > NVSHMEMI_IBGDA_MAX_QP_DEPTH) {
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INVALID_VALUE, out,
                           "NVSHMEM_QP_DEPTH can be at most %d.\n", NVSHMEMI_IBGDA_MAX_QP_DEPTH);
    }

    ibgda_num_requests_in_batch = options->IBGDA_NUM_REQUESTS_IN_BATCH;
    if (ibgda_num_requests_in_batch > 0) {
        ibgda_num_requests_in_batch = IBGDA_ROUND_UP_POW2_OR_0(ibgda_num_requests_in_batch);
    }
    if (ibgda_num_requests_in_batch <= 0) {
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INVALID_VALUE, out,
                           "NVSHMEM_IBGDA_NUM_REQUESTS_IN_BATCH must be a positive number.\n");
    } else if (ibgda_num_requests_in_batch > ibgda_qp_depth) {
        NVSHMEMI_ERROR_JMP(
            status, NVSHMEMX_ERROR_INVALID_VALUE, out,
            "NVSHMEM_IBGDA_NUM_REQUESTS_IN_BATCH must not be larger than QP depth.\n");
    }

    ibgda_num_fetch_slots_per_dci = options->IBGDA_NUM_FETCH_SLOTS_PER_DCI;
    if (ibgda_num_fetch_slots_per_dci > 0) {
        ibgda_num_fetch_slots_per_dci = IBGDA_ROUND_UP_POW2_OR_0(ibgda_num_fetch_slots_per_dci);
    }
    if (ibgda_num_fetch_slots_per_dci <= 0) {
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INVALID_VALUE, out,
                           "NVSHMEM_IBGDA_NUM_FETCH_SLOTS_PER_DCI must be a positive number.\n");
    }

    ibgda_num_fetch_slots_per_rc = options->IBGDA_NUM_FETCH_SLOTS_PER_RC;
    if (ibgda_num_fetch_slots_per_rc > 0) {
        ibgda_num_fetch_slots_per_rc = IBGDA_ROUND_UP_POW2_OR_0(ibgda_num_fetch_slots_per_rc);
    }
    if (ibgda_num_fetch_slots_per_rc <= 0) {
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INVALID_VALUE, out,
                           "NVSHMEM_IBGDA_NUM_FETCH_SLOTS_PER_RC must be a positive number.\n");
    }

    ibgda_state = (nvshmemt_ibgda_state_t *)calloc(1, sizeof(nvshmemt_ibgda_state_t));
    NVSHMEMI_NULL_ERROR_JMP(ibgda_state, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "p2p state allocation failed \n");
    transport->state = (void *)ibgda_state;

    ibgda_state->log_level = nvshmemt_common_get_log_level(options);

    if (nvshmemt_ibv_ftable_init(&ibv_handle, &ftable, ibgda_state->log_level)) {
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                           "Unable to dlopen libibverbs. Skipping devx transport.\n");
    }

    dev_list = ftable.get_device_list(&num_devices);
    NVSHMEMI_NULL_ERROR_JMP(dev_list, status, NVSHMEMX_ERROR_INTERNAL, out,
                            "get_device_list failed \n");

    ibgda_state->devices = calloc(MAX_NUM_HCAS, sizeof(struct ibgda_device));
    NVSHMEMI_NULL_ERROR_JMP(ibgda_state->devices, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "get_device_list failed \n");

    ibgda_state->dev_ids = (int *)malloc(MAX_NUM_PES_PER_NODE * sizeof(int));
    NVSHMEMI_NULL_ERROR_JMP(ibgda_state->dev_ids, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "malloc failed \n");

    ibgda_state->port_ids = (int *)malloc(MAX_NUM_PES_PER_NODE * sizeof(int));
    NVSHMEMI_NULL_ERROR_JMP(ibgda_state->port_ids, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "malloc failed \n");
    if (options->HCA_LIST_provided) {
        user_selection = 1;
        exclude_list = (options->HCA_LIST[0] == '^');
        hca_list_count = nvshmemt_parse_hca_list(options->HCA_LIST, hca_list, MAX_NUM_HCAS,
                                                 ibgda_state->log_level);
    }

    if (options->HCA_PE_MAPPING_provided) {
        if (hca_list_count) {
            NVSHMEMI_WARN_PRINT(
                "Found conflicting parameters NVSHMEM_HCA_LIST and NVSHMEM_HCA_PE_MAPPING, "
                "ignoring "
                "NVSHMEM_HCA_PE_MAPPING \n");
        } else {
            user_selection = 1;
            pe_hca_map_count = nvshmemt_parse_hca_list(
                options->HCA_LIST, pe_hca_mapping, MAX_NUM_PES_PER_NODE, ibgda_state->log_level);
        }
    }

    nic_mapping_memtype_request =
        ibgda_parse_nic_mapping_memtype_request(options->IBGDA_FORCE_NIC_BUF_MEMTYPE);
#ifdef NVSHMEM_IBGDA_SUPPORT_GPUMEM_ONLY
    if (nic_mapping_memtype_request == IBGDA_NIC_MAPPING_MEMTYPE_REQUEST_AUTO) {
        nic_mapping_memtype_request = IBGDA_NIC_MAPPING_MEMTYPE_REQUEST_GPUMEM;
    }
    if (nic_mapping_memtype_request != IBGDA_NIC_MAPPING_MEMTYPE_REQUEST_GPUMEM) {
        NVSHMEMI_ERROR_JMP(
            status, NVSHMEMX_ERROR_NOT_SUPPORTED, out,
            "GPU-initiated communication is compiled with GPU memory support only.\n");
    }
#endif

    INFO(ibgda_state->log_level,
         "Begin - Enumerating IB devices in the system ([<dev_id, device_name, num_ports>]) - \n");
    for (int i = 0; i < num_devices; i++) {
        device = (struct ibgda_device *)ibgda_state->devices + i;
        device->dev = dev_list[i];

        device->context = ftable.open_device(device->dev);
        if (!device->context) {
            INFO(ibgda_state->log_level, "open_device failed for IB device at index %d\n", i);
            continue;
        }

        const char *name = ftable.get_device_name(device->dev);
        NVSHMEMI_NULL_ERROR_JMP(name, status, NVSHMEMX_ERROR_INTERNAL, out,
                                "ibv_get_device_name failed \n");
        if (!strstr(name, "mlx5")) {
            NVSHMEMI_WARN_PRINT("device %s is not enumerated as an mlx5 device. Skipping...\n",
                                name);
            continue;
        }

        status = ftable.query_device(device->context, &device->device_attr);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "ibv_query_device failed \n");

        if (!nvshmemt_ib_common_query_mlx5_caps(device->context)) {
            NVSHMEMI_WARN_PRINT("device %s is not enumerated as an mlx5 device. Skipping...\n",
                                name);
            continue;
        }

        status = ibgda_check_gpu_mapping_nic_uar(device);
        if (status) {
            NVSHMEMI_WARN_PRINT("GPU cannot map UAR of device %s. Skipping...\n", name);
            continue;
        }

        status = ibgda_check_nic_mapping_memtypes(device, nic_mapping_memtype_request);
        if (status) {
            NVSHMEMI_WARN_PRINT(
                "device %s cannot allocate buffer on the specified memory type. Skipping...\n",
                name);
            continue;
        }

        INFO(ibgda_state->log_level,
             "Enumerated IB devices in the system - device id=%d (of %d), name=%s, num_ports=%d\n",
             i, num_devices, name, device->device_attr.phys_port_cnt);
        int device_used = 0;
        for (int p = 1; p <= device->device_attr.phys_port_cnt; p++) {
            int allowed_device = 1;
            int replicate_count = 1;
            if (hca_list_count) {
                // filter out based on user hca list
                allowed_device = exclude_list;
                for (int j = 0; j < hca_list_count; j++) {
                    if (!strcmp(hca_list[j].name, name)) {
                        if (hca_list[j].port == -1 || hca_list[j].port == p) {
                            hca_list[j].found = 1;
                            allowed_device = !exclude_list;
                        }
                    }
                }
            } else if (pe_hca_map_count) {
                // filter devices based on user hca-pe mapping
                allowed_device = 0;
                for (int j = 0; j < pe_hca_map_count; j++) {
                    if (!strcmp(pe_hca_mapping[j].name, name)) {
                        if (pe_hca_mapping[j].port == -1 || pe_hca_mapping[j].port == p) {
                            allowed_device = 1;
                            pe_hca_mapping[j].found = 1;
                            replicate_count = pe_hca_mapping[j].count;
                        }
                    }
                }
            }

            if (!allowed_device) {
                continue;
            } else {
                status = ftable.query_port(device->context, p, &device->port_attr[p - 1]);
                NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                      "ibv_port_query failed \n");

                // IBGDA supports IB and RoCE.
                if ((device->port_attr[p - 1].state != IBV_PORT_ACTIVE) ||
                    ((device->port_attr[p - 1].link_layer != IBV_LINK_LAYER_INFINIBAND) &&
                     (device->port_attr[p - 1].link_layer != IBV_LINK_LAYER_ETHERNET))) {
                    if (user_selection) {
                        NVSHMEMI_WARN_PRINT(
                            "found inactive port or port with non IB/RoCE link layer protocol, "
                            "skipping...\n");
                    }
                    continue;
                }

                status = ftable.query_gid(device->context, p, options->IB_GID_INDEX,
                                          &device->gid[p - 1]);
                NVSHMEMI_NULL_ERROR_JMP(dev_list, status, NVSHMEMX_ERROR_INTERNAL, out,
                                        "query_gid failed \n");

                device->pd = ftable.alloc_pd(device->context);
                NVSHMEMI_NULL_ERROR_JMP(device->pd, status, NVSHMEMX_ERROR_INTERNAL, out,
                                        "ibv_alloc_pd failed \n");

                for (int k = 0; k < replicate_count; k++) {
                    ibgda_state->dev_ids[offset] = i;
                    ibgda_state->port_ids[offset] = p;
                    offset++;
                }

                device_used = 1;
            }
        }

        if (!device_used) {
            status = ftable.close_device(device->context);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "ibv_port_query failed \n");
            continue;
        }

        /* Report whether we need to do atomic endianness conversions on 8 byte operands. */
        status = nvshmemt_ib_common_query_endianness_conversion_size(&atomic_host_endian_size,
                                                                     device->context);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "nvshmemt_ib_common_query_endianness_conversion_size failed.\n");
    }
    INFO(ibgda_state->log_level, "End - Enumerating IB devices in the system\n");

    ibgda_state->n_dev_ids = offset;
    INFO(ibgda_state->log_level,
         "Begin - Ordered list of devices for assignment (after processing user provdied env vars "
         "(if any))  - \n");
    for (int i = 0; i < ibgda_state->n_dev_ids; i++) {
        INFO(ibgda_state->log_level,
             "Ordered list of devices for assignment - idx=%d (of %d), device id=%d, port_num=%d\n",
             i, ibgda_state->n_dev_ids, ibgda_state->dev_ids[i], ibgda_state->port_ids[i]);

        device = (struct ibgda_device *)ibgda_state->devices + ibgda_state->dev_ids[i];
        nic_buf_on_gpumem &= device->support_nic_buf_on_gpumem;
        nic_buf_on_hostmem &= device->support_nic_buf_on_hostmem;
    }
    INFO(ibgda_state->log_level,
         "End - Ordered list of devices for assignment (after processing user provdied env vars "
         "(if any))\n");

    if (!ibgda_state->n_dev_ids) {
        INFO(
            ibgda_state->log_level,
            "no active IB device that supports GPU-initiated communication is found, exiting...\n");
        status = NVSHMEMX_ERROR_INTERNAL;
        goto out;
    }

    transport->n_devices = ibgda_state->n_dev_ids;
    transport->device_pci_paths = (char **)calloc(transport->n_devices, sizeof(char *));
    NVSHMEMI_NULL_ERROR_JMP(transport->device_pci_paths, status, NVSHMEMX_ERROR_INTERNAL, out,
                            "Unable to allocate paths for IB transport.\n");
    for (int i = 0; i < transport->n_devices; i++) {
        status = get_pci_path(i, &transport->device_pci_paths[i], transport);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "Failed to get paths for PCI devices.\n");
    }

    assert(nic_buf_on_gpumem || nic_buf_on_hostmem);
    if (nic_buf_on_gpumem) {
        ibgda_nic_buf_location = IBGDA_MEM_TYPE_GPU;
        INFO(ibgda_state->log_level, "NIC buffer will be on GPU memory.\n");
    } else {
        ibgda_nic_buf_location = IBGDA_MEM_TYPE_HOST;
        INFO(ibgda_state->log_level, "NIC buffer will be on host memory.\n");
    }

    // print devices that were not found
    if (hca_list_count) {
        for (int j = 0; j < hca_list_count; j++) {
            if (hca_list[j].found != 1) {
                NVSHMEMI_WARN_PRINT(
                    "cound not find user specified HCA name: %s port: %d, skipping\n",
                    hca_list[j].name, hca_list[j].port);
            }
        }
    } else if (pe_hca_map_count) {
        // filter devices based on user hca-pe mapping
        for (int j = 0; j < pe_hca_map_count; j++) {
            if (pe_hca_mapping[j].found != 1) {
                NVSHMEMI_WARN_PRINT(
                    "cound not find user specified HCA name: %s port: %d, skipping\n",
                    pe_hca_mapping[j].name, pe_hca_mapping[j].port);
            }
        }
    }

    status = cudaDeviceGetStreamPriorityRange(&lowest_stream_priority, &highest_stream_priority);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "stream priority query failed. \n");
    status = cudaStreamCreateWithPriority(&ibgda_state->my_stream, cudaStreamNonBlocking,
                                          highest_stream_priority);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "internal stream creation failed. \n");

    transport->host_ops.can_reach_peer = nvshmemt_ibgda_can_reach_peer;
    transport->host_ops.connect_endpoints = nvshmemt_ibgda_connect_endpoints;
    transport->host_ops.get_mem_handle = nvshmemt_ibgda_get_mem_handle;
    transport->host_ops.release_mem_handle = nvshmemt_ibgda_release_mem_handle;
    transport->host_ops.show_info = nvshmemt_ibgda_show_info;
    transport->host_ops.progress = nvshmemt_ibgda_progress;
    transport->host_ops.finalize = nvshmemt_ibgda_finalize;
    transport->host_ops.rma = NULL;
    transport->host_ops.amo = NULL;
    transport->host_ops.fence = NULL;
    transport->host_ops.quiet = NULL;
    transport->host_ops.enforce_cst = NULL;
    transport->host_ops.add_device_remote_mem_handles =
        nvshmemt_ibgda_add_device_remote_mem_handles;

    transport->attr = NVSHMEM_TRANSPORT_ATTR_CONNECTED;
    ibgda_state->options = options;
    transport->is_successfully_initialized = true;
    transport->max_op_len = 1ULL << 30;
    transport->atomic_host_endian_min_size = atomic_host_endian_size;
    transport->no_proxy = true;
    transport->type = NVSHMEM_TRANSPORT_LIB_CODE_IBGDA;
    transport->api_version = NVSHMEM_TRANSPORT_INTERFACE_VERSION;

    *t = transport;

    ibgda_state->dmabuf_support = false;
#if CUDART_VERSION >= 11070
    int flag;
    CUdevice gpu_device_id;

    status = CUPFN(ibgda_cuda_syms, cuCtxGetDevice(&gpu_device_id));
    if (status != CUDA_SUCCESS) {
        status = NVSHMEMX_ERROR_INTERNAL;
        goto out;
    }
    status =
        CUPFN(ibgda_cuda_syms,
              cuDeviceGetAttribute(&flag, CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED, gpu_device_id));
    if (status != CUDA_SUCCESS) {
        status = 0;
        cudaGetLastError();
    } else if (flag == 1) {
        ibgda_state->dmabuf_support = true;
    }
#endif

    if (ibgda_state->dmabuf_support == false) {
        if (nvshmemt_ib_common_nv_peer_mem_available() != NVSHMEMX_SUCCESS) {
            NVSHMEMI_ERROR_PRINT(
                "neither nv_peer_mem, or nvidia_peermem detected. Skipping transport.\n");
            status = NVSHMEMX_ERROR_INTERNAL;
            goto out;
        }
    }

out:
    if (status) {
        if (options) {
            free(options);
        }
    }
    // TODO: Implement cleanup
    return status;
}
