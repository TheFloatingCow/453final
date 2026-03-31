#include "pf_find_index.h"

// ============================================================
// two-pointer: O(N) algorithmic optimization.
//
// Key insight: both CDF[] and u[] are monotonically increasing.
//   CDF[0] <= CDF[1] <= ... <= CDF[N-1]   (cumulative weights)
//   u[0]   <= u[1]   <= ... <= u[N-1]     (u1 + j/N)
//
// Instead of searching the entire CDF for each u[j] (O(N^2)),
// we use a two-pointer merge: a single pointer into CDF that
// only moves forward. Total work = O(N) across all particles.
//
// Compared to comp-opt (parallel reverse scan, O(N^2/P)):
//   N=16384, P=8 => comp-opt ~33M iterations vs two-pointer ~32K
//
// Includes memory coalescing from mem-opt (max_widen_bitwidth=512).
// This version uses tiled ping-pong buffers for u/xj/yj to overlap
// load/compute/store between neighboring tiles.
// ============================================================

static void load_vector(const data_t* src, data_t dst[MAX_PARTICLES], int n_particles) {
    #pragma HLS INLINE off
    load_vector_loop: for (int i = 0; i < n_particles; ++i) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=16384
        #pragma HLS PIPELINE II=1
        dst[i] = src[i];
    }
}

static void load_tile(const data_t* src, data_t dst[PARTICLE_TILE], int base, int tile_count) {
    #pragma HLS INLINE off
    load_tile_loop: for (int i = 0; i < tile_count; ++i) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=256
        #pragma HLS PIPELINE II=1
        dst[i] = src[base + i];
    }
}

static void store_tile(data_t* dst, const data_t src[PARTICLE_TILE], int base, int tile_count) {
    #pragma HLS INLINE off
    store_tile_loop: for (int i = 0; i < tile_count; ++i) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=256
        #pragma HLS PIPELINE II=1
        dst[base + i] = src[i];
    }
}

static void compute_tile(
    const data_t cdf_buf[MAX_PARTICLES],
    const data_t array_x_buf[MAX_PARTICLES],
    const data_t array_y_buf[MAX_PARTICLES],
    const data_t u_buf[PARTICLE_TILE],
    data_t xj_buf[PARTICLE_TILE],
    data_t yj_buf[PARTICLE_TILE],
    int n_particles,
    int tile_count,
    int start_ptr,
    int* end_ptr
) {
    #pragma HLS INLINE off

    int ptr = start_ptr;
    if (ptr < 0) {
        ptr = 0;
    }
    if (ptr >= n_particles) {
        ptr = n_particles - 1;
    }

    compute_tile_loop: for (int j = 0; j < tile_count; ++j) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=256
        #pragma HLS PIPELINE II=1

        data_t value = u_buf[j];
        advance_ptr_loop: while ((ptr + 1) < n_particles && cdf_buf[ptr] < value) {
            #pragma HLS LOOP_TRIPCOUNT min=0 max=16383
            ptr++;
        }

        xj_buf[j] = array_x_buf[ptr];
        yj_buf[j] = array_y_buf[ptr];
    }

    *end_ptr = ptr;
}

extern "C" void find_index_kernel(
    const data_t* cdf,
    const data_t* u,
    const data_t* array_x,
    const data_t* array_y,
    data_t* xj,
    data_t* yj,
    int n_particles
) {
    // Memory coalescing (inherited from mem-opt)
    #pragma HLS INTERFACE m_axi port=cdf offset=slave bundle=gmem0 depth=16384 max_widen_bitwidth=512
    #pragma HLS INTERFACE m_axi port=u offset=slave bundle=gmem1 depth=16384 max_widen_bitwidth=512
    #pragma HLS INTERFACE m_axi port=array_x offset=slave bundle=gmem2 depth=16384 max_widen_bitwidth=512
    #pragma HLS INTERFACE m_axi port=array_y offset=slave bundle=gmem3 depth=16384 max_widen_bitwidth=512
    #pragma HLS INTERFACE m_axi port=xj offset=slave bundle=gmem4 depth=16384 max_widen_bitwidth=512
    #pragma HLS INTERFACE m_axi port=yj offset=slave bundle=gmem5 depth=16384 max_widen_bitwidth=512
    #pragma HLS INTERFACE s_axilite port=cdf bundle=control
    #pragma HLS INTERFACE s_axilite port=u bundle=control
    #pragma HLS INTERFACE s_axilite port=array_x bundle=control
    #pragma HLS INTERFACE s_axilite port=array_y bundle=control
    #pragma HLS INTERFACE s_axilite port=xj bundle=control
    #pragma HLS INTERFACE s_axilite port=yj bundle=control
    #pragma HLS INTERFACE s_axilite port=n_particles bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    // Keep CDF and source arrays cached on-chip.
    data_t cdf_buf[MAX_PARTICLES];
    data_t array_x_buf[MAX_PARTICLES];
    data_t array_y_buf[MAX_PARTICLES];

    // Tiled ping-pong buffers for streaming u in and xj/yj out.
    data_t u_buf_A[PARTICLE_TILE];
    data_t u_buf_B[PARTICLE_TILE];
    data_t xj_buf_A[PARTICLE_TILE];
    data_t xj_buf_B[PARTICLE_TILE];
    data_t yj_buf_A[PARTICLE_TILE];
    data_t yj_buf_B[PARTICLE_TILE];

    int particle_count = n_particles;
    if (particle_count <= 0) {
        return;
    }
    if (particle_count > MAX_PARTICLES) {
        particle_count = MAX_PARTICLES;
    }

    // Bulk load CDF and state arrays to on-chip BRAM.
    load_vector(cdf, cdf_buf, particle_count);
    load_vector(array_x, array_x_buf, particle_count);
    load_vector(array_y, array_y_buf, particle_count);

    int n_tiles = (particle_count + PARTICLE_TILE - 1) / PARTICLE_TILE;
    int cdf_ptr = 0;

    // Prologue: preload tile 0 into buffer A.
    {
        int first_tile_count = PARTICLE_TILE;
        if (first_tile_count > particle_count) {
            first_tile_count = particle_count;
        }
        load_tile(u, u_buf_A, 0, first_tile_count);
    }

    // Main tiled loop with ping-pong buffering.
    tile_loop: for (int t = 0; t < n_tiles; ++t) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=64

        int base = t * PARTICLE_TILE;
        int tile_count = PARTICLE_TILE;
        if (base + tile_count > particle_count) {
            tile_count = particle_count - base;
        }

        int next_base = (t + 1) * PARTICLE_TILE;
        int next_tile_count = PARTICLE_TILE;
        if (next_base + next_tile_count > particle_count) {
            next_tile_count = particle_count - next_base;
        }
        bool has_next = (t + 1) < n_tiles;

        int next_cdf_ptr = cdf_ptr;
        if ((t & 1) == 0) {
            #pragma HLS DATAFLOW
            compute_tile(cdf_buf, array_x_buf, array_y_buf, u_buf_A, xj_buf_A, yj_buf_A, particle_count, tile_count, cdf_ptr, &next_cdf_ptr);
            if (has_next) {
                load_tile(u, u_buf_B, next_base, next_tile_count);
            }
            store_tile(xj, xj_buf_A, base, tile_count);
            store_tile(yj, yj_buf_A, base, tile_count);
        } else {
            #pragma HLS DATAFLOW
            compute_tile(cdf_buf, array_x_buf, array_y_buf, u_buf_B, xj_buf_B, yj_buf_B, particle_count, tile_count, cdf_ptr, &next_cdf_ptr);
            if (has_next) {
                load_tile(u, u_buf_A, next_base, next_tile_count);
            }
            store_tile(xj, xj_buf_B, base, tile_count);
            store_tile(yj, yj_buf_B, base, tile_count);
        }
        cdf_ptr = next_cdf_ptr;
    }
}
