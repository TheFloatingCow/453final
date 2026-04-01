#include "pf_find_index.h"

// log2(MAX_PARTICLES) = log2(16384) = 14 binary search steps
#define BSEARCH_STEPS 14

// ============================================================
// ping-pong: builds on mem-opt (buffer + compute + memory opt).
// New in this version:
//   - Double-buffered u_buf, xj_buf, yj_buf arrays (A/B copies)
//   - #pragma HLS DATAFLOW inside the tile loop to overlap
//     load(tile N+1) with compute(tile N) and store(tile N-1)
//   - Binary search replaces O(N) linear scan per query -> O(log N)
//     Total compute per tile: O(TILE * log N) instead of O(TILE * N)
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

// Compute tile: binary search on sorted CDF for each u[j].
// Finds the smallest index such that CDF[index] >= u[j].
// Fixed BSEARCH_STEPS (14) iterations — fully deterministic for HLS.
static void compute_tile(
    const data_t cdf_buf[MAX_PARTICLES],
    const data_t array_x_buf[MAX_PARTICLES],
    const data_t array_y_buf[MAX_PARTICLES],
    const data_t u_buf[PARTICLE_TILE],
    data_t xj_buf[PARTICLE_TILE],
    data_t yj_buf[PARTICLE_TILE],
    int n_particles,
    int tile_count
) {
    #pragma HLS INLINE off

    compute_tile_loop: for (int j = 0; j < tile_count; ++j) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=256
        data_t value = u_buf[j];

        // Binary search: find first index where CDF[index] >= value
        int lo = 0;
        int hi = n_particles - 1;

        bsearch_loop: for (int step = 0; step < BSEARCH_STEPS; ++step) {
            #pragma HLS PIPELINE II=1
            int mid = lo + ((hi - lo) >> 1);
            if (cdf_buf[mid] < value) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        xj_buf[j] = array_x_buf[lo];
        yj_buf[j] = array_y_buf[lo];
    }
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

    data_t cdf_buf[MAX_PARTICLES];

    data_t array_x_buf[MAX_PARTICLES];
    data_t array_y_buf[MAX_PARTICLES];

    // Ping-pong double buffers for tile data.
    // While compute uses buffer A, load fills buffer B (and vice versa).
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

    load_vector(cdf, cdf_buf, particle_count);
    load_vector(array_x, array_x_buf, particle_count);
    load_vector(array_y, array_y_buf, particle_count);

    int n_tiles = (particle_count + PARTICLE_TILE - 1) / PARTICLE_TILE;

    // Prologue: load first tile into buffer A
    {
        int tile_count = PARTICLE_TILE;
        if (tile_count > particle_count) {
            tile_count = particle_count;
        }
        load_tile(u, u_buf_A, 0, tile_count);
    }

    // Main loop: overlap load(next) + compute(current) + store(prev)
    // using ping-pong buffers and DATAFLOW.
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
        bool has_next = (t + 1 < n_tiles);

        if ((t & 1) == 0) {
            // Even iteration: compute on A, load next into B
            #pragma HLS DATAFLOW
            compute_tile(cdf_buf, array_x_buf, array_y_buf, u_buf_A, xj_buf_A, yj_buf_A, particle_count, tile_count);
            if (has_next) {
                load_tile(u, u_buf_B, next_base, next_tile_count);
            }
            store_tile(xj, xj_buf_A, base, tile_count);
            store_tile(yj, yj_buf_A, base, tile_count);
        } else {
            // Odd iteration: compute on B, load next into A
            #pragma HLS DATAFLOW
            compute_tile(cdf_buf, array_x_buf, array_y_buf, u_buf_B, xj_buf_B, yj_buf_B, particle_count, tile_count);
            if (has_next) {
                load_tile(u, u_buf_A, next_base, next_tile_count);
            }
            store_tile(xj, xj_buf_B, base, tile_count);
            store_tile(yj, yj_buf_B, base, tile_count);
        }
    }
}
