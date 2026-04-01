#include "pf_find_index.h"

// Number of CDF elements compared in parallel per cycle.
// cdf_buf is cyclic-partitioned by this factor so each
// comparison reads from a separate BRAM bank.
#define PARALLEL_FACTOR 8

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

// Optimized compute: replaces the sequential break-based linear scan
// with a reverse-scan approach that enables pipelining and parallelism.
//
// Key changes from buffer version:
//   1. CDF is cyclic-partitioned by PARALLEL_FACTOR (array partitioning)
//   2. Inner comparison loop is fully unrolled (parallelization)
//   3. Outer chunk loop is pipelined at II=1 (pipelining)
//
// The reverse scan (high-to-low) eliminates the break: every qualifying
// cdf[idx] >= value overwrites index, and the last (lowest) write wins,
// producing the same result as the original forward scan with break.
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

    // Partition CDF into PARALLEL_FACTOR banks for parallel reads.
    // Indices 0,8,16,... -> bank 0; 1,9,17,... -> bank 1; etc.
    // Reading indices [base..base+7] hits 8 distinct banks simultaneously.
    #pragma HLS ARRAY_PARTITION variable=cdf_buf cyclic factor=8 dim=1

    int num_chunks = (n_particles + PARALLEL_FACTOR - 1) / PARALLEL_FACTOR;

    compute_tile_loop: for (int j = 0; j < tile_count; ++j) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=256
        data_t value = u_buf[j];
        int index = n_particles - 1;

        // Reverse scan: from highest chunk down to chunk 0.
        // Within each chunk, PARALLEL_FACTOR comparisons execute in parallel.
        find_chunk_loop: for (int c = num_chunks - 1; c >= 0; --c) {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=1024
            #pragma HLS PIPELINE II=1

            int base = c * PARALLEL_FACTOR;

            search_parallel: for (int p = PARALLEL_FACTOR - 1; p >= 0; --p) {
                #pragma HLS UNROLL
                int idx = base + p;
                if (idx < n_particles && cdf_buf[idx] >= value) {
                    index = idx;
                }
            }
        }

        xj_buf[j] = array_x_buf[index];
        yj_buf[j] = array_y_buf[index];
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
    #pragma HLS INTERFACE m_axi port=cdf offset=slave bundle=gmem0 depth=16384
    #pragma HLS INTERFACE m_axi port=u offset=slave bundle=gmem1 depth=16384
    #pragma HLS INTERFACE m_axi port=array_x offset=slave bundle=gmem2 depth=16384
    #pragma HLS INTERFACE m_axi port=array_y offset=slave bundle=gmem3 depth=16384
    #pragma HLS INTERFACE m_axi port=xj offset=slave bundle=gmem4 depth=16384
    #pragma HLS INTERFACE m_axi port=yj offset=slave bundle=gmem5 depth=16384
    #pragma HLS INTERFACE s_axilite port=cdf bundle=control
    #pragma HLS INTERFACE s_axilite port=u bundle=control
    #pragma HLS INTERFACE s_axilite port=array_x bundle=control
    #pragma HLS INTERFACE s_axilite port=array_y bundle=control
    #pragma HLS INTERFACE s_axilite port=xj bundle=control
    #pragma HLS INTERFACE s_axilite port=yj bundle=control
    #pragma HLS INTERFACE s_axilite port=n_particles bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    data_t cdf_buf[MAX_PARTICLES];
    #pragma HLS ARRAY_PARTITION variable=cdf_buf cyclic factor=8 dim=1

    data_t array_x_buf[MAX_PARTICLES];
    data_t array_y_buf[MAX_PARTICLES];
    data_t u_buf[PARTICLE_TILE];
    data_t xj_buf[PARTICLE_TILE];
    data_t yj_buf[PARTICLE_TILE];

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

    tile_loop: for (int base = 0; base < particle_count; base += PARTICLE_TILE) {
        int tile_count = PARTICLE_TILE;
        if (base + tile_count > particle_count) {
            tile_count = particle_count - base;
        }

        load_tile(u, u_buf, base, tile_count);
        compute_tile(cdf_buf, array_x_buf, array_y_buf, u_buf, xj_buf, yj_buf, particle_count, tile_count);
        store_tile(xj, xj_buf, base, tile_count);
        store_tile(yj, yj_buf, base, tile_count);
    }
}
