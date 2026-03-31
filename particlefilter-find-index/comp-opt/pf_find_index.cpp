#include "pf_find_index.h"

// Number of CDF elements compared in parallel per cycle.
// cdf_buf is cyclic-partitioned by this factor so each
// comparison reads from a separate BRAM bank.
#define PARALLEL_FACTOR 8

static void load_vector(const data_t* src, data_t dst[MAX_PARTICLES], int n_particles) {
    #pragma HLS INLINE off

    for (int i = 0; i < MAX_PARTICLES; ++i) {
        #pragma HLS PIPELINE II=1
        if (i < n_particles) {
            dst[i] = src[i];
        }
    }
}

static void load_tile(const data_t* src, data_t dst[PARTICLE_TILE], int base, int tile_count) {
    #pragma HLS INLINE off

    for (int i = 0; i < PARTICLE_TILE; ++i) {
        #pragma HLS PIPELINE II=1
        if (i < tile_count) {
            dst[i] = src[base + i];
        }
    }
}

static void store_tile(data_t* dst, const data_t src[PARTICLE_TILE], int base, int tile_count) {
    #pragma HLS INLINE off
store_tile_loop:
    for (int i = 0; i < PARTICLE_TILE; ++i) {
        #pragma HLS PIPELINE II=1
        if (i < tile_count) {
            dst[base + i] = src[i];
        }
    }
}

// Optimized compute: keeps the reverse-scan + parallel compare structure.
//
// Preserved optimizations:
//   1. CDF cyclic partitioning by PARALLEL_FACTOR
//   2. Inner comparison loop fully unrolled
//   3. Outer chunk loop pipelined at II=1
//
// Changes made only to help HLS estimate latency:
//   - fixed upper bound on j loop
//   - fixed upper bound on chunk loop
//   - guards inside loops instead of runtime loop limits
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

    #pragma HLS ARRAY_PARTITION variable=cdf_buf cyclic factor=8 dim=1

    const int NUM_CHUNKS = (MAX_PARTICLES + PARALLEL_FACTOR - 1) / PARALLEL_FACTOR;


    for (int j = 0; j < PARTICLE_TILE; ++j) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=256

        if (j < tile_count) {
            data_t value = u_buf[j];
            int index = n_particles - 1;

    
            for (int c = NUM_CHUNKS - 1; c >= 0; --c) {
                #pragma HLS LOOP_TRIPCOUNT min=1 max=2048
                #pragma HLS PIPELINE II=1

                int base = c * PARALLEL_FACTOR;

          
                for (int p = PARALLEL_FACTOR - 1; p >= 0; --p) {
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

    const int NUM_TILES = MAX_PARTICLES / PARTICLE_TILE;


    for (int t = 0; t < NUM_TILES; ++t) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=64

        int base = t * PARTICLE_TILE;

        if (base < particle_count) {
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
}