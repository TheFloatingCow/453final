#include "pf_find_index.h"
#include <ap_int.h>

#define PARALLEL_FACTOR 8
#define BUS_WIDTH_BITS 512
#define WORD_ELEMS 8

typedef ap_uint<BUS_WIDTH_BITS> bus_t;

// ============================================================
// Bit conversion helpers
// ============================================================
static data_t bits_to_double(ap_uint<64> bits) {
    union {
        unsigned long long u;
        data_t d;
    } conv;
    conv.u = (unsigned long long)bits;
    return conv.d;
}

static ap_uint<64> double_to_bits(data_t val) {
    union {
        unsigned long long u;
        data_t d;
    } conv;
    conv.d = val;
    return (ap_uint<64>)conv.u;
}

// ============================================================
// Wide AXI load/store helpers
// Each 512-bit word holds 8 doubles.
// ============================================================
static void load_vector_wide(
    const bus_t* src,
    data_t dst[MAX_PARTICLES],
    int n_particles
) {
    #pragma HLS INLINE off

    int num_words = (n_particles + WORD_ELEMS - 1) / WORD_ELEMS;

    for (int w = 0; w < num_words; ++w) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=2048
        #pragma HLS PIPELINE II=1

        bus_t word = src[w];

        for (int k = 0; k < WORD_ELEMS; ++k) {
            #pragma HLS UNROLL
            int idx = w * WORD_ELEMS + k;
            if (idx < n_particles) {
                ap_uint<64> bits = word.range((k + 1) * 64 - 1, k * 64);
                dst[idx] = bits_to_double(bits);
            }
        }
    }
}

static void load_tile_wide(
    const bus_t* src,
    data_t dst[PARTICLE_TILE],
    int base,
    int tile_count
) {
    #pragma HLS INLINE off

    int start_word = base / WORD_ELEMS;
    int end_word   = (base + tile_count - 1) / WORD_ELEMS;

    for (int w = start_word; w <= end_word; ++w) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=32
        #pragma HLS PIPELINE II=1

        bus_t word = src[w];

        for (int k = 0; k < WORD_ELEMS; ++k) {
            #pragma HLS UNROLL
            int global_idx = w * WORD_ELEMS + k;
            int local_idx = global_idx - base;

            if (local_idx >= 0 && local_idx < tile_count) {
                ap_uint<64> bits = word.range((k + 1) * 64 - 1, k * 64);
                dst[local_idx] = bits_to_double(bits);
            }
        }
    }
}

static void store_tile_wide(
    bus_t* dst,
    const data_t src[PARTICLE_TILE],
    int base,
    int tile_count
) {
    #pragma HLS INLINE off

    int start_word = base / WORD_ELEMS;
    int end_word   = (base + tile_count - 1) / WORD_ELEMS;

    for (int w = start_word; w <= end_word; ++w) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=32
        #pragma HLS PIPELINE II=1

        bus_t word = 0;

        for (int k = 0; k < WORD_ELEMS; ++k) {
            #pragma HLS UNROLL
            int global_idx = w * WORD_ELEMS + k;
            int local_idx = global_idx - base;

            ap_uint<64> bits = 0;
            if (local_idx >= 0 && local_idx < tile_count) {
                bits = double_to_bits(src[local_idx]);
            }

            word.range((k + 1) * 64 - 1, k * 64) = bits;
        }

        dst[w] = word;
    }
}

// ============================================================
// Compute tile: unchanged algorithmically
// ============================================================
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

    int num_chunks = (n_particles + PARALLEL_FACTOR - 1) / PARALLEL_FACTOR;

    for (int j = 0; j < tile_count; ++j) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=256

        data_t value = u_buf[j];
        int index = n_particles - 1;

        find_chunk_loop:
        for (int c = num_chunks - 1; c >= 0; --c) {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=2048
            #pragma HLS PIPELINE II=1

            int base_idx = c * PARALLEL_FACTOR;

            search_parallel:
            for (int p = PARALLEL_FACTOR - 1; p >= 0; --p) {
                #pragma HLS UNROLL
                int idx = base_idx + p;
                if (idx < n_particles && cdf_buf[idx] >= value) {
                    index = idx;
                }
            }
        }

        xj_buf[j] = array_x_buf[index];
        yj_buf[j] = array_y_buf[index];
    }
}

// ============================================================
// Top kernel with real 512-bit AXI ports
// New in this version:
//   - Double-buffered u_buf, xj_buf, yj_buf arrays (A/B copies)
//   - #pragma HLS DATAFLOW inside the tile loop to overlap
//     load(tile N+1) with compute(tile N) and store(tile N-1)
//   - Ping-pong selection toggled each iteration via (t & 1)
// ============================================================
extern "C" void find_index_kernel(
    const bus_t* cdf,
    const bus_t* u,
    const bus_t* array_x,
    const bus_t* array_y,
    bus_t* xj,
    bus_t* yj,
    int n_particles
) {
    #pragma HLS INTERFACE m_axi port=cdf     offset=slave bundle=gmem0 depth=2048 max_widen_bitwidth=512
    #pragma HLS INTERFACE m_axi port=u       offset=slave bundle=gmem1 depth=2048 max_widen_bitwidth=512
    #pragma HLS INTERFACE m_axi port=array_x offset=slave bundle=gmem2 depth=2048 max_widen_bitwidth=512
    #pragma HLS INTERFACE m_axi port=array_y offset=slave bundle=gmem3 depth=2048 max_widen_bitwidth=512
    #pragma HLS INTERFACE m_axi port=xj      offset=slave bundle=gmem4 depth=2048 max_widen_bitwidth=512
    #pragma HLS INTERFACE m_axi port=yj      offset=slave bundle=gmem5 depth=2048 max_widen_bitwidth=512

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

    load_vector_wide(cdf, cdf_buf, particle_count);
    load_vector_wide(array_x, array_x_buf, particle_count);
    load_vector_wide(array_y, array_y_buf, particle_count);

    int n_tiles = (particle_count + PARTICLE_TILE - 1) / PARTICLE_TILE;

    // Prologue: load first tile into buffer A
    {
        int tile_count = PARTICLE_TILE;
        if (tile_count > particle_count) {
            tile_count = particle_count;
        }
        load_tile_wide(u, u_buf_A, 0, tile_count);
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
                load_tile_wide(u, u_buf_B, next_base, next_tile_count);
            }
            store_tile_wide(xj, xj_buf_A, base, tile_count);
            store_tile_wide(yj, yj_buf_A, base, tile_count);
        } else {
            // Odd iteration: compute on B, load next into A
            #pragma HLS DATAFLOW
            compute_tile(cdf_buf, array_x_buf, array_y_buf, u_buf_B, xj_buf_B, yj_buf_B, particle_count, tile_count);
            if (has_next) {
                load_tile_wide(u, u_buf_A, next_base, next_tile_count);
            }
            store_tile_wide(xj, xj_buf_B, base, tile_count);
            store_tile_wide(yj, yj_buf_B, base, tile_count);
        }
    }
}