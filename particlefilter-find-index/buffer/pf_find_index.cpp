#include "pf_find_index.h"

static void load_vector(const data_t* src, data_t dst[MAX_PARTICLES], int n_particles) {
    #pragma HLS INLINE off
    load_vector_loop: for (int i = 0; i < n_particles; ++i) {
        #pragma HLS PIPELINE II=1
        dst[i] = src[i];
    }
}

static void load_tile(const data_t* src, data_t dst[PARTICLE_TILE], int base, int tile_count) {
    #pragma HLS INLINE off
    load_tile_loop: for (int i = 0; i < tile_count; ++i) {
        #pragma HLS PIPELINE II=1
        dst[i] = src[base + i];
    }
}

static void store_tile(data_t* dst, const data_t src[PARTICLE_TILE], int base, int tile_count) {
    #pragma HLS INLINE off
    store_tile_loop: for (int i = 0; i < tile_count; ++i) {
        #pragma HLS PIPELINE II=1
        dst[base + i] = src[i];
    }
}

static int find_first_geq(const data_t cdf[MAX_PARTICLES], int n_particles, data_t value) {
    #pragma HLS INLINE off
    int index = n_particles - 1;

    find_index_loop: for (int i = 0; i < n_particles; ++i) {
        if (cdf[i] >= value) {
            index = i;
            break;
        }
    }

    return index;
}

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
        int index = find_first_geq(cdf_buf, n_particles, u_buf[j]);
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
