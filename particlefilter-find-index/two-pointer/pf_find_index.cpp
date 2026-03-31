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
// No array partitioning or unrolling needed — single sequential
// pass at II=1 is already optimal.
// ============================================================

static void load_vector(const data_t* src, data_t dst[MAX_PARTICLES], int n_particles) {
    #pragma HLS INLINE off
    load_vector_loop: for (int i = 0; i < n_particles; ++i) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=16384
        #pragma HLS PIPELINE II=1
        dst[i] = src[i];
    }
}

static void store_vector(data_t* dst, const data_t src[MAX_PARTICLES], int n_particles) {
    #pragma HLS INLINE off
    store_vector_loop: for (int i = 0; i < n_particles; ++i) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=16384
        #pragma HLS PIPELINE II=1
        dst[i] = src[i];
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

    // Full-size on-chip buffers — no tiling needed for O(N) algorithm.
    data_t cdf_buf[MAX_PARTICLES];
    data_t u_buf[MAX_PARTICLES];
    data_t array_x_buf[MAX_PARTICLES];
    data_t array_y_buf[MAX_PARTICLES];
    data_t xj_buf[MAX_PARTICLES];
    data_t yj_buf[MAX_PARTICLES];

    int particle_count = n_particles;
    if (particle_count <= 0) {
        return;
    }
    if (particle_count > MAX_PARTICLES) {
        particle_count = MAX_PARTICLES;
    }

    // Bulk load all inputs to on-chip BRAM
    load_vector(cdf, cdf_buf, particle_count);
    load_vector(u, u_buf, particle_count);
    load_vector(array_x, array_x_buf, particle_count);
    load_vector(array_y, array_y_buf, particle_count);

    // Two-pointer merge: single pass through both sorted arrays.
    // ptr advances through CDF, j advances through u[].
    // Each iteration either advances ptr or j — never both.
    // Total iterations <= 2*N (ptr advances at most N-1, j advances exactly N).
    // Fixed upper bound (2*16384) ensures HLS can statically bound the loop.
    int ptr = 0;
    int j = 0;
    merge_loop: for (int iter = 0; iter < 2 * 16384; ++iter) {
        #pragma HLS LOOP_TRIPCOUNT min=2 max=32768
        #pragma HLS PIPELINE II=1

        if (j >= particle_count) {
            break;
        }

        if (ptr < particle_count - 1 && cdf_buf[ptr] < u_buf[j]) {
            // CDF[ptr] too small — advance pointer
            ptr++;
        } else {
            // CDF[ptr] >= u[j] — found the match
            xj_buf[j] = array_x_buf[ptr];
            yj_buf[j] = array_y_buf[ptr];
            j++;
        }
    }

    // Bulk store outputs
    store_vector(xj, xj_buf, particle_count);
    store_vector(yj, yj_buf, particle_count);
}
