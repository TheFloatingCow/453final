#include "pf_find_index.h"

#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>

// ============================================================
// Packing helpers for 512-bit bus words
// ============================================================
#define WORD_ELEMS 8
#define MAX_WORDS ((MAX_PARTICLES + WORD_ELEMS - 1) / WORD_ELEMS)

static bus_t pack_word(const data_t* src, int base, int n_valid) {
    bus_t word = 0;

    for (int k = 0; k < WORD_ELEMS; ++k) {
        union {
            uint64_t u;
            data_t d;
        } conv;

        if (k < n_valid) {
            conv.d = src[base + k];
        } else {
            conv.d = 0.0;
        }

        word.range((k + 1) * 64 - 1, k * 64) = (ap_uint<64>)conv.u;
    }

    return word;
}

static void unpack_word(bus_t word, data_t* dst, int base, int n_valid) {
    for (int k = 0; k < WORD_ELEMS; ++k) {
        if (k < n_valid) {
            union {
                uint64_t u;
                data_t d;
            } conv;

            conv.u = (uint64_t)word.range((k + 1) * 64 - 1, k * 64);
            dst[base + k] = conv.d;
        }
    }
}

static void pack_array(const data_t* src, bus_t* dst, int n_particles) {
    int num_words = (n_particles + WORD_ELEMS - 1) / WORD_ELEMS;

    for (int w = 0; w < num_words; ++w) {
        int base = w * WORD_ELEMS;
        int remaining = n_particles - base;
        int n_valid = (remaining >= WORD_ELEMS) ? WORD_ELEMS : remaining;
        dst[w] = pack_word(src, base, n_valid);
    }
}

static void unpack_array(const bus_t* src, data_t* dst, int n_particles) {
    int num_words = (n_particles + WORD_ELEMS - 1) / WORD_ELEMS;

    for (int w = 0; w < num_words; ++w) {
        int base = w * WORD_ELEMS;
        int remaining = n_particles - base;
        int n_valid = (remaining >= WORD_ELEMS) ? WORD_ELEMS : remaining;
        unpack_word(src[w], dst, base, n_valid);
    }
}

// Golden reference mapping to ex_particle_OPENMP_seq.c:
// findIndex() at lines 285-298.
static int openmp_find_index(const data_t* cdf, int length_cdf, data_t value) {
    int index = -1;
    for (int x = 0; x < length_cdf; ++x) {
        if (cdf[x] >= value) {
            index = x;
            break;
        }
    }
    if (index == -1) {
        return length_cdf - 1;
    }
    return index;
}

// Golden reference mapping to ex_particle_OPENMP_seq.c:
// resampling loop body at lines 487-494 inside the assignment window 453-494.
static void openmp_resample_reference(
    const data_t* cdf,
    const data_t* u,
    const data_t* array_x,
    const data_t* array_y,
    data_t* xj,
    data_t* yj,
    int n_particles
) {
    for (int j = 0; j < n_particles; ++j) {
        int i = openmp_find_index(cdf, n_particles, u[j]);
        if (i == -1) {
            i = n_particles - 1;
        }
        xj[j] = array_x[i];
        yj[j] = array_y[i];
    }
}

// Golden reference mapping to ex_particle_OPENMP_seq.c global LCG constants.
static int lcg_a = 1103515245;
static int lcg_c = 12345;
static long lcg_m = INT_MAX;

// Golden reference mapping to ex_particle_OPENMP_seq.c:
// randu() at lines 77-82, reused to match u1 generation in line 477.
static data_t openmp_randu(int* seed, int index) {
    int num = lcg_a * seed[index] + lcg_c;
    seed[index] = num % lcg_m;
    return std::fabs(seed[index] / (data_t)lcg_m);
}

// Golden reference mapping to ex_particle_OPENMP_seq.c:
// CDF/u setup at lines 471-481 inside the assignment window 453-494.
static void build_inputs_like_openmp(
    int n_particles,
    int seed,
    data_t* cdf,
    data_t* u,
    data_t* array_x,
    data_t* array_y
) {
    data_t weights[MAX_PARTICLES] = {0.0};
    int seed_arr[1] = {seed};

    data_t weight_sum = 0.0;
    for (int i = 0; i < n_particles; ++i) {
        weights[i] = 0.001 + openmp_randu(seed_arr, 0);
        weight_sum += weights[i];
        array_x[i] = 100.0 + (data_t)(i * 1.125);
        array_y[i] = -50.0 + (data_t)(i * 0.875);
    }

    for (int i = 0; i < n_particles; ++i) {
        weights[i] /= weight_sum;
    }

    cdf[0] = weights[0];
    for (int i = 1; i < n_particles; ++i) {
        cdf[i] = weights[i] + cdf[i - 1];
    }

    data_t u1 = (1.0 / (data_t)n_particles) * openmp_randu(seed_arr, 0);
    for (int i = 0; i < n_particles; ++i) {
        u[i] = u1 + i / (data_t)n_particles;
    }
}

// Runs one equivalence check for the find_index_kernel port.
static int run_case(int n_particles, int trial_seed) {
    data_t cdf[MAX_PARTICLES] = {0};
    data_t u[MAX_PARTICLES] = {0};
    data_t array_x[MAX_PARTICLES] = {0};
    data_t array_y[MAX_PARTICLES] = {0};

    data_t xj_hw[MAX_PARTICLES] = {0};
    data_t yj_hw[MAX_PARTICLES] = {0};
    data_t xj_sw[MAX_PARTICLES] = {0};
    data_t yj_sw[MAX_PARTICLES] = {0};

    bus_t cdf_bus[MAX_WORDS];
    bus_t u_bus[MAX_WORDS];
    bus_t array_x_bus[MAX_WORDS];
    bus_t array_y_bus[MAX_WORDS];
    bus_t xj_bus[MAX_WORDS];
    bus_t yj_bus[MAX_WORDS];

    for (int i = 0; i < MAX_WORDS; ++i) {
        cdf_bus[i] = 0;
        u_bus[i] = 0;
        array_x_bus[i] = 0;
        array_y_bus[i] = 0;
        xj_bus[i] = 0;
        yj_bus[i] = 0;
    }

    build_inputs_like_openmp(n_particles, trial_seed, cdf, u, array_x, array_y);

    pack_array(cdf, cdf_bus, n_particles);
    pack_array(u, u_bus, n_particles);
    pack_array(array_x, array_x_bus, n_particles);
    pack_array(array_y, array_y_bus, n_particles);

    openmp_resample_reference(cdf, u, array_x, array_y, xj_sw, yj_sw, n_particles);
    find_index_kernel(cdf_bus, u_bus, array_x_bus, array_y_bus, xj_bus, yj_bus, n_particles);

    unpack_array(xj_bus, xj_hw, n_particles);
    unpack_array(yj_bus, yj_hw, n_particles);

    int errors = 0;
    for (int i = 0; i < n_particles; ++i) {
        if (std::fabs(xj_hw[i] - xj_sw[i]) > 1e-9 || std::fabs(yj_hw[i] - yj_sw[i]) > 1e-9) {
            ++errors;
            if (errors < 6) {
                std::printf(
                    "Mismatch n=%d seed=%d idx=%d: expected (%0.6f, %0.6f), got (%0.6f, %0.6f)\n",
                    n_particles,
                    trial_seed,
                    i,
                    xj_sw[i],
                    yj_sw[i],
                    xj_hw[i],
                    yj_hw[i]
                );
            }
        }
    }

    return errors;
}

int main() {
    const int sizes[] = {1, 2, 3, 16, 255, 256, 257, 1024};
    const int n_sizes = (int)(sizeof(sizes) / sizeof(sizes[0]));
    int total_errors = 0;

    for (int s = 0; s < n_sizes; ++s) {
        for (int trial = 0; trial < 10; ++trial) {
            int seed = 1000 + sizes[s] * 31 + trial * 7;
            int errors = run_case(sizes[s], seed);
            total_errors += errors;
        }
    }

    if (total_errors == 0) {
        std::printf("find_index_kernel OpenMP-equivalence test PASSED\n");
        return 0;
    }

    std::printf("find_index_kernel OpenMP-equivalence test FAILED with %d mismatches\n", total_errors);
    return 1;
}