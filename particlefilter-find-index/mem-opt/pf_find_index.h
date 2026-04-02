#ifndef PF_FIND_INDEX_H
#define PF_FIND_INDEX_H

#include <ap_int.h>

typedef double data_t;
typedef ap_uint<512> bus_t;

#define MAX_PARTICLES 16384
#define PARTICLE_TILE 256

#ifdef __cplusplus
extern "C" {
#endif

void find_index_kernel(
    const bus_t* cdf,
    const bus_t* u,
    const bus_t* array_x,
    const bus_t* array_y,
    bus_t* xj,
    bus_t* yj,
    int n_particles
);

#ifdef __cplusplus
}
#endif

#endif