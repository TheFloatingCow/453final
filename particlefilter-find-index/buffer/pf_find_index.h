#ifndef PF_FIND_INDEX_H
#define PF_FIND_INDEX_H

typedef double data_t;

#define MAX_PARTICLES 16384
#define PARTICLE_TILE 256

#ifdef __cplusplus
extern "C" {
#endif

void find_index_kernel(
    const data_t* cdf,
    const data_t* u,
    const data_t* array_x,
    const data_t* array_y,
    data_t* xj,
    data_t* yj,
    int n_particles
);

#ifdef __cplusplus
}
#endif

#endif
