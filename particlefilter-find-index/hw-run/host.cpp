// ============================================================
// host.cpp — OpenCL host for find_index_kernel on Alveo U50
// hw-run: large input (16384 particles), looped for >1s timing
// ============================================================

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "xcl2.hpp"
#include "pf_find_index.h"

// hw-run uses full particle count for performance measurement
#define N_PARTICLES_RUN 16384
// Number of kernel invocations to exceed 1 second total
// (simulates multi-frame particle filter tracking)
#define NUM_FRAMES 10000

// -------------------- OpenMP-equivalent golden reference --------------------
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

// LCG constants from ex_particle_OPENMP_seq.c
static int lcg_a = 1103515245;
static int lcg_c = 12345;
static long lcg_m = INT_MAX;

static data_t openmp_randu(int* seed, int index) {
    int num = lcg_a * seed[index] + lcg_c;
    seed[index] = num % lcg_m;
    return std::fabs(seed[index] / (data_t)lcg_m);
}

static void build_inputs(
    int n_particles,
    int seed,
    data_t* cdf,
    data_t* u,
    data_t* array_x,
    data_t* array_y
) {
    int seed_arr[1] = {seed};

    data_t weight_sum = 0.0;
    for (int i = 0; i < n_particles; ++i) {
        data_t w = 0.001 + openmp_randu(seed_arr, 0);
        cdf[i] = w;
        weight_sum += w;
        array_x[i] = 100.0 + (data_t)(i * 1.125);
        array_y[i] = -50.0 + (data_t)(i * 0.875);
    }

    for (int i = 0; i < n_particles; ++i) {
        cdf[i] /= weight_sum;
    }

    for (int i = 1; i < n_particles; ++i) {
        cdf[i] = cdf[i] + cdf[i - 1];
    }

    data_t u1 = (1.0 / (data_t)n_particles) * openmp_randu(seed_arr, 0);
    for (int i = 0; i < n_particles; ++i) {
        u[i] = u1 + i / (data_t)n_particles;
    }
}

// -------------------- Main --------------------
int main(int argc, char** argv) {

    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <xclbin>\n";
        return EXIT_FAILURE;
    }
    std::string binaryFile = argv[1];

    int n_particles = N_PARTICLES_RUN;
    int num_frames = NUM_FRAMES;

    std::cout << "HOST: n_particles=" << n_particles
              << " num_frames=" << num_frames
              << " (hw-run performance)" << std::endl;

    // -------------------- Allocate host memory (page-aligned) --------------------
    std::vector<data_t, aligned_allocator<data_t>> cdf(n_particles);
    std::vector<data_t, aligned_allocator<data_t>> u(n_particles);
    std::vector<data_t, aligned_allocator<data_t>> array_x(n_particles);
    std::vector<data_t, aligned_allocator<data_t>> array_y(n_particles);
    std::vector<data_t, aligned_allocator<data_t>> xj_hw(n_particles, 0.0);
    std::vector<data_t, aligned_allocator<data_t>> yj_hw(n_particles, 0.0);

    std::vector<data_t> xj_sw(n_particles, 0.0);
    std::vector<data_t> yj_sw(n_particles, 0.0);

    // Build inputs using OpenMP-equivalent LCG
    build_inputs(n_particles, 42, cdf.data(), u.data(), array_x.data(), array_y.data());

    // Compute CPU golden reference (once, for verification)
    std::cout << "Running CPU golden reference..." << std::endl;
    for (int j = 0; j < n_particles; ++j) {
        int idx = openmp_find_index(cdf.data(), n_particles, u[j]);
        if (idx == -1) idx = n_particles - 1;
        xj_sw[j] = array_x[idx];
        yj_sw[j] = array_y[idx];
    }
    std::cout << "CPU reference done." << std::endl;

    // -------------------- OpenCL / XRT setup --------------------
    cl_int err = CL_SUCCESS;

    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];
    std::string device_name = device.getInfo<CL_DEVICE_NAME>();
    std::cout << "Found device: " << device_name << std::endl;

    cl::Context context(device, nullptr, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create context, err=" << err << "\n";
        return EXIT_FAILURE;
    }

    cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create command queue, err=" << err << "\n";
        return EXIT_FAILURE;
    }

    auto fileBuf = xcl::read_binary_file(binaryFile);
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};

    cl::Program program(context, {device}, bins, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to program device with xclbin, err=" << err << "\n";
        return EXIT_FAILURE;
    }

    cl::Kernel kernel(program, "find_index_kernel", &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create kernel, err=" << err << "\n";
        return EXIT_FAILURE;
    }

    // -------------------- Create device buffers --------------------
    size_t buf_bytes = sizeof(data_t) * n_particles;

    cl::Buffer buf_cdf(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, buf_bytes, cdf.data(), &err);
    cl::Buffer buf_u(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, buf_bytes, u.data(), &err);
    cl::Buffer buf_ax(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, buf_bytes, array_x.data(), &err);
    cl::Buffer buf_ay(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, buf_bytes, array_y.data(), &err);
    cl::Buffer buf_xj(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, buf_bytes, xj_hw.data(), &err);
    cl::Buffer buf_yj(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, buf_bytes, yj_hw.data(), &err);

    // -------------------- Set kernel arguments --------------------
    int arg_idx = 0;
    err  = kernel.setArg(arg_idx++, buf_cdf);
    err |= kernel.setArg(arg_idx++, buf_u);
    err |= kernel.setArg(arg_idx++, buf_ax);
    err |= kernel.setArg(arg_idx++, buf_ay);
    err |= kernel.setArg(arg_idx++, buf_xj);
    err |= kernel.setArg(arg_idx++, buf_yj);
    err |= kernel.setArg(arg_idx++, n_particles);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to set kernel args, err=" << err << "\n";
        return EXIT_FAILURE;
    }

    // -------------------- Transfer data: CPU → FPGA --------------------
    std::cout << "Transferring data to FPGA..." << std::endl;
    err = q.enqueueMigrateMemObjects({buf_cdf, buf_u, buf_ax, buf_ay}, 0);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to migrate inputs, err=" << err << "\n";
        return EXIT_FAILURE;
    }
    q.finish();

    // -------------------- First run: verify correctness --------------------
    std::cout << "Running single verification pass..." << std::endl;
    err = q.enqueueTask(kernel);
    q.finish();

    err = q.enqueueMigrateMemObjects({buf_xj, buf_yj}, CL_MIGRATE_MEM_OBJECT_HOST);
    q.finish();

    int err_cnt = 0;
    for (int i = 0; i < n_particles; i++) {
        double diff_x = std::fabs(xj_sw[i] - xj_hw[i]);
        double diff_y = std::fabs(yj_sw[i] - yj_hw[i]);
        if (diff_x > 1e-9 || diff_y > 1e-9) {
            err_cnt++;
            if (err_cnt < 10) {
                std::printf("Mismatch at %d: expected (%0.6f,%0.6f) got (%0.6f,%0.6f)\n",
                            i, xj_sw[i], yj_sw[i], xj_hw[i], yj_hw[i]);
            }
        }
    }

    if (err_cnt > 0) {
        std::cout << "*** VERIFICATION FAILED with " << err_cnt << " errors ***\n";
        return EXIT_FAILURE;
    }
    std::cout << "Verification PASSED." << std::endl;

    // -------------------- Performance loop --------------------
    // Re-migrate inputs (reset output buffers)
    err = q.enqueueMigrateMemObjects({buf_cdf, buf_u, buf_ax, buf_ay}, 0);
    q.finish();

    std::cout << "Running " << num_frames << " kernel invocations for timing..." << std::endl;
    auto t_start = std::chrono::high_resolution_clock::now();

    for (int f = 0; f < num_frames; ++f) {
        q.enqueueTask(kernel);
    }
    q.finish();

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    double per_call_ms = total_time_ms / num_frames;

    // -------------------- Read back final result --------------------
    err = q.enqueueMigrateMemObjects({buf_xj, buf_yj}, CL_MIGRATE_MEM_OBJECT_HOST);
    q.finish();

    std::cout << "============================================\n";
    std::cout << "*** TEST PASSED ***\n";
    std::cout << "n_particles:       " << n_particles << "\n";
    std::cout << "num_frames:        " << num_frames << "\n";
    std::cout << "Total kernel time: " << total_time_ms << " ms\n";
    std::cout << "Per-call time:     " << per_call_ms << " ms\n";
    std::cout << "Throughput:        " << (1000.0 / per_call_ms) << " frames/sec\n";
    std::cout << "============================================\n";

    return EXIT_SUCCESS;
}
