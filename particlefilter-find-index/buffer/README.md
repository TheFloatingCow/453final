# Particle Filter `find_index_kernel` Buffer Baseline

This folder contains the first HLS implementation stage for the selected Particle Filter kernel.

Kernel scope:
- Inputs: `cdf`, `u`, `array_x`, `array_y`, `n_particles`
- Outputs: `xj`, `yj`
- Behavior: for each particle `j`, find the first `i` where `cdf[i] >= u[j]`, then set `xj[j] = array_x[i]` and `yj[j] = array_y[i]`

Files:
- `pf_find_index.h`: constants and top-level declaration
- `pf_find_index.cpp`: HLS kernel with explicit load/compute/store decomposition
- `pf_find_index_tb.cpp`: C testbench against a software reference
- `run_hls.tcl`: C simulation and synthesis script

Current design intent:
- This is the `buffer` version required by the project brief.
- The kernel uses local on-chip buffers for `cdf`, `array_x`, and `array_y`.
- `u`, `xj`, and `yj` are processed in tiles.

Current assumptions:
- `MAX_PARTICLES` is set to 16384 for this baseline.
- If `n_particles` exceeds `MAX_PARTICLES`, the kernel clamps to `MAX_PARTICLES`.
- The baseline preserves the original OpenMP sequential search behavior.
