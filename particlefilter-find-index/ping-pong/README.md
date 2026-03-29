# ping-pong — Double-Buffered Dataflow Optimization

Builds on **mem-opt** (buffer + compute + memory coalescing).

## What's new

| Feature | Purpose |
|---|---|
| Ping-pong double buffers (`u_buf_A/B`, `xj_buf_A/B`, `yj_buf_A/B`) | Two copies of tile-sized arrays allow simultaneous read/write |
| `#pragma HLS DATAFLOW` inside tile loop | Enables HLS to overlap load(tile N+1) with compute(tile N) and store(tile N) |
| Prologue load | First tile is pre-loaded before the main loop starts, enabling full overlap from iteration 1 onward |

This reduces total latency by overlapping memory transfers with computation across tile iterations.

## How to run

```bash
vitis_hls -f run_hls.tcl
```

## Expected results

- **csim**: PASSED (identical to all previous versions)
- **csynth**: Similar Fmax; reduced overall latency due to dataflow overlap of load/compute/store phases
