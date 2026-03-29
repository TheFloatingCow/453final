# mem-opt — Memory-Coalescing Optimization

Builds on **comp-opt** (buffer + compute optimization).

## What's new

| Pragma / Feature | Purpose |
|---|---|
| `max_widen_bitwidth=512` on all 6 `m_axi` interfaces | Widens the AXI data bus from 64 bits (1 double/beat) to 512 bits (8 doubles/beat), enabling memory coalescing and more efficient burst transfers |
| `LOOP_TRIPCOUNT` annotations on load/store loops | Helps HLS estimate burst lengths for scheduling |

This reduces total DDR transactions by up to 8× for sequential accesses (load_vector, load_tile, store_tile).

## How to run

```bash
vitis_hls -f run_hls.tcl
```

## Expected results

- **csim**: PASSED (identical to buffer and comp-opt)
- **csynth**: Similar Fmax to comp-opt; improved burst/bandwidth utilization visible in the interface summary
