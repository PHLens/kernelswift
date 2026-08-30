# Final Summary — music_flamingo_rotary_embedding (S60 Epoch 2)

## Delivery Result (SUCCESS — accepted, beats base)

- status: **complete — accepted**
- **deliverable**: `triton_music_flamingo_rotary_embedding_e2_001.py` (correctness-PASS)
- **speedup vs base**: **1.11x (+9.55%)** — S60's 2nd operator to beat base
- **speedup vs epoch-1**: **~1.23x (0.9x → 1.11x)**

## Intervention (partial fusion)

Single direct-launched Triton kernel (grid=(B,seq_len)=(4,32), num_warps=1, HALF=32)
computes the freqs elementwise chain (div/mul/repeat_interleave/broadcast/cat/mul-angle)
into an intermediate [4,32,128] buffer; cos/sin remain **vendor** `torch.cos`/`torch.sin`.

## Why it wins (and why epoch-1 full fusion lost)

| metric | base | candidate |
|---|---:|---:|
| launches/call | 13.0 (topsLaunchKernel @118.85us) | 3.0 (1 Triton @13.59us + 2 vendor cos/sin @26.54us) |
| launch-API | ~118.85us | ~40.13us |
| tl.cos/tl.sin in kernel | — | **0** |

epoch-1 FULL fusion (cos/sin via tl.cos/tl.sin) was -13%: GCU's math-dialect trig is
~44% slower than the vendor trig library. Partial fusion splits the difference —
collapse ~10 elementwise launches while keeping the fast vendor trig, net +9.55%.

## Correctness

4/4 PASS (exact-match, deterministic). Tuple output (cos, sin) each [4,32,128] fp32.

## Terminal Classification

- terminal_result: accepted (+9.55%, above +5% bar, all 3 pairs positive)
- stop_reason: accepted; remaining 2 vendor cos/sin launches must stay vendor (the
  trig is the fast path), no further launch-collapse prize.

## Note

- No tl.dot; primary contract math.elementwise.
- position_angles/inv_freq reconstructed in __init__ exactly as base, register_buffer
  registered (state_dict keys unchanged).
