# Evidence-backed Anti-patterns

This catalog records failed approaches for consultation by Designer. Each entry
is conditional evidence, not a universal backend rule. Apply it only when the
listed preconditions match the current runtime fingerprint, shape, and lowering.

## Winner tree for repeated expert selection

**Evidence revision**

`39defa58d2d1c72bc42e122eb697ffcd89ea31a3` (`bd80f49^`),
`groupedtopk/log.md`, Entry 011.

**Preconditions**

MLU590-H8, Triton 3.2.0, grouped top-k with 256 experts arranged as eight
32-expert groups, four selected groups, top-eight expert selection, and the
observed compiler lowering from that revision. The same-trace baseline used
eight 256-lane indexed argmax rounds and measured 20.0288 us per call.

**Attempt**

Replace each wide selection with eight parallel 32-lane group argmax operations
followed by one 8-lane argmax over group winners, updating the winning group for
each of eight rounds.

**Observed failure**

The source-level hierarchy lowered to 16 separate argmax operations plus
value/index state updates. Device time rose to 45.0560 us, MLISA grew to 3126
lines, and GPR use rose from about 674 to 2243 despite correctness passing.

**Reconsider when**

The matched target/compiler exposes a fused hierarchical value/index selection
primitive, or an isolated microbenchmark proves that the two reduction levels
lower without duplicated state and beat the current accepted selection path.

## Sort-32 plus sort-64 selection network

**Evidence revision**

`39defa58d2d1c72bc42e122eb697ffcd89ea31a3` (`bd80f49^`),
`groupedtopk/log.md`, Entry 012.

**Preconditions**

The same grouped top-k shape, MLU590-H8 runtime, and 20.0288 us same-trace
baseline as Entry 011. Stable tie ordering required carrying both logit values
and global expert IDs.

**Attempt**

Fully expand bitonic sort-32 inside every group, retain each local top eight,
then fully expand sort-64 over the 64 survivors using 64-bit value/ID keys.

**Observed failure**

Compile-time compare/swap expansion and paired value/ID state produced 15,445
MLISA lines, 5,344 GPR, and 46.1 KB NRAM. Correct output took 170.6424 us, an
approximately 752% regression against the same-trace baseline.

**Reconsider when**

A small standalone 32- or 64-lane partial top-k network, containing only the
comparators needed for top eight and preserving value/ID semantics, first shows
a resource-bounded win on the matched compiler. Do not infer that every partial
selection network fails from this full-sort result.

## Dynamic tl.gather compaction

**Evidence revision**

`39defa58d2d1c72bc42e122eb697ffcd89ea31a3` (`bd80f49^`),
`groupedtopk/log.md`, Entry 013.

**Preconditions**

Four dynamically selected contiguous 32-expert groups were compacted from an
already loaded 256-element on-chip tensor into 128 candidates on the recorded
MLU Triton runtime. The same-trace non-compacted baseline was 20.0288 us.

**Attempt**

Construct 128 dynamic source offsets, use generic `tl.gather` for on-chip
compaction, and replace eight 256-lane indexed argmax rounds with eight 128-lane
rounds without reloading global memory.

**Observed failure**

Generic dynamic gather did not lower to a cheap contiguous-window movement. Its
control and rearrangement cost exceeded the narrower reductions, producing
21.9048 us, a 9.37% regression, while correctness passed.

**Reconsider when**

The selected data is not expressible as cheaper contiguous windows, a changed
backend has an efficient generic on-chip gather, or a same-runtime microbenchmark
shows that compaction cost is lower than the measured reduction savings.

## Cumsum compaction of selected groups

**Evidence revision**

`39defa58d2d1c72bc42e122eb697ffcd89ea31a3` (`bd80f49^`),
`groupedtopk/log.md`, Entry 016.

**Preconditions**

An eight-lane selected-group mask had to produce four ascending compact slots
before contiguous-window gather in the compact-128 grouped top-k path. The
same-trace accepted path measured 19.9832 us.

**Attempt**

Run an eight-lane cumsum over the selected mask, use its prefix positions to
collect group IDs, then execute the existing window gather and 128-lane expert
selection.

**Observed failure**

The short source expression still generated prefix/reduction work and
intermediate control state. Device time increased to 20.9864 us, a 5.02%
regression, even though it was cheaper than earlier explicit prefix and sort
variants.

**Reconsider when**

The target offers a cheap native prefix/compaction primitive, the selected-lane
count is materially different, or isolated lowering and profiler evidence shows
the prefix path beating the accepted compaction under the current fingerprint.
