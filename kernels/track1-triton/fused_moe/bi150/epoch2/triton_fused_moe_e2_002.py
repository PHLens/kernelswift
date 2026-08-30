"""Fused MoE — bi150 epoch-2 round-002 candidate (decision-002).

Round 002 is a HOST/BOUNDARY-ONLY change over the round-001 accepted source
``triton_fused_moe_e2_001.py`` (decision-002 ``change_scope: host``, G1 option
ii). The two Triton kernel bodies, their grids, the counting sort, the routing
prelude and the tier machinery are carried over semantically unchanged.

WHAT THIS ROUND ACTUALLY DELIVERS (correctness-hardening, NOT speed):

  The round was chartered to remove a 16.219 us/call allocation by reusing one
  persistent copy-out destination. That premise was FALSIFIED by measurement
  (p14/p15): ``torch.empty_like`` costs ~4 us/call (Orchestrator re-measurement;
  0.005 us under my probe's graph-replayed method, 4.13 us under theirs), so
  G1's ceiling is BELOW the 10.99 us adoption gate no matter how it is
  implemented. The round therefore does NOT clear the 5% wall gate and is
  shipped for its correctness property, not for speed.

  THE REAL PRODUCT: the retention guarantee, made explicit and tested.

  ``out_dest`` [83,128] fp16 is allocated ONCE outside the capture region and
  serves as the SHAPE/DTYPE TEMPLATE for the per-call output tensor. The graph
  result is copied from ``out_ws`` DIRECTLY into a fresh tensor each call:

      out_ws  --copy_-->  fresh   (single copy; out_dest is NOT written)

  C3, not the sketch's literal two-hop. The two-hop (``out_ws -> out_dest ->
  fresh``) was measured at +5.113 us/call vs round 001 because it adds a second
  full copy; C3 measures -0.022 us. Orchestrator RULING: the literal
  ``op_copy_out`` span (``writes: [out_dest]``) was predicated on the
  now-falsified premise that persisting the destination pays for itself, so the
  two-hop is formalism over evidence and is OVERRIDDEN.

  WHY THE FRESH TENSOR IS NON-NEGOTIABLE: ``auto_bench.compare_case`` retains
  ``v1_output`` and later hands that very tensor to ``export_profile`` as the
  profile reference output, after 150 forwards have run. A returned persistent
  buffer would silently corrupt it. Option (i) -- return ``out_dest`` directly
  -- was DENIED for exactly this reason, and "no per-call allocation" was
  proven mutually exclusive with "never aliases across calls" below ~150
  forwards (p12). That proof terminates the line of work.

  Residual retained: 22.497 us/call (aten::copy_ 13.936 + cudaMemcpyAsync
  6.160 + Memcpy DtoD 2.401) -- the price of returning a non-aliased tensor.

Family ``manual-graph-replay-fused``. The epoch-1 ungrouped one-launch kernel
(grid (E,) = 8 programs, BLOCK_M=256, 12.34x replicated GEMM arithmetic, half
the 16 SMs idle) is replaced by a two-Triton-launch grouped-dispatch pipeline
replayed through a MANUAL torch.cuda.CUDAGraph over static workspaces:

1. DIRECT-ADDRESS replay tier — the whole routing + counting-sort +
   grouped-expert pipeline is captured ONCE as a manual
   ``torch.cuda.CUDAGraph`` bound to the CALLER'S OWN hidden_states /
   router_logits addresses (ZERO copy-ins: a data_ptr match guarantees the
   replayed kernels read the live caller bytes) plus the instance's own
   parameter storage; per served call the host does only the guard predicate,
   ONE replay submission, and ONE copy-out of the static ``out_ws`` into a
   fresh invocation-owned tensor;
2. COPY-IN replay tier — static ``x_in`` / ``rl_in`` placeholders captured
   once; per call two copy-ins + ONE replay + the same single copy-out;
   serves any target-regime call whose pointers mismatch the tier-1 anchors;
3. EAGER tier — the uncaptured two-launch pipeline for non-target regimes
   and after any replay failure.

Every tier binds PERMANENTLY downward on its own capture/replay exception
(monotone flags, at most once each) while the triggering call still returns
correct results through the next tier. Recapture is bounded (<= 4 lifetime,
first-seen pointer sets only; same-set revisits ride tier-2 so alternation
never re-binds).

Static-shape discipline (decision-001 invariant, binding): the captured region
has NO host data-dependent branch, NO host data-dependent grid, no ``.item()``,
and no D2H read. The expert kernel's grid is the CONSTEXPR ``(E, NUM_TILES)``
with ``NUM_TILES = ceil(T*K / BLOCK_M)`` derived from SHAPES alone; the
per-expert row count is read on-device into a register and empty tiles exit
through an on-device ``if tile_n > 0`` guard. The counting sort runs on the
constexpr grid ``(1,)``.

Graph-pool discipline (decision-001/002 invariant, binding): ``out_ws`` is
graph-pool-backed and is NEVER returned. ``out_dest`` is a persistent
non-graph-pool buffer used as the shape/dtype TEMPLATE for the per-call output;
it is never written on the served path (C3), is never returned, and is never
aliased into a returned tensor. Both public surfaces hand back either a fresh
``torch.empty_like`` filled by ``copy_`` (forward) or the caller's own buffer
filled by ``copy_`` (run_out).

Correctness pedigree: torch.softmax, torch.topk (tie semantics verbatim), the
renormalize divide, and the fp16 casts of w1/w2 remain ATEN ops inside the
captured region, exactly as in the epoch-1 canonical source
``../triton_fused_moe_002.py``. w1/w2 stay fp32 ``nn.Parameter``; the cast
happens inside the captured region.
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl

# Target-regime shape constants. BLOCK_M is the sketch's preferred value (16,
# the proven tl.dot minimum) and the value the p02 pre-adoption sweep selected.
# The per-shape tile count is NOT a module constant: it is computed as
# ceil(T*K / BLOCK_M) inside _pipeline so the tier-3 eager path stays
# SHAPE-GENERIC (for the target shape T=83, K=2 it is ceil(166/16) = 11, and
# it is a SHAPE constant that never depends on routing outcomes).
_BLOCK_M = 16
_RECAPTURE_BUDGET = 4
_TOKENS = 83
_HIDDEN = 128
_EXPERTS = 8
_TOPK = 2
_INTERMEDIATE = 64


@triton.jit
def _counting_sort_kernel(
    flat_ids_ptr,      # [T*K] int64 (torch.topk output, narrowed in-kernel)
    flat_w_ptr,        # [T*K] fp16
    sorted_rows_ptr,   # [BLOCK_ROWS] int32, token index per sorted slot
    sorted_w_ptr,      # [BLOCK_ROWS] fp16, routing weight per sorted slot
    counts_ptr,        # [E] int32
    offsets_ptr,       # [E] int32, exclusive prefix sum of counts
    NROWS: tl.constexpr,   # T*K = 166
    E: tl.constexpr,       # 8
    K: tl.constexpr,       # 2
    BLOCK_ROWS: tl.constexpr,  # power of two >= NROWS (256)
):
    """grid (1,) — one program, so the whole compaction needs NO cross-block
    synchronization. Buckets the NROWS (token, k) rows by expert so the expert
    kernel can walk each expert's rows contiguously at BLOCK_M instead of
    masking a full 256-row tile per expert.

    TWO RUNTIME FINDINGS from the decision-scoped probes on THIS rig
    (log/probes/ p01b, p01c, p01d, p02c, p02d, p02e, p02f) shape this body:

    (1) The [E]-tile lane-deposit idioms are silently wrong: adding a
        [BLOCK_ROWS] 0/1 mask straight into an [E] accumulator is a hard
        compile error (shapes 8 vs 256), and both ``tl.where(be == b, x, 0)``
        and ``(be == b).to(tl.int32) * x`` for the [E]-tile prefix sum COMPILE
        BUT YIELD ZEROS. Every tile deposit here therefore REDUCES over the
        tile.

    (2) The scatter's rank must use a [BLOCK_ROWS, E] matrix (256 x 8), not a
        [BLOCK_ROWS, BLOCK_ROWS] one (256 x 256). The latter is what a naive
        per-bucket "count the earlier lanes" loop builds; measured at 554
        us/call, it made this kernel 95% of the whole pipeline. The [256, 8]
        form computes every bucket's per-lane rank in ONE matrix and reduces
        on the 8-wide axis: 26-37 us/call, a 15-21x reduction, verified
        bitwise-equal to a host stable counting sort on 18 id vectors across
        6 activation patterns and deterministic over 20 runs.

    An atomic one-pass cursor (6.5 us) was measured and REJECTED: it is fast
    but WRONG -- a vector-pointer atomic gives colliding destinations, so the
    result is neither deterministic nor even allclose to the stable
    reference.

    ``tl.sum`` here is an int32 elementwise reduction over a small tile; it is
    NOT the fp32 row-softmax reduction whose waiver decision-001 refused.
    softmax / topk / renormalize / the fp16 casts remain aten outside this
    kernel.
    """
    r = tl.arange(0, BLOCK_ROWS)
    valid = r < NROWS
    ids = tl.load(flat_ids_ptr + r, mask=valid, other=0).to(tl.int32)
    w = tl.load(flat_w_ptr + r, mask=valid, other=0.0)

    # token index = row // K, carried alongside so the expert kernel can
    # gather hidden_states rows and scatter results back without recomputing.
    token = (r // K).to(tl.int32)
    be = tl.arange(0, E)

    # Bucket counts.
    counts = tl.zeros((E,), dtype=tl.int32)
    for b in tl.static_range(E):
        cnt_b = tl.sum(tl.where((ids == b) & valid, 1, 0).to(tl.int32), axis=0)
        counts = counts + (be == b).to(tl.int32) * cnt_b

    # Exclusive prefix sum -> per-expert base offset into the sorted arrays.
    offsets = tl.zeros((E,), dtype=tl.int32)
    for b in tl.static_range(E):
        acc = tl.sum(tl.where(be < b, counts, 0), axis=0)
        offsets = offsets + (be == b).to(tl.int32) * acc

    tl.store(counts_ptr + be, counts)
    tl.store(offsets_ptr + be, offsets)

    # [BLOCK_ROWS, E] rank matrix: earlier[i, b] = number of lanes j < i whose
    # row belongs to bucket b. Reducing on the 8-wide axis after masking gives
    # each lane's within-bucket rank; the base offset is gathered the same
    # way. Stable (ranks follow flat row order), deterministic.
    earlier = tl.sum(
        tl.where((r[None, :] < r[:, None])[:, :, None]
                 & (ids[None, :, None] == be[None, None, :])
                 & valid[None, :, None], 1, 0).to(tl.int32), axis=1)
    belong = (ids[:, None] == be[None, :]) & valid[:, None]
    rank = tl.sum(tl.where(belong, earlier, 0), axis=1)
    base = tl.sum(tl.where(belong, offsets[None, :], 0), axis=1)
    dest = base + rank

    # Garbage lanes (row >= NROWS) write nothing.
    tl.store(sorted_rows_ptr + dest, token, mask=valid)
    tl.store(sorted_w_ptr + dest, w, mask=valid)


@triton.jit
def _grouped_expert_kernel(
    hidden_ptr,        # [T, H] fp16
    sorted_rows_ptr,   # [BLOCK_ROWS] int32 token index per sorted slot
    sorted_w_ptr,      # [BLOCK_ROWS] fp16 routing weight per sorted slot
    offsets_ptr,       # [E] int32
    counts_ptr,        # [E] int32
    w1_ptr,            # [E, 2*I, H] fp16
    w2_ptr,            # [E, H, I] fp16
    out_ptr,           # [T, H] fp16 workspace, zero-initialized before launch
    H: tl.constexpr,
    I: tl.constexpr,
    BLOCK_M: tl.constexpr,
    NUM_TILES: tl.constexpr,
    NROWS_TOTAL: tl.constexpr,
):
    """grid (E, NUM_TILES) — program_id(0) is the expert, program_id(1) is the
    row tile. The grid is a pure shape constant: it does NOT depend on how many
    rows any expert actually won. Each program reads its expert's row count
    into a register and exits ON-DEVICE when the tile is empty, so no host
    ever learns the routing outcome and no D2H read exists.

    Each program handles BLOCK_M rows of ONE expert, so the three dots run on
    real work only: the 12.34x replication of the epoch-1 ungrouped kernel is
    gone and the grid exposes E*NUM_TILES programs to the 16 SMs instead of 8.

    NROWS_TOTAL (= T*K) is passed in rather than hardcoded to 166 so the
    kernel is SHAPE-GENERIC: the tier-3 eager path serves non-target shapes
    with the same compiled source.
    """
    e = tl.program_id(0)
    tile = tl.program_id(1)

    base = tl.load(offsets_ptr + e)     # device -> register only
    n_rows = tl.load(counts_ptr + e)    # device -> register only
    start = base + tile * BLOCK_M
    tile_n = n_rows - tile * BLOCK_M    # rows actually present in this tile

    if tile_n > 0:                      # ON-DEVICE early exit; grid unchanged
        m = tl.arange(0, BLOCK_M)
        slot = start + m
        in_tile = (m < tile_n) & (slot < NROWS_TOTAL)

        tok = tl.load(sorted_rows_ptr + slot, mask=in_tile, other=0)
        w = tl.load(sorted_w_ptr + slot, mask=in_tile,
                    other=0.0).to(tl.float32)

        rk = tl.arange(0, H)
        rn = tl.arange(0, I)

        # Gather x rows: hidden[token, :] -> [BLOCK_M, H] fp16.
        x = tl.load(hidden_ptr + tok[:, None] * H + rk[None, :],
                    mask=in_tile[:, None], other=0.0)

        # w1[e] is [2*I, H]: first I rows are the gate weights, last I the up
        # weights. x @ w1[e].T splits into two dots of N=I each.
        w1_base = w1_ptr + e * (2 * I * H)
        gate_w = tl.load(w1_base + rn[:, None] * H + rk[None, :])
        up_w = tl.load(w1_base + (I + rn)[:, None] * H + rk[None, :])

        acc_g = tl.zeros((BLOCK_M, I), dtype=tl.float32)
        acc_u = tl.zeros((BLOCK_M, I), dtype=tl.float32)
        gate = tl.dot(x, tl.trans(gate_w), acc_g)
        up = tl.dot(x, tl.trans(up_w), acc_u)

        act = gate * tl.sigmoid(gate) * up        # SiLU(gate) * up

        # w2[e] is [H, I]: act @ w2[e].T -> [BLOCK_M, H].
        w2_base = w2_ptr + e * (H * I)
        w2e = tl.load(w2_base + rk[:, None] * I + rn[None, :])
        acc_y = tl.zeros((BLOCK_M, H), dtype=tl.float32)
        y = tl.dot(act.to(tl.float16), tl.trans(w2e), acc_y)

        contrib = y * w[:, None]

        # Rows beyond the tile's real extent are masked off, so padding lanes
        # contribute exactly nothing to the atomic accumulation.
        tl.atomic_add(out_ptr + tok[:, None] * H + rk[None, :], contrib,
                      mask=in_tile[:, None])


class ModelNew(nn.Module):
    """Fused MoE with grouped dispatch replayed through a manual CUDA graph.

    Public contract unchanged: ``ModelNew(num_experts, top_k, hidden_size,
    intermediate_size, renormalize=True).forward(hidden_states,
    router_logits)`` plus the preallocated-output surface
    ``run_out(hidden_states, router_logits, out)``.

    The three tiers are bitwise-equal for identical input bits: they run the
    same two kernels over the same bytes, and the copy boundaries preserve
    bits. On the replay tiers the python Triton launcher never executes, which
    is the mechanism decision-001 monetizes (two launches' worth of launcher
    tax removed against the R + F terms).
    """

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        renormalize: bool = True,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.renormalize = renormalize

        self.w1 = nn.Parameter(
            torch.empty(num_experts, 2 * intermediate_size, hidden_size)
        )
        self.w2 = nn.Parameter(
            torch.empty(num_experts, hidden_size, intermediate_size)
        )
        nn.init.normal_(self.w1, std=0.02)
        nn.init.normal_(self.w2, std=0.02)

        # Declared state set (Host Plan): two graph handles, the static
        # workspaces, the persistent copy-out destination, pointer anchors, the
        # bound-set history, the recapture counter, and two monotone tier flags.
        self.graph_direct = None
        self.graph_copyin = None
        self.out_ws = None
        # Persistent copy-out TARGET (round 002). Allocated once, outside the
        # capture region, and reused as the copy_ destination on every replay
        # call so the per-call torch.empty_like and its aten::empty_strided
        # disappear from the timed path. It is a DESTINATION ONLY: it is never
        # returned, never aliased into the returned tensor, and never handed to
        # a caller. forward() still produces a fresh tensor per call.
        self.out_dest = None
        self.x_in = None
        self.rl_in = None
        self.sorted_rows = None
        self.sorted_w = None
        self.expert_counts = None
        self.expert_offsets = None
        self.anchor_x = 0
        self.anchor_rl = 0
        self.anchor_w1 = 0
        self.anchor_w2 = 0
        self.bound_sets = ()
        self.recapture_budget = _RECAPTURE_BUDGET
        self.direct_replay_failed = False
        self.copyin_replay_failed = False
        self._ws_device = None

    # ------------------------------------------------------------------
    # target regime and guards
    # ------------------------------------------------------------------

    def _is_target_regime(self, hidden_states, router_logits):
        """Fixed-shape target regime: [83,128] fp16 hidden, [83,8] fp32
        router logits, the benchmark constructor configuration, contiguous, on
        the workspace device. Anything else routes to tier-3 eager and touches
        zero graph artifacts (so the graph is never captured for a shape whose
        constexpr grid would be wrong)."""
        if not (self.num_experts == _EXPERTS and self.top_k == _TOPK
                and self.hidden_size == _HIDDEN
                and self.intermediate_size == _INTERMEDIATE):
            return False
        if hidden_states.device != router_logits.device:
            return False
        device = hidden_states.device
        if device.type != "cuda":
            return False
        if self._ws_device is not None and device != self._ws_device:
            return False
        if (hidden_states.dtype != torch.float16
                or router_logits.dtype != torch.float32):
            return False
        if tuple(hidden_states.shape) != (_TOKENS, _HIDDEN):
            return False
        if tuple(router_logits.shape) != (_TOKENS, _EXPERTS):
            return False
        for tensor in (hidden_states, router_logits):
            if not (tensor.is_contiguous() and tensor.is_cuda):
                return False
        return True

    def _anchors_match(self, hidden_states, router_logits):
        """Four-way data_ptr guard including the parameter storage: the
        strongest available cache key. A match guarantees the replayed kernels
        read the live caller bytes AND the live weights at the captured
        addresses, which is exactly why tier-1 needs ZERO copy-ins."""
        return (hidden_states.data_ptr() == self.anchor_x
                and router_logits.data_ptr() == self.anchor_rl
                and self.w1.data_ptr() == self.anchor_w1
                and self.w2.data_ptr() == self.anchor_w2)

    # ------------------------------------------------------------------
    # the captured pipeline
    # ------------------------------------------------------------------

    def _pipeline(self, hidden_states, router_logits, out):
        """The captured region: routing arithmetic (aten, unchanged from the
        epoch-1 canonical source), the counting-sort launch, the out_ws
        zero-init, and the grouped-expert launch. Two Triton launches.

        Host-data-dependent control flow: NONE. Both grids are constexpr
        ((1,) and (E, NUM_TILES)); the per-expert row count is consumed on
        device. There is no .item(), no D2H read, and no print in here.

        num_tiles is derived from SHAPES alone (T and BLOCK_M), never from
        routing outcomes, so the grid stays a compile-time constant for a
        given shape. It is computed from T rather than pinned to a module
        constant so the tier-3 eager path remains SHAPE-GENERIC: a non-target
        shape (e.g. T=128) gets its own correct tile count instead of
        silently reusing the T=83 one.
        """
        T = hidden_states.shape[0]
        K = self.top_k
        dtype = hidden_states.dtype
        E = self.num_experts
        H = self.hidden_size
        I = self.intermediate_size
        # shape-derived tile count: static per shape, never host-data-dependent
        num_tiles = (T * K + _BLOCK_M - 1) // _BLOCK_M

        # --- routing (verbatim epoch-1 semantics; aten, tie order frozen) ---
        scores = torch.softmax(router_logits.float(), dim=-1)
        topk_weights, topk_ids = torch.topk(scores, self.top_k, dim=-1)
        if self.renormalize:
            topk_weights = topk_weights / topk_weights.sum(-1, keepdim=True)
        topk_weights = topk_weights.to(dtype)

        flat_ids = topk_ids.view(-1)        # [T*K] int64
        flat_w = topk_weights.view(-1)      # [T*K] fp16

        w1 = self.w1.to(dtype)              # [E, 2*I, H] fp16 (in-captured cast)
        w2 = self.w2.to(dtype)              # [E, H, I] fp16

        nrows = T * K
        block_rows = triton.next_power_of_2(nrows)

        # Sort buffers: the static ones are sized for the TARGET regime
        # (BLOCK_ROWS=256). A non-target shape on the tier-3 eager path
        # allocates its own correctly-sized buffers, so the eager path is
        # SHAPE-GENERIC. The captured path (tiers 1-2) always uses the static
        # buffers, which is what keeps the graph's shapes fixed.
        if self.sorted_rows is None or self.sorted_rows.shape[0] < block_rows:
            sorted_rows = torch.zeros(block_rows, dtype=torch.int32,
                                      device=hidden_states.device)
            sorted_w = torch.zeros(block_rows, dtype=flat_w.dtype,
                                   device=hidden_states.device)
        else:
            sorted_rows = self.sorted_rows
            sorted_w = self.sorted_w
        if self.expert_counts is None or self.expert_counts.shape[0] != E:
            expert_counts = torch.zeros(E, dtype=torch.int32,
                                        device=hidden_states.device)
            expert_offsets = torch.zeros(E, dtype=torch.int32,
                                         device=hidden_states.device)
        else:
            expert_counts = self.expert_counts
            expert_offsets = self.expert_offsets

        # --- launch 1: counting sort into the static index/weight buffers ---
        _counting_sort_kernel[(1,)](
            flat_ids, flat_w,
            sorted_rows, sorted_w,
            expert_counts, expert_offsets,
            NROWS=nrows, E=E, K=K, BLOCK_ROWS=block_rows,
            num_warps=_BEST_NUM_WARPS,
        )

        out.zero_()

        # --- launch 2: grouped expert GEMM over the static (E, num_tiles) ---
        _grouped_expert_kernel[(E, num_tiles)](
            hidden_states,
            sorted_rows, sorted_w,
            expert_offsets, expert_counts,
            w1, w2, out,
            H=H, I=I, BLOCK_M=_BLOCK_M, NUM_TILES=num_tiles,
            NROWS_TOTAL=nrows,
            num_warps=_BEST_NUM_WARPS,
        )

    # ------------------------------------------------------------------
    # workspace and capture machinery
    # ------------------------------------------------------------------

    def _alloc_workspace(self, device):
        if self.out_ws is None:
            T, H, E = _TOKENS, _HIDDEN, _EXPERTS
            block_rows = triton.next_power_of_2(T * self.top_k)
            self.out_ws = torch.zeros((T, H), dtype=torch.float16,
                                      device=device)
            self.sorted_rows = torch.zeros(block_rows, dtype=torch.int32,
                                           device=device)
            self.sorted_w = torch.zeros(block_rows, dtype=torch.float16,
                                        device=device)
            self.expert_counts = torch.zeros(E, dtype=torch.int32,
                                             device=device)
            self.expert_offsets = torch.zeros(E, dtype=torch.int32,
                                              device=device)
            # Persistent copy-out destination, allocated ONCE here (outside
            # the capture region) and reused thereafter.
            self.out_dest = torch.zeros((T, H), dtype=torch.float16,
                                        device=device)
            self._ws_device = device

    def _warmup_and_capture(self, x, rl, out_target):
        """torch.cuda.graph recommended pattern: three warmup iterations on a
        dedicated side stream (freezing the JIT specialization BEFORE capture),
        then a verbatim capture of the pipeline. All workspace allocation
        happens before the window; only graph-pool intermediates for the
        routing temporaries arise inside it (supported pattern)."""
        stream = torch.cuda.Stream(device=x.device)
        stream.wait_stream(torch.cuda.current_stream(x.device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                self._pipeline(x, rl, out_target)
        torch.cuda.current_stream(x.device).wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._pipeline(x, rl, out_target)
        return graph

    def _capture_direct(self, hidden_states, router_logits):
        """Bind (or rebind, bounded) the tier-1 graph to THIS pointer set. The
        initial binding is free; every rebinding spends one unit of the
        irreversibly-decrementing recapture budget."""
        was_bound = self.graph_direct is not None
        self._alloc_workspace(hidden_states.device)
        self.graph_direct = None
        graph = self._warmup_and_capture(hidden_states, router_logits,
                                         self.out_ws)
        self.graph_direct = graph
        self.anchor_x = hidden_states.data_ptr()
        self.anchor_rl = router_logits.data_ptr()
        self.anchor_w1 = self.w1.data_ptr()
        self.anchor_w2 = self.w2.data_ptr()
        self.bound_sets = (self.bound_sets +
                           ((self.anchor_x, self.anchor_rl, self.anchor_w1,
                             self.anchor_w2),))[-5:]
        if was_bound:
            self.recapture_budget -= 1

    def _build_copyin_graph(self, device):
        """Tier-2 machinery: static input placeholders captured once; the
        per-call copy-ins happen OUTSIDE the graph."""
        self._alloc_workspace(device)
        self.x_in = torch.zeros((_TOKENS, _HIDDEN), dtype=torch.float16,
                                device=device)
        self.rl_in = torch.zeros((_TOKENS, _EXPERTS), dtype=torch.float32,
                                 device=device)
        self.graph_copyin = None
        graph = self._warmup_and_capture(self.x_in, self.rl_in, self.out_ws)
        self.graph_copyin = graph

    def _bind_direct_failed(self):
        """PERMANENT-ONCE, monotone: once set, the flag never clears and the
        tier-1 graph handle is dropped so no later call can replay it. The
        flag is authoritative (not just the handle), so setting it directly
        also disables the tier."""
        self.direct_replay_failed = True
        self.graph_direct = None

    def _bind_copyin_failed(self):
        """PERMANENT-ONCE, monotone: drops BOTH the graph handle and the
        copy-in placeholders, so tier-2 cannot be re-entered even if the flag
        is set without an exception."""
        self.copyin_replay_failed = True
        self.graph_copyin = None
        self.x_in = None
        self.rl_in = None

    def _tier1_usable(self, hidden_states, router_logits):
        return (not self.direct_replay_failed
                and self._is_target_regime(hidden_states, router_logits))

    def _tier2_usable(self, hidden_states, router_logits):
        return (not self.copyin_replay_failed
                and self._is_target_regime(hidden_states, router_logits))

    def _dest_buffer(self):
        """Return the persistent copy-out destination, allocating it lazily on
        first use if the eager path was reached before any replay tier
        allocated the workspaces. Reused forever after; never returned."""
        if self.out_dest is None:
            self._alloc_workspace(self._ws_device or torch.device("cuda"))
        return self.out_dest

    def _direct_serve(self, hidden_states, router_logits, destination):
        """Tier-1 service. Returns the filled buffer, or None to route to
        tier-2 (recapture denied by budget, or a same-set revisit).

        Round-002 boundary change (C3, per Orchestrator ruling):

        The persistent ``out_dest`` is allocated ONCE, outside the capture
        region, and is used as the SHAPE/DTYPE TEMPLATE for the per-call
        output tensor. The graph result is copied **directly** from ``out_ws``
        into that fresh tensor -- a SINGLE copy. ``out_dest`` is NOT written
        on the served path.

        This is C3 rather than the sketch's literal two-hop
        (``out_ws -> out_dest -> fresh``). The two-hop was measured at
        +5.113 us/call against round 001 (p15 C2), because it adds a second
        full copy, while C3 measures -0.022 us -- cost-neutral. Orchestrator
        ruling: the literal `op_copy_out` span (`writes: [out_dest]`) was
        predicated on the now-falsified premise that persisting the copy
        destination pays for itself; since it does not, the two-hop is
        formalism over evidence and is OVERRIDDEN. C3 satisfies the sketch's
        INTENT in full: ``out_ws`` (graph-pool memory) is never returned, and
        the returned tensor never aliases across calls.

        For ``run_out`` (``destination`` is the caller's buffer) the caller's
        buffer is filled directly from ``out_ws``.
        """
        if self.graph_direct is None:
            self._capture_direct(hidden_states, router_logits)
        elif not self._anchors_match(hidden_states, router_logits):
            key = (hidden_states.data_ptr(), router_logits.data_ptr(),
                   self.w1.data_ptr(), self.w2.data_ptr())
            if self.recapture_budget <= 0 or key in self.bound_sets:
                return None
            self._capture_direct(hidden_states, router_logits)
        # --- op_copy_out (C3): out_ws -> fresh, single copy ---------------
        # out_dest is the shape/dtype template only; it is NOT written here.
        self.graph_direct.replay()
        if destination is None:
            destination = torch.empty_like(self._dest_buffer())
        destination.copy_(self.out_ws)
        return destination

    def _copyin_serve(self, hidden_states, router_logits, destination):
        """Tier-2 service: two copy-ins, ONE replay, copy-out. The parameter
        storage is read live inside the graph (self.w1 / self.w2 are the
        module's own tensors), so only the two activations are copied.

        Same boundary shape as tier 1 (C3): single copy from ``out_ws`` into
        the fresh tensor, with ``out_dest`` serving only as the shape/dtype
        template and never being written.
        """
        if self.graph_copyin is None:
            self._build_copyin_graph(hidden_states.device)
        self.x_in.copy_(hidden_states)
        self.rl_in.copy_(router_logits)
        self.graph_copyin.replay()
        if destination is None:
            destination = torch.empty_like(self._dest_buffer())
        destination.copy_(self.out_ws)
        return destination

    # ------------------------------------------------------------------
    # public surfaces
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,   # [T, H] fp16
        router_logits: torch.Tensor,   # [T, E] fp32
    ) -> torch.Tensor:
        if self._tier1_usable(hidden_states, router_logits):
            try:
                served = self._direct_serve(hidden_states, router_logits, None)
                if served is not None:
                    return served
            except Exception:
                self._bind_direct_failed()
        if self._tier2_usable(hidden_states, router_logits):
            try:
                served = self._copyin_serve(hidden_states, router_logits, None)
                if served is not None:
                    return served
            except Exception:
                self._bind_copyin_failed()
        # Tier-3 eager: never returns workspace memory either. The buffer is
        # sized from the ACTUAL input shape so the eager path stays
        # SHAPE-GENERIC (the static out_ws is sized for the target regime and
        # must not be used here).
        out = torch.zeros((hidden_states.shape[0], self.hidden_size),
                          dtype=hidden_states.dtype,
                          device=hidden_states.device)
        self._pipeline(hidden_states, router_logits, out)
        result = torch.empty_like(out)
        result.copy_(out)
        return result

    def run_out(self, hidden_states: torch.Tensor, router_logits: torch.Tensor,
                out: torch.Tensor) -> None:
        """Preallocated-output surface required by kernel-mode profiling
        (auto_bench.make_profile_call calls
        ``run_out(gating_input, *reference_outputs, **model.run_kwargs)``).

        Kernel-mode profiling is UNAVAILABLE for this 3-arg surface: the
        harness passes ``run_out(router_logits, out)`` (last input + all
        outputs), which is 2 args, while this signature takes 3. Per the
        fused-moe project contract and the sibling precedent, forward-mode
        profiling is therefore canonical and NO synchronization or
        accommodation is added here.

        The caller buffer is filled by copy-out OUTSIDE the replay boundary on
        tiers 1-2 and is never aliased to workspace or graph-pool memory;
        returns None; bitwise-equal to forward for identical inputs."""
        if self._tier1_usable(hidden_states, router_logits):
            try:
                if self._direct_serve(hidden_states, router_logits,
                                      out) is not None:
                    return None
            except Exception:
                self._bind_direct_failed()
        if self._tier2_usable(hidden_states, router_logits):
            try:
                if self._copyin_serve(hidden_states, router_logits,
                                      out) is not None:
                    return None
            except Exception:
                self._bind_copyin_failed()
        scratch = torch.zeros((hidden_states.shape[0], self.hidden_size),
                              dtype=hidden_states.dtype,
                              device=hidden_states.device)
        self._pipeline(hidden_states, router_logits, scratch)
        out.copy_(scratch)
        return None


# Pinned by the decision-001 pre-adoption sweep (probe p02): see
# log/probes/p02_r001_config_sweep_result.json. This is a module-level literal
# assignment, retained by the harness AST loader.
_BEST_NUM_WARPS = 1


def get_inputs():
    num_tokens, hidden_size, num_experts = 83, 128, 8
    hidden_states = torch.randn(num_tokens, hidden_size, dtype=torch.float16,
                                device="cuda")
    router_logits = torch.randn(num_tokens, num_experts, dtype=torch.float32,
                                device="cuda")
    return [hidden_states, router_logits]


def get_init_inputs():
    return [8, 2, 128, 64]


if __name__ == "__main__":
    init_inputs = get_init_inputs()
    model = ModelNew(*init_inputs).cuda().eval()
    inputs = get_inputs()
    with torch.no_grad():
        out = model(*inputs)
    if isinstance(out, (tuple, list)):
        for o in out:
            if hasattr(o, "shape"):
                print(o.shape)
    else:
        print(out.shape)
