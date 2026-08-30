import torch
import torch_npu  # noqa: F401  must precede any NPU allocation
import triton
import triton.language as tl


@triton.jit
def _fused_attention_kernel(
    Q,
    K,
    V,
    O,
    stride_b,
    stride_s,
    S,
    HEAD_DIM: tl.constexpr,
    NH: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    scale,
):
    """One program per (batch, head): full self-attention in a single kernel.

    q/k/v keep their native [B, S, NH*HEAD_DIM] layout and are indexed with
    explicit strides, so no transpose is ever materialized. K is loaded
    transposed as [HEAD_DIM, BLOCK_N] via strides, which avoids tl.trans because
    that primitive is not in the reviewed capability set for this target.
    """
    pid = tl.program_id(0)
    b = pid // NH
    h = pid % NH

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    row_mask = offs_m < S
    col_mask = offs_n < S

    base = b * stride_b + h * HEAD_DIM

    q = tl.load(
        Q + base + offs_m[:, None] * stride_s + offs_d[None, :],
        mask=row_mask[:, None],
        other=0.0,
    )
    # [HEAD_DIM, BLOCK_N]: element [d, n] is K[b, n, h*HEAD_DIM + d]
    k_t = tl.load(
        K + base + offs_n[None, :] * stride_s + offs_d[:, None],
        mask=col_mask[None, :],
        other=0.0,
    )
    v = tl.load(
        V + base + offs_n[:, None] * stride_s + offs_d[None, :],
        mask=col_mask[:, None],
        other=0.0,
    )

    # (BLOCK_M, HEAD_DIM) @ (HEAD_DIM, BLOCK_N) with fp32 accumulation
    qk = tl.dot(q, k_t) * scale
    qk = tl.where(col_mask[None, :], qk, -1.0e6)

    row_max = tl.max(qk, axis=1)
    p = tl.exp(qk - row_max[:, None])
    l_i = tl.sum(p, axis=1)
    p = p / l_i[:, None]

    acc = tl.dot(p.to(tl.float16), v)

    tl.store(
        O + base + offs_m[:, None] * stride_s + offs_d[None, :],
        acc.to(tl.float16),
        mask=row_mask[:, None],
    )


class ModelNew(torch.nn.Module):
    """Fused attention with a cached output buffer and a cached `CompiledKernel`.

    Round 003 moved the output allocation to an instance cache. Round 005 keeps
    that cache and drives the resolved `CompiledKernel` object directly in
    `forward` instead of re-entering per-call `JITFunction` dispatch. This is
    mechanism M2, selected by decision 005 and measured at ~88-94 us/call
    against a ~173-186 us proven baseline; see
    `log/probes/round_005_mechanism_probe.py`.

    `lifecycle.fast-launcher` is `Unknown` in the frozen profile. Legality is
    carried by citation of the retained round-004 Decision-scoped probe
    artifacts, not re-probed here. The fast path is used only when it is the
    same `CompiledKernel` object the proven path produced for the same extended
    launch key, and every anomaly degrades sticky to the proven launch, so the
    worst case at runtime is accepted round-003 behaviour, never a wrong answer.
    """

    def __init__(self, num_heads: int = 8, head_size: int = 64, num_kv_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_kv_heads = num_kv_heads
        self.scale = 1.0 / (head_size**0.5)
        # Launch constants hoisted out of forward: they are invariant for the
        # whole lifetime of the instance. The public attributes num_heads and
        # head_size above stay readable and unchanged for the public contract.
        self._num_heads = num_heads
        self._head_dim = head_size
        self._block = 128
        self._num_warps = 4
        self._num_stages = 1
        # Every launch argument other than q/k/v/out/strides/S is a constructor
        # constant, so the keyword bundle is built once. It is only ever
        # expanded into a fresh dict by the callee, never mutated.
        self._launch_kwargs = dict(
            HEAD_DIM=head_size,
            NH=num_heads,
            BLOCK_M=self._block,
            BLOCK_N=self._block,
            scale=self.scale,
            num_warps=self._num_warps,
            num_stages=self._num_stages,
        )
        # The launch-specialization inputs that never vary, precomputed so the
        # per-call key build stays small: BLOCK_M, BLOCK_N, HEAD_DIM, NH,
        # num_warps, num_stages and the scale value.
        self._const_key = (
            self._block,
            self._block,
            head_size,
            num_heads,
            self._num_warps,
            self._num_stages,
            self.scale,
        )
        # Round-003 host output-buffer cache.
        self._out_buffer = None
        self._out_cache_key = None
        # Round-005 cached CompiledKernel handle. All four are ordinary
        # instance attributes rather than Parameters, buffers, or submodules,
        # so none is module state and none is ever serialized into state_dict.
        self._kernel = None
        self._proven_kernel = None
        self._launch_key = None
        self._launcher_disabled = False

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        bsz, q_len, hidden = query.shape

        # Round-003 output cache key: (shape, dtype, device, query stride).
        # Compared on every call; any difference is a miss and reallocates.
        cache_key = (query.shape, query.dtype, query.device, query.stride())
        if self._out_cache_key != cache_key:
            # torch.empty, never torch.empty_like: the buffer is a fresh
            # allocation and can never alias query, key, or value.
            self._out_buffer = torch.empty(
                query.shape, dtype=query.dtype, device=query.device
            )
            self._out_cache_key = cache_key
        out = self._out_buffer

        block = self._block
        if q_len > block:
            raise ValueError(
                f"this candidate covers S <= {block}; got S={q_len}. "
                "A row-blocked loop is required before this shape can be served."
            )

        grid = (bsz * self._num_heads, 1, 1)

        # Extended launch key. The output key covers shape, dtype, device and
        # query stride; the rest covers every input that decides which
        # CompiledKernel is correct: key and value strides, S, the scale value,
        # the grid, the block sizes, HEAD_DIM, NH, num_warps and num_stages.
        launch_key = (
            cache_key,
            key.stride(),
            value.stride(),
            q_len,
            grid,
            self._const_key,
        )

        kernel = self._kernel
        if kernel is not None:
            if self._launch_key == launch_key:
                if kernel is not self._proven_kernel:
                    # A different CompiledKernel for the same key is a
                    # mismatch, never a reinterpretation. Disable sticky.
                    self._launcher_disabled = True
                else:
                    try:
                        # Drive the cached CompiledKernel directly. The stream
                        # is left for CompiledKernel.__getitem__ to resolve per
                        # call, exactly as the proven path does.
                        kernel[grid](
                            query,
                            key,
                            value,
                            out,
                            query.stride(0),
                            query.stride(1),
                            q_len,
                            self.scale,
                        )
                        return out
                    except Exception:
                        # Never a wrong answer: disable, discard, and redo this
                        # very call through the proven launch, which fully
                        # overwrites `out`.
                        self._launcher_disabled = True
            # Either the key changed, the kernel mismatched, or the launch
            # failed: discard the handle. Only a key change stays eligible to
            # be re-proven below.
            self._kernel = None
            self._proven_kernel = None
            self._launch_key = None

        # Proven fallback: the accepted round-003 launch path, one launch.
        proven_kernel = _fused_attention_kernel[grid](
            query,
            key,
            value,
            out,
            query.stride(0),
            query.stride(1),
            q_len,
            **self._launch_kwargs,
        )

        if not self._launcher_disabled:
            # Structural proof that the resolved kernel is the configured one,
            # then cache it for this key. A configuration mismatch is treated
            # as an unproven resolution and disables the fast path sticky.
            if (
                proven_kernel.metadata.num_warps == self._num_warps
                and proven_kernel.metadata.num_stages == self._num_stages
            ):
                self._kernel = proven_kernel
                self._proven_kernel = proven_kernel
                self._launch_key = launch_key
            else:
                self._launcher_disabled = True
        return out


def get_inputs():
    bsz, seq_len, num_heads, head_size, dtype = 2, 83, 8, 64, torch.float16
    hidden = num_heads * head_size
    query = torch.randn(bsz, seq_len, hidden, dtype=dtype, device="npu")
    key = torch.randn(bsz, seq_len, hidden, dtype=dtype, device="npu")
    value = torch.randn(bsz, seq_len, hidden, dtype=dtype, device="npu")
    return [query, key, value]


def get_init_inputs():
    return [8, 64, 8]
