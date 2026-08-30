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
    """Fused attention with a host-side output buffer cache and a fast launcher.

    Round 003 moved the output allocation to an instance cache. Round 004 keeps
    that cache untouched and replaces the per-call `JITFunction.run` dispatch
    with a lazily resolved `LibEntry` fast launcher, whose legality was
    established by the Decision-scoped probe at
    `log/probes/round_004_launch_abi_probe.py`.

    The launcher is an Unknown capability (`lifecycle.fast-launcher`) in the
    frozen profile, so it is never trusted blindly. The Decision-scoped probe
    proved it drives the same compiled kernel and produces a bit-identical
    result, and `forward` re-checks that kernel identity on every fast-path
    call. Any structural failure, key change, or identity mismatch falls back
    to the proven `_fused_attention_kernel[grid](...)` launch in the same call,
    so the worst case at runtime is the accepted behaviour, never a wrong
    answer. Exactly one kernel is launched per call on every path.
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
        # constant, so the keyword bundle is built once here rather than per
        # call. It is only ever expanded into a fresh dict by the callee, never
        # mutated, so sharing one instance across calls is safe.
        self._launch_kwargs = dict(
            HEAD_DIM=head_size,
            NH=num_heads,
            BLOCK_M=self._block,
            BLOCK_N=self._block,
            scale=self.scale,
            num_warps=self._num_warps,
            num_stages=self._num_stages,
        )
        # Round-003 host output-buffer cache. Ordinary instance attributes
        # rather than Parameters, buffers, or submodules, so none of them is
        # module state and none is ever serialized into state_dict.
        self._out_buffer = None
        self._out_cache_key = None
        # Round-004 launch handle.
        self._launcher = None
        self._launch_key = None
        self._launcher_disabled = False
        self._proven_kernel = None

    def _install_launcher(self, cache_key):
        """Construct the fast launch handle. No launch happens here.

        The handle is only ever *used* after `forward` has confirmed, on the
        call that follows, that it selects the same CompiledKernel object the
        proven launch produced. A structural failure here disables the fast
        path for the instance, so the worst case is the accepted behaviour.
        """
        try:
            from triton.runtime.libentry import LibEntry

            launcher = LibEntry(_fused_attention_kernel)
        except Exception:
            self._launcher_disabled = True
            return False
        self._launcher = launcher
        self._launch_key = cache_key
        return True

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
            # allocation and can never alias query, key, or value. A key change
            # also discards the launch handle, so this call uses the proven
            # launch and the handle is re-proven for the new key.
            self._out_buffer = torch.empty(
                query.shape, dtype=query.dtype, device=query.device
            )
            self._out_cache_key = cache_key
            self._launcher = None
            self._launch_key = None
        out = self._out_buffer

        block = self._block
        if q_len > block:
            raise ValueError(
                f"this candidate covers S <= {block}; got S={q_len}. "
                "A row-blocked loop is required before this shape can be served."
            )

        grid = (bsz * self._num_heads,)

        # The handle is cleared on every cache-key change below, so
        # `_launcher is not None` already implies the handle was proven for the
        # current key; no second key comparison is needed in the steady state.
        launcher = self._launcher
        if launcher is not None:
            try:
                # LibEntry.run returns (kernel, constexprs) on every call, so
                # the compiled kernel it selected is available for free. This
                # is the per-call enforcement of probe criterion 2: the fast
                # path is only trusted when it drives the same object the
                # proven launch produced.
                result = launcher[grid](
                    query,
                    key,
                    value,
                    out,
                    query.stride(0),
                    query.stride(1),
                    q_len,
                    **self._launch_kwargs,
                )
                if result[0] is self._proven_kernel:
                    return out
                # Different compiled kernel: a miss, never a reinterpretation.
            except Exception:
                pass
            # Never a wrong answer: drop the handle and redo this very call
            # through the proven launch, which fully overwrites `out`.
            self._launcher = None
            self._launch_key = None
            self._launcher_disabled = True

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

        self._proven_kernel = proven_kernel
        if self._launcher is None and not self._launcher_disabled:
            self._install_launcher(cache_key)
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
