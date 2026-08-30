import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelNew(nn.Module):
    """Causal scaled-dot-product attention (flexattention), decision-001.

    Family manual-cuda-graph-workspace-replay (change_scope=host): the
    byte-frozen base pipeline — view prelude, the retained vendor call
    F.scaled_dot_product_attention(q, k, v, scale=0.125, is_causal=True),
    squeeze/transpose/reshape epilogue — is captured ONCE as a manual
    torch.cuda.CUDAGraph over instance-owned static workspace buffers
    (q_in/k_in/v_in [83,8,64] fp16 placeholders plus attn_flat_ws [83,512]
    fp16 result placeholder). Every target-regime call performs only:
    strict regime guard evaluation, three small fp16 copy-ins of the live
    query/key/value into the static buffers, ONE graph replay submission,
    and one small copy-out OUTSIDE the replay boundary into an
    invocation-owned fresh buffer (forward) or the caller-provided buffer
    (run_out). Any exception during workspace setup, side-stream warmup,
    capture, or a later replay binds this instance permanently down-tier to
    the framework-eager tier (flag ``replay_failed`` transitions downward at
    most once) while keeping the triggering call correct through the lower
    tier. Off-regime inputs construct NOTHING and route eager without ever
    consulting or creating replay artifacts (selectivity, not poison).
    No Triton kernel and no Triton matrix-multiply primitive is introduced;
    no compiler machinery exists anywhere in this module. The retained vendor
    SDPA call stays the only attention computation, so all device kernels,
    their argument values, and output bits remain identical to the accepted
    reference by construction.
    """

    def __init__(self, num_heads: int = 8, head_size: int = 64,
                 scale: float = None, num_kv_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale or 1.0 / head_size ** 0.5
        self.num_kv_heads = num_kv_heads
        # Two-tier permanent chain state owned by the instance (Host Plan).
        # ``replay_failed`` is the named permanent tier-binding flag; it can
        # only transition downward (manual-replay -> framework-eager).
        self.replay_failed = False
        self.cuda_graph = None
        self.q_in = None
        self.k_in = None
        self.v_in = None
        self.attn_flat_ws = None
        self._ws_device = None

    def _is_target_regime(self, query, key, value):
        """Strict target-regime gate: fixed config (num_heads=8, head_size=64,
        num_kv_heads=8, scale None->0.125) plus three contiguous fp16
        [83,8,64] cuda tensors on one device matching the captured device.
        cache_key components are shape/dtype/device; anything else routes to
        the eager tier and constructs no artifacts."""
        if self.replay_failed:
            return False
        if not (self.num_heads == 8 and self.head_size == 64
                and self.num_kv_heads == 8 and self.scale == 0.125):
            return False
        device = query.device
        if key.device != device or value.device != device:
            return False
        if device.type != 'cuda':
            return False
        if self._ws_device is not None and device != self._ws_device:
            return False
        for tensor in (query, key, value):
            if tensor.dtype != torch.float16 or tensor.shape != (83, 8, 64):
                return False
            if not tensor.is_contiguous() or not tensor.is_cuda:
                return False
        return True

    def _pipeline_body(self):
        """Captured region: the exact base pipeline body over static
        workspace placeholders ONLY. Contains zero branches, zero host
        reads, and no Triton kernel or matrix-multiply primitive at all;
        internal SDPA
        output and relayout temporaries allocate from the framework-owned
        graph-private pool during capture and keep stable addresses after.
        The retained vendor call keeps identical argument values, dtypes,
        ordering and causal semantics; GQA broadcast is omitted by
        construction because num_kv_heads == num_heads == 8."""
        qw = self.q_in.unsqueeze(0).transpose(1, 2)
        kw = self.k_in.unsqueeze(0).transpose(1, 2)
        vw = self.v_in.unsqueeze(0).transpose(1, 2)
        attn = F.scaled_dot_product_attention(qw, kw, vw, scale=self.scale, is_causal=True)
        flat = attn.squeeze(0).transpose(0, 1).reshape(83, self.num_heads * self.head_size)
        self.attn_flat_ws.copy_(flat)

    def _capture_once(self, device):
        """One-time construction of the manual CUDA graph (torch.cuda.graph
        recommended pattern): static workspace allocation OUTSIDE any
        capture window, warmup iterations on a dedicated side stream, then
        verbatim capture of _pipeline_body against the static addresses.
        The caller catches every exception and binds the eager tier
        permanently."""
        self.q_in = torch.zeros((83, 8, 64), dtype=torch.float16, device=device)
        self.k_in = torch.zeros((83, 8, 64), dtype=torch.float16, device=device)
        self.v_in = torch.zeros((83, 8, 64), dtype=torch.float16, device=device)
        self.attn_flat_ws = torch.zeros((83, 512), dtype=torch.float16, device=device)
        self._ws_device = device
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                self._pipeline_body()
        torch.cuda.current_stream(device).wait_stream(stream)
        self.cuda_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.cuda_graph):
            self._pipeline_body()

    def _bind_eager_permanently(self):
        """Permanent down-tier binding (manual-replay -> framework-eager):
        set the named flag once and drop every replay-tier artifact so the
        lower tier stays reachable and the failure leaves no partial state."""
        self.replay_failed = True
        self.cuda_graph = None
        self.q_in = None
        self.k_in = None
        self.v_in = None
        self.attn_flat_ws = None
        self._ws_device = None

    def _forward_replay(self, query, key, value):
        """Per-call target-regime flow: three fp16 copy-ins, ONE replay
        submission, one small copy-out into an invocation-owned fresh
        buffer allocated OUTSIDE the boundary. Workspace contents are fully
        overwritten transient computation state and are never returned."""
        if self.cuda_graph is None:
            self._capture_once(query.device)
        self.q_in.copy_(query)
        self.k_in.copy_(key)
        self.v_in.copy_(value)
        self.cuda_graph.replay()
        result = torch.empty_like(self.attn_flat_ws)
        result.copy_(self.attn_flat_ws)
        return result

    def _forward_eager(self, query, key, value):
        """Framework-eager tier: byte-frozen baseline_adapter.py body kept
        verbatim for any regime; permanently reachable and correct."""
        num_tokens = query.shape[0]
        q = query.unsqueeze(0).transpose(1, 2)
        k = key.unsqueeze(0).transpose(1, 2)
        v = value.unsqueeze(0).transpose(1, 2)
        if self.num_kv_heads != self.num_heads:
            r = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(r, dim=1)
            v = v.repeat_interleave(r, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, scale=self.scale, is_causal=True)
        return out.squeeze(0).transpose(0, 1).reshape(
            num_tokens, self.num_heads * self.head_size)

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor) -> torch.Tensor:
        try:
            if self._is_target_regime(query, key, value):
                return self._forward_replay(query, key, value)
        except Exception:
            # Any warmup/capture/replay failure binds the eager tier
            # permanently and still serves THIS call correctly through it.
            self._bind_eager_permanently()
        return self._forward_eager(query, key, value)

    def run_out(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor, out: torch.Tensor) -> None:
        """Preallocated-output surface (project.md public_contract; unlocks
        canonical profile_mode=kernel profiling). Fills the caller-provided
        buffer with results bitwise-equal to forward for identical inputs;
        the copy-out happens OUTSIDE the replay boundary every call and the
        buffer is never aliased to workspace. Returns None."""
        try:
            if self._is_target_regime(query, key, value):
                if self.cuda_graph is None:
                    self._capture_once(query.device)
                self.q_in.copy_(query)
                self.k_in.copy_(key)
                self.v_in.copy_(value)
                self.cuda_graph.replay()
                out.copy_(self.attn_flat_ws)
                return None
        except Exception:
            self._bind_eager_permanently()
        out.copy_(self._forward_eager(query, key, value))
        return None


def get_inputs():
    (num_tokens, num_heads, head_size) = (83, 8, 64)
    dtype = torch.float16
    query = torch.randn(num_tokens, num_heads, head_size, dtype=dtype, device='cuda')
    key = torch.randn(num_tokens, num_heads, head_size, dtype=dtype, device='cuda')
    value = torch.randn(num_tokens, num_heads, head_size, dtype=dtype, device='cuda')
    return [query, key, value]


def get_init_inputs():
    return [8, 64, None, 8]
