import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelNew(nn.Module):

    def __init__(self, num_experts: int, top_k: int, hidden_size: int, intermediate_size: int, renormalize: bool=True):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.renormalize = renormalize
        self.w1 = nn.Parameter(torch.empty(num_experts, 2 * intermediate_size, hidden_size))
        self.w2 = nn.Parameter(torch.empty(num_experts, hidden_size, intermediate_size))
        nn.init.normal_(self.w1, std=0.02)
        nn.init.normal_(self.w2, std=0.02)

    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]
        dtype = hidden_states.dtype
        scores = torch.softmax(router_logits.float(), dim=-1)
        (topk_weights, topk_ids) = torch.topk(scores, self.top_k, dim=-1)
        if self.renormalize:
            topk_weights = topk_weights / topk_weights.sum(-1, keepdim=True)
        topk_weights = topk_weights.to(dtype)
        flat_ids = topk_ids.view(-1)
        flat_w = topk_weights.view(-1)
        x_rep = hidden_states.unsqueeze(1).expand(-1, self.top_k, -1).reshape(-1, self.hidden_size)
        w1 = self.w1.to(dtype)
        w2 = self.w2.to(dtype)
        expert_out = torch.zeros_like(x_rep)
        for e in range(self.num_experts):
            mask = flat_ids == e
            if not mask.any():
                continue
            x_e = x_rep[mask]
            gate_up = x_e @ w1[e].T
            (gate, up) = gate_up.chunk(2, dim=-1)
            act = F.silu(gate) * up
            expert_out[mask] = act @ w2[e].T
        expert_out = expert_out * flat_w.unsqueeze(-1)
        return expert_out.view(num_tokens, self.top_k, self.hidden_size).sum(dim=1)

def get_inputs():
    (num_tokens, hidden_size, num_experts) = (83, 128, 8)
    hidden_states = torch.randn(num_tokens, hidden_size, dtype=torch.float16, device='cuda')
    router_logits = torch.randn(num_tokens, num_experts, dtype=torch.float32, device='cuda')
    return [hidden_states, router_logits]

def get_init_inputs():
    return [8, 2, 128, 64]
if __name__ == '__main__':
    init_inputs = get_init_inputs()
    model = Model(*init_inputs).eval()
    inputs = get_inputs()
    with torch.no_grad():
        out = model(*inputs)
    if isinstance(out, (tuple, list)):
        for o in out:
            if hasattr(o, 'shape'):
                print(o.shape)
    else:
        print(out.shape)
