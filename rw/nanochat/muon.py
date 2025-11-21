import torch
import torch.distributed as dist
from torch import Tensor

@torch.compile
def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert (
        G.ndim >= 2
    )  # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = (
            b * A + c * A @ A
        )  # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

class Muon(torch.optim.Optimizer):
    """
    SGD + orthogonalization    
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_steps": ns_steps,
        }
        params: list[Tensor] = [*params]
        param_groups = []
        for size in {p.numel() for p in params}:
            group = {
                "params": [p for p in params if p.numel() == size]
            }
            param_groups.append(group)
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            for p in params:
                g = p.grad
                assert g is not None
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf: Tensor = state["momentum_buffer"]
                buf.lerp_(g, 1 - group["momentum"])
                g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                p.add(g, alpha=-group["lr"] * max(1, p.size(-2) / p.size(-1)) ** 0.5)

class DistMuon(torch.optim.Optimizer):
    """
    Muon: SGD-momentum + (optional) Nesterov, then orthogonalize the 2D update via Newton-Schulz,
    finally apply aspect-ratio scaled step. Performs its own distributed synchronization:
        - reduce_scatter(AVG) for gradient averaging
        - all_gather to replicate updated weights

    Notes:
        * Designed for 2D parameters (e.g. linear/conv kernels reshaped to 2D). Do not use for 0D/1D
        * Momentum buffers are maintained only on the owner rank for each parameter (rank chosen by block-cyclic assignment below). If you checkpoint optimizer state on a single rank, consolidate states beforehand.
        
    Args:
        params: iterable of Tensors
        lr: learning rate
        momentum: momentum coefficient in [0, 1)
        nesterov: if True, Nesterov-style update (g <- lerp(g, buf, momentum)); else use buf
        ns_steps: number of Newton-Schulz iterations for the orthogonalization
    """
    def __init__(self):
        pass
