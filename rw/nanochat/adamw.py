import torch


class DistAdamW(torch.optim.Optimizer):
    def __init__(
        self,
        param_groups,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        pass