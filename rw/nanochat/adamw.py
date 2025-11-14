import torch
import torch.distributed as dist
from torch import Tensor

class DistAdamW(torch.optim.Optimizer):
    """
    Distributed AdamW optimizer.
    In the style of ZeRO-2, i.e. shared optimizer states and gradient reduction
    """
    
    def __init__(
        self,
        param_groups,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(param_groups, defaults)
    
    @torch.compile # 计算密集型，频繁调用的方法，如 step(), forward(), training_step()
    @torch.no_grad() # 用于不需要计算梯度的方法，如  optimizer, inference, init_params, statistics
    def step(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        reduce_scatter_futures: list[torch.Future] = []
        all_reduce_futures: list[torch.Future] = []
        grad_slices = []
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            # empty_like 会返回一个指定 size 的未初始化的 tensor，适用于其内容会马上被覆盖的场景
            grad = torch.empty_like(
                params[-1]
            )
            for base_i in range(len(params)):
                grad = params[base_i].grad
                rank_size = grad.shape[0] // world_size
                grad_slice = torch.empty_like(grad[:rank_size])
                reduce_scatter_futures.append(
                    dist.reduce_scatter_tensor(
                        grad_slice, 
                        grad, 
                        op=dist.ReduceOp.AVG,
                        async_op=True
                    ).get_future()
                )
                grad_slices.append(grad_slice)
        
        idx = 0
            
    