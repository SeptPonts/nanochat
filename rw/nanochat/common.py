import os


# ddp - distributed data parallel
def is_ddp():
    # TODO is there a proper way
    return int(os.environ.get("RANK", -1)) != -1

# dist - torch.distributed, 用于多 GPU/多机分布式训练
def get_dist_info():
    if is_ddp():
        assert all(var in os.environ for var in ["RANK", "LOCAL_RANK", "WORLD_SIZE"])
        # 全局进程编号 (0 ~ world_size - 1)
        ddp_rank = int(os.environ["RANK"])
        # 当前节点内的本地进程编号
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        # 总进程数 (通常等于 GPU 数量)
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        return True, ddp_rank, ddp_local_rank, ddp_world_size
    else:
        return False, 0, 0, 1