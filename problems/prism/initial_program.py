GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.

    Args:
        gpu_num: Number of GPUs
        models: List of models to place

    Returns:
        A placement of models to GPUs
    """

    # Greedy KVPR-minimizing placement based on Algorithm 1 (without τ check)
    # 1) Sort models by r_j / s_j in descending order
    sorted_models = sorted(models, key=lambda m: (m.req_rate / m.slo), reverse=True)

    # 2) Initialize per-GPU states
    placement = {gpu_id: [] for gpu_id in range(gpu_num)}
    shared_kv = [GPU_MEM_SIZE for _ in range(gpu_num)]  # remaining memory per GPU
    weighted_req_rate = [0.0 for _ in range(gpu_num)]   # sum of r_j / s_j per GPU

    # 3) Assign each model to the GPU that minimizes current KVPR while fitting in memory
    for model in sorted_models:
        best_idx = None
        best_ratio = float('inf')

        for gpu_id in range(gpu_num):
            if model.model_size <= shared_kv[gpu_id] and shared_kv[gpu_id] > 0:
                current_ratio = weighted_req_rate[gpu_id] / shared_kv[gpu_id]
                if current_ratio < best_ratio:
                    best_ratio = current_ratio
                    best_idx = gpu_id

        # Failure: if no GPU can fit, raise an error instead of overcommitting
        if best_idx is None:
            raise ValueError(
                f"Unable to place model of size {model.model_size} GB on any GPU. "
                f"Remaining per-GPU memory: {shared_kv}"
            )

        placement[best_idx].append(model)
        weighted_req_rate[best_idx] += model.req_rate / model.slo
        shared_kv[best_idx] -= model.model_size

    return placement

# EVOLVE-BLOCK-END

