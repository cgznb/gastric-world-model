"""Read-only backend comparison on a saved fold bundle; prints aggregates only."""

import argparse
import json
from pathlib import Path

import torch

from stageworld.artifacts import read_json
from stageworld.generated700_data import fit_inputs, load_pool
from stageworld.generated700_inference import predict_bundle
from stageworld.generated700_models import AnchoredClassifier


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("project", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.cuda.set_per_process_memory_fraction(0.1)
    root = args.project / "artifacts/formal"
    groups = read_json(root / "partitions/fold-0/groups.json")
    pool = load_pool(args.project / "artifacts/pool")
    x, _ = fit_inputs(
        pool, groups["train"] + groups["validation"], root / "partitions/fold-0/refit.pt"
    )
    rows = pool.indices(groups["outer"])
    base = root / "fits/generated_v2_frozen_bce/17/fold-0/bundle"
    bundle = torch.load(base / "inference.pt", weights_only=True, map_location="cpu")
    state = bundle["model_state"]
    result = {}
    for fastpath in (True, False):
        torch.backends.mha.set_fastpath_enabled(fastpath)
        model = AnchoredClassifier(
            361, bundle["family"], state["anchor_weight"], state["anchor_bias"]
        )
        model.load_state_dict(state)
        model.eval()
        cpu = model(x[rows], pool.ct0[rows], pool.ct0_valid[rows]).sigmoid()
        cpu_batches = torch.cat(
            [model(x[r], pool.ct0[r], pool.ct0_valid[r]).sigmoid() for r in rows.split(32)]
        )
        model.cuda()
        gpu = torch.cat(
            [
                model(x[r].cuda(), pool.ct0[r].cuda(), pool.ct0_valid[r].cuda()).sigmoid().cpu()
                for r in rows.split(32)
            ]
        )
        replay = predict_bundle(
            base,
            [pool.clinical[p] for p in groups["outer"]],
            [pool.treatments[p] for p in groups["outer"]],
            pool.interval[rows],
            pool.ct0[rows],
            pool.ct0_valid[rows],
        )["probabilities"].float()
        result[str(fastpath)] = {
            "CPU_batch_difference": float((cpu - cpu_batches).abs().max()),
            "CPU_GPU_difference": float((cpu - gpu).abs().max()),
            "portable_CPU_difference": float((cpu - replay).abs().max()),
            "portable_GPU_difference": float((gpu - replay).abs().max()),
            "residual_scales": state["residual_scale"].tolist(),
        }
        del model
    print(json.dumps(result))


if __name__ == "__main__":
    main()
