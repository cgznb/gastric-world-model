"""Run an exported classifier without dataset bindings or post-treatment files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import torch

    from stageworld.binary_improvement_inference import predict_query

    torch.set_num_threads(4)
    print(json.dumps(predict_query(args.bundle, args.query, args.output)))


if __name__ == "__main__":
    main()
