"""Verify real baseline-only replay while denying every unrelated input-file read."""

from __future__ import annotations

import argparse
import builtins
import io
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch

from stageworld.artifacts import atomic_write_private_json, read_json
from stageworld.generated_inference import predict_generated_query


def verify(study: Path) -> dict[str, Any]:
    study = study.resolve()
    arms = {}
    torch.set_num_threads(4)
    for mode in ("history_generated", "history_only", "generated_only"):
        config = study / mode / "inference.json"
        query = study / "baseline_only_query.json"
        output = study / mode / "independent_replay.json"
        allowed = {config, query, study / mode / "inference.pt", study / "baseline_only_example.pt"}
        reads: set[str] = set()

        def guarded(original: Any, permitted: Any = allowed, recorded: Any = reads) -> Any:
            def opened(file: Any, access: str = "r", *args: Any, **kwargs: Any) -> Any:
                if isinstance(file, (str, Path)) and "r" in access:
                    path = Path(file).resolve()
                    if path not in permitted:
                        raise RuntimeError("Input read outside portable baseline-only allowlist")
                    recorded.add(path.name)
                return original(file, access, *args, **kwargs)

            return opened

        with (
            patch.object(builtins, "open", guarded(builtins.open)),
            patch.object(io, "open", guarded(io.open)),
        ):
            predict_generated_query(config, query, output)
        expected = torch.load(
            study / mode / "predictions.pt", map_location="cpu", weights_only=True
        )["rates"][0, 1]
        result = read_json(output)
        actual = torch.tensor(result["rates"])[0, :, 0]
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        arms[mode] = {
            "status": "passed",
            "allowed_input_files_read": len(reads),
            "ct1_and_outcome_files_denied": True,
            "max_rate_difference_cpu_single_vs_saved_batch": float((actual - expected).abs().max()),
        }
    result = {"status": "passed", "arms": arms, "test_used": False}
    atomic_write_private_json(study / "independent_inference_verification.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("study", type=Path)
    print(json.dumps(verify(parser.parse_args().study)))
