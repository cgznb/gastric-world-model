"""CPU parity of the adapted two-way fusion against the named CLARITY source."""

import argparse
import importlib.util
import json
from pathlib import Path

import torch

from stageworld.generated700_models import TwoWayFusion


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clarity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    source = args.clarity / "Predictor/models/survival_module.py"
    spec = importlib.util.spec_from_file_location("reference_clarity_survival", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.manual_seed(17)
    reference = module.TwoWayTransformer(128, num_heads=4, num_layers=1, dropout=0.1).eval()
    actual = TwoWayFusion(128).eval()
    mapping = {
        "attention.0": "layers.0.cross_attn_1to2",
        "attention.1": "layers.0.cross_attn_2to1",
        "feedforward.0": "layers.0.ffn1",
        "feedforward.1": "layers.0.ffn2",
        "norms.0": "layers.0.norm_q1",
        "norms.1": "layers.0.norm_q2",
        "norms.2": "layers.0.norm_ffn1",
        "norms.3": "layers.0.norm_ffn2",
        "final.0": "final_norm_seq1",
        "final.1": "final_norm_seq2",
    }
    state = reference.state_dict()
    adapted = {}
    for name in actual.state_dict():
        matches = [
            (prefix, original)
            for prefix, original in mapping.items()
            if name.startswith(prefix + ".")
        ]
        assert len(matches) == 1
        prefix, original = matches[0]
        adapted[name] = state[original + name[len(prefix) :]]
    actual.load_state_dict(adapted)
    errors, gradient_errors = [], []
    for scale in (0.01, 1.0, 10.0):
        inputs = [torch.randn(2, 27, 128) * scale for _ in range(2)]
        first = [v.clone().requires_grad_() for v in inputs]
        second = [v.clone().requires_grad_() for v in inputs]
        expected, output = reference(*first), actual(*second)
        for a, b in zip(expected, output, strict=True):
            torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-6)
            errors.append(float((a - b).abs().max().detach()))
        sum(v.square().mean() for v in expected).backward()
        sum(v.square().mean() for v in output).backward()
        for a, b in zip(first, second, strict=True):
            torch.testing.assert_close(a.grad, b.grad, atol=2e-6, rtol=2e-6)
            gradient_errors.append(float((a.grad - b.grad).abs().max()))
    report = {
        "status": "passed",
        "input_scales": [0.01, 1.0, 10.0],
        "maximum_output_error": max(errors),
        "maximum_input_gradient_error": max(gradient_errors),
        "private_data_read": False,
    }
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
