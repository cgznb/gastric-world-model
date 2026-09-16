# Generated S1 V2: Source Adaptations

This is a gastric frozen-feature experiment, not an unmodified reproduction of
AF, CLARITY or MeWM. No upstream weights or private upstream data are used.
Reference source snapshots are the local files inspected on 2026-09-16.
Existing references remain read-only. No repository checksum is recorded.

- CLARITY `Predictor/models/full_model.py:LatentPredictor`: separate condition,
  Fourier-time and image tokens; four Transformer layers. Adapted to six
  structured CT6/treatment/time tokens and 27 gastric image tokens, hidden128.
- CLARITY `Predictor/models/survival_module.py:TwoWayCrossAttentionLayer`:
  adapted pre-norm, sequential bidirectional attention with residual FFNs.
  MIT copyright notice is retained in `CLARITY_LICENSE.txt`. Endpoints are
  independent binary pCR/recorded recurrence heads, not survival outputs.
- AF `scripts/models_predictor.py:PredictorJEPA` and
  `models/v2/action_conditioned_world_model.py`: spatial residual dynamics and
  conditional state/readout separation inform the new independent implementation.
  The local public checkout has no top-level license; no AF source is vendored.
  Its post-baseline event inputs and outer-validation selection are not adopted.
- MeWM `Synthesis/Diffusion/ddpm/unet.py` and
  `Survival/model/dim1/{TransMIL,ABMIL}.py`: conditional spatial processing,
  multiscale spatial aggregation and gated attention inform the implementation.
  Its pixel/VQ diffusion decoder is incompatible with 3x3x3 Swin feature grids
  and is not loaded. Source license file says CC BY-NC4.0, although README also
  contains an inconsistent 2.0 statement. No MeWM source is vendored.

New engineering adaptations: depthwise 3D kernels at dilation1/2 within each
transition layer; residual 27x768 feature decoding; permutation-invariant CT
distribution supervision; fixed clinical-logit residual correction; inner-only
per-endpoint shrinkage. These are hypotheses, not established improvements.

The CT set objective uses global SmoothL1+cosine, 0.25 times squared sliced
Wasserstein on 64 fixed unit directions, and 0.1 times feature-standard-deviation
SmoothL1. CT1 tokens are detached. No spatial correspondence between independently
cropped CT0/CT1 is assumed. These losses supervise feature sets, not CT pixels.
