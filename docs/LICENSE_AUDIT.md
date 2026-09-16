# License and model-use audit

Audit date: 2026-09-07; selected-route execution update: 2026-09-09. This is an
engineering inventory, not legal advice.

| Component | Source license | Weight/data terms | Current decision |
|---|---|---|---|
| Merlin | MIT | Current public Hugging Face card says MIT and ungated; loader automatically downloads unless wrapped | Candidate default CT encoder after explicit storage/download approval and parity |
| TITAN | GitHub has no top-level license file; README/package declare CC-BY-NC-ND-4.0 | Gated CC-BY-NC-ND-4.0; academic noncommercial only; individual institutional approval; no redistribution; derivatives trained on outputs restricted | Interface only; blocked from execution and redistribution |
| TRIDENT | CC-BY-NC-ND-4.0 | Each encoder has its own terms; some factory paths download automatically | Run as isolated, unmodified upstream tool only after institutional review |
| UNI / UNI2-h | CC-BY-NC-ND-4.0 | Gated, noncommercial, individual approval, no redistribution and derivative restrictions | Optional independent comparator; blocked from execution |
| PRISM2 | Custom model code under model repository terms | Manual gated CC-BY-NC-ND-4.0; academic-only; expressly excludes clinical/diagnostic/treatment use | Research-only optional comparator; never deployment default; blocked |
| CLARITY | MIT | BrainIAC and MedGemma dependencies have separate terms | Ideas/interfaces may be cleanly adapted; no dependency import planned |
| V-JEPA 2 | MIT; three documented data utilities are Apache-2.0 | Individual checkpoints/datasets require their notices | Architectural reference only |
| DreamerV3 | MIT | No weights used | Formula/design reference; PyTorch reimplementation |
| pycox | BSD-2-Clause | Not weight based | Numerical reference and optional dependency |
| Dynamic-DeepHit | No license file | No official weight terms identified | Do not copy; paper-based clean-room baseline only |
| CTSMamba | No license file | Claimed checkpoint is not present in inspected repository | Do not copy; paper/source behavior informs independent comparator |
| Swin UNETR contribution | Apache-2.0 | Public release `0.8.1`; a locally supplied copy was approved for offline frozen-feature engineering and remains outside the releasable project | Selected ADR-0012 CT route executed with 126/126 backbone tensor coverage, exact eight-head exclusion, and runtime parity; no redistribution or clinical-validity inference |
| Cardiac world model | No license file | Repository artifacts have no explicit reuse grant | Do not copy; methods only |
| MeWM | LICENSE says CC-BY-NC-4.0; README says CC-BY-NC-2.0 | GPT-4o API and component models have separate terms | No code import; external-image upload path forbidden |
| Clin-JEPA | MIT | MIMIC-IV is credentialed PhysioNet data; Qwen weights have separate terms | Mask/EMA concepts only |
| scikit-survival | GPL-3.0 | Not weight based | Optional evaluation environment only; do not impose it on core package |
| TRIPOD+AI | Publication/checklist terms | Not applicable | Citation/reporting guidance |

## Gates

- Model-card approval is not inferred from public metadata. TITAN, UNI2-h and PRISM2 remain disabled until the user/institution accepts the exact terms outside this code path.
- `trust_remote_code=True` is forbidden by default. Approved use requires a locally reviewed, immutable revision and offline loading.
- No restricted weights, source snapshots or patient-derived feature caches will be redistributed.
- A missing license is not treated as permissive. Cardiac WM, Dynamic-DeepHit and CTSMamba source can inform behavior but is not copied.
- Merlin's ungated status does not authorize an unbudgeted 62.8 GB repository download. The main checkpoint subset and storage plan require approval.
- ADR-0012 authorizes only the specified local Swin UNETR release and frozen,
  outcome-blind engineering scope. It does not authorize redistribution,
  fine-tuning, OS-supervised use, or any other model asset by implication.
