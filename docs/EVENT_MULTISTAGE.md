# Event Multistage Terminal Classification

Current event workflow, added to this source release on 2026-09-16. The original
experiment ran full training without a small real-data trial. Release validation
uses synthetic data and does not repeat that clinical experiment. Generated V2
651 and original700 entrypoints remain available as separate comparisons.

## Cohort and events

Use the existing651 pool and its unchanged five patient folds. pCR has127
positive/524 negative labels; recurrence has147 positive/504 negative labels.
Training/validation is520/131 once and521/130 four times. The original700 pool
has47 missing pCR labels and three incomplete CT pairs; the source workbook has
956 records. These are different populations. Missing targets never become
negatives; runtime masks also support missing CT1, pCR and recurrence labels.

Spreadsheet bindings: AQ surgery (1/0), CE postoperative chemotherapy (yes/no),
BJ pCR (1 positive/2 negative), CG recurrence/metastasis (1/0). Neoadjuvant status
comes from cleaned treatment-method flags. Separate absent/present/unknown/
conflict codes are retained. Only present events execute a transition. Other
codes preserve the state; unknown/conflict also mark an incomplete history.
pCR supervision requires both a valid label and a confirmed surgical event.

CE is recorded as yes even in unoperated patients. Per the user's decision,
accept a recorded yes only after confirmed surgery; otherwise flag conflict and
skip that event. All selected651 have surgery and recorded postoperative yes.
This cohort cannot identify effects of surgery versus no surgery or postoperative
chemotherapy versus no chemotherapy. Postoperative regimen details are unavailable
to this model; its transition uses an event token only. Five selected patients
have recorded recurrence on/before surgery. The user chose event-only modeling
without time exclusions. Retain these labels and describe the endpoint as
recorded status, not future new recurrence following treatment.

## Forward path

CT6 uses sex, age, BMI, cT, cN, cM with existing missingness and N+ encoding.
Its32 features and four82-feature no-cycle treatment tokens yield360 values.
All transforms and feature statistics are fitted on training patients only.
Dates, intervals, cycles, observed pathology, outcomes and CT1 are excluded from
the input interface.

Frozen external Swin CT0 features27x768 and clinical information initialize S0.
S0 -> neoadjuvant -> S1 -> surgery -> S2 -> postoperative chemotherapy -> S3.
Each state is27x128. A shared four-layer transition uses four-head self-attention,
clinical/event/action conditioning and spatial depthwise local/dilated residual
blocks, dropout0.1. Regimen/drug tokens describe neoadjuvant therapy only and
enter only that event. Later events cannot change earlier states. Skips are
identity maps. No observed-state updater is used.

S1 decodes a CT1 feature residual over CT0. Separate query/two-way-attention/
gated-pooling/MLP heads read S2 for auxiliary pCR and the last effective state
for terminal recurrence. No intermediate recurrence head, survival head, KL
term or logistic anchor. Ordinary clinical/treatment/event logistic regression,
C=1, is fitted once per fold as a standalone recurrence reference.

The spatial modules reuse the generated V2 adaptations; see
GENERATED_V2_SOURCES.md for AF/CLARITY/MeWM sources and license notes. This is
an engineering adaptation, not an exact reproduction of those papers.
CT0/CT1 crop correspondence is unverified, so supervision is a permutation-
invariant feature-set loss: global SmoothL1+cosine +0.25 fixed sliced projection
distance +0.1 spread loss. No matched spatial reconstruction claim is made.

### Tensor dimensions

B is the batch size, normally32. Clinical encodings use3+2+2+12+9+4=32 values
for sex/age/BMI/cT/cN/cM. Each raw treatment token has39 value slots,39 observation
slots and4 group slots; all360 flattened values are standardized using the
training fold, so constant slots become zero. Event codes are0absent,1present,
2unknown and3conflict, in neoadjuvant/surgery/postoperative column order.

| Operation | Tensor shape |
| --- | --- |
| Cached frozen CT0 features | B x27 x768 |
| CT projection plus position and broadcast clinical | B x27 x128, initial S0 |
| Three integer event codes; select current stage column | B x3 -> B |
| Current stage's independent embedding lookup and token axis | B -> B x128 -> B x1 x128 |
| Clinical / current event / four action slots | B x1 x128 / B x1 x128 / B x4 x128 |
| Conditions concatenated with previous state | B x33 x128 |
| Each four-head attention and FFN block | B x33 x128 -> B x33 x512 -> B x33 x128 |
| Spatial branch within each block | B x27 x128 -> B x128 x3 x3 x3 -> B x27 x128 |
| Four-block output followed by residual state update | B x27 x128 |
| Stacked S0/S1/S2/S3 trajectory | B x4 x27 x128 |
| S1 CT decoder | B x27 x128 -> B x27 x256 -> B x27 x768 |
| Each endpoint learned query and state | B x1 x128 and B x27 x128 |
| Two-way attention, gated pool, query concatenation | B x256 |
| Each endpoint MLP, squeeze and sigmoid | B x256 -> B x128 -> B x64 -> B x1 -> B |
| Both endpoint probabilities stacked for inference | B x2 |

The four spatial blocks have distinct weights; the same four-block stack is
reused for all three event transitions. A block applies pre-normalized attention
and FFN, followed by GroupNorm8, condition-derived FiLM, two depthwise3D
convolutions with dilation1/2, a1x1 mixing convolution and a spatial residual.
The CT decoder predicts a residual scaled by training CT0 channel standard
deviations and added to CT0. It predicts features, not CT pixels. The endpoint
heads consume hidden states, not decoded CT1 or an earlier endpoint prediction.

Parameters excluding frozen Swin: world1,384,064; pCR471,809; recurrence471,809;
total2,327,682. During pretraining the recurrence head receives no loss gradient.
The postoperative event embedding has no nonzero pretraining loss gradient,
although materialized zero gradients can still permit AdamW weight decay.

## Training and reporting

Seeds17,43,97,131,173,211,257,307,359,419 share five folds:50fresh pretrains and
50matching joint fits, serially. No inner split, extra refit or prior cohort-
trained initialization.

- Pretrain: CT loss +0.5 ordinary masked pCR BCE.
- Joint: balanced masked recurrence BCE +0.5 ordinary pCR BCE +0.1 CT loss.
- Positive recurrence weight = training negative/positive count. Weighted BCE
  divides by the sum of valid sample weights. CT1 and labels are detached.
  Entirely unsupervised batches are skipped.
- AdamW, WD0.01, batch32, BF16 CUDA, clip1; no scheduler or runtime cap.
- Pretrain LR0.0002; joint world LR0.00002 and heads LR0.0002. Joint training
  updates generator and heads; external Swin remains frozen.
- Each phase max100 epochs, patience15, min_delta0.0001. Raw-best selection
  is separate from significant-improvement early stopping.
- Pretrain minimizes validation CT+pCR loss; joint maximizes validation
  recurrence AUPRC. Joint loads this fold/seed's best pretraining checkpoint.
- selected/latest/final include model, optimizer, RNG and epoch recovery state.

Report each seed's fivefold mean and sample SD separately, never across seeds.
Metrics: AUROC, AUPRC, accuracy, sensitivity/recall, specificity, precision, F1,
NPV, MCC, balanced accuracy, BCE, Brier, counts and confusion matrix. Unsupported
metrics stay empty. Threshold0.5, no calibration. Weighted scores are not
calibrated clinical probabilities. Compare generated CT features with copy-CT0
and training-mean controls before and after joint training.

The reported validation folds also select checkpoints. Results are developmental
and selection-biased, not independent external or sealed-test evaluation.

## Runtime and validation

Use `python run.py gastric scripts/run_event_multistage.py` so imports come
from this clone. Its required arguments are `--source-pool` (the complete651
directory containing pool.pt/folds.json), `--bindings` (private JSON paths),
`--pool` (event-cache directory) and `--output` (study output). Optional modes are
`--prepare-only` and `--verify-only`. See README.md and DATA_AND_PATHS.md for the
binding template and exact commands. There is no real-data smoke mode.
Preparation performs no optimizer updates. Formal training and full-study
verification require BF16 CUDA and cannot silently fall back to CPU. Standalone
bundle prediction through event_inference.predict_bundle supports CPU.

The runner sets four CPU threads, CUDA allocator fraction0.1 and no runtime cap;
it does not install a service or enforce an operating-system memory limit. The
original experiment used an external service with CPUQuota400%, MemoryHigh8G,
MemoryMax12G and swap0. A file lock prevents duplicate controllers sharing the
output parent. Source binding uses file sizes and modification times. Resume
requires the same source metadata; retain the original source for an existing
run and use a new output directory with a new clone. Private inputs, predictions
and weights are not distributed in this repository.

Final verification replays selected/final scores, patient/parent bindings,
portable bundles with source-file reads denied, and all endpoint metrics without
fitting. Passing software checks does not establish clinical model quality.
