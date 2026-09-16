# Sources and Licenses

Snapshot date: 2026-09-16. No original Git history is included.

| Source key / destination | Original project | Version context | License context |
|---|---|---|---|
| `gastric` / `.` | gastric_generated651_cv5_10seeds_20260916/source | complete651 generated V2 | Research code; third-party components retain their own licenses |
| `gastric_event_multistage` / event modules, entrypoint, tests and protocol | gastric_event_multistage_20260916/source | event-only S0-S3, complete651 fivefold ten seeds | Same research-code and retained third-party terms |
| `event_v2_research` / event_v2 modules, runner and tests | Authored in this repository on 2026-09-17 | Seed-specific internal holdouts, field tokens, gated stage adapters and ensemble heads | Independent implementations of cited mathematical ideas; no upstream implementation copied |

Event V2 references and actual implementation differences, including upstream
licenses, are documented in [EVENT_V2_RESEARCH.md](EVENT_V2_RESEARCH.md).
The new code does not import external research repositories or their checkpoints.

## Packaging Changes

- Original Python module names, model mathematics, losses and tensor contracts are retained.
- The event update copies its eight runtime modules, runner and synthetic tests without changing their contents; its protocol and release documentation describe portable execution.
- Absolute source and artifact paths are resolved through `research_release.py` and private local settings.
- Old SSH-script introspection is replaced by explicit local SSH settings.
- Historical identity values and cohort exclusions are not emitted into release configs; real asset checks remain strict.
- Configuration unit tests use temporary synthetic assets and retain mismatch/rejection assertions.
- First-post/three-phase model construction reads architecture settings without requesting unrelated codec identities.
- ROI32 reporting converts BF16 arrays to FP32 before NumPy; its numerical regression is covered by a CPU test.
- No source from the original running worktrees is modified. No patient artifacts or restricted weights are included.

## Attribution

MeWM-derived code: https://github.com/scott-yjyang/MeWM . Original license texts remain beside each included source tree.
SymmFlow original copyright and the adapted VQ exception remain in its LICENSE and third-party notices.
The registration transform reader retains the MIT notice by Chengyue Wu.
StageWorld reference methods and component terms are documented in its retained license audit.
This private source collection does not grant additional rights to third-party weights or datasets.
No new blanket license is assigned to the collection. For project-specific details, use only rows present in the table above.

`SOURCE_FILES.json` maps copied source files. Newly written release tooling and documentation are recorded in `RELEASE_FILES.json`.
Source identity is described by project, branch/context and snapshot date; release records do not contain checksum values.
