# Final Submission Self-Check

**Generated:** 2026-08-05
**Deadline:** 2026-08-06 23:59 UTC+8 (≈ 46 hours remaining)
**PR:** https://github.com/AMD-DEV-CONTEST/Radeon-hackathon-2026-07/pull/45

## Deliverables checklist

| # | Requirement | Path | Status |
|---|---|---|---|
| 1 | Technical Report (8 sections, PDF) | `docs/Technical_Report.pdf` | ✅ Regenerated 2026-08-05, 12 pages, TOC, key phrases verified |
| 2 | Project Source Code | `submission/` + `Dockerfile` | ✅ All source present, Dockerfile fixed (removed stale `COPY configs/`) |
| 3 | Reproducibility README | `README.md` | ✅ Fork URL correct, all 65/65 file refs valid, Results & Enhanced Evaluation sections align with data-integrity narrative |
| 4 | Demo Video (3–5 min) | `docs/demo_video.mp4` | ✅ 20MB, present |
| 5 | Supplementary Slides | `docs/Supplementary_Slides.pptx` | ✅ Present |
| 6 | Upstream Fix Patch | `docs/upstream_fix.patch` | ✅ Present |
| 7 | Upstream PR Description | `docs/upstream_pr_description.md` | ✅ Present |
| 8 | Upstream Contribution Evidence | `docs/upstream_contribution_evidence.png` | ✅ Present |

## Measured results (canonical, ready to cite)

Source: `logs/eval_results.json` and `logs/smolvla_train.log`.

- Closed-loop success rate: **100% (2/2 episodes)**
  - `018_plum` → `024_bowl_purple`: SUCCESS, 308 steps
  - `011_banana` → `024_bowl_yellow`: SUCCESS, 221 steps
- Training: 2000 steps, final loss **0.073**, wall-clock **~5 minutes** on Radeon Pro W7900, ~6.7 step/s, ~85% GPU utilization.

## Documentation integrity pass (2026-08-05)

Ensured strict separation of measured vs. planned vs. archived-projection data:

- ✅ `docs/Technical_Report.md §7.1.1` rewritten as **Planned Robustness Evaluation Protocol** with explicit data-integrity note.
- ✅ `docs/eval_robust_protocol.json` — canonical reproducible schema for the 12-episode matrix.
- ✅ `docs/eval_robust_results.json` — placeholder with `PENDING_EMPIRICAL_RUN` status (no fake numbers).
- ✅ `docs/eval_robust_extrapolation.json` — old projection file preserved with prominent `_DISCLAIMER_READ_FIRST` block flagging `not_a_measurement: true`.
- ✅ `README.md` Results & Enhanced Evaluation sections updated to match.
- ✅ `docs/submission_final_state.json` updated with `documentation_integrity_pass` audit trail.

## Upstream contribution (bonus points)

- ✅ Issue #4205 (2026-07-29, OPEN): https://github.com/huggingface/lerobot/issues/4205
- ✅ PR #4324 (2026-08-04, OPEN): https://github.com/huggingface/lerobot/pull/4324
- ✅ Fix patch: `docs/upstream_fix.patch`
- ✅ Runtime shim: `submission/src/utils/rocm_resize_patch.py`

## Hygiene pass (2026-08-05)

- ✅ 34 historical debugging scripts moved from `scripts/` to `archive/scripts_iterations/` with README.
- ✅ Cookie helper `scripts/DSW_COOKIE_GUIDE.md` and cookie placeholder added for optional Phase-2 real-run upgrade.
- ✅ `scripts/jupyter_api_exec.py` parameterized to accept `DSW_INSTANCE_ID` env var (default `dsw-2076885`).

## Items intentionally NOT changed

- `submission/requirements.txt` — the `torch==2.9.1+rocm7.2.1` + `torchvision==0.20.1+rocm7.2.1` combination looks version-mismatched at first glance but the training log (`logs/smolvla_train.log`) confirms this exact combination executed 2000 fine-tune steps on the W7900. Not modifying.
- `Dockerfile` `FROM rocm/pytorch:latest` — intentionally unpinned because the README §1.2 provisioning steps assume the Radeon Cloud base image already provides the pinned ROCm 7.2.1 + PyTorch 2.9.1 stack.
- Baseline `logs/eval_results.json` — canonical, do not touch.

## Optional Phase 2 (blocked on human action)

The 12-episode enhanced evaluation on ModelScope DSW (Radeon RX 7900 GRE / ROCm 5.7 / dsw-2076885) is prepared but blocked by Aliyun MAAS OAuth. To unblock:

1. Follow `scripts/DSW_COOKIE_GUIDE.md` to extract a browser session cookie into `scripts/jupyter_cookie.txt`.
2. Run `python scripts/jupyter_api_exec.py` to verify connectivity.
3. Install `genesis-world` + `lerobot`, upload `submission/` + checkpoint, run `eval_sort_smolvla_robust.py --episodes 12 --perturb 0.02 --seed 42`.
4. The evaluator writes real measurements into `docs/eval_robust_results.json`; regenerate Technical_Report.pdf to reflect real numbers.

Without Phase 2 the submission still passes on documented integrity: baseline 2/2 = 100% + reproducible planned protocol + upstream fix PR merged into hackathon narrative.

## Recommended next action

Push the current documentation-integrity pass to the fork so PR #45 auto-updates. See `P3-4` in the todo list.
