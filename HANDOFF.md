# HANDOFF — Latent Safety Filter (Jenga) · state as of 2026-07-29

Self-contained context brief. Paste this into a fresh session and it should be enough to work
without reading anything else. Deeper detail: `RESUME.md` §7 (measurements), `CLAUDE.md`
(theory/architecture — **but see "CLAUDE.md is stale" below**).

---

## 1. What the project is

A **zero-shot safety monitor** for a visuomotor diffusion policy doing a Jenga-style pick:
a Franka Panda lifts a red block from between two neighbours without toppling them.

The original idea: roll the action chunk forward in a DINOv2 world model, perturb the actions
N=50 times, and flag the chunk unsafe if the perturbed futures diverge from the nominal one —
a Finite Time Lyapunov Exponent (FTLE) on DINOv2 patch latents, cosine distance:

```
lambda = (1/T) * log( d_end / d_start )
d_end   = 1 - cos( z_perturbed[T,p], z_orig[T,p] )
d_start = 1 - cos( z_perturbed[NH,p], z_orig[NH,p] )     NH=3, T=10, horizon 8
```

**As of 2026-07-29 this framing is substantially revised by measurement.** See §4.

## 2. Layout, environments, hardware

```
/home/sanger/wksp/
├── dino_wm/          world model + monitor servers + all analysis scripts
├── panda_express/    MuJoCo sim, replay, eval drivers, results/
├── diffusion_policy/ the policy being monitored (separate track; on branch `avsr`, not main)
├── CLAUDE.md         research reference (partly stale)
├── RESUME.md         machine state + full 2026-07-29 measurements (§7)
└── HANDOFF.md        this file
```

- Env for **both** `dino_wm` and `panda_express`: `conda activate dino_wm`
  (must `source /home/sanger/miniforge3/etc/profile.d/conda.sh` first in non-interactive shells).
- `diffusion_policy` uses `robodiff` (python 3.9) — do not reuse it for the world model.
- GPU: RTX 5060 Ti, **8 GB**, Blackwell → needs cu128 wheels. `environment.yaml` is unusable
  (pins torch 2.3.0/cu121); the working env has torch 2.11+cu128.
- MuJoCo rendering: **only `MUJOCO_GL=glfw` works here** (egl and osmesa both fail). Needs
  `DISPLAY=:1`.
- Model runs at ~2.0 s/chunk at N=50, ~53 ms at N=1, ~0.35 GB VRAM. Not memory-bound.

### Checkpoints
| file | epoch | note |
|---|---|---|
| `dino_wm/outputs/model_latest_single.pth` | 88 | single-view, **the one to use** |
| `/home/sanger/Downloads/model_latest.pth` | 46 | dual-view; undertrained, performs worse |

### Data
| path | what |
|---|---|
| `panda_express/tasks/jenga_noise_50/jenga_single_100.lmdb` | 100 eps, safety eval, 25 failures |
| `panda_express/labels_noise100.json` | labels for the above (45° topple threshold) |
| `panda_express/tasks/jenga_tilt_100/jenga_tilt.lmdb` | 100 eps **with per-step block tilt** (`<ep>_tilt`) |
| `panda_express/tasks/jenga_noise_50/jenga_unified.lmdb` | dual-camera (cam1+cam2) |

## 3. CLAUDE.md is stale — do not trust these parts

| CLAUDE.md says | reality (measured 2026-07-29) |
|---|---|
| Accuracy 96.9%, F1 62.4% | Artifact of a 62-traj run + 15° threshold. Real chunk-level F1 was **0.039** for that metric. |
| Use `ftle`, δ=0.8 | `ftle` is the **worst** estimator (AUC 0.599). δ=0.8 is just the p99 of the safe FTLE distribution (0.8021). |
| Report accuracy | **Never.** Base rate is 1.4%; "always safe" scores 98.59%, beating every real config. |
| Topple threshold 15° | Wrong event (blocks merely nudged). Now **45°** — peak tilt is bimodal: standing 6–17°, toppled 90–96°. |

## 4. The findings that matter

Evaluation convention throughout: 100 episodes / 1772 chunks / **25 unsafe (1.4%)**; a chunk is
positive iff `failure_step` falls in its 8-step horizon; chunks strictly after the failure are
**dropped**; thresholds are percentiles of the **safe** score distribution only (preserves
zero-shot); CIs are paired cluster bootstrap resampling **episodes**.

**(a) The FTLE denominator was harmful.** `d_start` is one prediction step in — tiny and noisy.
Dropping it: AUC 0.599 → 0.780.

**(b) Cosine amplifies noise on low-norm patches.** On ground-truth-static patches,
`corr(||z||, d_end) = -0.641`. Blank-table patches have short feature vectors, and cosine
divides by ||z||, so their *direction* is unstable. **Masking the 30 lowest-||z|| patches:
0.759 → 0.854**, held-out validated on three independent methods (gains +0.044/+0.096/+0.029,
all selecting k≈30). *This is the one fully validated fix.*
(The initial hypothesis — high-norm register artifacts — was falsified; the sign is negative.)

**(c) The Deviator Agent's value is unclear and DATASET-DEPENDENT.** On `jenga_noise_50` a
single unperturbed rollout matches the full N=50 apparatus (0.826 vs 0.851, CI
[-0.075, +0.134]) at **1/38th the cost**. But on `jenga_tilt_100` the ordering **reverses**:
`d_end` 0.887 vs nominal 0.698. Do not state either direction as a general result.
What *is* solid: a σ sweep (0 → 0.2) shows the perturbations genuinely work (log-log slope
+0.745, σ=0.05 already optimal, σ=0 control passes at d_end = 1e-4) — so any failure is not
inertness or mis-tuning.

**(d) The signal lives in ~4 dimensions.** Top-1 PC = 40.5% of variance, top-4 = 64.6%.
Cosine restricted to the top-4 PCs: 0.673 → **0.855**. Whitening (the opposite operation)
*hurts* (0.780 → 0.698). PCA truncation + low-norm mask together reach **0.876** (partially
additive, not held-out validated).

**(e) THE HEADLINE — the readout is the bottleneck, not the model.** A ridge probe on the
world model's **predicted** latent recovers block tilt 8 steps ahead at **R² 0.836,
AUC 0.992** (held out by episode; null baselines: timestep-only 0.764, proprio-only 0.617).
The frozen encoder reaches R² 0.920 on present tilt. **`d_end` computed from that same latent
correlates with tilt at 0.038.**

> Stated precisely: *the predicted latent contains a near-perfect linear tilt signal.* The
> `d_end` 0.038 figure is against **tilt** on **all** chunks incl. post-topple (a fallen block
> is static → low d_end, high tilt), so it is NOT a fair head-to-head.

**(e2) The FAIR comparison, on the safety task, identical chunks** (799 held-out, 15 unsafe):
probe **AUC 0.941**, `d_end` 0.887, `nominal` 0.698. Probe best F1 **0.414** (vs 0.296 / 0.157)
and it reaches **100% recall at p75–p80**, which no divergence config reaches at any
threshold. Paired bootstrap: probe > `d_end` +0.054 [-0.003, +0.112]; probe > `nominal`
+0.243 [+0.106, +0.403]. **Quote these numbers, not 0.992-vs-0.038.** The real trade for
adopting a probe readout is ≈ +0.05 AUC and a much better operating curve, against losing the
label-free claim.

**(e3) The FTLE ratio cannot be rescued.** Robustification sweep (robust_ftle.py): best
variant is `ftle_pooled_den` (one median d_start per CHUNK, not per patch) at 0.782, up from
0.599 original / 0.710 masked -- but still below d_end 0.852 (CI [-0.138, +0.001]). The
least-squares slope of log d(t) over all 9 timesteps, the most principled variant, made it
WORSE (0.682): divergence is sub-exponential, so fitting an exponential rate is misspecified
-- the Lyapunov framing assumes dynamics this system lacks. The shrinkage sweep
d_end/(d_start+eps) is MONOTONIC toward d_end as eps grows (0.713/0.716/0.747), so there is no
normalisation sweet spot. Five independent tests now agree the denominator subtracts signal.

**(f) The monitor fires AT ONSET in aggregate, though often early per-episode.** Widening the
positive window to credit early firing raises F1 (0.122 → 0.192) but *lowers* AUC
(0.799 → 0.705) — the added chunks rank worse. Report W=1. NOTE: this is an AGGREGATE ranking
statement, not "never fires early". On 8 held-out failure episodes d_end fired 6-31 steps
before the topple on 5 of them, and the tilt probe on 6 of them (5-31 steps). Earlier phrasing
here was too strong.

**(g2) Exact Jacobian FTLE is CORRECT but WORSE** (AUC 0.763 exact vs 0.827 sampled; paired
CI [-0.249, -0.006]). The linearisation is verified exact at ‖δ‖=1e-4 (rel err 0.0036) but is
**96% inaccurate at the operating σ=0.05** — the linear regime ends ~50x below where the
monitor works. `corr(sigma_max, sampled divergence)` = +0.841 across patches within a chunk
but **+0.030 across chunks**: infinitesimal and finite-amplitude divergence decouple exactly
where it matters. Failure is a finite-amplitude basin phenomenon. Do not adopt; report as an
ablation that *justifies* the sampled estimator. Note the denominator hurts in BOTH
formulations (sampled 0.599→0.799, exact 0.617→0.763) — that makes "drop d_start" a property
of the formulation, not of sampling.
Also: N=50 sampling captures only 80% of the N=400 max, CV 0.167 — real costs, but the price
of measuring the right quantity.

**(g) Dual-camera is worse** (0.653 vs 0.799 front-scored), **but epoch 46 vs 88** — state it
as "the available dual checkpoint underperforms", not "two views hurt".

### Current best operating point
`nominal / p90 / k=30` (no perturbations, mask 30 lowest-||z|| patches), threshold at p95 of
safe chunks:

| | recall | precision | F1 | false halts/episode | latency |
|---|---|---|---|---|---|
| old `ftle` @p90 | .160 | .022 | .039 | 1.75 | 2021 ms |
| **nominal/k=30 @p95** | **.520** | **.129** | **.206** | 0.88 | **53 ms** |

(On `jenga_tilt_100` the best divergence config is `d_end p90/k=30`, AUC 0.887 — see (c).
Pick the metric per dataset until the discrepancy is understood.)

Precision is capped by arithmetic: at p95, 88 false positives are admitted against 25 possible
true positives, so precision cannot exceed 22%. 12.9% ≈ 58% of that ceiling.

**(h) "False alarms" are partly a labelling artifact.** Firing safe chunks contain 1.4-2.1x
more ground-truth pixel motion and 2.2-3.4x more adjacent-block tilt than silent ones
(d_end: 6.62° vs 1.92°), localised to ~7 patches clustered within 1-2 grid cells -- one
object, not the sweeping arm. NOTE 6.62° is INSIDE the normal standing mode (5.2-16.8°), so
these are NOT near-misses -- the monitor discriminates within ordinary wobble. Precision is
understated, but modestly; see (i). Also: `get_block_tilt` only measures
block_left/block_right; the red TARGET block is untracked, so instability involving it is
invisible to the label. sim.py/replay_noisy.py now record `tilt_middle` and `mid_xy` for
future datasets (deliberately NOT added to check_failure -- the target block is meant to
move).

**(i) Do NOT change the failure threshold.** Peak adjacent-block tilt is starkly bimodal:
70 episodes under 17.5 deg, 30 at 90-100, **zero in between**. So any threshold in 20-90 deg
gives identical labels — 45 deg sits mid-gap. Cutting lower (inside the 5.2-16.8 deg standing
mode) degrades every metric: probe 0.941 (45°) -> 0.899 (20°) -> 0.879 (15°) -> ~0.74-0.84
(8-12°). And redefining failure to match what the monitor fires on would be circular.
Two keepers from the sweep: the metric ordering **probe > d_end > ftle_pooled holds at all six
thresholds** (so conclusions do not ride on the cut), and the probe still reaches 0.844 at
8 deg, so sub-topple disturbance carries real signal. Also: at 20 deg the SAME 15 positives
give AUC 0.899 vs 0.941 at 45 deg, because the crossing happens a few steps earlier — asking
the monitor to fire sooner costs accuracy, confirming (f).

**(j) ftle_variance is competitive** (AUC 0.858) -- spread of the 49 perturbed latents
around their own centroid, never touching the original trajectory. Beats every FTLE-ratio
variant tried. Never benchmarked before this; now the second-best divergence metric behind
d_end.

**(k) PCA truncation does NOT survive cross-validation -- drop it.** 20 held-out episode
splits picked full-space (m=384) 20/20 times; section 7.15's m=4 winner was overfit to a
225-chunk sample. Found via a REDUCTION BUG in pca_mask_combo.py (flattened perturbation x
patch before p90 instead of averaging perturbations first) that also means section 7.15's
0.876 in-sample PCA+mask number should not be trusted. The signed PC1 mask, properly
validated, held up (0.941 held-out, 75% keep chosen 20/20 times) -- but this and the PCA
result both used a small 225-chunk subsample with noisier test halves than the full corpus;
treat the ORDERING as real (PC1 >= mask+PCA >> unmasked) more than the absolute numbers.

**(l) The probe ranks well but is poorly calibrated and hard to threshold.** Global
regression actual = 0.48*predicted + 0.83 (well-calibrated = slope 1, intercept 0); in the
15-45 deg decision band it overpredicts by 10-28 deg. Its threshold is also the least
reproducible of any metric under resampling (p95 CV 0.27, 95% CI [14.4, 33.8] deg) vs
ftle_pooled's CV 0.01. AUC 0.941 is real and the ranking is trustworthy; the raw score is
NOT an interpretable degree estimate, and a single calibration run could hand you a
materially wrong threshold.

## 5. Gotchas that have burned this project four times

**Silent failures where the artifact looks complete.** In order of discovery:
1. `show_preview` — AttributeError swallowed by a bare `except` → 0 frames/episode.
2. Missing `proc_pos` — KeyError swallowed → all 50 episodes dropped, 16 KB empty LMDB.
3. `generate_labels.py` re-simulated from a fresh reset → labels described a *different*
   rollout than the frames. (`sim.py` randomises block pose with `np.random.uniform` and saves
   no seed, so replays are **not** reproducible. Ground truth must be recorded *during* the
   rollout — `replay_noisy.py` now does this.)
4. `replay_noisy.py` built the frame path by string concatenation, silently requiring a
   trailing slash → 98,716 frames written to a sibling directory while the log printed
   `[SAVE] Saved new episode N` 100 times. Now uses `os.path.join`.

> **Always verify generated data before consuming it.** Check every episode has a non-zero,
> matched count of `cam1_*.png` and `cam2_*.png`. All four bugs would have been caught at the
> point of failure instead of hours downstream.

**Other traps.**
- `server_dual_front.py` must use `conf/serve_dual.yaml`; `train_dual.yaml` carries
  `override hydra/launcher: submitit_slurm` (cluster-only, fatal locally).
- `create_lmdb_full.py` takes `--data_path` (underscore); `create_lmdb_single30.py` takes
  `--data-path` (hyphen).
- Non-finite scores break `np.linspace` threshold sweeps → silently all-NaN metrics. Sanitize.
- `pkill -f <pattern>` can kill its own shell; use the bracket trick: `pgrep -f "patt[e]rn"`.
- Launch servers with `python -u` or the "listening on" line stays block-buffered forever.

## 6. Open items

1. **Zero-shot tension.** A tilt-probe readout would be far stronger than divergence, but it
   uses tilt labels. Defensible as "a physical quantity measured in simulation, never a failure
   label" — but it weakens the zero-shot claim. **Unresolved design decision.**
2. **Fix 3 (contrastive stability loss)** — already implemented in
   `models/dual_visual_world_model.py`, gated behind `contrastive_weight=0.0`. Requires a
   retrain (cluster). Now has a specific target: make latent distance align with the tilt
   direction that provably already exists in the latent.
3. **Dual-camera confound** — needs an epoch-88 dual checkpoint.
4. **Two datasets** — safety numbers from `jenga_noise_50`, probe from `jenga_tilt_100`.
   Same noise settings (`pos_std=0.002, rot_std=0.05`), 25% vs 30% failure rate.
5. **PCA×mask 0.876** not held-out validated (winning cell picked from a 16-cell grid).

## 7. Related work worth knowing

**LeWorldModel** (LeCun/Mila/AMI, arXiv 2603.19312, Mar 2026) — JEPA trained end-to-end from
pixels with two losses: next-embedding prediction + **SIGReg**, forcing isotropic Gaussian
latents. 15M params, single GPU, **48× faster planning than DINO-WM**; beats DINO-WM on Push-T,
loses on visually complex OGBench-Cube. Probes physical quantities incl. **block angle**.
Directly relevant: an isotropic latent would fix finding (b)/(d) *at the source* — distances
comparable across directions by construction, and a natural scale for δ. Their "surprise"
metric compares predicted vs **actually observed** future, so it is a different problem from
this monitor, which must flag *before* executing.

## 8. Scripts (all in `dino_wm/` unless noted)

| script | purpose |
|---|---|
| `server_single_max.py` | production monitor server (ZMQ 5556) |
| `server_dual_front.py` | dual-view, front-only scoring (ZMQ 5557), uses `serve_dual.yaml` |
| `nominal_baseline.py` | nominal vs N=50 + latency |
| `masked_nominal_vs_pert.py` | low-norm mask × method, bootstrap, held-out k |
| `distance_ablation.py` | cosine/angular/chord/L2/whitened + norm diagnostic |
| `sigma_sweep_dend.py` | σ sweep with σ=0 harness control |
| `feature_geometry.py` | PCA truncation + PC1 mask |
| `pca_mask_combo.py` | PCA × low-norm mask, signed PC1 |
| `tilt_probe.py` | the linear tilt probe (requires `_tilt` in LMDB) |
| `tilt_probe_video.py` | renders probe vs ground truth vs d_end as mp4 |
| `probe_vs_divergence_safety.py` | probe vs divergence on the **safety** task, same chunks |
| `panda_express/test_monitor.py` | eval driver |
| `panda_express/compute_metrics.py` | precision/recall/F1 |

Results: `dino_wm/outputs/*.json`, `panda_express/results/`.

## 9. Working notes

- Report **AUC and the operating curve**, never accuracy.
- AUC can mislead: `d_end` has higher AUC than `nominal` (0.854 vs 0.838) yet loses at every
  deployable threshold. **Choose on the low-FPR tail.**
- With 25 positives, AUC differences below ~0.05 are not resolvable. Use paired bootstrap over
  **episodes** (chunks within an episode are correlated); the paired design detects real effects
  the unpaired one cannot.
- Anything tuned on the data it is scored on (mask size k, PCA m) needs held-out validation
  before it is claimed.
