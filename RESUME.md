# RESUME — Picking the Latent Safety Filter Back Up

**Last real work:** 2026-07-28 / 07-29 · **Previous gap:** ~5 weeks (June → 2026-07-27)

`CLAUDE.md` describes the *research* (theory, architecture, intended commands). This file
describes the **actual state of the machine**. As of 2026-07-29 the runtime is fully rebuilt
and the monitor has been measured end to end — **§7 supersedes CLAUDE.md's reported results
and its recommended metric.**

---

## Headline: the metric was the bottleneck, not the model

A linear probe on the world model's **predicted** latent recovers future block tilt at
**R² 0.836 / AUC 0.992**. The divergence metric computed from that same latent correlates
with tilt at **0.038**. Perception and dynamics are fine; the readout was discarding the
signal. Full numbers in §7.

Production metric changed: **`nominal / p90 / k=30`** (no perturbations, mask the 30
lowest-‖z‖ patches) — 5× the F1 of the old `ftle`, and 38× faster. See §7.6.

---

## Session log — 2026-07-27

What changed, in order. Details in the linked sections.

1. **Committed** five weeks of untracked June work to a new branch `auto_failure_check` in
   both `dino_wm` and `panda_express` (§2). Not pushed.
2. **Rebuilt the `dino_wm` conda env** — torch 2.11+cu128 for the Blackwell GPU; the pinned
   `environment.yaml` was unusable (§3 Step 1).
3. **Validated the automatic failure check** and found the 15° topple threshold encodes the
   wrong event. Raised it to 45° and made the tilt metric yaw-immune (§5b).
4. **Corrected two metric semantics bugs** so precision/recall measure prediction rather
   than observation (§5c).
5. Added five scripts that run without a checkpoint: `visualize_failure_check.py`,
   `failure_frames.py`, `survey_tilts.py`, `eyetest_sheet.py`, `demo_metric_truncation.py`.

**Nothing about the monitor's real accuracy was measured** — that still requires the world
model checkpoint (§3 Step 2), which remains the single blocker. All of the above is upstream
of it: environment, label quality, and metric correctness — three things that would have
quietly corrupted the numbers once the checkpoint came back.

## TL;DR

You left the project in a **good code state but a stripped runtime state**. In June you built a
complete evaluation harness (labels → scores → metrics → ablation sweep) and then apparently
cleared out the heavy artifacts. What's gone:

- the `dino_wm` conda environment
- the trained world-model checkpoint
- the LMDB eval dataset
- `labels.json` and all `results/`

The code that consumes them is intact and uncommitted. **The one true blocker is the world model
checkpoint** — it lived on the Alliance Canada cluster (`$SCRATCH/outputs/model_latest.pth`) and
was never copied back here. Everything else is rebuildable locally in an afternoon.

---

## 1. Inventory: what exists vs. what's missing

### Present ✅

| Thing | Where | Notes |
|---|---|---|
| All three repos, full source | `wksp/{dino_wm,panda_express,diffusion_policy}` | |
| Raw Jenga episodes | `panda_express/tasks/jenga_mujoco/episodes/` | 102 episodes, 2.4 GB, dual-cam PNGs + `trajectory_*.json` |
| DINOv2 pretrained weights | `~/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth` | offline load works, no download needed |
| `robodiff` conda env | torch 2.7.0+cu128, CUDA available | for `diffusion_policy` only |
| MASc defense slides | `wksp/defense_slides.html` | best single-file refresher on the whole method |
| GPU | RTX 5060 Ti, 8 GB, driver 580, CUDA 13 | Blackwell — needs cu128 wheels |

### Missing ❌

| Thing | Expected path | How to recover |
|---|---|---|
| **World model checkpoint** | `dino_wm/outputs/model_latest_single.pth` | **Only on the cluster.** `scp` from `$SCRATCH/outputs/` — see §3. |
| ~~`dino_wm` conda env~~ | — | ✅ **rebuilt 2026-07-27**, see §3 Step 1 |
| Eval LMDB | `panda_express/tasks/jenga_mujoco_noise/jenga_single.lmdb` | rebuild — §4 |
| Noisy replay dataset (1600 traj) | `panda_express/tasks/jenga_mujoco_noise/episodes/` | regenerate with `replay_noisy.py` |
| `labels.json` | `panda_express/labels.json` | regenerate with `generate_labels.py` (needs mujoco) |
| `results/`, `patch_stats.npz` | — | outputs of the eval runs |
| Jenga Zarr (diffusion policy data) | `diffusion_policy/data/jenga/task_replay.zarr` | rebuild with `create_dual_zarr.py` if you need the policy |

> Note: only `mujoco`, `lmdb`, and `timm` are missing from any usable env — `robodiff` has torch,
> zmq, cv2, hydra, zarr, diffusers. But `robodiff` is python 3.9 and is *not* the right env for
> `dino_wm`; don't try to reuse it for the world model.

---

## 2. Uncommitted work ✅ COMMITTED 2026-07-27

Five weeks of the newest and most valuable work was untracked in git — the June evaluation
harness, the thing that turned the project from "a monitor" into "a paper with baselines".

It is now committed to a **new branch `auto_failure_check`** in both repos (branched off `main`
in dino_wm and off `mujoco_sim` in panda_express). Neither has been pushed — push when ready:

```bash
cd /home/sanger/wksp/dino_wm && git push -u origin auto_failure_check
```
```bash
cd /home/sanger/wksp/panda_express && git push -u origin auto_failure_check
```

What went in:

### `dino_wm` → `auto_failure_check` (`775c25a`)
```
?? calibrate_patches.py          NEW — per-patch shadow normalization (Fix 1 calibration)
?? server_ablation.py            NEW — all 10 metric modes behind mode= Hydra override
 M conf/train_dual.yaml
 M models/dual_visual_world_model.py
 D server.py server2.py server3.py server_cluster.py server_dist.py
 D server_pca.py server_pca_single.py server_single.py
```
The 8 deleted `server*.py` files were consolidated into `server_ablation.py`. Deliberate cleanup.

Four further commits landed on `panda_express/auto_failure_check` the same day, all from the
failure-check and metric work in §5b/§5c:

| commit | what |
|---|---|
| `83a3467` | `visualize_failure_check.py`, `failure_frames.py` — replay + detection visuals, `--repeat` label-stability sweep |
| `c521b7f` | `survey_tilts.py` — peak-tilt distribution; documents that 15° is too low |
| `4ac3e6c` | `TOPPLE_THRESHOLD_DEG = 45.0`; `get_block_tilt` now measures tilt from vertical; `eyetest_sheet.py` |
| `51e4849` | `chunk_bounds()` — score only chunks that could have predicted the failure |

Plus `7604ecb`, which dropped post-failure chunks from the metrics.

### `panda_express` → `auto_failure_check` (`69eb845`)
```
?? test_monitor.py               NEW — argparse eval driver, replaces test_traj_noise_*
?? compute_metrics.py            NEW — precision/recall/F1, episode + step level, threshold sweep
?? generate_labels.py            NEW — auto-labels topples by replaying LMDB through MuJoCo
?? run_ablations.sh              NEW — sweeps 9 modes, emits metrics_summary.json
 M sim.py
 D 8 legacy test_traj_noise_*.py / client_dev*.py / eval_distances*.py
```

### `diffusion_policy` — you are on branch `avsr`, not `main`
This is a **different project track**. On 2026-06-16 you trained an `avsr_cup` task (12 runs,
36 GB of outputs, resnet18 @ 240×320, real-robot flavored) — unrelated to Jenga. The Jenga policy
work is on `main`. Don't confuse the two:

```bash
cd /home/sanger/wksp/diffusion_policy && git checkout main   # for Jenga work
```
Two `avsr_cup` runs have usable checkpoints (`epoch=0100-train_loss=0.006.ckpt`). 36 GB — consider
pruning intermediate epochs if you need disk.

---

## 3. Rebuild path (in order)

### Step 1 — recreate the `dino_wm` environment ✅ DONE 2026-07-27

**Do not use `environment.yaml`.** It pins python 3.9 + torch 2.3.0 + cu121, and
`req_env_general.txt` pins torch 2.5.1 — both predate Blackwell and cannot drive an sm_120 GPU.

The env was rebuilt from scratch and verified working:
```bash
conda create -n dino_wm python=3.11 -y
conda run -n dino_wm pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
conda run -n dino_wm pip install "numpy<2" opencv-python einops omegaconf "hydra-core==1.3.2" \
    tqdm lmdb mujoco pyzmq imageio imageio-ffmpeg scipy wandb accelerate timm psutil \
    pyyaml websockets submitit decord gym
```
Resulting stack: **torch 2.11.0+cu128, mujoco 3.10.0, numpy 1.26.4** (pinned `<2` deliberately —
this codebase predates the numpy 2 API changes), CUDA available on the RTX 5060 Ti (sm_120).

Verified: all imports clean, GPU matmul works, and DINOv2-S loads **offline** from the torch hub
cache and produces correct patch tokens. `polymetis` and `pyrealsense2` were intentionally
skipped — they're real-robot only and not needed for sim or the monitor.

### Step 2 — retrieve the world model checkpoint 🔑
This is the blocker. Nothing on this machine has it; `find / -name "*.pth" -size +10M` returns
only the DINOv2 cache. It should still be on Rorqual/Narval:
```bash
scp <user>@rorqual.alliancecan.ca:~/links/scratch/outputs/model_latest.pth /home/sanger/wksp/dino_wm/outputs/model_latest_single.pth
```
`server_single_max.py:13` hardcodes that exact path, and falls back to
`{cfg.ckpt_base_path}/outputs/model_latest.pth`. Also grab `dataset_stats.pt` from the same
directory if it's there — the server loads it for normalization stats when present.

**If the checkpoint is gone from the cluster too**, you're looking at a retrain (Experiment 4 in
CLAUDE.md, ~7 h on 4×H100). Check this *before* investing in steps 3–4.

### Step 3 — regenerate the noisy dataset
`replay_noisy.py` reads `tasks/jenga_mujoco/` (present) and writes `tasks/jenga_mujoco_noise/`
(missing), generating 1000 episodes with ±2 mm positional / ~1° rotational noise:
```bash
cd /home/sanger/wksp/panda_express && python replay_noisy.py
```
Config is at the top of the file (`N_EPISODES_TO_GENERATE = 1000`) — drop it to ~50 for a fast
smoke test.

### Step 4 — build the LMDB
```bash
cd /home/sanger/wksp/dino_wm && python create_lmdb_single30.py
```
⚠️ `create_lmdb_single30.py:12` has `DATA_PATH = tasks/jenga_mujoco_noise` hardcoded. If you skip
step 3, point it at `tasks/jenga_mujoco` instead — the 102 clean episodes are a perfectly good
source for a monitor smoke test, since the LMDB is only a trajectory replay source at eval time.
It uses `cam2` only (single-view, matching `server_single_max.py`).

### Step 5 — regenerate labels, then rerun the pipeline
Follow CLAUDE.md Experiments 0 → 1 → 2 as written; they're accurate once the artifacts above
exist.

---

## 4. Discrepancies to resolve (CLAUDE.md vs. the code)

Worth settling before you quote numbers in the thesis:

1. **Threshold δ**: CLAUDE.md says `δ = 0.8` throughout. `test_monitor.py:50` defaults to `0.87`,
   and so does `run_ablations.sh`. The 96.9%/62.4% headline presumably came from one of them.
   `compute_metrics.py` does a 100-point threshold sweep — use it to re-derive the operating
   point rather than trusting either constant.
2. **Perturbation noise ε**: CLAUDE.md specifies `ε = 0.005` ("~positional error of EE"), but
   `test_monitor.py:47` defaults to `--noise-std 0.05` — **10× larger** — and Experiment 1 in
   CLAUDE.md passes `0.05` explicitly. One of the two is wrong; this materially changes the FTLE
   scale and therefore δ. Resolve this first, since it likely explains why δ drifted to 0.87.
3. **`ftle_calibrated` is excluded from `run_ablations.sh`** (9 modes, not 10) because it needs
   `patch_stats.npz` generated first. Correct behavior, but the sweep won't cover it — run it
   manually per CLAUDE.md Experiment 3.
4. ~~**8 GB VRAM may OOM at N=50.**~~ **Resolved 2026-07-27 — not a problem.** Measured on the
   rebuilt env: a full N=50 batch through DINOv2-S peaks at **0.35 GB of 8.07 GB**. The predictor
   rollout over T=8 adds to that, but the headroom is large. No need to reduce N.

5. **Patch grid = 14×14 (196), not 16×16 — this is correct, don't "fix" it.** Confusing at first
   glance: DINOv2 `vits14` on a 224² image would give 16×16 = 256 patches. But
   `dual_visual_world_model.py:104-110` resizes to `encoder_image_size = (224//16) * 14 = 196`
   before the encoder, so patch-14 on 196² yields exactly **14×14 = 196**. Verified 2026-07-27.
   Consequently `MASK_TOP_ROWS = 28` (2 rows × 14) is right, and `test_monitor.py:230`'s
   `(idx // 14) * 16` is right too (14×14 grid displayed on a 224 px crop → 16 px per patch).
   The `decoder_scale = 16` constant is inherited from the VQVAE, not the DINO patch size.

---

## 5. What to run *right now* to refresh your memory

Ranked by time-to-signal. The first two need zero setup.

### A. Reread your own defense slides — 10 min, no setup
```bash
xdg-open /home/sanger/wksp/defense_slides.html
```
Single self-contained file, "Stability-Centric Safety Monitor — MASc Defense". The fastest route
back into the *why*: FTLE formulation, the δ≠0 argument, results. Start here.

### B. Read the June harness you don't remember writing — 20 min, no setup
In this order, they tell a clean story:
1. `panda_express/test_monitor.py` — the eval loop. Note the `TemporalAggregator` (max/mean/ema)
   at line 85, which is Fix 2 for the recall bottleneck.
2. `dino_wm/server_ablation.py` — all 10 metric modes side by side. Reading the mode dispatch is
   the single best refresher on what each baseline/ablation actually computes.
3. `panda_express/compute_metrics.py` — episode-level vs step-level scoring, threshold sweep.

### C. Run the simulator — ready now, env already built
The only *executable* thing that doesn't need the checkpoint:
```bash
conda activate dino_wm && cd /home/sanger/wksp/panda_express && python sim.py
```
Franka + 3 Jenga blocks, dual camera, 30 FPS, `check_failure()` topple detection. Watching a few
picks is the fastest way to re-anchor what "unsafe" physically means here.

### C2. Exercise the failure check and the metrics — ready now, no checkpoint needed
```bash
python visualize_failure_check.py --episodes 1 2 3 --speed 3   # replay + tilt plots
python failure_frames.py --episodes 2 3                        # close-ups at detection
python survey_tilts.py --n 14                                  # peak-tilt distribution
python eyetest_sheet.py --n 12 --compare 15                    # verdicts vs visual review
python demo_metric_truncation.py                               # metric semantics (synthetic)
```
These are the only parts of the pipeline that run today. See §5b and §5c for what they found.

### D. Regenerate data end-to-end (steps 3–4) — ~1 h
Gets you a working LMDB and proves the data path. Still can't score anything without the
checkpoint, but it de-risks everything downstream.

### E. Full pipeline — blocked on §3 Step 2
Once the checkpoint is back: CLAUDE.md Experiments 0 → 1, then `run_ablations.sh` for the table.

**Suggested first session:** §2 (commit) and §3 Step 1 (env) are done. So: A → B to rebuild
context, then C to see the task run. In parallel, `ssh` the cluster and confirm the checkpoint
still exists — that single fact determines whether the next month is "finish the paper" or
"retrain the world model".

---

## 5b. Automatic failure check — validated 2026-07-27

Run it yourself (needs no LMDB, no checkpoint):
```bash
conda activate dino_wm && cd /home/sanger/wksp/panda_express && python visualize_failure_check.py --episodes 1 2 3 --speed 3
```
```bash
python failure_frames.py --episodes 2 3        # close-up frames at the detection instant
```
```bash
python visualize_failure_check.py --episodes 1 2 3 --repeat 5 --no-video   # label stability
```

**How it works.** `SIM.check_failure(threshold_deg=15.0)` watches **two** bodies —
`block_left` and `block_right`. The middle red block is deliberately excluded: it's the pick
target, so it's *supposed* to move. The verdict is a **max over those two**: if either exceeds
the threshold, the episode is a failure.

Tilt per block ([sim.py:85](panda_express/sim.py:85)) is the geodesic angle between the block's
current quaternion and a reference recorded at the last reset:
`tilt = degrees(2 · arccos(|q_curr · q_ref|))`. The `abs()` handles quaternion double-cover.
The reference is captured **post-reset** ([sim.py:72](panda_express/sim.py:72)), so the ±3°
random yaw applied at reset is correctly cancelled out.

**Verified working.** Tilt reads exactly 0.000 until contact, then rises smoothly — the metric
and reference quaternions are sound. On a clear topple it fires at 15° roughly 25 steps before
the blocks reach 90° flat, so it's a genuine early warning, not a post-hoc observation.

### ⚠️ The 15° threshold is wrong — raise it to ~45°

**"Toppled" means lying flat, not leaning.** A block at 15–25° has been *perturbed*; it is still
standing. The default threshold encodes the wrong event.

`survey_tilts.py` measured peak tilt-from-vertical over 14 episodes. The distribution is
**strongly bimodal, with a completely empty gap between them**:

| band | episodes | peak tilt | physical state |
|---|---|---|---|
| standing | **8** | 6.4 – 15.5° | untouched or nudged, still upright |
| *(gap)* | **0** | 15.5 – 94.6° | — nothing lands here — |
| toppled | **6** | 94.6 – 96.4° | lying flat on the table |

Verdict vs. threshold — note how flat it is once you clear the standing cluster:

| threshold | 10° | **15° (current)** | 20° | 30° | **45°** | 60° | 75° |
|---|---|---|---|---|---|---|---|
| failures | 12 (86%) | **7 (50%)** | 6 | 6 | **6 (43%)** | 6 | 6 |

**Any threshold from 20° to 75° gives the identical, correct answer of 6.** The current 15° cuts
straight through the middle of the standing cluster, producing a false positive on an episode
that merely leans. Setting `--topple-threshold 45` is both correct *and* maximally robust — it
sits in a ~55°-wide dead zone where the answer cannot change.

This single change fixes three things at once:

1. **Definition** — labels now mean "lying flat", which is the actual failure mode.
2. **Label instability** — replaying ep1 5× at 15° gave **3 failure / 2 success** (peaks
   12.9–18.9°, straddling the threshold), a coin flip that `--n-replays 3` majority voting
   cannot rescue. At 45° that episode is unambiguously a success every time. The instability was
   never inherent to the sim; it was an artifact of the threshold sitting inside the standing
   cluster.
3. **Yaw contamination** — `get_block_tilt` measures angular deviation about *any* axis, so a
   block that slides and spins without tipping still registers (ep6: quat 14.0° vs. true tilt
   7.4°; ep10: 16.1° vs. 6.4°). That contamination is ~10° in magnitude, so it only matters when
   the threshold is ~15°. At 45° it is irrelevant.

Optionally also make the metric yaw-immune (one line in [sim.py:85](panda_express/sim.py:85)),
which is the physically correct measure of "tilt from vertical":
```python
R = self.data.xmat[body_id].reshape(3, 3)
return float(np.degrees(np.arccos(np.clip(R[2, 2], -1.0, 1.0))))
```
With a 45° threshold this is a refinement rather than a fix — but it makes `failure_step` timing
honest, which the step-level metrics in `compute_metrics.py` depend on.

> Caveat on scope: the above replays *raw teleop* episodes, which are open-loop-replayed from a
> different random reset than they were recorded under, so they topple far more often than the
> real pipeline would. `generate_labels.py` replays sim-generated LMDB episodes recorded under
> the same reset distribution, which should be more faithful. **Re-run this validation on the
> LMDB once it's rebuilt** before trusting `labels.json`.

## 5c. Metric semantics — two corrections applied 2026-07-27

Both fixes are in `compute_metrics.py` and both aim at the same thing: make the metrics
measure **prediction**, not **observation**. Neither has been validated against real scores —
they cannot be until the checkpoint returns.

**The root cause.** `generate_labels.py` stops replaying at the first topple
([generate_labels.py:84](panda_express/generate_labels.py:84)) but `test_monitor.py` keeps
scoring to the end of the episode ([test_monitor.py:153](panda_express/test_monitor.py:153)).
So `scores.json` contains chunks that observe a scene where the blocks are already down. The
world model correctly reports huge divergence there — and the old metrics mishandled it in
two opposite directions.

**Fix 1 — drop post-failure chunks.** Previously they were labeled *safe* and kept, so the
monitor was charged a false positive for correctly flagging a collapsed tower (depressing
step precision); and the per-episode `max` ran over all chunks, so a monitor that never
predicted the fall but spiked afterwards was booked a true positive (inflating recall).
`--no-truncate` restores the old behaviour.

**Fix 2 — score only chunks that could have predicted.** `scores.json` is keyed by **chunk
start index, not timestep** — `test_monitor.py` strides by `num_pred`, so there is one score
per 8 timesteps. What the code calls "step-level" is really chunk-level. A chunk at `start`:

```
observes:  start ... start+num_hist-1        (3 frames)
predicts:  start+num_hist ... start+num_hist+num_pred-1   (8 steps)
```

The old unsafe window `[fs-num_pred, fs]` included the chunk starting at `fs`, whose
observation frames already contain the topple. `chunk_bounds()` now requires both that
observations end before `fs` and that the horizon still reaches it:

```
[fs - num_hist - num_pred + 1,  fs - num_hist]
```

For `fs=40, num_hist=3, num_pred=8` that is `[30, 37]` → at stride 8, exactly chunk 32.
The old window `[32, 40]` also credited chunk 40, which is *watching* the topple.
`--label-window legacy` restores it.

**See it without a checkpoint:**
```bash
conda activate dino_wm && cd /home/sanger/wksp/panda_express && python demo_metric_truncation.py
```
Runs all three configurations on synthetic input. ⚠️ **Its numbers are invented, not
measurements** — the script prints a banner saying so. It is a regression check on the
arithmetic, nothing more.

> These fixes depend on `failure_step` being accurate, which is the other reason the 45°
> threshold in §5b mattered: at 15°, `failure_step` fired early on blocks that were only
> leaning, which would shift every window above.

## 5d. Checkpoint recovered — pipeline runs 2026-07-27

**The blocker is gone.** All four artifacts were in Google Drive under `linux_transfer/`:
`model_latest_single.pth` (378 MB), `model_latest.pth` (569 MB), `latest.ckpt` (5.7 GB — very
likely the Jenga diffusion policy), and `jenga_mujoco.tar.gz`. The single-view checkpoint is
now at `dino_wm/outputs/model_latest_single.pth`, exactly where
[server_single_max.py:13](dino_wm/server_single_max.py:13) expects it.

**Verified from the weights themselves:** the predictor's `pos_embedding` is `(1, 588, 404)`.
588 = 3 frames × **196** tokens = 14×14, and 404 = 384 + 10 + 10. This settles §4 item 5 —
`MASK_TOP_ROWS = 28` is correct, confirmed against ground truth rather than inferred.

**First real measurement.** 139 chunks over 6 clean expert episodes, δ=0.87:
```
range 0.461 .. 0.855   mean 0.639   median 0.631
p90 0.734   p95 0.760   p99 0.826
> 0.87 :  0 / 139  (0.0%)
```
Zero false alarms on expert demonstrations. Note this refines the δ story: safe trajectories
do **not** sit at ~0.8 — their mean is **0.64**. δ was calibrated just above the safe
distribution's *maximum* (0.855), a margin of only ~0.015. That is thin and worth stating
explicitly rather than describing the baseline as "~0.8".

Throughput: **4.3 s/chunk** on the 5060 Ti → ~2.6 h for 100 episodes, ~24 h for the 9-mode
ablation sweep. Budget accordingly.

### ⚠️ `replay_noisy.py` was thoroughly broken — three separate faults

It had not produced usable output since the sim.py refactor. All three are fixed, but the
pattern is worth noting: **two of the three were masked by a bare `except Exception`**, so the
script reported success while doing nothing.

**1. Zero frames written.** It called `recorder.show_preview()`, a method of `cam.py`'s
real-camera Recorder that `SimRecorder` does not have. The `AttributeError` was swallowed, so
every episode "completed" having written no images.

**1b. Episodes silently dropped at LMDB build.** `create_lmdb_*.py` reads action from
`position`/`gripper` and proprio from `proc_pos`/`proc_gripper`. `replay_noisy.py` wrote only
the achieved pose, as `position`, with no `proc_*` — so `process_episode` raised `KeyError`
into its own bare `except`, returned `None` for all 50 episodes, and produced a 16 KB empty
LMDB. Waypoints now carry both: `position`/`orientation` = commanded (noisy) target,
`proc_pos`/`proc_quat`/`proc_gripper` = what the robot reached. This also explains why
CLAUDE.md's `ACTION_MEAN` and `PROPRIO_MEAN` are close but not identical.

**1c. Not resumable.** `get_next_episode_id()` indexed off the *source* directory, so a second
run restarted at the same id and overwrote. Now indexes off the target.

**2. The noise constant is 3.5× its documented value.** `NOISE_POS_STD = 0.007` is 7 mm,
beside a comment reading "2mm position noise". Measured effect, using the ground truth now
recorded during each rollout:

| `--noise-pos` | failures | median peak tilt |
|---|---|---|
| `0.007` (existing default) | **92%** (12/13) | 93.6° |
| `0.002` (what the comment says) | **33%** (4/12) | 12.2° |

92% failures cannot be the distribution behind the reported results — CLAUDE.md describes
safe timesteps far outnumbering unsafe, and 33% episode-level failures puts step-level
positives near 2–3%, which fits. **Regenerate with `--noise-pos 0.002`.**

At 2 mm the peak tilts are again cleanly bimodal (8.9–14.2° standing, 90.5–94.8° flat,
nothing between) — a third independent confirmation of the 45° threshold.

### Labels are now recorded at generation time

`generate_labels.py` reconstructs labels by re-simulating from a **fresh random reset**, which
produces a *different rollout* than the frames the monitor is scored on. Given the measured
replay variance (§5b), the label often describes a topple that never occurred in the recorded
frames. `replay_noisy.py` now calls `SIM.check_failure()` as the rollout executes and stores
`outcome`, `failure_step`, `failure_timestamp`, `failure_block`, `peak_tilt_deg` in the episode
metadata — exact, free, and describing the right trajectory. `generate_labels.py` is no longer
needed for the noisy set.

> **Still to write:** a small script to extract `labels.json` from the new episode metadata
> (mapping `failure_timestamp` → LMDB chunk index, since alignment drops ~0.3% of waypoints).

### Remaining path to real numbers
```bash
python replay_noisy.py --n-episodes 1600 --noise-pos 0.002     # ~8 h, resumable
python ../dino_wm/create_lmdb_single30.py --data-path tasks/jenga_mujoco_noise
# extract labels.json from episode metadata   <- script not yet written
python test_monitor.py --lmdb .../jenga_single.lmdb            # ~2.6 h
python compute_metrics.py --scores ... --labels labels.json
```

## 6. Open threads from June

Inferred from the code you left; confirm against your own notes.

- **Fix 1 (shadow artifacts)** — `calibrate_patches.py` + `ftle_calibrated` written but never run;
  no `patch_stats.npz` exists. Targets the 65.9% precision bottleneck.
- **Fix 2 (temporal accumulation)** — implemented in `test_monitor.py` as `--temporal-window` /
  `--temporal-agg`, no results saved. Targets the 59.2% recall bottleneck.
- **Fix 3 (contrastive stability loss)** — design sketched in CLAUDE.md "Planned Extensions",
  not implemented in `train_dual.py`. Highest novelty, requires a retrain. Since a retrain may be
  forced anyway (§3 Step 2), consider folding Fix 3 in rather than reproducing the old checkpoint.
- **`ftle_variance`** — implemented, framed as the principled-threshold alternative (natural
  threshold ≈ 0.02 vs. 0.8). Never benchmarked against the main method.

---

## 7. Measured results — 2026-07-28 / 07-29

First session with the checkpoint restored, so these are the **first real numbers** the
project has produced. They replace the figures in CLAUDE.md's "Results" section, which came
from a 62-trajectory run with the old 15° topple threshold and the pre-correction metric
semantics.

**Evaluation setup (all of §7 unless stated).** 100 episodes / 1772 chunks / 25 unsafe
(1.4% base rate), `jenga_noise_50/jenga_single_100.lmdb`, labels from
`labels_noise100.json` (recorded during replay, 45° threshold). Chunk is positive when
`failure_step` falls in its 8-step prediction horizon; chunks strictly after the failure are
dropped so nothing is rewarded for detecting the aftermath. Thresholds are percentiles of
the **safe** score distribution only, so the zero-shot claim survives. CIs are paired
cluster bootstrap resampling **episodes** (chunks within an episode are correlated).

### 7.1 The estimator was the problem — drop the FTLE denominator

`d_start` is measured after a single prediction step: tiny, noisy, unrelated to stability.
Dividing by it reorders patches by how quiet they happened to start.

| metric | AUC |
|---|---|
| `ftle` (production, double max) | 0.599 |
| `ftle_maxpatch_meanpert` | 0.685 |
| `dend_mean` | 0.780 |
| `dend_std` | 0.787 |
| `dend_p90` | **0.799** |

### 7.2 Cosine's denominator is a noise amplifier — mask low-‖z‖ patches

Hypothesis going in was that register-free DINOv2's **high**-norm artifact tokens caused the
spurious background divergence visible in the videos. **Falsified:** on ground-truth-static
patches, `corr(‖z‖, d_end) = −0.641`. High-norm patches are the *stable* ones; divergence
concentrates on **low**-norm patches, because cosine divides by ‖z‖ and a blank table patch
has a short, poorly-determined feature vector.

Masking the 30 lowest-‖z‖ patches (on top of the existing row mask):

| method | k=0 | k=30 | held-out gain | bootstrap CI |
|---|---|---|---|---|
| nominal | 0.791 | 0.838 | +0.044 | [+0.020, +0.050] |
| `d_end` mean | 0.759 | **0.854** | +0.096 | [+0.038, +0.147] |
| `d_end` std | 0.768 | 0.810 | +0.029 | — |

*k* chosen on one half of the episodes and scored on the other (20 splits); all three methods
independently select *k*≈30 and keep their gain out of sample. **This is the one fully
validated fix of the session.**

### 7.3 The Deviator Agent does not earn its cost

`nominal[p] = 1 − cos(z_orig[NH,p], z_orig[T,p])` — a single unperturbed rollout, measuring
only how much the predicted scene changes — matches the full 50-rollout apparatus.

| | AUC | latency |
|---|---|---|
| nominal (N=1) | 0.826 | **53 ms** |
| `d_end` mean (N=50) | 0.851 | 2021 ms |

Difference +0.025, CI **[−0.075, +0.134]**. On *this dataset*:

1. Statistically indistinguishable from nominal (bootstrap).
2. Masking does not rescue it — both improve together.
3. Worse than nominal at every deployable threshold (§7.6).
4. Worse than nominal at every perturbation magnitude (§7.4).

> ⚠️ **Points 1 and 3 do NOT replicate on `jenga_tilt_100` (§7.16).** There `d_end` reaches
> AUC 0.887 against nominal's 0.698 and beats it at the loose thresholds. The ordering of
> nominal vs `d_end` is **dataset-dependent**; do not state it as a general result. The σ
> sweep (§7.4) and the mask findings (§7.2) are unaffected.

Caveat: 25 positives limits power. But the *same* test returned P(A>B)=1.000 twice for the
mask, so "underpowered" is not a complete explanation.

### 7.4 σ sweep — the perturbations work, they just do not help

| σ | d_end safe | d_end unsafe | ratio | AUC d_end | AUC nominal |
|---|---|---|---|---|---|
| 0.000 | 0.0001 | 0.0001 | 1.00 | 0.518 | 0.799 |
| 0.010 | 0.0257 | 0.0307 | 1.20 | 0.597 | 0.799 |
| 0.020 | 0.0427 | 0.0531 | 1.24 | 0.636 | 0.799 |
| **0.050** | 0.0666 | 0.0891 | **1.34** | **0.760** | 0.799 |
| 0.100 | 0.1252 | 0.1465 | 1.17 | 0.745 | 0.799 |
| 0.200 | 0.2641 | 0.2816 | 1.07 | 0.719 | 0.799 |

log-log slope **+0.745** — divergence scales nearly linearly with σ, so the predictor
genuinely responds to action noise. The failure is not inertness or mis-tuning. **σ=0.05 is
already optimal** (peak ratio and peak AUC), and the ratio collapses at larger σ as
perturbations overwhelm the physics. σ=0 is a harness control: d_end = 1e-4 (the epsilon),
AUC 0.518 ≈ chance, as it must be when all 50 rollouts are identical.

### 7.5 Feature geometry — the signal lives in ~4 dimensions

Effective dimensionality of the predicted-latent space is tiny: top-1 PC = 40.5% of
variance, top-4 = 64.6%, top-16 = 88.5%.

| distance | patch AUC (vs GT motion) | chunk AUC |
|---|---|---|
| cosine | 0.960 | 0.780 |
| angular | 0.964 | 0.752 |
| chord | 0.964 | 0.752 |
| L2 | 0.960 | 0.729 |
| whitened cosine | 0.869 | 0.698 |

Cosine survives — but *not* for the stated norm-robustness reason (§7.2). Two structural
findings:

- **Whitening hurts** (0.780 → 0.698) despite genuine 7.3× per-dimension anisotropy: the
  high-variance directions carry the signal, the quiet tail is nuisance.
- **PCA truncation is the opposite operation and helps a lot:** cosine restricted to the
  top-4 PCs gives p90 AUC **0.855** vs 0.673 at full 384 (225-chunk subset). Sharp cliff
  between m=8 (0.831) and m=16 (0.703) — components beyond ~8 are actively harmful.
- `chord` is monotone in cosine, so identical patch ranking, yet chunk AUC differs by 0.028.
  **The distance acts through the aggregation, not the ranking.**

### 7.6 Operating points — AUC is not the deciding number

`nominal / p90 / k=30`:

| thr | recall | precision | accuracy | F1 | false alarms/episode |
|---|---|---|---|---|---|
| p85 | .720 | .064 | .8482 | .118 | 2.62 |
| p90 | .680 | .089 | .8967 | .157 | 1.75 |
| **p95** | **.520** | **.129** | .9436 | **.206** | **0.88** |
| p99 | .080 | .100 | .9769 | .089 | 0.18 |

vs the old production metric `ftle` @p90: recall .160, precision .022, F1 .039.
**≈3× recall, 6× precision, 5× F1, 38× faster.**

`d_end` has higher AUC (0.854 vs 0.838) but **loses to nominal at every threshold** — AUC
integrates over regions you would never deploy in. Choose on the operating curve.

**Do not report accuracy.** At a 1.4% base rate, predicting "safe" always scores **98.59%**,
better than every row above. CLAUDE.md's 96.9% is this artifact.

**Precision is capped by arithmetic:** a p95 threshold admits 88 false positives against 25
possible true positives, so precision cannot exceed 22%. Nominal reaches 12.9% ≈ 58% of the
achievable ceiling.

### 7.7 The monitor detects onset, not onset-in-advance

Widening the positive window to credit firing 1–2 chunks early:

| window | positives | AUC | best F1 |
|---|---|---|---|
| W=1 (concurrent) | 25 | **0.799** | 0.122 |
| W=2 | 50 | 0.746 | 0.165 |
| W=3 | 75 | 0.705 | 0.192 |

F1 rises and **AUC falls** — the added chunks rank *worse*, so the F1 gain is only the
precision ceiling moving with the base rate. Fraction of achievable precision drops
55% → 46% → 43%. Report **W=1**; the sweep is a clean sensitivity analysis showing the
monitor fires *with* the topple, not before it.

### 7.8 Dual-camera model is worse (with a confound)

`server_dual_front.py` + `test_monitor_dual.py`, scoring the **front** view only
(patches 224–307, 84/392 after masking), identical 1772 chunks:

| metric | dual | single | CI on difference |
|---|---|---|---|
| `dend_p90` | 0.653 | 0.799 | [−0.292, −0.017] |
| `dend_mean` | 0.633 | 0.780 | [−0.291, −0.011] |
| `dend_std` | 0.677 | 0.787 | [−0.208, −0.020] |

**Confound: dual checkpoint is epoch 46, single is epoch 88.** State this as "the available
dual checkpoint underperforms", not "two views hurt". To disentangle cheaply, run
`SCORE_VIEW=wrist` and `SCORE_VIEW=both` (~3.2 h each): if front-only is specifically
depressed, that is cross-view contamination from the ego-centric wrist camera; if all three
are equally low, it is just checkpoint maturity.

Also: dual costs 6.9 s/chunk vs 2.0 s single, dominated by ZMQ transfer of two views, not by
the model (GPU util was 31%).

### 7.9 **Linear tilt probe — the decisive result**

Required regenerating data: `sim.py`'s reset randomises block pose via `np.random.uniform`
without saving a seed, so re-simulating produces a *different* rollout than the stored
frames. `replay_noisy.py` now records `tilt_left`/`tilt_right` per waypoint during the
rollout, and `create_lmdb_single30.py` carries them through the same timestamp alignment as
actions/proprio into an `<ep>_tilt` key. Data: `tasks/jenga_tilt_100/` (100 episodes,
98,716 frames), LMDB `jenga_tilt.lmdb`. 1838 chunks, held out by episode (919/919).

| probe | R² | corr | AUC>10° | AUC>45° |
|---|---|---|---|---|
| *null:* timestep only | 0.074 | 0.312 | 0.709 | 0.764 |
| *null:* proprio only | −0.006 | 0.173 | 0.610 | 0.617 |
| ENCODER z_t [pool] | 0.900 | 0.949 | 0.985 | 0.999 |
| ENCODER z_t [concat] | **0.920** | 0.961 | 0.992 | 0.999 |
| **PREDICTOR z_t+8 [pool]** | **0.836** | 0.915 | 0.951 | **0.992** |
| PREDICTOR z_t+8 [concat] | 0.833 | 0.913 | 0.962 | 0.992 |
| **`d_end p90` (current metric)** | — | **0.038** | **0.535** | **0.499** |

The null baselines matter: **elapsed time alone reaches AUC 0.764**, because failures happen
late. Any probe result must be read against 0.764, not 0.5.

**Conclusion.** The frozen encoder linearly encodes present tilt (R² 0.92). The world model's
predicted latent encodes tilt **8 steps ahead** (R² 0.836, AUC 0.992). The divergence metric
computed from that same latent correlates with tilt at **0.038**. Perception is not the
bottleneck; dynamics prediction is not the bottleneck; **the readout is the entire
bottleneck.**

*Caveat, stated precisely:* the `d_end` row is scored against **tilt** over **all** chunks
including post-topple frames, where a fallen block is static (low d_end, high tilt), which
inverts the relationship. 0.499 is not "d_end fails at its own task" — on its actual task it
reaches 0.78–0.85. The defensible claim is narrower: *the predicted latent contains a
near-perfect linear tilt signal that the divergence readout does not access.*

### 7.10 What this implies for the paper

- **Headroom is bounded, not guessed.** Best safety AUC today 0.854; a linear readout of the
  same latent reaches 0.99 on the physical quantity underlying failure.
- **Fix 3 gains a specific target.** Not "better latent geometry" but: make distance align
  with the tilt direction that provably already exists in the latent.
- **A direct alternative exists** — score safety with a linear tilt probe on the predicted
  latent instead of with divergence. Tension to resolve deliberately: it uses tilt labels.
  Defensible as "a physical quantity measured in simulation, never a failure label", but it
  weakens the zero-shot claim and a reviewer will press on it.
- **δ=0.8 is explained.** It is the p99 of the safe FTLE distribution (computed: 0.8021), so
  the hand-tuned value was doing safe-calibration by eye. The offset exists because distance
  in frozen DINOv2 space has no physical calibration — see LeWorldModel below.

### 7.11 Related work found this session

**LeWorldModel** (LeCun, Mila, AMI Labs, arXiv 2603.19312, March 2026) — first JEPA training
stably end-to-end from pixels with two loss terms: next-embedding prediction + **SIGReg**, a
regularizer forcing isotropic Gaussian latents (random projections + Epps–Pulley normality
tests; Cramér–Wold gives jointness). 15M params, single GPU, hours. **48× faster planning
than DINO-WM.** Benchmarks against DINO-WM directly: beats it on Push-T, loses on the
visually complex OGBench-Cube (which favours your cluttered-Jenga setting). Also probes
physical quantities including **block angle**, with DINO-WM's frozen features probing well —
consistent with §7.9.

Relevance: SIGReg attacks §7.2 and §7.5 at the source. An isotropic latent makes distances
comparable across directions *by construction* — no norm masking, no PCA truncation, and a
natural scale for δ. Their "surprise" evaluation compares predicted vs **actually observed**
future, so it is a *different problem* from this monitor, which must flag before executing;
worth stating explicitly in related work. Feasible locally at 15M params — a strong paper
section would be: same monitor, two representations, threshold offset as dependent variable.

### 7.12 Scripts added this session

In `dino_wm/`: `nominal_baseline.py`, `distance_ablation.py`, `masked_nominal_vs_pert.py`,
`sigma_sweep_dend.py`, `feature_geometry.py`, `tilt_probe.py`, `pca_mask_combo.py`,
`conf/serve_dual.yaml`.

Results in `dino_wm/outputs/*.json` and `panda_express/results/eval100_dual/scores.json`.

Fixes: `server_dual_front.py` now uses `conf/serve_dual.yaml` (the old `train_dual.yaml`
carries `override hydra/launcher: submitit_slurm` plus Slurm-only keys, fatal locally);
`create_lmdb_single30.py` carries `_tilt`; `replay_noisy.py` records per-step tilt **and**
builds the frame path with `os.path.join` (see below).

### 7.13 Silent-failure pattern — read before generating data

Fourth instance this project of *an operation fails, nothing raises, the artifact looks
structurally complete*:

1. `show_preview` — AttributeError swallowed by a bare `except` → 0 frames per episode.
2. Missing `proc_pos` — KeyError swallowed → all 50 episodes dropped, 16 KB empty LMDB.
3. `generate_labels.py` re-simulating from a fresh reset → labels described a different
   rollout than the frames.
4. **2026-07-29:** `replay_noisy.py` built the frame path by string concatenation
   (`TARGET_TASK_DIR+"episodes/"`), silently requiring a trailing slash. The hardcoded
   default had one; `--target-dir` did not. All 98,716 frames went to
   `jenga_tilt_100episodes/` while the trajectory JSONs (using `os.path.join`) landed
   correctly. The log printed `[SAVE] Saved new episode N` 100 times. Recovered by moving
   the directory — no re-simulation needed.

> **Add a post-generation assertion** to `replay_noisy.py`: every episode directory must
> contain a non-zero, matched count of `cam1_*.png` and `cam2_*.png`. All four bugs would
> have been caught at the point of failure instead of hours downstream.

### 7.14 Open items

1. ~~**PCA × low-norm mask**~~ ✅ **answered — partially additive, see §7.15**
2. **Dual-camera confound** — needs an epoch-88 dual checkpoint (cluster work).
3. **Zero-shot tension** in the tilt-probe readout (§7.10).
4. **Re-run §7.1–7.8 conclusions on the regenerated tilt dataset** — currently the safety
   numbers come from `jenga_noise_50` and the probe from `jenga_tilt_100`. They are separate
   generations with the same noise settings (`pos_std=0.002, rot_std=0.05`, 25% vs 30%
   failure rate), which is fine for the claims made but not ideal for a single table.

### 7.15 PCA × low-norm mask, and PC1 done properly (`pca_mask_combo.py`)

p90 AUC, 225-chunk subset (25 unsafe / 200 safe), identical rollouts across all cells:

| m \ k | k=0 | k=10 | k=20 | k=30 |
|---|---|---|---|---|
| **m=4** | 0.855 | 0.855 | 0.865 | **0.876** |
| m=8 | 0.831 | 0.837 | 0.849 | 0.862 |
| m=16 | 0.703 | 0.704 | 0.699 | 0.685 |
| m=384 | 0.799 | 0.824 | 0.849 | 0.852 |

Single-fix margins are 0.855 (PCA alone, m=4/k=0) and 0.852 (mask alone, m=384/k=30);
combined **0.876**. **Partially additive** — perfect additivity would predict ~0.908, so the
two share most of their effect (both delete weak background patches) while each contributes
something the other misses. Net **+0.021** over the better single fix.

`m=16` stays at ~0.70 for every *k*: masking cannot rescue a bad subspace, reinforcing that
PCs 9–16 inject noise no patch-level mask removes.

> **Not held-out validated.** The winning cell was chosen from a 16-cell grid on 25
> positives — optimistic by construction, unlike the *k* in §7.2 which survived a split.
> Treat 0.876 as promising, not banked.

**Signed PC1 mask.** Foreground direction determined from the data (PC1-positive patches
have higher mean ‖z‖: 48.5 vs 45.5), not assumed:

| keep % | signed p90 | earlier \|PC1\| p90 |
|---|---|---|
| 100 | 0.799 | 0.799 |
| **75** | **0.817** | 0.793 |
| 50 | 0.656 | 0.785 |
| 25 | 0.633 | 0.739 |

The correction was real — at 75% the signed version beats the absolute one, confirming
`|PC1|` was the wrong operation. But it collapses past 75% and never reaches the low-norm
mask (0.817 vs 0.852). **PC1 masking is not worth pursuing; ‖z‖ is the better foreground
proxy.**

### 7.16 Probe vs divergence on the SAFETY task — the fair comparison

§7.9's headline (probe AUC 0.992 vs `d_end` corr 0.038) is **not** a like-for-like comparison:
different target (tilt angle vs safety label) and, decisively, the probe was scored on ALL
chunks including post-topple ones, where "still fallen 8 steps from now" is trivial.
`probe_vs_divergence_safety.py` scores all three methods on identical chunks under the
monitor's own rules (post-failure dropped, safe-calibrated thresholds, probe fit on one half
of the episodes and scored on the other), on `jenga_tilt_100`.

**799 held-out chunks, 15 unsafe (1.9%).** Trivial "always safe" accuracy = 0.9812.

| method | AUC |
|---|---|
| **probe (tilt at t+8)** | **0.941** |
| `d_end p90 / k=30` | 0.887 |
| `nominal p90 / k=30` | 0.698 |

Probe operating points:

| thr | recall | precision | F1 | FP/episode |
|---|---|---|---|---|
| p75 | **1.000** | .071 | .133 | 3.92 |
| p80 | **1.000** | .087 | .160 | 3.14 |
| p85 | .867 | .099 | .178 | 2.36 |
| p90 | .800 | .132 | .226 | 1.58 |
| p95 | .533 | .167 | .254 | 0.80 |
| **p99** | .400 | **.429** | **.414** | 0.16 |

Best F1: probe **0.414**, nominal 0.296, `d_end` 0.157. **The probe reaches 100% recall at
p75–p80**; no divergence configuration reaches it at any threshold.

| comparison | A | B | A−B | 95% CI | P(A>B) |
|---|---|---|---|---|---|
| probe vs `d_end` | 0.941 | 0.887 | +0.054 | [−0.003, +0.112] | 0.968 |
| probe vs `nominal` | 0.941 | 0.698 | +0.243 | **[+0.106, +0.403]** | 1.000 |

**Three corrections this forces:**

1. **The true gap is 0.941 vs 0.887, not 0.99 vs 0.04.** The probe is meaningfully better —
   especially at the extremes — but divergence is not the disaster §7.9's framing implied.
   Quote §7.16 for the comparison and §7.9 only for "the information is present in the latent".
2. **`nominal` vs `d_end` is dataset-dependent** (see the warning in §7.3). Here `d_end`
   (0.887) is far ahead of `nominal` (0.698) and wins at loose thresholds; on
   `jenga_noise_50` the ordering was reversed at deployable thresholds.
3. **The k=30 mask transferred out of sample.** It was tuned on `jenga_noise_50` and applied
   here without retuning — a genuine cross-dataset check, and it held.

**Zero-shot trade, quantified.** Adopting a probe readout buys ≈ +0.05 AUC over `d_end`, plus
a much better operating curve (100% recall available; 43% precision available), at the cost of
requiring tilt labels. Labels come from simulation physics, never from failure annotation —
defensible, but a weaker claim than the current one. Open decision (§7.14 item 3).

### 7.17 Exact Jacobian FTLE — correct, and worse (`jacobian_ftle.py`)

Motivated by §7.18's coverage result: the sampled max is a Monte Carlo lower bound on the
true maximum expansion, so compute the real thing —
`J = dz_T/da`, `lambda = (1/T) ln sigma_max(J)`, with J projected onto the tangent space at
z_T (cosine distance is scale-invariant, so radial growth must be projected out) and
sigma_max from the largest eigenvalue of J^T J.

**The Jacobian is correct.** Direct linearisation test against the model:

| ‖δ‖ | relative error | cos(actual, predicted) |
|---|---|---|
| 1e-4 | 0.0036 | 1.0000 |
| 1e-3 | 0.0477 | 0.9995 |
| 1e-2 | 0.5833 | 0.8210 |
| **5e-2 = operating σ** | **0.9633** | **0.5380** |

**But the linear regime ends around 1e-3 — 50× below the σ actually used.** At σ=0.05 the
linearisation is 96% inaccurate and its predicted displacement direction has only 0.54 cosine
with the truth.

Results on 225 chunks (25 unsafe), identical chunks for both methods:

| metric | AUC |
|---|---|
| `dend_p90` — **sampled N=50** | **0.827** |
| `jac_sigma_end_max` — exact | 0.763 |
| `jac_sigma_end` — exact, p90 | 0.706 |
| `jac_ftle` — exact, with denominator | 0.617 |

Paired bootstrap: **−0.121, CI [−0.249, −0.006], P(exact>sampled) = 0.022.** Significantly
worse, not tied.

**Mechanism.** `corr(sigma_max, sampled divergence)` is **+0.841 across patches within a
chunk** but **+0.030 across chunks**. The infinitesimal and finite-amplitude quantities agree
locally and decouple at the level the monitor scores.

> The script's built-in "weak correlation → unreliable" guard fires here. That guard exists to
> catch a *buggy* Jacobian; J was independently verified correct (rel err 0.0036 at ‖δ‖=1e-4),
> so the weak correlation is the FINDING, not an error. Do not re-derive this as a bug.

**Consistency result worth quoting.** The denominator hurts in *both* formulations —
sampled 0.599 → 0.799, exact 0.617 → 0.763. Two independent estimators agree, so "drop the
denominator" is a property of the formulation, not an artifact of sampling.

**Cost.** `jacfwd` 3.9 s/chunk vs 2.0 s for sampled N=50 — 2× *more* expensive, because
forward-mode AD requires the MATH SDPA backend (the memory-efficient kernel adopted earlier
for a 1.6× speedup does not implement forward AD). The finite-difference fallback (34
sequential rollouts on the fast kernel) was faster at 1.9 s.

**Verdict.** Do not adopt. Report as an ablation: *the mathematically correct FTLE
underperforms the crude finite-perturbation estimator, because failure is a finite-amplitude
basin phenomenon and the linearisation is invalid at the operating scale.* This converts an
apparent methodological weakness into a justified design choice, and pre-empts the reviewer
objection that "max over random samples" is not a Lyapunov exponent.

### 7.18 Perturbation coverage and the nonlinearity question (`perturbation_coverage.py`)

The perturbation lives in **span(11) × 3 = 33 dimensions** (not 24 — earlier arithmetic in
this file was wrong). A random direction has expected alignment ~1/√33 = 0.17 with the top
singular vector.

**Saturation** — `max_j d_end` vs N, 125 chunks:

| N | max (unsafe) | max (safe) | fraction of N=400 |
|---|---|---|---|
| 10 | 0.1992 | 0.1353 | 0.643 |
| 25 | 0.2232 | 0.1623 | 0.746 |
| **50** | 0.2247 | 0.1770 | **0.798** |
| 100 | 0.2549 | 0.1943 | 0.880 |
| 200 | 0.2676 | 0.2102 | 0.942 |
| 400 | 0.2772 | 0.2248 | 1.000 |

N=50 captures only **80%** of the N=400 max, which has itself not plateaued. Estimator noise:
**CV = 0.167** for the N=50 max across disjoint blocks of 50.

**Bimodality** (per-perturbation d_end distribution):

| group | BC (>0.555 = bimodal) | fraction BC>0.555 |
|---|---|---|
| unsafe | 0.509 | 0.36 |
| safe | 0.530 | 0.38 |

No contrast between unsafe and safe.

> ⚠️ **This test does NOT establish linearity, and I initially over-read it.** Bimodality
> measures distribution *shape*; §7.17's direct test shows the map is severely nonlinear at
> σ=0.05 despite unimodal output. The σ-slope of +0.745 (§7.4) misleads the same way — a
> smooth power law in the *mean* is compatible with strong per-direction nonlinearity. To ask
> "is a linearisation valid", compare `J·δ` against `f(δ) − f(0)` at the operating scale.
> Everything else is a proxy.

### 7.19 Robust FTLE variants — the denominator has no salvageable value (`robust_ftle.py`)

Five ways to keep the exponential-growth-rate idea while removing the fragility identified in
§7.16 (d_start is 7x smaller than d_end, CV 0.42 vs 0.28, and carries same-direction signal at
AUC 0.607 so dividing cancels signal). All on identical rollouts, k=30 mask, p90:

| variant | p90 AUC |
|---|---|
| `ftle_2pt` (current) | 0.710 |
| `ftle_2pt_median` | 0.701 |
| **`ftle_slope`** (least-squares slope of log d(t) over all 9 timesteps) | **0.682** |
| `ftle_slope_median` | 0.675 |
| `ftle_slope_of_mean` | 0.692 |
| `ftle_ratio_of_means` | 0.708 |
| **`ftle_pooled_den`** (one median d_start per chunk) | **0.782** |
| `ftle_shrunk_0.5x / 1x / 4x` | 0.713 / 0.716 / 0.747 |
| `d_end` (reference) | **0.852** |

**The slope fit — the most theoretically motivated variant — made things WORSE** (0.682 vs
0.710). A least-squares slope of log d(t) assumes *exponential* growth; the divergence is
sub-exponential (sigma-sweep slope +0.745, saturating at large sigma), so the model is
misspecified and adding the early timesteps (smallest d, worst relative noise) hurts.
**The "Lyapunov exponent" framing assumes dynamics this system does not have.**

**Pooling the denominator is the real win: 0.710 -> 0.782.** One median d_start per chunk
instead of per patch. Confirms the diagnosis exactly — per-patch denominator noise was the
dominant cost — while preserving the log-ratio form.

Full arc for the ratio: **0.599 -> 0.710 (mask) -> 0.782 (pooled denominator)**.

> **The shrinkage sweep is the decisive diagnostic and it is monotonic:** 0.713 -> 0.716 ->
> 0.747 as eps grows, heading to d_end's 0.852 at eps -> infinity. No intermediate optimum.
> There is no normalisation sweet spot; the denominator is pure cost.

`ftle_pooled_den` vs `d_end`: −0.069, CI [−0.138, +0.001], P(A>B) = 0.025.

**Verdict.** If FTLE framing is wanted for the paper, use `ftle_pooled_den` — principled (one
reference scale per chunk), keeps the exponential-rate form, and lifts 0.599 -> 0.782. But the
case against the ratio is now consistent across **five independent tests**: sampled (§7.1),
sampled with all fixes (§7.16), exact Jacobian (§7.17), shrinkage monotonicity (here), and
robustification (here). The honest framing is that FTLE motivated the method and measurement
showed its denominator subtracts signal.

### 7.20 Pooled-denominator FTLE videos — the fix is partial (`metric_compare_video.py`)

`ftle_pooled` (§7.19, AUC 0.782) rendered on the same ten episodes as `d_end` and `ftle`.
Threshold p90 of its own safe distribution = 0.3985.

| episode | topple | `ftle` | `ftle_pooled` |
|---|---|---|---|
| ep50 | 159 | 24 (135)* | 24 (135)* — unchanged |
| ep54 | 72 | 0 (72)* | **miss** — spurious catch removed |
| ep59 | 45 | 40 (5) | 40 (5) |
| ep61 | 62 | miss | miss |
| ep65 | 70 | 24 (46)* | 24 (46)* — unchanged |
| ep78 / ep79 / ep82 | — | miss | miss |
| ep51 | none | silent | silent |
| ep52 | none | false alarm @112 | false alarm @72 |

\* fired before anything happened. **`ftle` 4/8 → `ftle_pooled` 3/8 caught.**

Pooling removed **one of three** spurious early alarms. That splits the denominator problem:

- **per-patch noise** — pooling fixes it; this is the 0.710 → 0.782 gain
- **quiet chunks have small denominators** — structural, survives pooling. Early in an
  episode the arm is far from the blocks, so `d_start` is *genuinely* small at chunk level,
  and dividing by it inflates the score precisely where nothing is happening. Same mechanism
  as `d_start` carrying same-direction signal (AUC 0.607 alone).

> Higher AUC did NOT mean more topples caught: `ftle_pooled` (0.782) catches fewer than
> `ftle` (0.710) on these ten. AUC ranks ~200 chunks globally; catch-count at a fixed
> threshold on 8 failures is a far noisier statistic. Do not read 0.782 as "catches more".

### 7.21 The "false alarms" are not arbitrary (`false_positive_probe.py`)

`get_block_tilt` only measures `block_left` / `block_right`. The red **target** block is not
tracked, so instability involving it is invisible to the label. On the success episodes
rendered as videos, adjacent tilt peaks at 6.4–8.4 deg yet all three metrics fired.

Tested without new simulation (which would be unreproducible — sim.py randomises block pose
with no saved seed) by comparing ground-truth PIXEL motion at firing vs silent safe chunks.
250 held-out safe chunks, p90 thresholds, 25 firings each:

| metric | motion (fire) | motion (silent) | ratio | adj tilt (fire) | adj tilt (silent) |
|---|---|---|---|---|---|
| `d_end` | 1.445 | 0.689 | **2.10x** | **6.62°** | 1.92° |
| `probe` | 1.146 | 0.722 | **1.59x** | **4.71°** | 2.13° |
| `ftle_pooled` | 1.024 | 0.736 | 1.39x | 3.33° | 2.29° |

Spatial spread of the firing patches (fixed camera, so patch index is a stable location):

| metric | patches | row spread | col spread | centre |
|---|---|---|---|---|
| `probe` | 7.2 | 1.00 | 1.88 | (2, 7) |
| `d_end` | 7.2 | 0.86 | 1.49 | (3, 6) |
| `ftle_pooled` | 7.4 | 0.91 | 1.89 | (2, 5) |

~7 of 84 patches, clustered within 1–2 patches on a 14x14 grid — the signature of a single
object, not the sweeping arm (which would spread across many columns).

**Conclusion.** Firing safe chunks contain 1.4–2.1x more real motion and 2.2–3.4x more
adjacent-block tilt than silent ones, localised to a small fixed region. The binary
topple/no-topple label scores a 15 deg wobble identically to a motionless scene, so
**precision is understated** — the monitor is detecting a continuum the label discards.
Not unlimited: 6.6 deg is real wobble, not a near-miss.

Note the ordering: `d_end`'s false alarms are the most justified (2.10x, 6.62°) and
`ftle_pooled`'s the least (1.39x, 3.33°), matching their AUC ordering. The better metric's
mistakes are more defensible mistakes.

**Instrumentation added for next time.** `sim.py` now tracks `block_middle` in
`_record_ref_quats` and exposes `get_block_xy()`; `replay_noisy.py` records `tilt_middle`
and `mid_xy` per waypoint. `block_middle` is deliberately NOT added to `check_failure` — the
target block is meant to move, and counting it would corrupt the labels. A future
regeneration can then test the target-block hypothesis directly instead of by inference.

### 7.22 Should the failure definition change? No — and the data proves it can't matter

Peak adjacent-block tilt over all 100 episodes of `jenga_tilt_100`:

```
  5- 10 deg :  23      20- 90 deg :   0
 10- 15 deg :  35      90-100 deg :  30
 15- 20 deg :  12
```

**Zero episodes between 17.5 and 90 deg.** The distribution is starkly bimodal — blocks stay
under ~17 deg or end flat at ~90. No partial topples, no near-misses. **Any threshold in
20-90 deg produces identical labels**, so the 45 deg cut sits mid-gap and is about as robust
as a binary choice can be. Changing it in that range is cosmetic.

> Corrects §7.21's framing: the false alarms land at 6.62 deg mean tilt, which is *inside*
> the normal standing mode (5.2-16.8, median 11.1) — **not** near-misses. The monitor
> discriminates within ordinary wobble. Precision is understated by less than §7.21 implied.

The only redefinition that changes anything cuts INSIDE the standing mode
(`failure_threshold_sweep.py`, 913 chunks, 50 held-out episodes, labels recomputed from
scratch at each level, identical rollouts throughout):

| threshold | positives | base rate | probe | `d_end` | `ftle_pooled` |
|---|---|---|---|---|---|
| 8° | 41 | 7.7% | 0.844 | 0.815 | 0.700 |
| 10° | 35 | 5.9% | 0.808 | 0.778 | 0.664 |
| 12° | 30 | 4.5% | 0.737 | 0.790 | 0.732 |
| 15° | 21 | 2.8% | 0.879 | 0.818 | 0.698 |
| 20° | 15 | 1.9% | 0.899 | 0.866 | 0.748 |
| **45° (current)** | 15 | 1.9% | **0.941** | **0.887** | **0.785** |

**Verdict: keep 45 deg.** (a) 20-90 deg is a no-op, (b) going lower degrades every metric,
(c) redefining failure to match what the monitor fires on is circular — a change must be
justified by the task ("a 15 deg wobble is operationally bad"), never by the metric.

**Two results worth keeping from the sweep:**

1. **The metric ordering is invariant.** probe > `d_end` > `ftle_pooled` at *all six*
   thresholds without exception. Conclusions about which readout is best do not depend on
   where the failure line is drawn — a stronger claim than any single-threshold comparison,
   and it pre-empts the reviewer objection that the headline numbers ride on a hand-chosen cut.
2. **The monitor does rank sub-topple disturbance.** At 8 deg, deep inside normal wobble, the
   probe still reaches 0.844. The "false alarms" are not noise; there is real signal about
   degrees of instability.

**A subtle confirmation of finding (f).** At 20 deg there are the SAME 15 positives as at
45 deg, yet AUC drops 0.941 -> 0.899. Crossing 20 deg happens a few steps earlier, so an
earlier chunk becomes the positive one and the metrics score it lower. Asking the monitor to
fire a few steps sooner costs measurable accuracy — independent evidence that it tracks the
fall as it develops rather than predicting it far ahead.

### 7.23 ftle_variance benchmarked — competitive, and cheaper conceptually

Never benchmarked before (`metric_family_eval.py`). Spread of the 49 PERTURBED final
latents around their own centroid, never referencing the original/unperturbed trajectory:

```
centroid[p] = mean_j z_pert_j[T,p]
ftle_variance[p] = mean_j (1 - cos(z_pert_j[T,p], centroid[p]))
```

799 held-out chunks, identical to section 7.16/7.22's setup:

| metric | AUC |
|---|---|
| probe | 0.941 |
| `d_end` | 0.887 |
| **`ftle_variance`** | **0.858** |
| `ftle_pooled` | 0.785 |

Beats every FTLE-ratio variant from section 7.19, including the best one (`ftle_pooled`).
Sidesteps the whole d_start problem by construction (no division by anything near the noise
floor), at the cost of never comparing to what the unperturbed action would have done — it
answers "how much do perturbations disagree with each other", not "how far does the
perturbed future diverge from the actual one".

### 7.24 Held-out validation of PCA truncation and the PC1 mask — corrects section 7.15

**PCA truncation does not survive cross-validation.** 20 episode splits, (m,k) grid searched
on train half only (`pca_mask_heldout.py`, corrected reduction — see caveat below):

| | held-out AUC | selection |
|---|---|---|
| PCA x low-norm mask | 0.895 | **m=384 (no truncation) chosen 20/20 times**, k=30 (18/20) or k=20 (2/20) |
| unmasked baseline | 0.700 | — |

Every single split picked the FULL feature space. Section 7.15's m=4 winner (0.855/0.876)
was overfit to the 225-chunk sample it was picked on. **Drop PCA truncation from the
pipeline** — only the low-norm mask component was ever real.

**The (corrected, signed) PC1 mask held up better than expected:**

| | held-out AUC | selection |
|---|---|---|
| PC1 mask | **0.941** | **75% keep chosen 20/20 times** (zero variance) |

Now scores ABOVE the low-norm mask's own held-out result (section 7.2: mean 0.848),
reversing section 7.15's in-sample read where PC1 underperformed.

> **Reduction bug found and fixed mid-task.** The score() this borrowed from
> pca_mask_combo.py flattened over (perturbation, patch) before taking p90 (~4000 pooled
> values) instead of averaging over perturbations FIRST then taking p90 over patches
> (~54-84 values) -- the convention used everywhere else since section 7.1. The buggy
> version gave AUC 0.92-0.94 for PCA+mask; fixed, it gives 0.895. **This means section
> 7.15's 0.876 in-sample PCA+mask reference has the same bug and should not be trusted
> either.**
>
> Even after the fix, both numbers here (0.895, 0.941) sit above section 7.2's full-corpus
> held-out figures (0.838-0.854, 1772 chunks). This run used the same 225-chunk/91-episode
> subsample as pca_mask_combo.py -- smaller, noisier test halves (~112 chunks, ~12-13
> positives) than the full pool. Trust the RELATIVE ordering (PC1 >= PCA+mask >> unmasked,
> PCA truncation contributes nothing) since both share the identical sample; do not cite
> 0.895 or 0.941 as replacements for section 7.2's numbers without a full-corpus rerun.

### 7.25 Probe calibration — good ranking, poor face-value magnitude

`metric_family_eval.py`, all held-out chunks, predicted vs actual future tilt:

| predicted bin | n | mean pred | mean actual | bias |
|---|---|---|---|---|
| 0-5° | 260 | 2.37 | 1.39 | -0.99 |
| 15-20° | 20 | 16.64 | 6.69 | -9.96 |
| 20-30° | 18 | 24.94 | 7.95 | -16.99 |
| **30-45°** | 19 | 37.30 | 8.92 | **-28.38** |
| 45-70° | 11 | 55.37 | 27.84 | -27.53 |

**Global regression: actual = 0.48 x predicted + 0.83.** Well-calibrated would be slope~1,
intercept~0. Right where halt decisions get made (predicted 15-45 deg, between the p80 and
p99 thresholds), the probe systematically overpredicts by 10-28 deg -- a "37 deg" prediction
averages 9 deg true tilt.

**AUC 0.941 says ranking is sound; the raw score is not an interpretable degree estimate.**
Fine for percentile-thresholded classification (what section 7.16/7.22 do); would need
isotonic regression or similar before reporting it as a face-value physical prediction.

### 7.26 Threshold stability under resampling — the probe is the hardest to calibrate

2000-rep episode bootstrap of the SAFE chunk pool, threshold value (not AUC) per resample:

| metric | p80 CV | p90 CV | p95 CV | p99 CV | p95 95% CI |
|---|---|---|---|---|---|
| **probe** | 0.09 | 0.18 | **0.27** | 0.12 | **[14.4, 33.8] deg** |
| `d_end` | 0.03 | 0.03 | 0.04 | 0.11 | [0.064, 0.074] |
| **`ftle_pooled`** | 0.01 | 0.01 | 0.01 | 0.03 | [0.410, 0.426] |
| `ftle_variance` | 0.04 | 0.03 | 0.02 | 0.06 | [0.036, 0.039] |

**The probe's p95 threshold could land anywhere from 14 deg to 34 deg** depending on which
50 safe episodes happen to be in the calibration sample -- a >2x range. `ftle_pooled` is the
most reproducible across every quantile (CV 0.01-0.03), consistent with section 7.25: the
same heavy-tailed, compressed score distribution that causes the calibration bias also makes
the probe's percentiles unstable under resampling.

**Trade-off to state plainly in the paper:** the probe discriminates best (AUC 0.941) but is
the least reliable of the four to calibrate from limited held-out data. A single calibration
run on this system could hand you a materially wrong threshold.

### 7.27 Miss analysis: ep78 and ep82 are "flash topples" with no precursor signal

Both consistently missed by the probe across every threshold tested (section 7.16, 7.22),
and by `d_end` on one of the two. Full pre-failure trajectory (not just the tail):

**ep78**: tilt flat at 0.0 deg for 80 steps, then in the single TERMINAL chunk:
0.0 -> 7.6 deg (observed) -> 90 deg (predicted horizon). Max tilt anywhere in the entire
pre-failure history: **7.6 deg**.

**ep82**: flat at 0.0 deg for 56 steps, then 0.0 -> 10.8 -> 85.1 in one window. Max tilt
anywhere pre-failure: **10.8 deg**.

For comparison, section 7.21's *unjustified* false alarms averaged 6.62 deg adjacent-block
tilt. The state immediately preceding these catastrophic falls is LESS extreme than plenty
of chunks the monitor correctly ignores elsewhere -- nothing in the 3 observed frames
visually distinguishes "about to fall in 8 steps" from ordinary wobble.

Contrast with caught episodes: **ep65** ramps clearly across consecutive chunks before the
fall (probe: 1.4 -> -3.4 -> -0.0 -> 2.7 -> 17.5 -> 78.6). **ep50** is caught via an EARLIER,
separate wobble bout (tilt to 10 deg, three chunks before the actual fall) that happens to
trip the alarm, even though its terminal event is equally instantaneous (0 -> 90 deg in one
window) -- the catch is opportunistic, not a detection of the actual falling mechanism.

`d_end` catches ep78 comfortably (0.0788 vs threshold 0.0605) but ep82 by the barest
possible margin (0.0606 vs 0.0605, a difference of 0.0001). The probe misses both, further
below threshold in each case (9.32 and 11.07 vs 12.78).

**Conclusion: a genuine information limit, not a metric weakness.** Given a 3-frame
observation window and 8-step horizon, a fast physical event with zero visible run-up
cannot be forecast by construction -- there is nothing to detect. Distinct from section
7.7's "fires with the topple, not before it": that finding was about aggregate ranking of
chunks WITH some precursor signal; this is about chunks with NONE. No metric tuning closes
this gap; only a longer horizon or higher sampling rate could, and only if the precursor is
visually detectable before the block starts moving at all.

### 7.28 Why the probe's threshold is unstable: two distinct mechanisms, one dominant

Diagnosed directly from `metric_family_eval.json`'s cached per-chunk scores (no new GPU
run). Distribution of probe scores on the 784 safe chunks: mean 4.57, median 2.75, std 9.95,
**skewness 2.64, excess kurtosis 10.84** -- a heavily right-tailed, non-normal distribution.

**Episode 60 alone supplies 15 of the top 39 scores (38% of the entire top-5% tail)**, and
it is a real, sustained behavioural pattern rather than noise: tilt wobbles to 13 deg around
step 112-136, settles back to a flat 0 by step 176 -- and the probe's prediction never
recovers, staying at 25-53 deg for the final 14 consecutive chunks after the true tilt is
already zero. This directly explains section 7.25's calibration bias (the 30-45 deg
predicted bin averaging only ~9 deg actual): episode 60-type chunks are a large part of that
bin's mass. Because it is sustained rather than a one-off, this pattern would recur on any
calibration sample that happens to include an episode like it -- it is a property of the
representation after a resolved disturbance, not sampling luck.

**Episode 76 shows a different failure**: one isolated single-chunk spike (probe=75.38 at
one step, the highest value across all 784 safe chunks) surrounded by otherwise normal
values immediately before and after. Dropping this single episode barely moves the
threshold (p95: 23.10 -> 22.60) -- transient single-frame noise, cheap to average away and
not the dominant driver.

**Two distinct mechanisms feed the heavy tail**: transient single-frame noise (episode 76,
minor contributor) and sustained post-disturbance drift where the latent state does not
fully reset after a resolved wobble (episode 60, the dominant contributor). The second is a
genuine property of the probe worth fixing -- e.g. an explicit decay/reset term, or
including "recently disturbed but now settled" examples in the probe's training data -- not
merely a calibration artifact to shrug off.
