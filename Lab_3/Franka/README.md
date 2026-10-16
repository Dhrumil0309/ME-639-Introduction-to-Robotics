# Tutorial 4 – Pick and Place in MuJoCo with IK

A Franka Emika Panda picks a randomly placed cube from a table and drops it into a
tray. Inverse kinematics is done four ways and compared on identical episodes:

| Method | File / class | Notes |
|---|---|---|
| **Mink** (off-the-shelf, Stage 1) | `ik_solvers.MinkIK` | https://github.com/kevinzakka/mink — FrameTask + PostureTask, joint/velocity limits, collision-avoidance limit |
| **Pseudo-inverse** (own choice) | `ik_solvers.PinvIK` | Newton–Raphson, `dq = J⁺ K e`, no damping |
| **DLS-IK** | `ik_solvers.DLSIK` | `dq = (JᵀJ+λI)⁻¹Jᵀ K e` + null-space joint centring |
| **QP-IK** | `ik_solvers.QPIK` | DLS cost as a QP with joint-limit and velocity constraints (quadprog) |

## 1. Install

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

The Panda model (from [mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie),
Apache-2.0) is already in `assets/panda/`.

Headless Linux (no screen): `export MUJOCO_GL=egl` before running anything that renders.

## 2. Run

```bash
python ik_solvers.py                          # self-test: all 4 solvers hit one target
python workspace.py                           # Sec 5-6: workspace plots + table design (~9 min)
python workspace.py --reuse                   #   re-plot from saved data (seconds)
python pick_place.py --method mink --episodes 25            # Sec 9-10: batch run + log
python pick_place.py --method dls  --episodes 3 --viewer    # watch it live
python compare.py --run                       # Sec 13: all methods x 3 experiments (~7 min)
bash make_videos.sh                           # Sec 15: 1-min video per method
```

`pick_place.py` options: `--method {mink,pinv,dls,qp}`, `--episodes N`, `--seed S`
(episode *i* uses seed *S+i* → identical cubes for every method), `--region {designed,full}`,
`--naive` (switches off the smart choices, for the failure study), `--retries K`
(alternate grasp yaws), `--video file.mp4`, `--viewer`.
On macOS the live viewer needs `mjpython pick_place.py --viewer`.

## 3. Files

| File | What it does |
|---|---|
| `config.py` | every number (table, tray, speeds, gains) |
| `scene.py` | builds robot + table + cube + tray with `MjSpec` |
| `kinematics.py` | FK, Jacobian, quaternion error (as in the notes), slerp |
| `ik_solvers.py` | the 4 IK solvers behind one `.solve()` interface |
| `workspace.py` | task/dexterous workspace, table height/size selection |
| `pick_place.py` | planner (IK along Cartesian segments), physics execution, checks, logs |
| `recorder.py` | mp4 recording with the TCP trail drawn in |
| `compare.py` | runs/aggregates experiments, summary table and plots |
| `GUIDE.md` | **step-by-step explanation of everything** |
| `REPORT_DRAFT.md` | report skeleton (5 sections) with the measured numbers |

## 4. Outputs

* `results/ws_*.png`, `results/table_design.md` – workspace analysis & table design
* `results/logs_<method>[_stress|_naive].csv` – one row per episode
* `results/summary.md` / `summary.csv` – metrics table
* `results/plots/cmp_*.png` – comparison plots
* `results/failures/*.png` – screenshot of every failed episode
* `videos/*.mp4` – one minute per IK method, failure clip, joined 3-min video

## 5. Headline results (25 episodes each, seeds 0–24)

| Experiment | Mink | Pinv | DLS | QP |
|---|---|---|---|---|
| nominal (designed spawn region) | 100 % | 100 % | 100 % | 100 % |
| stress (cube anywhere on the table) | 100 % | 100 % | 100 % | 100 % |
| naive (smart choices OFF) | 16 % | 24 % | 28 % | 16 % |

Details and interpretation: `results/summary.md` and `REPORT_DRAFT.md`.
