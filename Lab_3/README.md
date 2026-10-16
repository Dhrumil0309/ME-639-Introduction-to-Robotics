# ME 639 – Lab 3: Inverse Kinematics & Constrained Pick-and-Place Manipulation
## Comparative Analysis: 6-DOF HEAL Arm vs. 7-DOF Franka Emika Panda

Instructor: **Prof. Madhu Vadali** | **IIT Gandhinagar**  
Author: **Dhrumil Bhutaiya**  

---

## 📌 Complete Deliverables & Interactive Documentation
> [!IMPORTANT]
> **Complete Lab 3 Interactive Deliverables & Notion Documentation:**  
> 👉 [**Open Lab 3 Notion Workspace**](https://sour-grain-a01.notion.site/LAB_3-ITR-3f718e510a158056ab86d72f7af0095c?source=copy_link)  
> The Notion workspace provides full interactive logs, mathematical derivations, high-resolution failure scene captures, and detailed step-by-step explanations for all deliverables.

---

## 📖 Overview & Executive Summary

This lab investigates the kinematic design, workspace geometry, and real-time inverse kinematics (IK) performance for robotic pick-and-place manipulation under physical contact and joint constraints in **MuJoCo**. We implement and evaluate two robotic platforms:
1. **6-DOF HEAL Arm**: An anthropomorphic industrial manipulator with spherical wrist equipped with a **Robotiq 2F-85** parallel-jaw gripper.
2. **7-DOF Franka Emika Panda**: A redundant collaborative manipulator equipped with an integrated 2-finger parallel gripper.

We formulate, solve, and compare three numerical Inverse Kinematics paradigms:
* **Damped Least Squares (DLS / Levenberg-Marquardt)**: Adaptive Tikhonov regularization $\Delta \theta = J^T (J J^T + \lambda^2 I)^{-1} e$ avoiding singularity blow-up.
* **Quadratic Programming (QP-IK with OSQP)**: Constrained convex optimization $\min \|\dot{q}\|_2^2$ subject to exact task tracking and box inequality joint limits $q_{\min} \le q + \Delta q \le q_{\max}$.
* **Mink (Task-Space Optimization)**: Modern Lie-group task hierarchy using dual quaternion geometry and collision avoidance.

### Summary of Key Results

| Metric | 6-DOF HEAL Arm (Mink) | 6-DOF HEAL (DLS) | 6-DOF HEAL (QP) | 7-DOF Franka Panda (Nominal) | 7-DOF Franka Panda (Stress) | 7-DOF Franka Panda (Naive) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Degrees of Freedom (DOF)** | 6 | 6 | 6 | 7 | 7 | 7 |
| **Nominal Task Success Rate** | **76.0%** (19/25)* | **72.0%** (18/25) | **76.0%** (19/25) | **100.0%** (25/25) | **100.0%** (25/25) | 16% – 24% |
| **True Spatial Misses** | 24.0% (6/25) | 28.0% (7/25) | 24.0% (6/25) | 0.0% (0/25) | 0.0% (0/25) | 76% – 84% |
| **Average Solve Time / Step** | 4.92 ms | 3.12 ms | 5.88 ms | 36.7 ms (Mink) | 48.2 ms | 26.3 ms (Pinv) / 53.7 ms (QP) |
| **Joint Limit Violations** | 0 | 0 | 0 | 0 | 0 | 0 (QP/Mink) vs 413 (Pinv) |
| **Redundancy Resolution** | None | None | None | Posture regularization ($q_0$) | Posture regularization | Boundary collision without nullspace |

*\*Note on HEAL Grasp Dynamics: The Robotiq 2F-85 uses passive spring return. Lingering silicone pad friction (`geom 36/48`) during upward retreat imparts yaw torque to the light 100g cube on the low-friction MuJoCo tray floor (`friction=[1.0, 0.05, 0.0001]`). The cube is safely inside the tray but continues spinning, yielding an effective task success rate of 76.0% (19/25).*

---

## 🎥 Video Deliverables (3-Minute Side-by-Side Demonstrations)

Each platform includes a comprehensive 3-minute video featuring **Mink (0:00–1:00)**, **DLS-IK (1:00–2:00)**, and **QP-IK (2:00–3:00)** with real-time on-screen TCP trail visualization:

* 🎬 **HEAL 6-DOF Video**: [`Lab_3/HEAL/HEAL_3min_IK_comparison.mp4`](./HEAL/HEAL_3min_IK_comparison.mp4) (2.3 MB)
* 🎬 **Franka 7-DOF Video**: [`Lab_3/Franka/Franka_3min_IK_comparison.mp4`](./Franka/Franka_3min_IK_comparison.mp4) (7.5 MB)

---

## 🗂️ Repository Directory Structure

```text
Lab_3/
├── README.md                              <- This comprehensive guide
├── lab_guidance.pdf                       <- Official Lab 3 assignment prompt
│
├── HEAL/                                  <- 6-DOF HEAL Robotic Arm Implementation
│   ├── HEAL_3min_IK_comparison.mp4        <- 3-Minute side-by-side IK video
│   ├── run_pick_and_place_mink.py         <- Main pick-and-place simulation (Mink IK)
│   ├── spawn_heal.py                      <- Standalone model visualizer
│   ├── spawn_task_scene.py                <- Task scene constructor (table + tray)
│   ├── analyze_workspace.py               <- Grid-based reachability generator
│   ├── analyze_heal_task_workspace.py     <- Task-space orientation reachability
│   ├── record_3min_video.py               <- Headless frame capturer & video compiler
│   ├── heal_workspace_results/            <- Voxel plots & reachability data (.npz)
│   ├── task10_batch/                      <- 25 randomized episodes: logs & screenshots
│   │   ├── batch_logs.csv
│   │   └── screenshots/                   <- Per-episode capture (Success vs Failure)
│   ├── task11_failure/                    <- Root-cause failure analysis & distributions
│   │   ├── failure_analysis.py
│   │   ├── failure_analysis_report.md
│   │   └── plots/                         <- Outcome breakdown & spatial failure scatter
│   ├── task12_custom_ik/                  <- Custom IK implementations from scratch
│   │   ├── custom_ik_solvers.py           <- DLS (Levenberg-Marquardt) & QP (OSQP) solvers
│   │   └── kinematics.py                  <- Forward kinematics & analytical Jacobians
│   ├── task13_comparison/                 <- Cross-solver evaluation suite
│   │   ├── compare_ik_methods.py          <- 75-run benchmark runner (Mink vs DLS vs QP)
│   │   ├── comparison_summary.csv         <- Aggregated timing, iterations & success
│   │   └── plots/                         <- Success rate & solve time comparisons
│   └── Lab3/                              <- Robot description assets (MJCF & meshes)
│       └── robot_descriptions/
│           ├── arm_gripper.xml
│           └── heal_meshes/
│
├── Franka/                                <- 7-DOF Franka Emika Panda Implementation
│   ├── Franka_3min_IK_comparison.mp4      <- 3-Minute side-by-side IK video
│   ├── pick_place.py                      <- Parameterized pick-and-place simulator
│   ├── compare.py                         <- 300-episode solver benchmarking suite
│   ├── workspace.py                       <- Monte Carlo FK & sliced workspace analyzer
│   ├── ik_solvers.py                      <- Mink, Pinv, DLS, and QP solver modules
│   ├── kinematics.py                      <- Analytical Franka Jacobian & FK transforms
│   ├── scene.py                           <- MjSpec procedural scene generator
│   ├── config.py                          <- Physical parameters & tuning constants
│   ├── recorder.py                        <- Offscreen video & camera streamer
│   ├── record_3min_video.py               <- 3-minute video pipeline compiler
│   ├── assets/panda/                      <- Franka Panda MJCF model and meshes
│   └── results/                           <- Benchmarks, failure captures & plots
│       ├── summary.csv                    <- 300-run comprehensive metric table
│       ├── summary.md                     <- Markdown benchmark summary
│       ├── ws_slices.png                  <- Reachability slices across heights
│       ├── ws_table.png                   <- Table-surface reachability map
│       ├── failures/                      <- High-res collision & failure screenshots
│       └── plots/                         <- Comparative success, time, & constraint graphs
│
└── report/                                <- 5-Page IEEE Conference Paper
    ├── main.tex                           <- Complete LaTeX source code (IEEE twocolumn)
    └── figures/                           <- 10 high-resolution publication figures
        ├── heal_ws.png
        ├── heal_outcomes.png
        ├── heal_timing.png
        ├── heal_constraints.png
        ├── heal_success.png
        ├── franka_table.png
        ├── franka_slices.png
        ├── franka_success.png
        ├── franka_failures.png
        └── franka_constraints.png
```

---

## 🛠️ Step-by-Step Task Breakdown & Execution

### 1. Prerequisites & Environment Setup
Ensure your virtual environment is active with MuJoCo, Mink, and OSQP:
```bash
source ~/mujoco_env/bin/activate
pip install mujoco mink qpsolvers osqp imageio-ffmpeg matplotlib pandas
```

---

### 2. HEAL 6-DOF Arm Workflows (`Lab_3/HEAL/`)

#### A. Workspace Analysis (Tasks 5, 6, 7)
Computes the 3D reachable and dexterous workspace over the tabletop domain ($x \in [0.15, 0.35]$, $y \in [-0.45, -0.05]$, $z \in [0.30, 0.40]$):
```bash
cd Lab_3/HEAL
python analyze_heal_task_workspace.py
```
* Generates 3D voxel heatmaps in `heal_workspace_results/heal_dexterous_workspace_3d.png`.

#### B. Pick-and-Place Baseline & Batch Benchmark (Tasks 8, 9, 10)
Runs 25 randomized episodes with procedural cube spawning and logs outcomes:
```bash
cd Lab_3/HEAL
python task10_batch_demonstrations.py
```
* Logs written to `task10_batch/batch_logs.csv`.
* Screenshots saved to `task10_batch/screenshots/`.

#### C. Failure Taxonomy & Root Cause Analysis (Task 11)
Analyzes failure modes, contact forces, and yaw torques:
```bash
cd Lab_3/HEAL
python task11_failure/failure_analysis.py
```
* Generates outcome distribution and spatial scatter plots in `task11_failure/plots/`.

#### D. Custom IK Solvers & Comparative Benchmark (Tasks 12, 13)
Executes a 75-episode Monte Carlo benchmark comparing Mink, DLS, and QP:
```bash
cd Lab_3/HEAL
python task13_comparison/compare_ik_methods.py
```
* Output plots saved in `task13_comparison/plots/`.

---

### 3. Franka Emika Panda 7-DOF Workflows (`Lab_3/Franka/`)

#### A. Workspace Reachability & Table Design
Evaluates the 7-DOF dexterous workspace using Monte Carlo forward kinematics and sliced grid projections:
```bash
cd Lab_3/Franka
python workspace.py
```
* Output plots: `results/ws_table.png` and `results/ws_slices.png`.

#### B. Pick-and-Place Execution
Run interactive pick-and-place with selectable solvers (`mink`, `pinv`, `dls`, `qp`):
```bash
cd Lab_3/Franka
python pick_place.py --solver mink --episodes 5
```

#### C. Comprehensive 300-Episode Benchmark (Nominal, Stress, Naive)
Runs 3 regimes $\times$ 4 solvers $\times$ 25 episodes:
```bash
cd Lab_3/Franka
python compare.py
```
* Results aggregated in `results/summary.csv` and plotted in `results/plots/`.

---

## 📑 5-Page IEEE Conference Paper (`Lab_3/report/`)

The accompanying IEEE-style research report is located in `Lab_3/report/main.tex`. It adheres strictly to the 5-page suggested structure:
1. **Introduction & Goals**: Manipulator kinematics, degrees of freedom, and problem definition.
2. **Methods**: Kinematic modeling, workspace characterization, contact mechanics, and IK formulations (DLS, QP-IK, Mink).
3. **Results**: Metric comparisons, convergence iterations, solve latencies, and workspace coverage.
4. **Failure Analysis & Mitigations**: Contact dynamics of parallel-jaw grippers, pad stick-slip, and trajectory optimization.
5. **Discussion & Conclusions**: Practical engineering tradeoffs between analytical DLS, convex QP, and task-space optimization.

To compile the LaTeX report to PDF:
```bash
cd Lab_3/report
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

---

## 🔗 Related Coursework Links
* [Course Repository](https://github.com/Dhrumil0309/ME-639-Introduction-to-Robotics)
* [Homework 1: Foundations & Rotations](../homework%201/README.md)
* [Lab 3 Notion Full Deliverables Page](https://sour-grain-a01.notion.site/LAB_3-ITR-3f718e510a158056ab86d72f7af0095c?source=copy_link)
