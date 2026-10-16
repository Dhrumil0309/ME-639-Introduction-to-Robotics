# Step-by-step guide – Tutorial 4: Pick and Place with IK

This guide follows the assignment in order. For every step it says **what you
do**, **why**, and **which code does it**. Read a step, run its command, look at
the output, and only then move on. The tutor warns that this code will be reused
in later assignments, so the goal is that you can explain every function.

---

## Step 0 – The big picture in one paragraph

A robot arm is a chain of 7 rotating joints. If you know the 7 angles **q**, you
can compute where the gripper is; that is **forward kinematics (FK)**. The task
needs the reverse: you know where the cube is, so where must the joints be?
That is **inverse kinematics (IK)**. IK has no simple formula for a 7-joint arm,
so we solve it **iteratively**: guess **q**, see how wrong the gripper pose is,
nudge **q** in the direction that reduces the error, and repeat. The four IK
methods differ only in how they compute that nudge.

```
cube pose ──► plan gripper poses ──► IK (q for each pose) ──► send q to motors ──► physics ──► check result
```

---

## Step 1 – Set up the environment (Prerequisites, Section 3)

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
python ik_solvers.py
```

The last command should print four lines ending in `converged=True`. If it does,
MuJoCo, Mink and the QP solver all work.

**Robot model:** we use the Franka Panda from Google DeepMind's
`mujoco_menagerie` (in `assets/panda/`). It has 7 arm joints and a 2-finger
gripper. Every joint has a **position actuator**, so you send a target angle and
an internal PD controller drives the joint there.

**For the report:** the Mink repository is https://github.com/kevinzakka/mink.

---

## Step 2 – Build the scene (Section 7) → `scene.py`

Instead of writing a long XML file, we load the Panda XML and **add** objects to
it from Python using `mujoco.MjSpec`:

| What | Why |
|---|---|
| `tcp` site 10.34 cm below the hand | the **Tool Centre Point**, the point between the finger pads. Every IK target refers to this point. |
| table (top + 4 legs) | size from `config.py` (chosen in Step 4) |
| tray = floor + 4 thin walls | the place target |
| cube with a **free joint** | a free joint gives 6 DoF, so physics can move it, drop it or let it slip |
| `gravcomp = 1` on the robot links | gravity compensation, which real Panda controllers also do. Without it the arm sagged 3–8 mm below every target and nearly touched the table. |
| higher friction on fingers/cube | so the grasp holds |

`build_model()` has switches because we need three versions of the world:

* `cube=True, table=True` – the full simulation.
* `cube=False, table=True` – the "imagination" model the IK uses. It needs the
  table and tray for collision checks, but not the moving cube.
* `cube=False, table=False` – the bare robot, for workspace analysis.

**Try it:** `python pick_place.py --method mink --episodes 2 --viewer`

---

## Step 3 – Kinematics basics → `kinematics.py`

### 3a. FK
```python
kin.set_q(q)            # put joints at q, mj_kinematics computes all link poses
pos, quat = kin.tcp_pose()
```

### 3b. Jacobian
`J` (6×7) answers: *if the joints move by dq, how does the tool move?*
`dx = J dq`. The top 3 rows are linear velocity and the bottom 3 rows are
angular velocity. MuJoCo computes it for us with `mj_jacSite`.

### 3c. Orientation error (the "Quaternion computations" part of the notes)
Position error is just `p_target − p_current`. Orientation error needs quaternions:

```python
qe = q_target ⊗ q_current⁻¹          # rotation still missing
if qe.w < 0: qe = -qe                # hemisphere continuity: take the short way round
θ = 2·arccos(qe.w);  error = θ · qe.xyz / sin(θ/2)    # axis × angle
```
⚠ **MuJoCo stores quaternions as `[w, x, y, z]` (scalar first)**, but the notes
use `[x, y, z, w]`. That is why the code checks `qe[0]` for w. Mixing the two
orders is the most common bug.

### 3d. "Gripper pointing down" with a yaw
`quat_down(yaw)` builds the rotation *rotate π about x (flip the gripper to face
down), then rotate `yaw` about the vertical*. At the home pose the yaw is π/2.

---

## Step 4 – Workspace analysis and table design (Sections 5–6) → `workspace.py`

```bash
python workspace.py          # ~9 min on 2 cores
```

**What it does.** It tests a 5 cm grid of (x, y) points at 10 heights. At each
point it asks the QP-IK, which respects joint limits, to reach the point with the
gripper pointing **down** at **8 yaw angles** (every 45°). A point counts only if:
1. IK converged, **and**
2. all joints are inside their limits, **and**
3. the elbow/wrist links stay **5 cm above the table** of that height. Without
   this rule a "table" at floor level won, because the solver let the elbow pass
   straight through it.

* **Task workspace** = at least 1 yaw works (light blue in the plots).
* **Dexterous workspace** = all 8 yaws work (dark blue). A cube anywhere in
  this region can be grabbed whatever its rotation.

**Choosing the table.** For each candidate height H (0, 0.1, 0.2, 0.3, 0.4 m) we
need the grasp height (H + 2 cm, the cube centre) **and** the pre-grasp/transit
height (H + 15 cm) to be dexterous. Inside that region we find the **largest
rectangle** (a classic algorithm, `largest_rectangle`) that starts at x ≥ 0.30 m
so the table stays clear of the robot base. The biggest rectangle wins, and ties
go to the higher table.

**Result** (`results/table_design.md`):

| Parameter | Value | Why |
|---|---|---|
| H | 0.10 m | tied for the most usable area (0.38 m²) with H = 0; higher wins the tie because it is a real table and keeps the links further from the floor |
| L | 0.35 m (x 0.30 → 0.65) | largest rectangle inside the dexterous region along x; starts 0.30 m from the base |
| W | 0.90 m (y −0.45 → 0.45) | same rectangle along y |

Note that H is measured from the **robot base**. On a real setup with the robot
on a pedestal, the table top would sit 0.10 m above the robot's mounting plate.

We then placed the **tray** on the −y side and the **cube spawn region** on the
+y side, both inside the dark area (`results/ws_table.png`).

Plots to put in your report: `ws_fk_cloud.png`, `ws_slices.png`,
`ws_area_vs_h.png`, `ws_table.png`.

---

## Step 5 – The four IK solvers (Sections 9 & 12) → `ik_solvers.py`

All custom solvers share this loop. Learn it by heart:

```python
for it in range(max_iters):
    set_q(q)                               # FK
    e = pose_error(target, current)        # 6-vector
    if |e_pos| < 1 mm and |e_rot| < 0.01 rad:  return q   # converged
    J = jacobian()
    q = q + step(q, e, J)                  # <- the only part that differs
```

| Method | `step()` | Strength | Weakness |
|---|---|---|---|
| **Pinv** | `dq = J⁺ (K e)` | simplest, fastest | near a singularity a singular value σ→0, so 1/σ explodes; ignores joint limits |
| **DLS** | `dq = (JᵀJ+λI)⁻¹Jᵀ(K e)` + null-space term | stable near singularities, because each direction is scaled by σ/(σ²+λ) instead of 1/σ | needs λ tuning; joint limits are only *encouraged*, not enforced |
| **QP** | `min ½dqᵀH dq + cᵀdq  s.t.  G dq ≤ h` | **hard** joint and velocity limits | needs a QP solver; more iterations per waypoint |
| **Mink** | builds and solves its own QP from *tasks* and *limits* | quick to use, has collision avoidance | less control over the internals |

**DLS derivation** (from the notes): minimise ‖J dq − dx‖² + λ‖dq‖². Set the
gradient to zero: (JᵀJ + λI) dq = Jᵀdx. The λ term "damps" large joint motions.

**QP derivation:** substitute dq = Δq/Δt into the same cost to get
H = 2/Δt²(JᵀJ+λI) and c = −(1/Δt)Jᵀẋ_d. The constraints form a box:
`max(q_min − q, −v_max Δt) ≤ Δq ≤ min(q_max − q, v_max Δt)`, written as
`G = [I; −I]`, `h = [upper; −lower]`. `quadprog` wants the form
`min ½xᵀGx − aᵀx  s.t.  Cᵀx ≥ b`, so we pass `a = −c`, `C = −Gᵀ`, `b = −h`.

### 🔎 A lesson we hit: the null-space projector
The notes give `N = I − (JᵀJ+λI)⁻¹JᵀJ` for the secondary task (pulling joints
toward the middle of their range). With λ > 0 that matrix is **not** an exact
null space: part of the secondary motion leaks into the gripper motion. Our DLS
solver then **stalled 2 mm from the target** and never converged. Using the
exact projector `N = I − J⁺J` fixed it (converges in about 5 iterations). This
is worth a sentence in your report.

**Mink setup:**
`FrameTask("tcp")` is the gripper pose we want, and `PostureTask` is a weak pull
to the home posture. The limits are `ConfigurationLimit` (joint limits),
`VelocityLimit`, and `CollisionAvoidanceLimit` (gripper/wrist vs table + tray,
4 mm minimum distance).

---

## Step 6 – Planning one episode (Section 9) → `Planner` in `pick_place.py`

The motion is split into 6 straight-line **segments**:

```
home → PRE-GRASP (12 cm above cube) → GRASP (TCP at cube centre)  | close gripper
     → LIFT (transit height) → ABOVE TRAY → PLACE (cube 1.2 cm above tray floor) | open
     → RETREAT
```

Each segment is cut into waypoints every 1 cm (or 0.1 rad of rotation). IK is
solved for each waypoint, **warm-started** from the previous answer, so every
solve starts close to the solution. That is why most waypoints converge in 1–7
iterations. If any waypoint fails, the episode is marked `ik_infeasible`.

**Smart choices (they decide the 20 points for cube–tray collision avoidance):**

1. **Grasp yaw using cube symmetry.** A cube looks the same every 90°, so there
   are 4 valid gripper yaws. `yaw_candidates()` sorts them so the wrist
   (joint 7) stays near the middle of its range. If planning fails, the next
   yaw is tried (`--retries`). This is the "alternate grasp yaw" mitigation.
2. **Transit height above the tray rim.**
   `TRANSIT_Z = H + max(wall + half cube + 5 cm margin, pre-grasp height)`, so
   the bottom of the carried cube always passes ≥ 5 cm over the rim. The motion
   is *up → across → straight down*, never diagonally over a wall.
3. **Place yaw aligned with the tray.** The cube is held square to the fingers,
   so we rotate to the nearest multiple of 90°. Its edges then line up with the
   walls, using the least wrist rotation.
4. **Planned collision check.** Every few waypoints we compute the distance from
   gripper/wrist to table/tray (`mj_geomDistance`). Negative means penetration,
   and the episode is marked `planned_collision`.

---

## Step 7 – Executing with physics and checking (Sections 9–10) → `Sim`

`track()` sends the joint waypoints to the position actuators, linearly
interpolated, at a speed that moves the TCP at about 0.2 m/s. Then MuJoCo physics
runs at 500 Hz. The grasp is **real friction**, so the cube can slip or be
knocked off.

Checks every step:

| Check | Failure label |
|---|---|
| robot link touches table/tray/floor | `robot_collision` |
| fingers fully closed after grasp (nothing between them) | `grasp_miss` |
| cube not lifted | `grasp_slip` |
| cube more than 3 cm from TCP during transit | `dropped_in_transit`, or `cube_tray_collision` if it touched a tray wall |
| final cube not inside the tray | `place_miss` |
| IK did not converge | `ik_infeasible (jointN at limit …)` |

Logged per episode (`results/logs_<method>.csv`): initial pose, success, failure
reason, planning time, IK time, IK iterations (mean/max per waypoint), joint-limit
violations, velocity violations, max joint speed / limit, min clearance to
obstacles, min cube–tray clearance, cube–tray contact, grasp/place yaw, and final
cube position.

```bash
python pick_place.py --method mink --episodes 25        # Stage 1 batch
```

---

## Step 8 – Fair comparison (Section 13) → `compare.py`

```bash
python compare.py --run
```

**Fair** means the same seeds: episode *i* always uses seed *i*, so all 4
methods see exactly the same cubes. Three experiments:

| Experiment | Cube spawn | Smart choices | Purpose |
|---|---|---|---|
| nominal | designed region | ON | the deliverable behaviour |
| stress | anywhere on the table | ON | does the design hold at the edges? |
| naive | anywhere | **OFF** (`--naive`) | creates failures to analyse |

### Results (25 episodes per cell)

| | Mink | Pinv | DLS | QP |
|---|---|---|---|---|
| nominal success | 100 % | 100 % | 100 % | 100 % |
| stress success | 100 % | 100 % | 100 % | 100 % |
| naive success | 16 % | 24 % | 28 % | 16 % |
| planning time (nominal) | 57 ms | 38 ms | 46 ms | 85 ms |
| IK iters / waypoint | 1.0 | 2.0 | 2.1 | 5.9 |
| joint-limit violations (naive) | 0 | **413** | **375** | 0 |
| max joint speed / limit (naive) | **3.27** | 0.81 | 0.81 | 1.44 |

**How to read it:**

* With the smart choices, **every method succeeds**. The design (workspace →
  table → grasp yaw → transit height) matters more than the IK method.
* With the smart choices off, the methods fail **differently**:
  * **Pinv and DLS** never refuse. They just plan **through the joint limits**
    (hundreds of waypoints outside the range). The motors clamp those angles, so
    the real arm does something slightly different from the plan.
  * **QP and Mink** respect the limits, so they report `ik_infeasible
    (joint7 at limit)` instead. Failing honestly is safer than violating limits.
    Mink also fails where its **collision-avoidance limit** refuses to drag the
    hand through the tray rim.
  * Mink once **jumped to another IK branch** (3.3× the joint speed limit),
    because a single solve can hop between solutions.
  * The main naive failure for every method is `cube_tray_collision`. Transit at
    6 cm drags the cube into the rim. This is direct evidence for smart choice #2.
* Why does Mink need only 1 iteration per waypoint? Its frame task corrects the
  full error in one QP step. QP-IK needs about 6 because its per-iteration
  velocity limit caps the step size.

Plots: `cmp_success.png`, `cmp_iters_time.png`, `cmp_failures.png`,
`cmp_constraints.png`, `cmp_positions.png`.

---

## Step 9 – Failure analysis (Section 11)

Screenshots of every failure are in `results/failures/`. Pick one per category:

| Category | Cause | Mitigation (already in the code) |
|---|---|---|
| cube_tray_collision | transit too low | transit height = rim + cube + margin, motion up → across → down |
| ik_infeasible (joint7 at limit) | grasp yaw needs too much wrist rotation | 90° cube symmetry: pick the yaw nearest the wrist's mid-range, retry others |
| ik_infeasible (Mink, no convergence) | collision limit blocks the path | better approach direction / higher via-point |
| joint-limit violations (DLS/pinv) | method has no hard constraints | null-space centring (DLS), or use the QP |
| velocity spike (Mink) | IK branch switch | warm start, smaller waypoint step, posture task |
| grasp slip (seen during development) | arm sagging under gravity | gravity compensation, higher pad friction |

---

## Step 10 – Video (Section 15, "Important")

```bash
bash make_videos.sh       # Windows: run the python lines inside it by hand
```

This creates `videos/mink_1min.mp4`, `dls_1min.mp4`, `qp_1min.mp4` (+ `pinv_1min.mp4`),
`failures_naive.mp4`, and **`three_methods_3min.mp4`** (the required 3-minute
video). Each clip shows the TCP **trail**: the current episode is bright and
previous episodes are faded, so different trials are visible.

---

## Step 11 – Submit (Google Classroom: repository link)

1. Create a GitHub repository and push this folder (code + README + results + videos).
   Videos over 100 MB need Git LFS, or link them from Google Drive.
2. Write the report (max 5 pages). Start from `REPORT_DRAFT.md`, but **rewrite it
   in your own words** and add your understanding.
3. Submit: repo link, report, video.

---

## Questions you should be able to answer

1. What is the difference between task and dexterous workspace? How did you compute them?
2. Why is H = 0.10 m? What would go wrong with H = 0.40 m?
3. Write the DLS formula. What does λ do to a singular value σ?
4. Why did the damped null-space projector stall, and what fixed it?
5. Why does the QP have box constraints? Where do the velocity limits come from?
6. Why can a cube be grasped at 4 different yaws? How do you choose one?
7. How do you guarantee the cube does not hit the tray?
8. What does "identical seeds" mean, and why does it make the comparison fair?
9. Why do DLS/pinv show joint-limit violations but QP/Mink don't?
10. What is the TCP and why is it 10.34 cm from the hand?
