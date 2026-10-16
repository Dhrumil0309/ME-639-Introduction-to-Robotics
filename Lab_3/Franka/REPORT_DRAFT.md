# Pick and Place in MuJoCo with Inverse Kinematics — Report (draft)

> Draft with the measured numbers filled in. Rewrite in your own words, add your
> name/roll number, and keep it within 5 pages. Figures are in `results/`.

## 1. Introduction and goals
A Franka Emika Panda (7-DoF, parallel gripper) must pick a 4 cm cube placed at a
random position and yaw on a table and place it in a tray. Goals: (i) analyse the
task and dexterous workspace and design the table from it; (ii) build a
randomized, logged pick-and-place pipeline with an off-the-shelf IK library (Mink,
https://github.com/kevinzakka/mink); (iii) replace it with our own IK solvers
(pseudo-inverse, damped least squares, QP with joint/velocity limits) and compare
all methods on identical episodes; (iv) analyse failures and propose mitigations.

## 2. Methods
**Environment.** MuJoCo 3.x, Panda model from mujoco_menagerie, position actuators
with gravity compensation, 2 ms timestep. The scene (table, tray of 16×16 cm with
5 cm walls, free-floating cube of 50 g) is generated with `MjSpec` from parameters
in `config.py`. A TCP site is defined 10.34 cm below the hand flange.

**Workspace analysis.** For a 5 cm grid at 10 TCP heights, a top-down grasp was
solved at 8 yaws (45° steps) with the QP-IK. A point counts only if IK converges
inside the joint limits and the elbow/wrist links stay 5 cm above the candidate
table. Task workspace = ≥1 feasible yaw; dexterous = all 8 (Fig. `ws_slices.png`).
For table heights H ∈ {0, 0.1, 0.2, 0.3, 0.4} m we intersected the dexterous maps
at grasp (H+0.02) and transit (H+0.15) heights and found the largest inscribed
rectangle with x ≥ 0.30 m (Fig. `ws_area_vs_h.png`).

| Parameter | Value | Rationale |
|---|---|---|
| L | 0.35 m (x 0.30–0.65) | largest dexterous rectangle along x, clear of the base |
| W | 0.90 m (y −0.45–0.45) | same rectangle along y |
| H | 0.10 m | largest usable rectangle (0.38 m², tied with H = 0; higher chosen for link clearance); H ≥ 0.2 shrinks the area to 0.34 → 0.23 m² |

**Randomisation.** Episode *i* uses seed *i*: x ∈ [0.34, 0.61], y ∈ [−0.14, 0.40],
yaw ∈ [−π, π] (nominal); the stress test samples the whole table except the tray.

**Planning and execution.** Six Cartesian segments (pre-grasp 12 cm above the
cube → grasp → lift → above tray → place → retreat) are discretised at 1 cm /
0.1 rad and solved by IK with warm start. Grasp yaw exploits the cube's 90°
symmetry, choosing the candidate that keeps joint 7 near mid-range (with retries).
The transit height keeps the cube bottom ≥ 5 cm above the tray rim and the
place yaw aligns the cube with the walls. Planned configurations are checked for
gripper–table/tray distance (`mj_geomDistance`); during execution all contacts
and clearances are monitored and the outcome is classified.

**IK methods.** Common loop: pose error e = [Δp; ε_ω] with ε_ω from the
quaternion error q_t ⊗ q_c⁻¹ (hemisphere-corrected, log map); stop at 1 mm /
0.01 rad.
* Pseudo-inverse: Δq = J⁺Ke.
* DLS: Δq = (JᵀJ+λI)⁻¹JᵀKe + (I − J⁺J)·α(q_mid − q), λ = 0.01, K = 0.8, α = 0.05.
  The damped projector I − (JᵀJ+λI)⁻¹JᵀJ from the notes leaked into the task
  space and made the solver stall ≈2 mm from the target; the exact projector fixed it.
* QP: min ½ΔqᵀHΔq + cᵀΔq with H = 2/Δt²(JᵀJ+λI), c = −(1/Δt)Jᵀẋ_d, subject to
  joint limits (0.03 rad margin) and |Δq| ≤ v_max·Δt (Δt = 0.05 s), solved with quadprog.
* Mink: FrameTask + PostureTask, configuration, velocity and collision-avoidance
  limits, DAQP solver.

## 3. Results
25 episodes per method and experiment, same seeds (Fig. `cmp_success.png`).

| | Mink | Pinv | DLS | QP |
|---|---|---|---|---|
| Success – nominal | 25/25 | 25/25 | 25/25 | 25/25 |
| Success – stress (whole table) | 25/25 | 25/25 | 25/25 | 25/25 |
| Success – naive (smart choices off) | 4/25 | 6/25 | 7/25 | 4/25 |
| Mean planning time / episode (nominal) | 57 ms | 38 ms | 46 ms | 85 ms |
| IK iterations / waypoint (mean, max) | 1.0, 1 | 2.0, 2 | 2.1, 3 | 5.9, 7 |
| Min gripper clearance to table/tray | 11.3 mm | 11.6 mm | 11.5 mm | 11.9 mm |
| Min cube–tray clearance (nominal) | 54 mm | 54 mm | 54 mm | 54 mm |
| Joint-limit violations, naive (waypoints) | 0 | 413 | 375 | 0 |
| Max joint speed / limit, naive | 3.27 | 0.81 | 0.81 | 1.44 |

No cube–tray contact occurred in any nominal or stress episode.

## 4. Failure analysis and mitigations
Failures were generated with the naive configuration (Fig. `cmp_failures.png`,
screenshots in `results/failures/`).
1. **Cube–tray collision** (the main mode for every method): with a 6 cm transit
   the cube is dragged into the rim and knocked out of the grip. *Mitigation:*
   transit height from rim + cube + margin, vertical descent into the tray (0 contacts).
2. **IK infeasible at joint limits** (Mink, QP): using the raw cube yaw demands
   wrist angles outside joint 7's range. *Mitigation:* the 4 symmetric grasp
   yaws, ordered by predicted wrist angle, with retries.
3. **Joint-limit violation** (Pinv, DLS): unconstrained methods plan beyond the
   limits; the actuators clamp, so execution deviates from the plan.
   *Mitigation:* null-space centring or hard constraints (QP).
4. **Collision-limit blocking** (Mink): the collision-avoidance limit refuses to
   move the hand through the rim → no convergence. *Mitigation:* higher via-point
   or a different approach direction.
5. **IK branch jump** (Mink, one episode at 3.3× the speed limit). *Mitigation:*
   smaller waypoint spacing, stronger posture regularisation, velocity check before execution.
6. **Gravity sag** (development): without gravity compensation the arm tracked
   3–8 mm low, giving 2 mm finger–table clearance. *Mitigation:* gravity compensation.

## 5. Discussion and conclusions
With a workspace-driven table design and the grasp/transit strategy, all four IK
methods achieve 100 % success. The choice of IK method mattered far less than the
task design. The methods differ in *how* they fail: unconstrained
methods (pinv, DLS) silently violate joint limits; constrained ones (QP, Mink)
fail explicitly, which is safer. DLS is the most robust unconstrained method;
QP costs ~2× the planning time of DLS but guarantees limits; Mink is fastest to
integrate and adds collision avoidance, but gives less control and showed one
branch jump. Future work: collision constraints inside our own QP, orientation-
aware approach directions for cubes near the reach boundary, and sim-to-real
gripper modelling.
