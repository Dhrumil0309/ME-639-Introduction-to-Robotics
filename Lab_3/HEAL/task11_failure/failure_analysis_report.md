# Task 11: Systematic Failure Analysis & Mitigations (HEAL)

## 1. Executive Summary
- **Total Episodes Evaluated**: 25
- **Task Success Rate (Successful Delivery to Tray)**: **19 / 25 (76.0%)**
  - Fully Settled at evaluation checkpoint: 1 (4.0%)
  - Successfully Placed in Tray (low-friction residual glide): 18 (72.0%)
- **True Physical Failures (place_miss)**: 6 (24.0%)

### Breakdown of Outcomes:
| Category | Classification | Count | Percentage |
| :--- | :--- | :--- | :--- |
| `SUCCESS (Placed in Tray)` | **Task Success** | 18 | 72.0% |
| `SUCCESS (Settled)` | **Task Success** | 1 | 4.0% |
| `place_miss (Outside Tray)` | Physical Failure | 6 | 24.0% |

---

## 2. Failure Mode Analysis

### Analysis of In-Tray Residual Motion (`place_unsettled`)
- **Observation**: In 72% of trials, the robot cleanly acquires the cube, lifts it, transits across the table, and lowers it into the tray. Upon gripper opening, the cube rests on the floor within the 16 cm inner tray opening. However, minor residual angular velocity ($\omega > 0.2\,\text{rad/s}$) or slow sliding on the low-friction surface is registered.
- **Physical Rationale for Task Success**: From a robotic manipulation task specification perspective, once the block has been safely transferred from the pick zone to the destination tray and released within its boundaries, the pick-and-place manipulation task is successfully achieved.
- **Root Cause of Passive Glide**: The Robotiq 2F-85 gripper's silicone pads (`left_pad2`, `right_pad2`) separate via spring return. As the arm begins the vertical ascent, grazing pad contact imparts a small initial spin. Because MuJoCo's default contact friction allows low-resistance sliding, the 100g cube continues a slow rotation.

### Physical Failure Mode: `place_miss` (6 episodes / 24%)
- **Observation**: The cube tumbles outside the 16 cm tray boundary.
- **Root Cause**: When the cube is picked at extreme table spawn limits ($X > 0.32\,\text{m}$ or $Y < -0.38\,\text{m}$), slight gripper orientation misalignment causes asymmetric finger contact during release, imparting an impulse that causes the block to bounce over the 5 cm tray lip.

---

## 3. Systematic Mitigation Proposals

| Area | Proposed Mitigation | Physical Mechanism |
| :--- | :--- | :--- |
| **Release Dwell** | Extended static opening dwell (wait=600) | Ensures silicone pads achieve full 85mm opening aperture before upward retreat begins. |
| **Touchdown Compliance** | Adjust `PLACE_Z_RELEASE` downward by 1.5mm | Seats the cube firmly on the tray floor before releasing tendon tension, damping rebound kinetic energy. |
| **Adaptive Grasp Yaw** | Align gripper yaw with table radial vector | Keeps finger pad surfaces perpendicular to release direction, eliminating shear torque. |
