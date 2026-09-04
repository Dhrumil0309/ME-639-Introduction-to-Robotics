# ME-639: Introduction to Robotics — Homework 1

This repository contains the complete implementation and mathematical verification for **Homework 1** of **ME-639: Introduction to Robotics** (IIT Gandhinagar). The assignment explores the kinematics of 3D rotations, non-commutativity on $SO(3)$, skew-symmetric matrices ($\mathfrak{so}(3)$), coordinate frame transformations, MuJoCo physics simulation, and ROS 2 TF/Interactive Marker visualization.

---

## Table of Contents
1. [General Setup & Prerequisites](#general-setup--prerequisites)
   - [Repository Setup](#repository-setup)
   - [MuJoCo Python Environment](#mujoco-python-environment)
   - [ROS 2 Humble Workspace](#ros-2-humble-workspace)
2. [Question 7 (Task 1): MuJoCo Rotation Sandbox](#question-7-task-1---mujoco-rotation-sandbox)
   - [How to Run](#how-to-run-task-1)
   - [In-Viewer Keyboard Controls & Custom Angle Buffer](#in-viewer-keyboard-controls--custom-angle-buffer)
   - [Kinematics & Theoretical Foundations](#kinematics--theoretical-foundations-task-1)
3. [Question 8 (Task 2): Skew-Symmetric Matrix Properties](#question-8-task-2---skew-symmetric-properties)
   - [How to Run](#how-to-run-task-2)
   - [Interactive Controls & Side-by-Side Proofs](#interactive-controls--side-by-side-proofs)
   - [Rigorous Mathematical Proofs (LaTeX)](#rigorous-mathematical-proofs)
   - [Numerical Residuals vs. Continuous Proofs](#numerical-residuals-vs-continuous-proofs)
4. [Question 9 (Task 3): ROS 2 TF Broadcaster & RViz2 Interactive Markers](#question-9-task-3---ros-2-tf-broadcaster)
   - [How to Build & Launch](#how-to-build--launch-task-3)
   - [3D Interactive Markers & RViz2 Context Menu](#3d-interactive-markers--rviz2-context-menu)
   - [Kinematic Processing & Frame Composition](#kinematic-processing--frame-composition)

---

## General Setup & Prerequisites

### Repository Setup
Clone the repository and enter the project directory:
```bash
git clone https://github.com/Dhrumil0309/ME-639-Introduction-to-Robotics.git
cd ME-639-Introduction-to-Robotics
```

### MuJoCo Python Environment
Tasks 1 and 2 utilize the native Python bindings for **MuJoCo 3.x** and **NumPy**.
If you already have a virtual environment configured (e.g. `~/mujoco_env`):
```bash
source ~/mujoco_env/bin/activate
```
Alternatively, to create a fresh virtual environment:
```bash
python3 -m venv ~/mujoco_env
source ~/mujoco_env/bin/activate
pip install --upgrade pip
pip install mujoco numpy
```

### ROS 2 Humble Workspace
Task 3 uses **ROS 2 Humble** and `interactive_markers`. Ensure ROS 2 Humble is installed and sourced:
```bash
source /opt/ros/humble/setup.bash
sudo apt-get install ros-humble-interactive-markers ros-humble-rviz2 ros-humble-tf2-ros
```

---

## Question 7 (Task 1) - MuJoCo Rotation Sandbox

The rotation sandbox ([`01_rotation_sandbox.py`](file:///home/dhrumil/Homework_1/ME-639-Introduction-to-Robotics/homework%201/code-templates/hw01-mujoco-rotations/scripts/01_rotation_sandbox.py)) provides a 3D interactive viewer in MuJoCo for rotating an asymmetric rigid body (a dart model). All interaction occurs directly inside the MuJoCo passive viewer window with live on-screen matrix and status text overlays.

### How to Run (Task 1)
```bash
cd "homework 1/code-templates/hw01-mujoco-rotations/scripts"
source ~/mujoco_env/bin/activate
python 01_rotation_sandbox.py
```
*Optional argument:*
- `--step-duration <seconds>`: Control SLERP interpolation duration (default: `1.2` seconds).

### In-Viewer Keyboard Controls & Custom Angle Buffer
All keypresses are captured directly by GLFW event callbacks in the viewer:

| Keys | Function | Mathematical Formulation |
| :--- | :--- | :--- |
| **`1`**, **`2`**, **`3`** | Rotate $\pm\theta$ about **Fixed (Space)** $X, Y, Z$ axes | $R_{\text{new}} = R_{\text{step}} \, R_{\text{old}}$ (Pre-multiplication) |
| **`4`**, **`5`**, **`6`** | Rotate $\pm\theta$ about **Current (Body)** $X, Y, Z$ axes | $R_{\text{new}} = R_{\text{old}} \, R_{\text{step}}$ (Post-multiplication) |
| **`0` – `9`** | **Dynamic Angle Buffer**: Type any angle in degrees | Live on-screen buffer (e.g. typing `4`, `5` sets $45^\circ$) |
| **`-`** | Negate Angle | Toggles positive/negative angle in buffer |
| **`Backspace`** | Delete digit | Erases the last typed digit |
| **`Enter`** | Confirm angle | Locks in buffered angle without rotating |
| **`R`** | **Reset** | Restores orientation to identity $R = I$ and resets buffer |

*Note on Animation & Stability:*
- Each rotation is interpolated using **Spherical Linear Interpolation (SLERP)** across unit quaternions.
- Hemispheric sign checks ($\mathbf{q}_0 \cdot \mathbf{q}_1 < 0 \implies \mathbf{q}_1 \leftarrow -\mathbf{q}_1$) ensure shortest-path geodesics on $SO(3)$, preventing $270^\circ$ wrapping glitches.
- Sequential rotations are strictly processed one-at-a-time, eliminating animation stacking or mid-motion discontinuities.

### Kinematics & Theoretical Foundations (Task 1)

#### Non-Commutativity of Rotations
Rotations in 3D Euclidean space form the Lie group **$SO(3)$**:
$$SO(3) = \{ R \in \mathbb{R}^{3 \times 3} \mid R^T R = I, \; \det(R) = +1 \}$$
Matrix multiplication on $SO(3)$ is non-abelian (non-commutative). In general:
$$R_A R_B \neq R_B R_A$$
For example, rotating $90^\circ$ about $Z$ followed by $90^\circ$ about $X$ yields a drastically different orientation than applying $X$ followed by $Z$.

#### Pre-Multiplication vs. Post-Multiplication
Let $R_b^s \in SO(3)$ represent the rotation matrix transforming coordinates from the moving body frame $\{b\}$ to the fixed space frame $\{s\}$:
$$p_s = R_b^s \, p_b$$

1. **Rotation about the Fixed (Space) Frame (Pre-multiplication / Left):**
   If an incremental rotation $R_{\text{step}}$ is taken about the axes of the fixed space frame $\{s\}$, the new orientation is:
   $$R_{\text{new}} = R_{\text{step}} \, R_{\text{old}}$$
   *Rationale:* The rotation operates directly on coordinates already expressed in the fixed space frame $\{s\}$.

2. **Rotation about the Current (Body) Frame (Post-multiplication / Right):**
   If the incremental rotation $R_{\text{step}}$ is taken about the instantaneous axes of the moving body frame $\{b\}$, the new orientation is:
   $$R_{\text{new}} = R_{\text{old}} \, R_{\text{step}}$$
   *Rationale:* In the local coordinate frame, $p_b = R_{\text{step}} p_{b'}$, which upon substitution into $p_s = R_{\text{old}} p_b$ yields $p_s = R_{\text{old}} R_{\text{step}} p_{b'}$.

---

## Question 8 (Task 2) - Skew-Symmetric Properties

The skew-symmetric verification script ([`02_verify_skew_properties.py`](file:///home/dhrumil/Homework_1/ME-639-Introduction-to-Robotics/homework%201/code-templates/hw01-mujoco-rotations/scripts/02_verify_skew_properties.py)) spins the body model under a dynamically varying angular velocity $\omega(t) \in \mathbb{R}^3$, evaluates the live rotation matrix $R(t)$, and tests fundamental Lie algebra identities against randomly generated vectors.

### How to Run (Task 2)
```bash
cd "homework 1/code-templates/hw01-mujoco-rotations/scripts"
source ~/mujoco_env/bin/activate
python 02_verify_skew_properties.py
```
*Headless/CI mode:*
```bash
python 02_verify_skew_properties.py --headless --duration 5.0
```

### Interactive Controls & Side-by-Side Proofs
The simulation runs continuously without an arbitrary time cutoff:
- **`Spacebar`**: Toggle pause/resume of the physics simulation.
- **`Enter`**: Grabs current $R(t)$, generates independent random vectors $v, w, \omega \sim \mathcal{N}(0, I)$, and prints full side-by-side component comparisons.
- **`T`**: Manually prints a compact numerical statistical residual row.
- **`Q` or `Escape`**: Cleanly shuts down the simulation.

---

### Rigorous Mathematical Proofs

Let $R \in SO(3)$, and let $v, w, \omega \in \mathbb{R}^3$.
The skew-symmetric (hat) operator $(\;\widehat{\cdot}\;) : \mathbb{R}^3 \to \mathfrak{so}(3)$ is defined such that:
$$\widehat{\omega} = \begin{bmatrix} 0 & -\omega_3 & \omega_2 \\ \omega_3 & 0 & -\omega_1 \\ -\omega_2 & \omega_1 & 0 \end{bmatrix}, \quad \widehat{\omega} \, v = \omega \times v \quad \forall v \in \mathbb{R}^3$$

---

#### Proof of Identity 1: $R(v \times w) = (Rv) \times (Rw)$

**Method A: Using Levi-Civita / Determinant Properties**
Recall that for any $3 \times 3$ matrix $M$, the triple scalar product satisfies:
$$\det(M) \, (u \cdot (v \times w)) = (Mu) \cdot ((Mv) \times (Mw))$$
Since $R \in SO(3)$, we have $\det(R) = +1$ and $R^T R = I$ (preserves the inner product: $(Ru) \cdot (Rz) = u \cdot z$).
Let $z = v \times w$. Then:
$$u \cdot (v \times w) = (Ru) \cdot (R(v \times w))$$
Equating this with the identity above for $M = R$:
$$(Ru) \cdot (R(v \times w)) = (Ru) \cdot ((Rv) \times (Rw))$$
Since this equality holds for all $u \in \mathbb{R}^3$, the vector $Ru$ spans all of $\mathbb{R}^3$. Therefore:
$$R(v \times w) = (Rv) \times (Rw) \quad \blacksquare$$

---

#### Proof of Identity 2: $R \widehat{\omega} R^T = \widehat{(R\omega)}$

Let $x \in \mathbb{R}^3$ be an arbitrary test vector.
Consider the action of the left-hand side matrix on $x$:
$$\text{LHS} \cdot x = \left( R \widehat{\omega} R^T \right) x = R \left( \widehat{\omega} (R^T x) \right)$$
By the definition of the cross product via the skew-symmetric matrix ($\widehat{\omega} y = \omega \times y$):
$$\widehat{\omega} (R^T x) = \omega \times (R^T x)$$
Multiplying by $R$:
$$R \left( \omega \times (R^T x) \right)$$
Now, apply **Identity 1** ($R(a \times b) = (Ra) \times (Rb)$) with $a = \omega$ and $b = R^T x$:
$$R \left( \omega \times (R^T x) \right) = (R\omega) \times (R R^T x)$$
Since $R \in SO(3)$, $R R^T = I$, so $R R^T x = x$:
$$(R\omega) \times (R R^T x) = (R\omega) \times x$$
By definition of the skew-symmetric operator on the vector $(R\omega)$:
$$(R\omega) \times x = \widehat{(R\omega)} \, x$$
Thus:
$$\left( R \widehat{\omega} R^T \right) x = \widehat{(R\omega)} \, x \quad \forall x \in \mathbb{R}^3$$
Since this holds for every $x \in \mathbb{R}^3$, the matrices must be identical:
$$R \widehat{\omega} R^T = \widehat{(R\omega)} \quad \blacksquare$$

---

### Numerical Residuals vs. Continuous Proofs

When running `02_verify_skew_properties.py`, the terminal logs residuals defined by the Frobenius and Euclidean norms:
$$\epsilon_{\text{cross}} = \| R(v \times w) - (Rv) \times (Rw) \|_2, \qquad \epsilon_{\text{skew}} = \| R \widehat{\omega} R^T - \widehat{(R\omega)} \|_F$$
Across all simulated time steps, the measured residuals are on the order of:
$$\epsilon \approx 10^{-16} \sim 10^{-15}$$
which is near IEEE 754 double-precision machine epsilon ($\varepsilon_{\text{mach}} \approx 2.22 \times 10^{-16}$).

**Epistemological Distinction:**
- **Why Numerical Verification is Valuable:** Simulation confirms that implementations, matrix indexing, trigonometric libraries, and integrator representations adhere to theoretical behavior. It rules out implementation bugs, transcription errors, and coordinate convention mismatches.
- **Why it Cannot Replace Analytic Proof:** Numerical checks only evaluate identities at a *finite, discrete* set of points in the continuous manifold $SO(3) \times \mathbb{R}^3$. A test at $10^4$ states cannot mathematically guarantee that an edge-case discontinuity, singularity, or domain invalidity does not exist elsewhere on the manifold. The symbolic proof guarantees universality across the entire continuous domain; the numerical verification confirms implementation fidelity.

---

## Question 9 (Task 3) - ROS 2 TF Broadcaster

Task 3 implements a ROS 2 package ([`hw01_tf_demo`](file:///home/dhrumil/Homework_1/ME-639-Introduction-to-Robotics/homework%201/ros_ws/src/hw01_tf_demo/)) that broadcasts moving transform frames (`space_frame` and `body_frame`) and integrates **3D Interactive Markers** and context menus directly inside **RViz2**.

### How to Build & Launch (Task 3)

1. Open a terminal and source ROS 2 Humble:
   ```bash
   source /opt/ros/humble/setup.bash
   ```
2. Navigate to the ROS 2 workspace and build using `colcon`:
   ```bash
   cd "/home/dhrumil/Homework_1/ME-639-Introduction-to-Robotics/homework 1/ros_ws"
   colcon build --symlink-install
   ```
3. Source the install overlay:
   ```bash
   source install/setup.bash
   ```
4. Launch both the TF broadcaster node and RViz2 simultaneously:
   ```bash
   ros2 launch hw01_tf_demo hw01.launch.py
   ```

---

### 3D Interactive Markers & RViz2 Context Menu

Once launched, RViz2 opens pre-configured with the Grid, TF display (large axes), and the `InteractiveMarkers` display loaded on `/rotation_controls/update`.

```
                  [RViz2 3D Viewport]
                     ┌──────────┐
                     │body_frame│ (moving child: z = 0.8)
                     │ ┌──────┐ │
                     │ │  ●   │ │  <- 3D Rotation Rings (X, Y, Z)
                     │ └──────┘ │
                     └────┬─────┘
                          │
                     ┌────┴─────┐
                     │space_frame│ (fixed origin: z = 0.0)
                     └──────────┘
```

1. **Interact Tool**: Ensure the **Interact** tool is selected on the top toolbar of RViz2 (keyboard shortcut: `i`).
2. **Dragging Rotation Rings**:
   - **Red Ring (X)**, **Green Ring (Y)**, and **Blue Ring (Z)** allow intuitive 3D dragging.
   - Click and drag any ring to rotate the body frame in real-time.
3. **Right-Click Context Menu**:
   - Right-click anywhere on the marker's center or rings to bring up the context menu:
     - **`Set Mode: Current (Body) Frame`**: Checked for body-frame post-multiplication.
     - **`Set Mode: Fixed (Space) Frame`**: Checked for space-frame pre-multiplication.
     - **`Reset to Identity`**: Restores the body frame orientation back to identity $R = I$.

---

### Kinematic Processing & Frame Composition

In [`tf_broadcaster_node.py`](file:///home/dhrumil/Homework_1/ME-639-Introduction-to-Robotics/homework%201/ros_ws/src/hw01_tf_demo/hw01_tf_demo/tf_broadcaster_node.py), incremental user dragging generates relative delta rotation matrices:
$$R_{\text{delta}} = R_{\text{marker\_current}} \, R_{\text{prev\_drag}}^T$$

Depending on the mode selected in the context menu:
- **Current (Body) Frame:**
  $$R_{\text{body}} \leftarrow R_{\text{body}} \, R_{\text{delta}} \quad \text{(Post-multiplication)}$$
- **Fixed (Space) Frame:**
  $$R_{\text{body}} \leftarrow R_{\text{delta}} \, R_{\text{body}} \quad \text{(Pre-multiplication)}$$

Singular Value Decomposition (SVD) projection ($R = U V^T$) guarantees that $R_{\text{body}}$ remains strictly orthonormal on $SO(3)$ with zero numerical drift over extended dragging sessions. Transforms are broadcasted at a continuous **60 Hz** on `/tf`.

---

## File Structure

```
ME-639-Introduction-to-Robotics/
├── homework 1/
│   ├── ME_639_Homework_1.pdf
│   ├── README.md                                <- Complete documentation and proofs
│   ├── code-templates/
│   │   └── hw01-mujoco-rotations/
│   │       ├── model/
│   │       │   └── asymmetric_body.xml         <- MuJoCo dart XML model
│   │       └── scripts/
│   │           ├── 01_rotation_sandbox.py      <- Task 1 interactive rotation viewer
│   │           ├── 02_verify_skew_properties.py<- Task 2 skew-identity verification
│   │           ├── utils.py                    <- Rotation matrices & quaternion math
│   │           └── skew_verification_residuals.csv
│   └── ros_ws/
│       └── src/
│           └── hw01_tf_demo/
│               ├── CMakeLists.txt / setup.py
│               ├── package.xml
│               ├── launch/
│               │   └── hw01.launch.py          <- Launch broadcaster + RViz2
│               ├── rviz/
│               │   └── demo.rviz               <- RViz2 display configuration
│               └── hw01_tf_demo/
│                   └── tf_broadcaster_node.py  <- Interactive Marker TF broadcaster
└── README.md
```
