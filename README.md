# ME 639 – Introduction to Robotics (AY 2026–27)

Instructor: Prof. Madhu Vadali | IIT Gandhinagar

Repository containing coursework, simulation models, and homework implementations for **ME 639: Introduction to Robotics**.

---

## Repository Structure

```text
ME-639-Introduction-to-Robotics/
├── README.md                                  <- Main repository guide & run instructions
├── homework 1/
│   ├── ME_639_Homework_1.pdf                 <- Official Homework 1 assignment prompt
│   ├── code-templates/
│   │   └── hw01-mujoco-rotations/
│   │       ├── model/
│   │       │   └── asymmetric_body.xml       <- MuJoCo 3D asymmetric dart model with frame axes
│   │       ├── scripts/
│   │       │   ├── 01_rotation_sandbox.py     <- Problem 7: Current vs. Fixed frame rotation sandbox
│   │       │   ├── 02_verify_skew_properties.py <- Problem 8: Time-varying skew identity verification
│   │       │   ├── skew_verification_residuals.csv <- Output residuals logged during simulation
│   │       │   └── utils.py                  <- Rotation matrices, quat conversion & hat/vee operators
│   │       ├── requirements.txt
│   │       └── README.md
│   └── ros_ws/                               <- Problem 9: ROS 2 Humble workspace
│       ├── README.md
│       └── src/
│           └── hw01_tf_demo/
│               ├── hw01_tf_demo/
│               │   └── tf_broadcaster_node.py <- TF broadcaster with live parameter & keyboard toggle
│               ├── package.xml
│               └── setup.py
```


---

## Homework 1: Foundations & Rotations

### Quick Links to Problem Files
* 📄 **Assignment Prompt**: [`homework 1/ME_639_Homework_1.pdf`](./homework%201/ME_639_Homework_1.pdf)
* 🎯 **3D MuJoCo Model**: [`homework 1/code-templates/hw01-mujoco-rotations/model/asymmetric_body.xml`](./homework%201/code-templates/hw01-mujoco-rotations/model/asymmetric_body.xml)
* 🔄 **Problem 7 Simulation**: [`homework 1/code-templates/hw01-mujoco-rotations/scripts/01_rotation_sandbox.py`](./homework%201/code-templates/hw01-mujoco-rotations/scripts/01_rotation_sandbox.py)
* Visual:[**Youtube link**](https://youtu.be/LcqWxAOiutM).
* 📐 **Problem 8 Simulation**: [`homework 1/code-templates/hw01-mujoco-rotations/scripts/02_verify_skew_properties.py`](./homework%201/code-templates/hw01-mujoco-rotations/scripts/02_verify_skew_properties.py)
* 📊 **Problem 8 Residuals CSV**: [`homework 1/code-templates/hw01-mujoco-rotations/scripts/skew_verification_residuals.csv`](./homework%201/code-templates/hw01-mujoco-rotations/scripts/skew_verification_residuals.csv)
* 🤖 **Problem 9 ROS 2 Node**: [`homework 1/ros_ws/src/hw01_tf_demo/hw01_tf_demo/tf_broadcaster_node.py`](./homework%201/ros_ws/src/hw01_tf_demo/hw01_tf_demo/tf_broadcaster_node.py)

---

## How to Run the Simulations

### Prerequisites
Activate your Python environment with MuJoCo and NumPy:
```bash
source ~/mujoco_env/bin/activate
```

---

### Problem 7: Rotation Sandbox (Does Order Matter?)

Run the interactive sandbox to apply elemental rotations ($R_x, R_y, R_z$) about the **current (body) frame** or **fixed (space) frame**:

```bash
cd "homework 1/code-templates/hw01-mujoco-rotations/scripts"

# 1. Compare Current vs. Fixed frame mathematically and visually:
python 01_rotation_sandbox.py --compare

# 2. Interactive mode (specify axis, angle in degrees, and frame step-by-step):
python 01_rotation_sandbox.py --interactive

# 3. Default animated demo (90° about z, then 90° about x in current frame):
python 01_rotation_sandbox.py
```

#### Mathematical Composition Rules:
* **Current (Body) Frame**: Post-multiplication (on the right):
  $$R_{\text{new}} = R_{\text{old}} \, R_{\text{step}}$$
* **Fixed (Space) Frame**: Pre-multiplication (on the left):
  $$R_{\text{new}} = R_{\text{step}} \, R_{\text{old}}$$

---

### Problem 8: Verifying Skew-Symmetric Properties in Simulation

Spins the asymmetric body in MuJoCo under a continuously **time-varying angular velocity** $\omega(t)$ and verifies two fundamental Lie group identities at multiple simulation steps:

1. **Cross-Product Preservation**:
   $$R(v \times w) = (R v) \times (R w)$$
2. **Adjoint / Skew Similarity**:
   $$R \, \hat{\omega} \, R^T = \widehat{(R \omega)}$$

Run the verification:
```bash
cd "homework 1/code-templates/hw01-mujoco-rotations/scripts"
python 02_verify_skew_properties.py
```

* Results are printed to the terminal and automatically exported to `skew_verification_residuals.csv`.
* Residuals hold to near machine epsilon ($\sim 10^{-16}$).

---

### Problem 9: ROS 2 TF Broadcaster & RViz2 Visualization

Broadcasts `space_frame` (static origin) and `body_frame` (moving orientation) for the rotation sequence with dynamic live switching between Current and Fixed frame composition.

#### 1. Build the ROS 2 Workspace:
```bash
cd "homework 1/ros_ws"
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

#### 2. Run the TF Broadcaster Node:
```bash
ros2 run hw01_tf_demo tf_broadcaster_node
```

#### 3. Visualize in RViz2:
In a new terminal:
```bash
source /opt/ros/humble/setup.bash
rviz2
```
* In RViz2, set **Fixed Frame** (top-left panel) to `space_frame`.
* Click **Add** -> select **TF** display. You will see `space_frame` and `body_frame` axes animating live.

#### 4. Live Toggle Between Current vs. Fixed Frame:
While the node and RViz2 are running, you can toggle the composition rule at any time:

* **Via ROS 2 Parameter**:
  ```bash
  # Switch to Fixed frame composition:
  ros2 param set /hw01_tf_broadcaster compose_frame fixed

  # Switch back to Current frame composition:
  ros2 param set /hw01_tf_broadcaster compose_frame current
  ```
* **Via Keyboard**:
  In the terminal running `tf_broadcaster_node`, type `t` (or `c` / `f`) and press Enter to toggle live.
