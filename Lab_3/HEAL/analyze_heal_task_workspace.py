"""Estimate HEAL task and dexterous workspaces and test a table proposal.

The lab requests arbitrary yaw orientations but does not prescribe a sample
resolution. This analysis uses 16 top-down yaw samples and reports that finite
orientation sampling as an approximation, not a continuous-yaw guarantee.
"""

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib
HAS_GRAPHICAL_DISPLAY = bool(
    os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
)
if not HAS_GRAPHICAL_DISPLAY:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import mujoco
import numpy as np

from analyze_workspace import MODEL_DIR, MODEL_PATH, load_model


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "heal_workspace_results"
DEFAULT_HEIGHTS = np.array([0.10, 0.20, 0.30, 0.40, 0.50])
ARM_JOINT_NAMES = [f"joint_{index}" for index in range(1, 7)]
PAD_BODY_NAMES = ("left_pad", "right_pad")
POSITION_TOLERANCE = 0.003
ROTATION_TOLERANCE = 0.03
ROTATION_WEIGHT = 0.15
IK_DAMPING = 0.04
MAX_IK_ITERATIONS = 100
PATH_STEP = 0.025
PLOT_DPI = 300
PROVISIONAL_TABLE_HEIGHT = 0.30
PROVISIONAL_TABLE_FOOTPRINT = (0.15, 0.35, -0.45, -0.05)
PROVISIONAL_TRAY_CENTER = (0.24, -0.16)
DIAGNOSTIC_EPISODES = (
    ("success_pick_yaw_0", (0.32, -0.38), 0.0),
    ("success_pick_yaw_90", (0.32, -0.38), 90.0),
    ("tray_edge_collision_yaw_45", (0.19, -0.27), 45.0),
    ("joint_limit_yaw_202_5", (0.19, -0.38), 202.5),
)


def save_figure(figure: plt.Figure, output_path: Path) -> None:
    """Save a high-resolution figure and report its absolute location."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    absolute_path = output_path.resolve()
    figure.savefig(absolute_path, dpi=PLOT_DPI, bbox_inches="tight")
    print(f"Saved PNG: {absolute_path}")


@dataclass
class IKResult:
    q: np.ndarray
    converged: bool
    position_error: float
    rotation_error: float
    iterations: int
    joint_limit_margin: float


@dataclass
class GripperOpenState:
    finger_joint_names: tuple[str, ...]
    qpos_values: np.ndarray
    pad_geom_names: tuple[str, ...]
    approach_axis_local: np.ndarray
    jaw_axis_local: np.ndarray


def calibrate_open_gripper(model: mujoco.MjModel) -> GripperOpenState:
    """Settle the gripper at its actuator-open command and measure its local axes."""
    data = mujoco.MjData(model)
    actuator_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, "fingers_actuator"
    )
    if actuator_id < 0:
        raise ValueError("Model has no fingers_actuator for open-state calibration")
    open_command = float(model.actuator_ctrlrange[actuator_id, 1])
    for _ in range(1500):
        mujoco.mj_forward(model, data)
        data.qfrc_applied[:] = data.qfrc_bias
        data.ctrl[actuator_id] = open_command
        mujoco.mj_step(model, data)

    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper_base")
    pad_geom_ids = np.array([
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in ("left_pad1", "left_pad2", "right_pad1", "right_pad2")
    ])
    if np.any(pad_geom_ids < 0):
        raise ValueError("Could not find the four fingertip pad collision geoms")
    left_pad_ids = pad_geom_ids[:2]
    right_pad_ids = pad_geom_ids[2:]
    left_center = data.geom_xpos[left_pad_ids].mean(axis=0)
    right_center = data.geom_xpos[right_pad_ids].mean(axis=0)
    pad_center = (left_center + right_center) / 2
    jaw_world = right_center - left_center
    rotation_flat = np.zeros(9)
    mujoco.mju_quat2Mat(rotation_flat, data.xquat[base_id])
    base_rotation = rotation_flat.reshape(3, 3)
    approach_axis = base_rotation.T @ (pad_center - data.xpos[base_id])
    jaw_axis = base_rotation.T @ jaw_world
    approach_axis /= np.linalg.norm(approach_axis)
    jaw_axis -= approach_axis * np.dot(jaw_axis, approach_axis)
    jaw_axis /= np.linalg.norm(jaw_axis)

    finger_joint_ids = [
        joint_id
        for joint_id in range(model.njnt)
        if (name := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id))
        and name.startswith(("left_", "right_"))
    ]
    addresses = model.jnt_qposadr[finger_joint_ids].copy()
    finger_names = tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        for joint_id in finger_joint_ids
    )
    pad_names = ("left_pad1", "left_pad2", "right_pad1", "right_pad2")
    return GripperOpenState(
        finger_names,
        data.qpos[addresses].copy(),
        pad_names,
        approach_axis,
        jaw_axis,
    )


class HealIK:
    """Damped least-squares pose IK for the six HEAL arm joints."""

    def __init__(self, model: mujoco.MjModel, gripper_state: GripperOpenState):
        self.model = model
        self.data = mujoco.MjData(model)
        self.gripper_state = gripper_state
        finger_joint_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in gripper_state.finger_joint_names
        ]
        if any(joint_id < 0 for joint_id in finger_joint_ids):
            raise ValueError("Could not resolve calibrated gripper joints in this model")
        finger_qpos_addresses = model.jnt_qposadr[finger_joint_ids]
        self.data.qpos[finger_qpos_addresses] = gripper_state.qpos_values
        self.joint_ids = np.array([
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in ARM_JOINT_NAMES
        ])
        if np.any(self.joint_ids < 0):
            raise ValueError("Combined model must define joint_1 through joint_6")
        self.qpos_addresses = model.jnt_qposadr[self.joint_ids]
        self.dof_addresses = model.jnt_dofadr[self.joint_ids]
        self.lower = model.jnt_range[self.joint_ids, 0].copy()
        self.upper = model.jnt_range[self.joint_ids, 1].copy()
        self.home = model.qpos0[self.qpos_addresses].copy()
        self.pad_geom_ids = np.array([
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in gripper_state.pad_geom_names
        ])
        if np.any(self.pad_geom_ids < 0):
            raise ValueError("Could not resolve calibrated fingertip geoms in this model")
        self.gripper_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "gripper_base"
        )
        if self.gripper_id < 0:
            raise ValueError("Could not find gripper_base")
        self.jacobian_position = np.zeros((3, model.nv))
        self.jacobian_rotation = np.zeros((3, model.nv))
        self.pad_jacobians = [np.zeros((3, model.nv)) for _ in self.pad_geom_ids]
        self.alternate_seeds = [
            np.clip(self.home + np.array([0, -1.2, 1.2, 0, 0, 0]), self.lower, self.upper),
            np.clip(self.home + np.array([0, 1.2, -1.2, 0, 0, 0]), self.lower, self.upper),
        ]

    def quaternion_for_yaw(self, yaw: float) -> np.ndarray:
        """Map the measured jaw and approach axes to horizontal jaw / downward approach."""
        jaw_world = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        approach_world = np.array([0.0, 0.0, -1.0])
        jaw_local = self.gripper_state.jaw_axis_local
        approach_local = self.gripper_state.approach_axis_local
        local_y = np.cross(approach_local, jaw_local)
        world_y = np.cross(approach_world, jaw_world)
        local_frame = np.column_stack([jaw_local, local_y, approach_local])
        world_frame = np.column_stack([jaw_world, world_y, approach_world])
        rotation = world_frame @ local_frame.T
        quaternion = np.zeros(4)
        mujoco.mju_mat2Quat(quaternion, np.ascontiguousarray(rotation.ravel()))
        return quaternion

    def update(self, q: np.ndarray) -> None:
        self.data.qpos[self.qpos_addresses] = q
        mujoco.mj_forward(self.model, self.data)

    def grasp_position(self) -> np.ndarray:
        return self.data.geom_xpos[self.pad_geom_ids].mean(axis=0).copy()

    def pose_error(
        self,
        target_position: np.ndarray,
        target_quaternion: np.ndarray,
    ) -> tuple[np.ndarray, float, float]:
        position_error = target_position - self.grasp_position()
        current_quaternion = self.data.xquat[self.gripper_id]
        inverse_current = current_quaternion * np.array([1.0, -1.0, -1.0, -1.0])
        error_quaternion = np.zeros(4)
        mujoco.mju_mulQuat(error_quaternion, target_quaternion, inverse_current)
        if error_quaternion[0] < 0:
            error_quaternion *= -1
        vector_norm = np.linalg.norm(error_quaternion[1:])
        if vector_norm < 1e-10:
            rotation_error_vector = np.zeros(3)
        else:
            angle = 2 * np.arctan2(vector_norm, max(error_quaternion[0], 0.0))
            rotation_error_vector = angle * error_quaternion[1:] / vector_norm
        return (
            np.concatenate([position_error, ROTATION_WEIGHT * rotation_error_vector]),
            float(np.linalg.norm(position_error)),
            float(np.linalg.norm(rotation_error_vector)),
        )

    def solve(
        self,
        target_position: np.ndarray,
        target_quaternion: np.ndarray,
        seed: np.ndarray,
    ) -> IKResult:
        q = np.clip(seed.copy(), self.lower, self.upper)
        for iteration in range(MAX_IK_ITERATIONS):
            self.update(q)
            error, position_norm, rotation_norm = self.pose_error(
                target_position, target_quaternion
            )
            joint_margin = float(min(np.min(q - self.lower), np.min(self.upper - q)))
            if position_norm <= POSITION_TOLERANCE and rotation_norm <= ROTATION_TOLERANCE:
                return IKResult(
                    q, True, position_norm, rotation_norm, iteration, joint_margin
                )

            for index, geom_id in enumerate(self.pad_geom_ids):
                mujoco.mj_jac(
                    self.model,
                    self.data,
                    self.pad_jacobians[index],
                    None,
                    self.data.geom_xpos[geom_id],
                    int(self.model.geom_bodyid[geom_id]),
                )
            mujoco.mj_jacBody(
                self.model,
                self.data,
                None,
                self.jacobian_rotation,
                self.gripper_id,
            )
            position_jacobian = np.mean(self.pad_jacobians, axis=0)[:, self.dof_addresses]
            rotation_jacobian = self.jacobian_rotation[:, self.dof_addresses]
            jacobian = np.vstack([
                position_jacobian,
                ROTATION_WEIGHT * rotation_jacobian,
            ])
            system = jacobian @ jacobian.T + IK_DAMPING**2 * np.eye(6)
            delta_q = jacobian.T @ np.linalg.solve(system, error)
            delta_norm = np.linalg.norm(delta_q)
            if delta_norm > 0.2:
                delta_q *= 0.2 / delta_norm
            q = np.clip(q + delta_q, self.lower, self.upper)

        self.update(q)
        _, position_norm, rotation_norm = self.pose_error(target_position, target_quaternion)
        joint_margin = float(min(np.min(q - self.lower), np.min(self.upper - q)))
        return IKResult(
            q, False, position_norm, rotation_norm, MAX_IK_ITERATIONS, joint_margin
        )

    def solve_with_seeds(
        self,
        target_position: np.ndarray,
        target_quaternion: np.ndarray,
        seed: np.ndarray,
    ) -> IKResult:
        seeds = [seed, self.home, *self.alternate_seeds]
        best = None
        for candidate in seeds:
            result = self.solve(target_position, target_quaternion, candidate)
            if result.converged:
                return result
            score = result.position_error + ROTATION_WEIGHT * result.rotation_error
            if best is None or score < best[0]:
                best = (score, result)
        return best[1]


def make_task_model(
    table_height: float,
    footprint: tuple[float, float, float, float],
    tray_center: tuple[float, float],
    tray_opening: float,
    tray_wall_height: float,
    tray_wall_thickness: float,
    table_thickness: float = 0.03,
    cube_size: float = 0.04,
) -> mujoco.MjModel:
    """Compile the HEAL model with an assumed tabletop and square tray."""
    x_min, x_max, y_min, y_max = footprint
    tray_x, tray_y = tray_center
    tray_half = tray_opening / 2
    wall_half = tray_wall_thickness / 2
    outer_half = tray_half + tray_wall_thickness
    original_dir = Path.cwd()
    try:
        os.chdir(MODEL_DIR)
        spec = mujoco.MjSpec.from_file(str(MODEL_PATH))
        spec.worldbody.add_geom(
            name="analysis_table_top",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[(x_min + x_max) / 2, (y_min + y_max) / 2, table_height - table_thickness / 2],
            size=[(x_max - x_min) / 2, (y_max - y_min) / 2, table_thickness / 2],
            contype=0,
            conaffinity=0,
            rgba=[0.5, 0.35, 0.22, 0.35],
        )
        spec.worldbody.add_geom(
            name="analysis_tray_floor",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[tray_x, tray_y, table_height + 0.002],
            size=[outer_half, outer_half, 0.002],
            contype=0,
            conaffinity=0,
            rgba=[0.2, 0.55, 0.85, 0.35],
        )
        wall_z = table_height + tray_wall_height / 2
        wall_specs = [
            ("px", tray_x + tray_half + wall_half, tray_y, wall_half, outer_half),
            ("nx", tray_x - tray_half - wall_half, tray_y, wall_half, outer_half),
            ("py", tray_x, tray_y + tray_half + wall_half, tray_half, wall_half),
            ("ny", tray_x, tray_y - tray_half - wall_half, tray_half, wall_half),
        ]
        for label, x, y, half_x, half_y in wall_specs:
            spec.worldbody.add_geom(
                name=f"analysis_tray_wall_{label}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[x, y, wall_z],
                size=[half_x, half_y, tray_wall_height / 2],
                contype=0,
                conaffinity=0,
                rgba=[0.2, 0.55, 0.85, 0.35],
            )
        cube_body = spec.worldbody.add_body(
            name="analysis_carried_cube",
            pos=[0.0, 0.0, 2.0],
        )
        cube_body.add_freejoint(name="analysis_cube_free")
        cube_body.add_geom(
            name="analysis_carried_cube_geom",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[cube_size / 2] * 3,
            contype=0,
            conaffinity=0,
            rgba=[0.86, 0.22, 0.18, 0.55],
        )
        return spec.compile()
    finally:
        os.chdir(original_dir)


def collision_geometry_ids(model: mujoco.MjModel) -> tuple[list[int], list[int]]:
    obstacle_names = [
        "analysis_table_top",
        "analysis_tray_floor",
        "analysis_tray_wall_px",
        "analysis_tray_wall_nx",
        "analysis_tray_wall_py",
        "analysis_tray_wall_ny",
    ]
    obstacles = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in obstacle_names]
    if any(geom_id < 0 for geom_id in obstacles):
        raise ValueError("Failed to build the assumed table/tray collision geometry")
    robot = [
        geom_id
        for geom_id in range(model.ngeom)
        if model.geom_contype[geom_id] != 0
        and model.geom_bodyid[geom_id] != 0
        and model.geom_bodyid[geom_id] < model.nbody
        and mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            int(model.geom_bodyid[geom_id]),
        ) != "analysis_carried_cube"
    ]
    return robot, obstacles


def set_carried_cube_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    center: np.ndarray,
    yaw: float,
) -> int:
    """Place the non-colliding-in-physics cube proxy for geometric distance checks."""
    joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "analysis_cube_free"
    )
    geom_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "analysis_carried_cube_geom"
    )
    qpos_address = int(model.jnt_qposadr[joint_id])
    half_yaw = yaw / 2
    data.qpos[qpos_address:qpos_address + 7] = [
        center[0], center[1], center[2], np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)
    ]
    mujoco.mj_forward(model, data)
    return geom_id


def geom_obstacle_distance(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom_id: int,
    obstacle_geoms: list[int],
) -> tuple[float, str | None]:
    best_distance = np.inf
    best_name = None
    for obstacle_id in obstacle_geoms:
        distance = float(mujoco.mj_geomDistance(
            model, data, geom_id, obstacle_id, 0.5, None
        ))
        if distance < best_distance:
            best_distance = distance
            best_name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_GEOM, obstacle_id
            )
    return best_distance, best_name


def self_collision_pairs(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    tolerance: float = 1e-4,
) -> list[tuple[str, str, float]]:
    """Return substantial robot self-penetrations; ignore tiny pad seam contacts."""
    mujoco.mj_forward(model, data)
    pairs = []
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        if contact.dist >= -tolerance:
            continue
        body1_id = int(model.geom_bodyid[contact.geom1])
        body2_id = int(model.geom_bodyid[contact.geom2])
        if body1_id == 0 or body2_id == 0:
            continue
        body1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body1_id)
        body2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body2_id)
        if body1 == "analysis_carried_cube" or body2 == "analysis_carried_cube":
            continue
        pairs.append((body1 or "unnamed", body2 or "unnamed", float(contact.dist)))
    return pairs


def body_mesh_xy_bounds(model: mujoco.MjModel, body_name: str) -> np.ndarray:
    """Compute a body's XY footprint from its compiled mesh vertices at qpos0."""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    vertices = []
    for geom_id in range(model.ngeom):
        if int(model.geom_bodyid[geom_id]) != body_id:
            continue
        mesh_id = int(model.geom_dataid[geom_id])
        if mesh_id < 0:
            continue
        start = int(model.mesh_vertadr[mesh_id])
        count = int(model.mesh_vertnum[mesh_id])
        mesh_vertices = model.mesh_vert[start:start + count]
        vertices.append(
            data.geom_xpos[geom_id]
            + mesh_vertices @ data.geom_xmat[geom_id].reshape(3, 3).T
        )
    if not vertices:
        raise ValueError(f"No mesh geometry found on {body_name}")
    xy = np.concatenate(vertices)[:, :2]
    return np.column_stack([xy.min(axis=0), xy.max(axis=0)])


def minimum_obstacle_distance(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    robot_geoms: list[int],
    obstacle_geoms: list[int],
) -> float:
    distances = [
        mujoco.mj_geomDistance(model, data, robot_geom, obstacle_geom, 0.5, None)
        for robot_geom in robot_geoms
        for obstacle_geom in obstacle_geoms
    ]
    return float(min(distances, default=np.inf))


def closest_obstacle_pair(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    robot_geoms: list[int],
    obstacle_geoms: list[int],
) -> tuple[float, str, str]:
    best = (np.inf, "none", "none")
    for robot_geom in robot_geoms:
        body_id = int(model.geom_bodyid[robot_geom])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or "unnamed_body"
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, robot_geom) or "unnamed_geom"
        for obstacle_geom in obstacle_geoms:
            distance = float(mujoco.mj_geomDistance(
                model, data, robot_geom, obstacle_geom, 0.5, None
            ))
            if distance < best[0]:
                obstacle_name = mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_GEOM, obstacle_geom
                ) or "unnamed_obstacle"
                best = (distance, f"{body_name}/{geom_name}", obstacle_name)
    return best


def ik_failure_reason(result: IKResult) -> str:
    if result.joint_limit_margin <= 0.01:
        return "IK_JOINT_LIMIT"
    return "IK_NONCONVERGENCE"


def check_robot_pose(
    model: mujoco.MjModel,
    solver: HealIK,
    robot_geoms: list[int],
    obstacle_geoms: list[int],
) -> tuple[float, tuple[str, str] | None, list[tuple[str, str, float]]]:
    distance, robot_name, obstacle_name = closest_obstacle_pair(
        model, solver.data, robot_geoms, obstacle_geoms
    )
    environment_collision = None
    if distance < -1e-5:
        environment_collision = (robot_name, obstacle_name)
    self_collisions = self_collision_pairs(model, solver.data)
    return distance, environment_collision, self_collisions


def trace_vertical_path(
    model: mujoco.MjModel,
    solver: HealIK,
    robot_geoms: list[int],
    obstacle_geoms: list[int],
    target_xyz: np.ndarray,
    yaw: float,
    clearance: float,
    carry_cube_on_withdrawal: bool = False,
) -> dict:
    """Trace pre-grasp, vertical approach, and withdrawal with categorized failures."""
    target_xyz = np.asarray(target_xyz, dtype=float)
    pregrasp_xyz = target_xyz + np.array([0.0, 0.0, clearance])
    quaternion = solver.quaternion_for_yaw(yaw)
    result_data = {
        "success": False,
        "reason": "UNTESTED",
        "stage": "pregrasp",
        "yaw_degrees": float(np.rad2deg(yaw) % 360),
        "target_xyz": target_xyz.copy(),
        "position_residual": np.nan,
        "orientation_residual": np.nan,
        "joint_limit_margin": np.nan,
        "minimum_environment_distance": np.inf,
        "closest_pair": None,
        "self_collision_pairs": [],
        "q_at_target": None,
        "q_at_pregrasp": None,
    }

    high = solver.solve_with_seeds(pregrasp_xyz, quaternion, solver.home)
    if not high.converged:
        reason = ik_failure_reason(high)
        print(f"[Failure: {reason}] Stage: pregrasp, Target: {pregrasp_xyz}, Yaw: {np.rad2deg(yaw):.1f}° | Pos Res: {high.position_error:.4f}m, Ori Res: {high.rotation_error:.4f}rad, Jnt Margin: {high.joint_limit_margin:.4f}rad")
        result_data.update({
            "reason": reason,
            "position_residual": high.position_error,
            "orientation_residual": high.rotation_error,
            "joint_limit_margin": high.joint_limit_margin,
            "iterations": high.iterations,
        })
        return result_data

    segments = max(1, int(np.ceil(clearance / PATH_STEP)))
    q = high.q
    q_at_pregrasp = q.copy()
    approach_waypoints = np.linspace(pregrasp_xyz, target_xyz, segments + 1)[1:]
    target_result = high
    for waypoint_index, waypoint in enumerate(approach_waypoints, start=1):
        stage = f"approach_{waypoint_index}/{segments}"
        result_data["stage"] = stage
        target_result = solver.solve_with_seeds(waypoint, quaternion, q)
        if not target_result.converged:
            reason = ik_failure_reason(target_result)
            print(f"[Failure: {reason}] Stage: {stage}, Target: {waypoint}, Yaw: {np.rad2deg(yaw):.1f}° | Pos Res: {target_result.position_error:.4f}m, Ori Res: {target_result.rotation_error:.4f}rad, Jnt Margin: {target_result.joint_limit_margin:.4f}rad")
            result_data.update({
                "reason": reason,
                "position_residual": target_result.position_error,
                "orientation_residual": target_result.rotation_error,
                "joint_limit_margin": target_result.joint_limit_margin,
                "iterations": target_result.iterations,
            })
            return result_data
        q = target_result.q
        solver.update(q)
        distance, collision_pair, self_pairs = check_robot_pose(
            model, solver, robot_geoms, obstacle_geoms
        )
        if distance < result_data["minimum_environment_distance"]:
            result_data["minimum_environment_distance"] = distance
            result_data["closest_pair"] = collision_pair or closest_obstacle_pair(
                model, solver.data, robot_geoms, obstacle_geoms
            )[1:]
            result_data["collision_distance"] = distance
        if collision_pair is not None:
            print(f"[Failure: COLLISION] Stage: {stage}, Collision: {collision_pair[0]} and {collision_pair[1]}, Distance: {distance:.4f}m")
            result_data.update({
                "reason": "TABLE_TRAY_COLLISION",
                "collision_pair": collision_pair,
                "collision_distance": distance,
                "position_residual": target_result.position_error,
                "orientation_residual": target_result.rotation_error,
                "joint_limit_margin": target_result.joint_limit_margin,
                "iterations": target_result.iterations,
            })
            return result_data
        if self_pairs:
            print(f"[Failure: SELF COLLISION] Stage: {stage}, Pairs: {self_pairs}")
            result_data.update({
                "reason": "SELF_COLLISION",
                "self_collision_pairs": self_pairs,
                "position_residual": target_result.position_error,
                "orientation_residual": target_result.rotation_error,
                "joint_limit_margin": target_result.joint_limit_margin,
                "iterations": target_result.iterations,
            })
            return result_data

    cube_geom_id = None
    if carry_cube_on_withdrawal:
        cube_geom_id = set_carried_cube_pose(
            model, solver.data, solver.grasp_position(), yaw
        )
        cube_distance, cube_obstacle = geom_obstacle_distance(
            model, solver.data, cube_geom_id, obstacle_geoms
        )
        if cube_distance < -1e-5:
            print(f"[Failure: CUBE COLLISION] Stage: at grasp, Obstacle: {cube_obstacle}, Distance: {cube_distance:.4f}m")
            result_data.update({
                "reason": "CUBE_TABLE_TRAY_COLLISION_AT_GRASP",
                "cube_obstacle": cube_obstacle,
                "cube_distance": cube_distance,
            })
            return result_data

    q_at_target = q.copy()
    withdrawal_waypoints = np.linspace(target_xyz, pregrasp_xyz, segments + 1)[1:]
    for waypoint_index, waypoint in enumerate(withdrawal_waypoints, start=1):
        stage = f"withdrawal_{waypoint_index}/{segments}"
        result_data["stage"] = stage
        result = solver.solve_with_seeds(waypoint, quaternion, q)
        if not result.converged:
            reason = ik_failure_reason(result)
            print(f"[Failure: {reason}] Stage: {stage}, Target: {waypoint}, Yaw: {np.rad2deg(yaw):.1f}° | Pos Res: {result.position_error:.4f}m, Ori Res: {result.rotation_error:.4f}rad, Jnt Margin: {result.joint_limit_margin:.4f}rad")
            result_data.update({
                "reason": reason,
                "position_residual": result.position_error,
                "orientation_residual": result.rotation_error,
                "joint_limit_margin": result.joint_limit_margin,
                "iterations": result.iterations,
            })
            return result_data
        q = result.q
        solver.update(q)
        distance, collision_pair, self_pairs = check_robot_pose(
            model, solver, robot_geoms, obstacle_geoms
        )
        if carry_cube_on_withdrawal:
            cube_geom_id = set_carried_cube_pose(
                model, solver.data, solver.grasp_position(), yaw
            )
            cube_distance, cube_obstacle = geom_obstacle_distance(
                model, solver.data, cube_geom_id, obstacle_geoms
            )
            if cube_distance < result_data["minimum_environment_distance"]:
                result_data["minimum_environment_distance"] = cube_distance
                result_data["closest_pair"] = ("carried_cube", cube_obstacle)
            if cube_distance < -1e-5:
                print(f"[Failure: CUBE COLLISION] Stage: {stage}, Obstacle: {cube_obstacle}, Distance: {cube_distance:.4f}m")
                result_data.update({
                    "reason": "CUBE_TABLE_TRAY_COLLISION",
                    "stage": stage,
                    "cube_obstacle": cube_obstacle,
                    "cube_distance": cube_distance,
                    "position_residual": result.position_error,
                    "orientation_residual": result.rotation_error,
                    "joint_limit_margin": result.joint_limit_margin,
                    "iterations": result.iterations,
                })
                return result_data
        if distance < result_data["minimum_environment_distance"]:
            result_data["minimum_environment_distance"] = distance
            result_data["closest_pair"] = closest_obstacle_pair(
                model, solver.data, robot_geoms, obstacle_geoms
            )[1:]
            result_data["collision_distance"] = distance
        if collision_pair is not None:
            print(f"[Failure: COLLISION] Stage: {stage}, Collision: {collision_pair[0]} and {collision_pair[1]}, Distance: {distance:.4f}m")
            result_data.update({
                "reason": "TABLE_TRAY_COLLISION",
                "collision_pair": collision_pair,
                "collision_distance": distance,
                "position_residual": result.position_error,
                "orientation_residual": result.rotation_error,
                "joint_limit_margin": result.joint_limit_margin,
                "iterations": result.iterations,
            })
            return result_data
        if self_pairs:
            print(f"[Failure: SELF COLLISION] Stage: {stage}, Pairs: {self_pairs}")
            result_data.update({
                "reason": "SELF_COLLISION",
                "self_collision_pairs": self_pairs,
                "position_residual": result.position_error,
                "orientation_residual": result.rotation_error,
                "joint_limit_margin": result.joint_limit_margin,
                "iterations": result.iterations,
            })
            return result_data

    result_data.update({
        "success": True,
        "reason": "PASS_IK_APPROACH_WITHDRAWAL",
        "stage": "complete",
        "position_residual": target_result.position_error,
        "orientation_residual": target_result.rotation_error,
        "joint_limit_margin": target_result.joint_limit_margin,
        "iterations": target_result.iterations,
        "q_at_target": q_at_target,
        "q_at_pregrasp": q_at_pregrasp,
    })
    return result_data


def trace_cube_transfer_and_place(
    model: mujoco.MjModel,
    solver: HealIK,
    robot_geoms: list[int],
    obstacle_geoms: list[int],
    pick_xy: tuple[float, float],
    tray_xy: tuple[float, float],
    table_height: float,
    cube_size: float,
    tray_wall_height: float,
    clearance: float,
    yaw: float,
    pick_path: dict,
) -> dict:
    """Check carried-cube transfer/descent and gripper retreat into the tray."""
    if not pick_path["success"]:
        return {"success": False, "reason": "PICK_PATH_FAILED", "stage": "pickup"}
    quaternion = solver.quaternion_for_yaw(yaw)
    pick_pre = np.array([
        pick_xy[0], pick_xy[1], table_height + cube_size / 2 + clearance
    ])
    place = np.array([
        tray_xy[0], tray_xy[1], table_height + 0.004 + cube_size / 2
    ])
    transit_z = max(
        pick_pre[2],
        table_height + tray_wall_height + cube_size / 2 + 0.03,
    )
    transit = np.array([tray_xy[0], tray_xy[1], transit_z])
    q = pick_path["q_at_pregrasp"].copy()
    min_cube_distance = np.inf
    min_robot_distance = np.inf
    segments = max(1, int(np.ceil(np.linalg.norm(transit - pick_pre) / PATH_STEP)))

    phases = [
        ("transfer", np.linspace(pick_pre, transit, segments + 1)[1:]),
        ("tray_descent", np.linspace(transit, place, max(1, int(np.ceil((transit_z - place[2]) / PATH_STEP))) + 1)[1:]),
    ]
    for phase, waypoints in phases:
        for waypoint_index, waypoint in enumerate(waypoints, start=1):
            result = solver.solve_with_seeds(waypoint, quaternion, q)
            if not result.converged:
                reason = ik_failure_reason(result)
                print(f"[Failure: {reason}] Stage: {phase}_{waypoint_index}, Target: {waypoint}, Yaw: {np.rad2deg(yaw):.1f}°")
                return {
                    "success": False,
                    "reason": reason,
                    "stage": f"{phase}_{waypoint_index}",
                    "position_residual": result.position_error,
                    "orientation_residual": result.rotation_error,
                    "joint_limit_margin": result.joint_limit_margin,
                }
            q = result.q
            solver.update(q)
            robot_distance, robot_collision, self_pairs = check_robot_pose(
                model, solver, robot_geoms, obstacle_geoms
            )
            min_robot_distance = min(min_robot_distance, robot_distance)
            if robot_collision:
                print(f"[Failure: COLLISION] Stage: {phase}_{waypoint_index}, Collision: {robot_collision[0]} and {robot_collision[1]}, Distance: {robot_distance:.4f}m")
                return {
                    "success": False,
                    "reason": "TABLE_TRAY_COLLISION",
                    "stage": f"{phase}_{waypoint_index}",
                    "collision_pair": robot_collision,
                    "distance": robot_distance,
                    "position_residual": result.position_error,
                    "orientation_residual": result.rotation_error,
                }
            if self_pairs:
                print(f"[Failure: SELF COLLISION] Stage: {phase}_{waypoint_index}, Pairs: {self_pairs}")
                return {
                    "success": False,
                    "reason": "SELF_COLLISION",
                    "stage": f"{phase}_{waypoint_index}",
                    "self_collision_pairs": self_pairs,
                }
            cube_geom = set_carried_cube_pose(
                model, solver.data, solver.grasp_position(), yaw
            )
            cube_distance, cube_obstacle = geom_obstacle_distance(
                model, solver.data, cube_geom, obstacle_geoms
            )
            min_cube_distance = min(min_cube_distance, cube_distance)
            if cube_distance < -1e-5:
                print(f"[Failure: CUBE COLLISION] Stage: {phase}_{waypoint_index}, Obstacle: {cube_obstacle}, Distance: {cube_distance:.4f}m")
                return {
                    "success": False,
                    "reason": "CUBE_TABLE_TRAY_COLLISION",
                    "stage": f"{phase}_{waypoint_index}",
                    "obstacle": cube_obstacle,
                    "distance": cube_distance,
                    "position_residual": result.position_error,
                    "orientation_residual": result.rotation_error,
                }

    # Check the placed cube at rest: floor contact is allowed; tray-wall penetration is not.
    cube_geom = set_carried_cube_pose(model, solver.data, place, yaw)
    placed_distances = [
        (float(mujoco.mj_geomDistance(model, solver.data, cube_geom, obstacle, 0.5, None)),
         mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, obstacle))
        for obstacle in obstacle_geoms
    ]
    worst_distance, worst_obstacle = min(placed_distances, key=lambda item: item[0])
    if worst_distance < -1e-5:
        print(f"[Failure: CUBE PLACEMENT] Stage: placed_cube_static_check, Obstacle: {worst_obstacle}, Distance: {worst_distance:.4f}m")
        return {
            "success": False,
            "reason": "PLACED_CUBE_OVERLAPS_TRAY_OR_TABLE",
            "stage": "placed_cube_static_check",
            "obstacle": worst_obstacle,
            "distance": worst_distance,
        }

    retreat = trace_vertical_path(
        model,
        solver,
        robot_geoms,
        obstacle_geoms,
        place,
        yaw,
        clearance,
        carry_cube_on_withdrawal=False,
    )
    if not retreat["success"]:
        retreat["stage"] = f"release_{retreat['stage']}"
        return retreat
    return {
        "success": True,
        "reason": "PASS_TRANSFER_PLACE_RELEASE_RETREAT",
        "stage": "complete",
        "minimum_cube_distance": min(min_cube_distance, worst_distance),
        "minimum_robot_distance": min_robot_distance,
        "placed_cube_clearance": worst_distance,
        "placed_cube_nearest_obstacle": worst_obstacle,
    }


def largest_rectangle(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Find the maximum-area all-true axis-aligned rectangle in a grid mask."""
    best_area = 0
    best_box = None
    heights = np.zeros(mask.shape[1], dtype=int)
    for row in range(mask.shape[0]):
        heights = np.where(mask[row], heights + 1, 0)
        stack = []
        for column in range(mask.shape[1] + 1):
            current = heights[column] if column < mask.shape[1] else 0
            start = column
            while stack and stack[-1][1] >= current:
                start, height = stack.pop()
                area = height * (column - start)
                if area > best_area:
                    best_area = area
                    best_box = (row - height + 1, row, start, column - 1)
            stack.append((start, current))
    return best_box


def test_orientation_workspace(
    solver: HealIK,
    xs: np.ndarray,
    ys: np.ndarray,
    table_height: float,
    cube_size: float,
    approach_clearance: float,
    yaws: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Test IK at grasp and pre-grasp positions for every grid point and yaw."""
    feasible = np.zeros((len(xs), len(ys), len(yaws)), dtype=bool)
    grasp_z = table_height + cube_size / 2
    transit_z = grasp_z + approach_clearance
    warm_seeds = [solver.home.copy() for _ in yaws]
    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            for yaw_index, yaw in enumerate(yaws):
                quaternion = solver.quaternion_for_yaw(float(yaw))
                grasp = solver.solve_with_seeds(
                    np.array([x, y, grasp_z]), quaternion, warm_seeds[yaw_index]
                )
                if not grasp.converged:
                    continue
                transit = solver.solve_with_seeds(
                    np.array([x, y, transit_z]), quaternion, grasp.q
                )
                if transit.converged:
                    feasible[ix, iy, yaw_index] = True
                    warm_seeds[yaw_index] = transit.q
    return feasible.any(axis=2), feasible.all(axis=2)


def table_rectangle(
    box: tuple[int, int, int, int],
    xs: np.ndarray,
    ys: np.ndarray,
    grid_step: float,
) -> tuple[float, float, float, float]:
    i0, i1, j0, j1 = box
    return (
        float(xs[i0] - grid_step / 2),
        float(xs[i1] + grid_step / 2),
        float(ys[j0] - grid_step / 2),
        float(ys[j1] + grid_step / 2),
    )


def proposed_target_points(
    footprint: tuple[float, float, float, float],
    tray_center: tuple[float, float],
    tray_opening: float,
    tray_wall_thickness: float,
    cube_size: float,
) -> dict[str, tuple[float, float]]:
    x_min, x_max, y_min, y_max = footprint
    margin = cube_size / 2 + 0.02
    corners = {
        "front_left": (x_min + margin, y_min + margin),
        "front_right": (x_min + margin, y_max - margin),
        "back_left": (x_max - margin, y_min + margin),
        "back_right": (x_max - margin, y_max - margin),
    }
    tray_half = tray_opening / 2 + tray_wall_thickness
    locations = {
        name: xy for name, xy in corners.items()
        if not (
            abs(xy[0] - tray_center[0]) <= tray_half + margin
            and abs(xy[1] - tray_center[1]) <= tray_half + margin
        )
    }
    center = ((x_min + x_max) / 2, (y_min + y_max) / 2)
    if not (
        abs(center[0] - tray_center[0]) <= tray_half
        and abs(center[1] - tray_center[1]) <= tray_half
    ):
        locations["table_center"] = center
    locations["tray_center"] = tray_center
    return locations


def check_approach_paths(
    model: mujoco.MjModel,
    solver: HealIK,
    robot_geoms: list[int],
    obstacle_geoms: list[int],
    target_xy: tuple[float, float],
    target_z: float,
    clearance: float,
    yaws: np.ndarray,
) -> tuple[int, int, int, float]:
    """Check target and discretized vertical approach/withdrawal against table/tray."""
    passed = 0
    ik_complete = 0
    total = len(yaws)
    global_minimum = np.inf
    segments = max(1, int(np.ceil(clearance / PATH_STEP)))
    fractions = np.linspace(0.0, 1.0, segments + 1)
    for yaw in yaws:
        quaternion = solver.quaternion_for_yaw(float(yaw))
        target = np.array([target_xy[0], target_xy[1], target_z])
        pregrasp = target + np.array([0.0, 0.0, clearance])
        high_result = solver.solve_with_seeds(pregrasp, quaternion, solver.home)
        if not high_result.converged:
            continue
        q = high_result.q
        path_clear = True
        path_ik_complete = True
        for fraction in fractions:
            position = pregrasp + fraction * (target - pregrasp)
            result = solver.solve_with_seeds(position, quaternion, q)
            if not result.converged:
                path_clear = False
                path_ik_complete = False
                break
            q = result.q
            solver.update(q)
            distance = minimum_obstacle_distance(model, solver.data, robot_geoms, obstacle_geoms)
            global_minimum = min(global_minimum, distance)
            if distance < -1e-5:
                path_clear = False
                break
        if path_ik_complete:
            ik_complete += 1
        if path_clear:
            passed += 1
    return passed, ik_complete, total, float(global_minimum)


def plot_xy_map(
    xs: np.ndarray,
    ys: np.ndarray,
    task_mask: np.ndarray,
    dexterous_mask: np.ndarray,
    footprint: tuple[float, float, float, float],
    tray_center: tuple[float, float],
    tray_opening: float,
    tray_wall_thickness: float,
    table_height: float,
    base_bounds: np.ndarray,
    episode_targets: dict[str, dict],
    fully_validated: bool,
    output_path: Path,
) -> plt.Figure:
    figure, axes = plt.subplots(figsize=(9, 8))
    image = np.full((*task_mask.shape, 3), [0.88, 0.89, 0.90], dtype=float)
    image[task_mask] = [0.95, 0.76, 0.30]
    image[dexterous_mask] = [0.20, 0.58, 0.34]
    axes.imshow(
        np.transpose(image, (1, 0, 2)),
        origin="lower",
        interpolation="nearest",
        extent=[xs[0] - (xs[1] - xs[0]) / 2, xs[-1] + (xs[1] - xs[0]) / 2,
                ys[0] - (ys[1] - ys[0]) / 2, ys[-1] + (ys[1] - ys[0]) / 2],
        aspect="equal",
    )
    x_min, x_max, y_min, y_max = footprint
    axes.add_patch(Rectangle(
        (x_min, y_min), x_max - x_min, y_max - y_min,
        fill=False,
        edgecolor="#2366a8" if fully_validated else "#b52f37",
        linestyle="-" if fully_validated else "--",
        linewidth=2.2,
    ))
    tray_half = tray_opening / 2 + tray_wall_thickness
    axes.add_patch(Rectangle(
        (tray_center[0] - tray_half, tray_center[1] - tray_half),
        2 * tray_half, 2 * tray_half,
        facecolor="#397fb5", edgecolor="#174e7a", alpha=0.55, label="Assumed tray outer footprint",
    ))
    axes.add_patch(Rectangle(
        (base_bounds[0, 0], base_bounds[1, 0]),
        base_bounds[0, 1] - base_bounds[0, 0],
        base_bounds[1, 1] - base_bounds[1, 0],
        facecolor="#565b60", edgecolor="#25282b", alpha=0.25,
    ))
    axes.scatter(0, 0, marker="^", color="black", s=100, zorder=5)
    plotted_success = False
    plotted_failure = False
    for case in episode_targets.values():
        is_success = bool(case["success"])
        axes.scatter(
            *case["xy"],
            marker="*" if is_success else "X",
            color="#27864b" if is_success else "#d24a43",
            edgecolor="white",
            linewidth=0.8,
            s=180,
            zorder=6,
            label=("Verified episode success target" if is_success and not plotted_success
                   else "Verified episode failure target" if not is_success and not plotted_failure
                   else None),
        )
        plotted_success |= is_success
        plotted_failure |= not is_success
    axes.set_xlabel("World X (m), forward from robot base")
    axes.set_ylabel("World Y (m)")
    axes.set_title(
        f"HEAL Top-Down Workspace at H={table_height:.2f} m\n"
        "Gold: any tested yaw; green: all tested yaws; fixed footprint is provisional"
    )
    axes.grid(alpha=0.18)
    axes.legend(handles=[
        Patch(facecolor="#278f52", label="All tested yaws feasible"),
        Patch(facecolor="#f2bf45", label="At least one tested yaw feasible"),
        Patch(facecolor="#e0e2e4", label="No tested IK solution (not proven unreachable)"),
        Rectangle((0, 0), 1, 1, fill=False,
                  edgecolor="#2366a8" if fully_validated else "#b52f37",
                  linestyle="-" if fully_validated else "--",
                  label="Fixed provisional tabletop (not globally certified)"),
        Patch(facecolor="#397fb5", alpha=0.55, label="Assumed tray outer footprint"),
        Patch(facecolor="#565b60", alpha=0.25, label="Base mesh XY footprint"),
         Line2D([0], [0], marker="*", color="w", markerfacecolor="#27864b",
             markeredgecolor="white", markersize=12, label="Verified success target"),
         Line2D([0], [0], marker="X", color="w", markerfacecolor="#d24a43",
             markeredgecolor="white", markersize=10, label="Verified failure target"),
    ], loc="upper left", fontsize=8)
    figure.tight_layout()
    save_figure(figure, output_path)
    return figure


def plot_height_sweep(
    heights: np.ndarray,
    task_areas: np.ndarray,
    dexterous_areas: np.ndarray,
    output_path: Path,
) -> plt.Figure:
    figure, axes = plt.subplots(figsize=(8, 5))
    axes.plot(heights, task_areas, "o-", label="Task area: any tested yaw")
    axes.plot(heights, dexterous_areas, "o-", label="Dexterous area: all tested yaws")
    axes.set_xlabel("Table-top height H above robot base plane (m)")
    axes.set_ylabel("Feasible sampled XY area (m²)")
    axes.set_title("HEAL Top-Down Feasibility Versus Table Height")
    axes.grid(alpha=0.25)
    axes.legend()
    figure.tight_layout()
    save_figure(figure, output_path)
    return figure


def workspace_points(
    masks: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    heights: np.ndarray,
    cube_size: float,
) -> np.ndarray:
    """Return XYZ coordinates of tested grasp-center cells in true masks."""
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
    points = []
    for level_index, table_height in enumerate(heights):
        mask = masks[level_index]
        if mask.any():
            points.append(np.column_stack([
                grid_x[mask],
                grid_y[mask],
                np.full(int(mask.sum()), table_height + cube_size / 2),
            ]))
    if not points:
        return np.empty((0, 3))
    return np.concatenate(points, axis=0)


def plot_task_workspace_3d(
    task_points: np.ndarray,
    output_path: Path,
) -> plt.Figure:
    figure = plt.figure(figsize=(10, 8))
    axes = figure.add_subplot(111, projection="3d")
    points = axes.scatter(
        task_points[:, 0], task_points[:, 1], task_points[:, 2],
        c=task_points[:, 2], cmap="viridis", s=22, alpha=0.85,
        depthshade=False,
    )
    axes.set_xlabel("World X (m)")
    axes.set_ylabel("World Y (m)")
    axes.set_zlabel("Grasp-center Z (m)")
    axes.set_title("HEAL Task Workspace\nPositions feasible for at least one tested yaw")
    axes.set_box_aspect(np.maximum(np.ptp(task_points, axis=0), 0.05))
    figure.colorbar(points, ax=axes, label="Grasp-center Z (m)", shrink=0.72)
    figure.tight_layout()
    save_figure(figure, output_path)
    return figure


def plot_reached_vs_not_reached_voxels(
    task_masks: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    heights: np.ndarray,
    cube_size: float,
    grid_step: float,
    output_path: Path,
) -> plt.Figure:
    """Draw tested grid cells as colored 3D voxel bars, not an exact boundary."""
    layer_centers = heights + cube_size / 2
    if len(layer_centers) > 1:
        layer_edges = np.empty(len(layer_centers) + 1)
        layer_edges[1:-1] = (layer_centers[:-1] + layer_centers[1:]) / 2
        layer_edges[0] = layer_centers[0] - (layer_edges[1] - layer_centers[0])
        layer_edges[-1] = layer_centers[-1] + (layer_centers[-1] - layer_edges[-2])
    else:
        layer_edges = np.array([
            layer_centers[0] - grid_step / 2,
            layer_centers[0] + grid_step / 2,
        ])
    dx = np.full(task_masks.size, grid_step * 0.88)
    dy = np.full(task_masks.size, grid_step * 0.88)
    x_grid, y_grid, z_grid = np.meshgrid(xs, ys, layer_centers, indexing="ij")
    reached = task_masks.transpose(1, 2, 0).reshape(-1)
    x_left = x_grid.reshape(-1) - dx / 2
    y_left = y_grid.reshape(-1) - dy / 2
    layer_indices = np.tile(np.arange(len(layer_centers)), len(xs) * len(ys))
    dz = layer_edges[layer_indices + 1] - layer_edges[layer_indices]
    z_bottom = z_grid.reshape(-1) - dz / 2

    figure = plt.figure(figsize=(11, 8))
    axes = figure.add_subplot(111, projection="3d")
    for mask, color, label in (
        (reached, "#27864b", "At least one yaw solved"),
        (~reached, "#d94b45", "No tested yaw solved (not proven unreachable)"),
    ):
        axes.bar3d(
            x_left[mask], y_left[mask], z_bottom[mask],
            dx[mask], dy[mask], dz[mask],
            color=color, alpha=0.68, shade=True, edgecolor="none",
            label=label,
        )
    axes.set_xlabel("World X (m)")
    axes.set_ylabel("World Y (m)")
    axes.set_zlabel("Grasp-center Z layer (m)")
    axes.set_title(
        "Tested HEAL Workspace Cells: Reached vs Not Reached\n"
        f"XY grid {grid_step:.2f} m; Z layers are discrete tested table heights"
    )
    axes.set_box_aspect([
        max(np.ptp(xs), grid_step),
        max(np.ptp(ys), grid_step),
        max(np.ptp(layer_centers), grid_step),
    ])
    axes.legend(loc="upper left", fontsize=8)
    figure.text(
        0.5, 0.02,
        "Red means no tested yaw converged at that sampled grid cell; it is not proof of unreachability.",
        ha="center", fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.045, 1, 1))
    save_figure(figure, output_path)
    return figure


def plot_workspace_with_reference_points(
    task_points: np.ndarray,
    default_grasp_center: np.ndarray,
    base_bounds: np.ndarray,
    output_path: Path,
) -> plt.Figure:
    figure = plt.figure(figsize=(10, 8))
    axes = figure.add_subplot(111, projection="3d")
    axes.scatter(
        task_points[:, 0], task_points[:, 1], task_points[:, 2],
        color="#4f8f70", s=16, alpha=0.55, depthshade=False,
        label="Task-workspace grid positions",
    )
    axes.scatter(
        0, 0, 0, marker="^", color="black", edgecolor="white", s=150,
        label="Robot base origin",
    )
    axes.scatter(
        *default_grasp_center, marker="*", color="#d24a43",
        edgecolor="white", s=220, label="Default open-gripper grasp center",
    )
    z_floor = task_points[:, 2].min()
    base_corners = np.array([
        [base_bounds[0, 0], base_bounds[1, 0], z_floor],
        [base_bounds[0, 1], base_bounds[1, 0], z_floor],
        [base_bounds[0, 1], base_bounds[1, 1], z_floor],
        [base_bounds[0, 0], base_bounds[1, 1], z_floor],
    ])
    axes.plot(
        *np.vstack([base_corners, base_corners[0]]).T,
        color="#303438", linewidth=2, label="Base mesh XY projection",
    )
    axes.set_xlabel("World X (m)")
    axes.set_ylabel("World Y (m)")
    axes.set_zlabel("World Z (m)")
    axes.set_title("HEAL Workspace with Robot Base and Default Grasp Center")
    axes.set_box_aspect(np.maximum(np.ptp(task_points, axis=0), 0.05))
    axes.legend(loc="upper left", fontsize=8)
    figure.tight_layout()
    save_figure(figure, output_path)
    return figure


def plot_workspace_xy_projection(
    task_mask: np.ndarray,
    dexterous_mask: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    base_bounds: np.ndarray,
    table_height: float,
    output_path: Path,
) -> plt.Figure:
    figure, axes = plt.subplots(figsize=(9, 8))
    image = np.full((*task_mask.shape, 3), [0.88, 0.89, 0.90], dtype=float)
    image[task_mask] = [0.95, 0.76, 0.30]
    image[dexterous_mask] = [0.20, 0.58, 0.34]
    step_x = float(np.median(np.diff(xs))) if len(xs) > 1 else 0.1
    step_y = float(np.median(np.diff(ys))) if len(ys) > 1 else 0.1
    axes.imshow(
        np.transpose(image, (1, 0, 2)),
        origin="lower",
        interpolation="nearest",
        extent=[xs[0] - step_x / 2, xs[-1] + step_x / 2,
                ys[0] - step_y / 2, ys[-1] + step_y / 2],
        aspect="equal",
    )
    axes.add_patch(Rectangle(
        (base_bounds[0, 0], base_bounds[1, 0]),
        base_bounds[0, 1] - base_bounds[0, 0],
        base_bounds[1, 1] - base_bounds[1, 0],
        facecolor="#565b60", edgecolor="#25282b", alpha=0.3,
    ))
    axes.scatter(0, 0, marker="^", color="black", s=100, zorder=5)
    axes.set_xlabel("World X (m), forward from robot base")
    axes.set_ylabel("World Y (m)")
    axes.set_title(
        f"HEAL Workspace X–Y Projection at H={table_height:.2f} m\n"
        "Gold: at least one tested yaw; green: all tested yaws"
    )
    axes.legend(handles=[
        Patch(facecolor="#278f52", label="All tested yaws feasible"),
        Patch(facecolor="#f2bf45", label="At least one tested yaw feasible"),
        Patch(facecolor="#e0e2e4", label="No tested IK solution; unknown, not proven unreachable"),
        Patch(facecolor="#565b60", alpha=0.3, label="Base mesh XY footprint"),
    ], loc="upper left", fontsize=8)
    axes.grid(alpha=0.18)
    figure.tight_layout()
    save_figure(figure, output_path)
    return figure


def plot_dexterous_workspace_3d(
    dexterous_points: np.ndarray,
    output_path: Path,
) -> plt.Figure:
    figure = plt.figure(figsize=(10, 8))
    axes = figure.add_subplot(111, projection="3d")
    if len(dexterous_points):
        axes.scatter(
            dexterous_points[:, 0], dexterous_points[:, 1], dexterous_points[:, 2],
            color="#27864b", s=32, alpha=0.9, depthshade=False,
            label="All tested yaw orientations feasible",
        )
    else:
        axes.text2D(0.2, 0.5, "No sampled points passed all tested orientations", transform=axes.transAxes)
    axes.set_xlabel("World X (m)")
    axes.set_ylabel("World Y (m)")
    axes.set_zlabel("Grasp-center Z (m)")
    axes.set_title("HEAL Dexterous Workspace\n16 sampled top-down yaw orientations")
    if len(dexterous_points):
        axes.set_box_aspect(np.maximum(np.ptp(dexterous_points, axis=0), 0.05))
        axes.legend(loc="upper left", fontsize=8)
    figure.tight_layout()
    save_figure(figure, output_path)
    return figure


def write_summary(
    path: Path,
    args: argparse.Namespace,
    table_height: float,
    footprint: tuple[float, float, float, float],
    tray_center: tuple[float, float],
    episode_results: dict[str, dict],
    legacy_path_results: list[dict],
    sweep_rows: list[tuple[float, int, int, float, float]],
    base_bounds: np.ndarray,
) -> None:
    x_min, x_max, y_min, y_max = footprint
    lines = [
        "# HEAL Task Workspace and Table Scenario",
        "",
        "This is a configurable analysis scenario, not a table/object specification found in the HEAL files.",
        "",
        "## Assumptions",
        f"- Top-down grasp; {args.yaw_count} world-Z yaw samples over 360 degrees.",
        f"- Cube side: {args.cube_size:.3f} m.",
        f"- Square tray opening: {args.tray_opening:.3f} m; wall height {args.tray_wall_height:.3f} m; wall thickness {args.tray_wall_thickness:.3f} m.",
        f"- Vertical approach/withdrawal clearance: {args.approach_clearance:.3f} m.",
        f"- XY grid spacing: {args.grid_step:.3f} m; IK position tolerance {POSITION_TOLERANCE:.3f} m; orientation tolerance {ROTATION_TOLERANCE:.3f} rad.",
        "- The analyzed H/L/W and tray center are held at the user's provisional values; they are not optimized in this diagnosis.",
        "- The cube is represented by a geometric proxy during carry/placement checks; it does not participate in MuJoCo dynamics or finger friction.",
        "- Self-collision checks ignore penetrations smaller than 0.1 mm to avoid classifying pad seam contact as a collision.",
        "- These are kinematic/geometric path checks, not actuator torque or dynamic grasp validation.",
        "",
        "## Diagnosed model issue",
        "The previous analyzer used left_pad/right_pad body origins as the grasp center. In this MJCF the physical pad collision boxes are offset from those origins. The corrected TCP is the mean of the four actual pad-box geom centers, and its Jacobian is evaluated at those points.",
        "The prior calibration also reused geom IDs from the bare model in a separately compiled table/tray model. Scene insertion changes geom IDs. The corrected code carries geom/joint names and resolves IDs per compiled model.",
        "Together, the 129.4 mm old TCP offset and stale scene geom indices explain why the previous aggregate 0/48 path result did not describe the intended fingertip TCP in the task model.",
        "",
        "## Fixed provisional table",
        "- H/L/W and tray center are unchanged from the requested diagnosis values.",
        "- This does not globally validate every point on the tabletop; only listed diagnostic cases are tested.",
        f"- Actual base mesh projection at the world origin: X=[{base_bounds[0, 0]:.3f}, {base_bounds[0, 1]:.3f}] m, Y=[{base_bounds[1, 0]:.3f}, {base_bounds[1, 1]:.3f}] m; search keeps at least 0.050 m between the base projection and tabletop edge.",
        f"- H = {table_height:.3f} m above the model floor/base plane.",
        f"- X = [{x_min:.3f}, {x_max:.3f}] m; L = {x_max - x_min:.3f} m.",
        f"- Y = [{y_min:.3f}, {y_max:.3f}] m; W = {y_max - y_min:.3f} m.",
        f"- Tray center = ({tray_center[0]:.3f}, {tray_center[1]:.3f}) m in world XY.",
        "",
        "## Episode diagnostics",
        "PASS requires IK through approach and withdrawal, robot/table/tray and significant self-collision checks, carried-cube transfer/descent clearance, static tray placement, and release retreat.",
    ]
    for name, result in episode_results.items():
        pick = result["pick"]
        final = result["final"]
        lines.append(
            f"- {name}: target=({result['xy'][0]:.3f}, {result['xy'][1]:.3f}) m, "
            f"yaw={result['yaw_degrees']:.1f} deg; success={result['success']}; "
            f"reason={final['reason']} at {final.get('stage', 'unknown')}; "
            f"pick={pick['reason']} at {pick.get('stage', 'unknown')}; "
            f"pick residuals=({pick.get('position_residual', np.nan):.4f} m, "
            f"{pick.get('orientation_residual', np.nan):.4f} rad); "
            f"joint margin={pick.get('joint_limit_margin', np.nan):.4f} rad; "
            f"minimum robot/environment distance={pick.get('collision_distance', pick.get('minimum_environment_distance', np.nan)):.4f} m; "
            f"obstacle pair={pick.get('collision_pair', pick.get('closest_pair'))}; "
            f"minimum cube distance={final.get('minimum_cube_distance', final.get('cube_distance', np.nan)):.4f} m."
        )
    failure_counts = {}
    for result in legacy_path_results:
        failure_counts[result["reason"]] = failure_counts.get(result["reason"], 0) + 1
    lines.extend([
        "",
        "## Recheck of historical 48 target/yaw paths",
        f"- Replayed 3 historical locations × {args.yaw_count} yaws = {len(legacy_path_results)} corrected-TCP approach/withdrawal tests.",
        f"- Outcome counts: {sum(result['success'] for result in legacy_path_results)} pass; failures by reason: {failure_counts}.",
        "- These 48 are vertical target paths only; they are not counted as full pick-and-place successes. Full transfer/place/release is checked only for the named episode cases above.",
    ])
    for result in legacy_path_results:
        if result["success"]:
            continue
        lines.append(
            f"- Failure: {result['target_name']} at yaw {result['yaw_degrees']:.1f} deg, "
            f"{result['reason']} during {result.get('stage')}; "
            f"residual=({result.get('position_residual', np.nan):.4f} m, "
            f"{result.get('orientation_residual', np.nan):.4f} rad), "
            f"joint margin={result.get('joint_limit_margin', np.nan):.4f} rad."
        )
    lines.extend([
        "",
        "## Height sweep",
        "H (m) | any-yaw grid points | all-yaw grid points | any-yaw area (m²) | all-yaw rectangle area (m²)",
        "---:|---:|---:|---:|---:",
    ])
    for height, task_count, dex_count, task_area, dex_area in sweep_rows:
        lines.append(f"{height:.3f} | {task_count} | {dex_count} | {task_area:.3f} | {dex_area:.3f}")
    lines.extend([
        "",
        "## Interpretation",
        "Position workspace maps are computed from the six-joint arm and corrected pad-geom TCP. Dexterous cells mean IK converged at grasp and pre-grasp for every tested discrete yaw; intermediate yaws are not certified. Cells with no sampled IK solution are unknown, not proven unreachable. Episode success labels apply only to the explicitly tested target/yaw cases.",
        "A reported 0.0000 m minimum clearance is tangency/support contact within the distance routine's precision; penetrations below -0.00001 m are rejected. Cube/table-floor contact at placement is intentional. The cube proxy is geometric only and does not validate dynamic grasp forces or friction.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yaw-count", type=int, default=16)
    parser.add_argument("--cube-size", type=float, default=0.04, help="assumed cube side in metres")
    parser.add_argument("--tray-opening", type=float, default=0.16, help="assumed square tray opening in metres")
    parser.add_argument("--tray-wall-height", type=float, default=0.05, help="assumed tray wall height in metres")
    parser.add_argument("--tray-wall-thickness", type=float, default=0.01, help="assumed tray wall thickness in metres")
    parser.add_argument("--approach-clearance", type=float, default=0.12, help="vertical clearance above grasp center in metres")
    parser.add_argument("--grid-step", type=float, default=0.10, help="XY grid spacing in metres")
    parser.add_argument("--x-range", type=float, nargs=2, default=(0.20, 0.80), help="forward-side XY search interval in world X")
    parser.add_argument("--y-range", type=float, nargs=2, default=(-0.50, 0.50), help="XY search interval in world Y")
    parser.add_argument("--heights", type=float, nargs="+", default=DEFAULT_HEIGHTS.tolist(), help="table-top heights above floor/base plane")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="save all figures without opening interactive windows",
    )
    args = parser.parse_args()

    if args.yaw_count < 1 or args.grid_step <= 0 or args.cube_size <= 0 or args.approach_clearance <= 0:
        parser.error("yaw-count, grid-step, cube-size, and approach-clearance must be positive")

    base_model = load_model()
    gripper_state = calibrate_open_gripper(base_model)
    solver = HealIK(base_model, gripper_state)
    base_bounds = body_mesh_xy_bounds(base_model, "base_link")
    yaw_angles = np.arange(args.yaw_count) * (2 * np.pi / args.yaw_count)
    safe_x_minimum = max(
        args.x_range[0],
        base_bounds[0, 1] + 0.05 + args.grid_step / 2,
    )
    if safe_x_minimum > args.x_range[1]:
        parser.error("X search range is too close to the measured base footprint")
    xs = np.round(np.arange(safe_x_minimum, args.x_range[1] + args.grid_step / 2, args.grid_step), 6)
    ys = np.round(np.arange(args.y_range[0], args.y_range[1] + args.grid_step / 2, args.grid_step), 6)
    heights = np.asarray(args.heights, dtype=float)
    if not np.any(np.isclose(heights, PROVISIONAL_TABLE_HEIGHT)):
        parser.error("--heights must include the fixed provisional H=0.30 m for this diagnosis")

    print("========== TASK SCENARIO (GENERIC ASSUMPTIONS) ==========")
    print(f"Robot base/world origin: (0, 0, 0) m; +X is the table search/front direction.")
    print(f"Measured base mesh XY footprint: X={base_bounds[0].tolist()}, Y={base_bounds[1].tolist()} m; table edge clearance=0.05 m.")
    print(f"Cube side={args.cube_size:.3f} m; square tray opening={args.tray_opening:.3f} m.")
    print(f"Top-down yaw samples={args.yaw_count}; approach/withdrawal clearance={args.approach_clearance:.3f} m.")
    print("Open gripper joint values (actuator-settled):", gripper_state.qpos_values)
    print("Measured local approach axis:", gripper_state.approach_axis_local)
    print("Measured local jaw axis:", gripper_state.jaw_axis_local)
    print(f"Search XY: X={tuple(args.x_range)} m, Y={tuple(args.y_range)} m, grid={args.grid_step:.3f} m.")
    print("These object/table values are assumptions, not found HEAL scene dimensions.")
    solver.update(solver.home)
    legacy_pad_body_ids = [
        mujoco.mj_name2id(base_model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in PAD_BODY_NAMES
    ]
    legacy_center = solver.data.xpos[legacy_pad_body_ids].mean(axis=0)
    corrected_center = solver.grasp_position()
    print("Legacy pad-body-origin center at default pose:", legacy_center)
    print("Corrected pad-geom center at default pose:", corrected_center)
    print("Legacy TCP position error (geom-center minus body-origin):", corrected_center - legacy_center)

    task_maps = []
    dex_maps = []
    sweep_rows = []
    for height in heights:
        print(f"\nTesting H={height:.3f} m ...", flush=True)
        task_mask, dex_mask = test_orientation_workspace(
            solver, xs, ys, float(height), args.cube_size,
            args.approach_clearance, yaw_angles,
        )
        task_maps.append(task_mask)
        dex_maps.append(dex_mask)
        box = largest_rectangle(dex_mask)
        area = 0.0 if box is None else (box[1] - box[0] + 1) * (box[3] - box[2] + 1) * args.grid_step**2
        task_area = int(task_mask.sum()) * args.grid_step**2
        sweep_rows.append((float(height), int(task_mask.sum()), int(dex_mask.sum()), task_area, area))
        print(f"  grid points: any yaw={task_mask.sum()}, all yaws={dex_mask.sum()}, largest all-yaw rectangle={area:.3f} m^2")
    chosen_height = PROVISIONAL_TABLE_HEIGHT
    chosen_footprint = PROVISIONAL_TABLE_FOOTPRINT
    tray_center = PROVISIONAL_TRAY_CENTER
    chosen_height_index = int(np.flatnonzero(np.isclose(heights, chosen_height))[0])
    chosen_task = task_maps[chosen_height_index]
    chosen_dex = dex_maps[chosen_height_index]
    x_min, x_max, y_min, y_max = chosen_footprint

    print("\n========== FIXED-SCENE PATH DIAGNOSTICS ==========")
    print(
        f"Keeping H={chosen_height:.2f} m, L={x_max - x_min:.2f} m, "
        f"W={y_max - y_min:.2f} m, tray center={tray_center}."
    )
    task_model = make_task_model(
        chosen_height,
        chosen_footprint,
        tray_center,
        args.tray_opening,
        args.tray_wall_height,
        args.tray_wall_thickness,
        cube_size=args.cube_size,
    )
    path_solver = HealIK(task_model, gripper_state)
    robot_geoms, obstacle_geoms = collision_geometry_ids(task_model)
    base_pad_geom_ids = [
        mujoco.mj_name2id(base_model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in gripper_state.pad_geom_names
    ]
    scene_pad_geom_ids = [
        mujoco.mj_name2id(task_model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in gripper_state.pad_geom_names
    ]
    print(
        "Pad geom IDs (bare model -> task model, resolved by name):",
        list(zip(gripper_state.pad_geom_names, base_pad_geom_ids, scene_pad_geom_ids)),
    )
    episode_results = {}
    for name, pick_xy, yaw_degrees in DIAGNOSTIC_EPISODES:
        yaw = np.deg2rad(yaw_degrees)
        pick_target = np.array([
            pick_xy[0], pick_xy[1], chosen_height + args.cube_size / 2
        ])
        pick = trace_vertical_path(
            task_model,
            path_solver,
            robot_geoms,
            obstacle_geoms,
            pick_target,
            yaw,
            args.approach_clearance,
            carry_cube_on_withdrawal=True,
        )
        if pick["success"]:
            final = trace_cube_transfer_and_place(
                task_model,
                path_solver,
                robot_geoms,
                obstacle_geoms,
                pick_xy,
                tray_center,
                chosen_height,
                args.cube_size,
                args.tray_wall_height,
                args.approach_clearance,
                yaw,
                pick,
            )
        else:
            final = {
                "success": False,
                "reason": pick["reason"],
                "stage": pick["stage"],
                **{key: pick[key] for key in (
                    "position_residual", "orientation_residual", "joint_limit_margin",
                    "collision_pair", "cube_obstacle", "cube_distance",
                ) if key in pick},
            }
        episode_results[name] = {
            "xy": pick_xy,
            "yaw_degrees": yaw_degrees,
            "pick": pick,
            "final": final,
            "success": bool(pick["success"] and final["success"]),
        }
        print(
            f"{name}: {'SUCCESS' if episode_results[name]['success'] else 'FAILURE'}; "
            f"reason={final['reason']} stage={final.get('stage')}; "
            f"pick reason={pick['reason']} stage={pick.get('stage')}; "
            f"position residual={pick.get('position_residual', np.nan):.4f} m; "
            f"orientation residual={pick.get('orientation_residual', np.nan):.4f} rad; "
            f"joint margin={pick.get('joint_limit_margin', np.nan):.4f} rad; "
            f"minimum environment distance={pick.get('collision_distance', pick.get('minimum_environment_distance', np.nan)):.4f} m; "
            f"environment pair={pick.get('collision_pair', pick.get('closest_pair'))}; "
            f"self-penetrations={len(pick.get('self_collision_pairs', []))}; "
            f"cube clearance={final.get('minimum_cube_distance', final.get('cube_distance', np.nan)):.4f} m"
        )
    legacy_target_locations = proposed_target_points(
        chosen_footprint,
        tray_center,
        args.tray_opening,
        args.tray_wall_thickness,
        args.cube_size,
    )
    legacy_path_results = []
    for target_name, target_xy in legacy_target_locations.items():
        for yaw in yaw_angles:
            target_height = (
                chosen_height + 0.004 + args.cube_size / 2
                if target_name == "tray_center"
                else chosen_height + args.cube_size / 2
            )
            path_result = trace_vertical_path(
                task_model,
                path_solver,
                robot_geoms,
                obstacle_geoms,
                np.array([target_xy[0], target_xy[1], target_height]),
                float(yaw),
                args.approach_clearance,
                carry_cube_on_withdrawal=(target_name != "tray_center"),
            )
            path_result["target_name"] = target_name
            path_result["xy"] = target_xy
            legacy_path_results.append(path_result)
    legacy_failures = [result for result in legacy_path_results if not result["success"]]
    print(
        f"Historical target-path replay: {len(legacy_path_results) - len(legacy_failures)} "
        f"passed, {len(legacy_failures)} failed."
    )
    for result in legacy_failures:
        print(
            f"  {result['target_name']} yaw={result['yaw_degrees']:.1f} deg: "
            f"{result['reason']} at {result.get('stage')}; "
            f"residual=({result.get('position_residual', np.nan):.4f} m, "
            f"{result.get('orientation_residual', np.nan):.4f} rad); "
            f"joint margin={result.get('joint_limit_margin', np.nan):.4f} rad"
        )
    fully_validated = all(result["success"] for result in episode_results.values())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    task_mask_at_chosen_height = chosen_task
    dexterous_mask_at_chosen_height = chosen_dex
    solver.update(solver.home)
    default_grasp_center = solver.grasp_position()
    episode_targets = {
        name: {"xy": result["xy"], "success": result["success"]}
        for name, result in episode_results.items()
    }
    task_points = workspace_points(
        np.stack(task_maps), xs, ys, heights, args.cube_size
    )
    dexterous_points = workspace_points(
        np.stack(dex_maps), xs, ys, heights, args.cube_size
    )
    figures = []
    figures.append(plot_task_workspace_3d(
        task_points,
        args.output_dir / "heal_task_workspace_3d.png",
    ))
    figures.append(plot_reached_vs_not_reached_voxels(
        np.stack(task_maps), xs, ys, heights, args.cube_size,
        args.grid_step, args.output_dir / "heal_reached_vs_not_reached_voxels.png",
    ))
    figures.append(plot_workspace_with_reference_points(
        task_points,
        default_grasp_center,
        base_bounds,
        args.output_dir / "heal_workspace_base_default_grasp_3d.png",
    ))
    figures.append(plot_workspace_xy_projection(
        task_mask_at_chosen_height,
        dexterous_mask_at_chosen_height,
        xs,
        ys,
        base_bounds,
        chosen_height,
        args.output_dir / "heal_workspace_xy_projection.png",
    ))
    figures.append(plot_dexterous_workspace_3d(
        dexterous_points,
        args.output_dir / "heal_dexterous_workspace_3d.png",
    ))
    figures.append(plot_xy_map(
        xs, ys, chosen_task, chosen_dex, chosen_footprint, tray_center,
        args.tray_opening, args.tray_wall_thickness, chosen_height,
        base_bounds, episode_targets, fully_validated,
        args.output_dir / "heal_workspace_xy.png",
    ))
    figures.append(plot_height_sweep(
        heights,
        np.asarray([row[3] for row in sweep_rows]),
        np.asarray([row[4] for row in sweep_rows]),
        args.output_dir / "heal_workspace_height_sweep.png",
    ))
    np.savez_compressed(
        args.output_dir / "heal_dexterous_workspace.npz",
        xs=xs,
        ys=ys,
        heights=heights,
        yaw_angles=yaw_angles,
        task_masks=np.stack(task_maps),
        dexterous_masks=np.stack(dex_maps),
        chosen_height=np.asarray(chosen_height),
        chosen_footprint=np.asarray(chosen_footprint),
        tray_center=np.asarray(tray_center),
        cube_size=np.asarray(args.cube_size),
        approach_clearance=np.asarray(args.approach_clearance),
        fixed_table_dimensions=np.asarray([
            chosen_height, x_max - x_min, y_max - y_min,
        ]),
        episode_names=np.asarray(list(episode_results)),
        episode_success=np.asarray([
            result["success"] for result in episode_results.values()
        ]),
        historical_target_names=np.asarray([
            result["target_name"] for result in legacy_path_results
        ]),
        historical_yaw_degrees=np.asarray([
            result["yaw_degrees"] for result in legacy_path_results
        ]),
        historical_path_success=np.asarray([
            result["success"] for result in legacy_path_results
        ]),
        historical_failure_reasons=np.asarray([
            result["reason"] for result in legacy_path_results
        ]),
    )
    write_summary(
        args.output_dir / "heal_table_workspace_summary.md",
        args, chosen_height, chosen_footprint, tray_center,
        episode_results, legacy_path_results, sweep_rows, base_bounds,
    )

    print("\n========== TABLE CANDIDATE ==========")
    print(f"H={chosen_height:.3f} m; L={x_max - x_min:.3f} m; W={y_max - y_min:.3f} m")
    print(f"World footprint X=[{x_min:.3f}, {x_max:.3f}], Y=[{y_min:.3f}, {y_max:.3f}] m.")
    print(f"Assumed tray center={tray_center}; dimensions remain fixed for this diagnosis.")
    print("Global table status: not globally validated; only explicit episode targets are path-tested.")
    print("Episode success requires IK, approach, withdrawal, cube transfer, tray placement, and release-retreat checks.")
    print("\n========== PLOT SUMMARY ==========")
    print(f"Sampled XY cells per height: {len(xs) * len(ys)}")
    print(f"Height layers: {len(heights)}; total tested position layers: {len(heights) * len(xs) * len(ys)}")
    print(f"XY grid resolution: {args.grid_step:.3f} m")
    print(f"Sampled grasp-center XYZ bounds: {task_points.min(axis=0)} to {task_points.max(axis=0)} m")
    print(f"Tested top-down yaw orientations: {args.yaw_count}; yaw step: {360 / args.yaw_count:.1f} degrees")
    print(
        "Position-orientation combinations tested: "
        f"{len(heights) * len(xs) * len(ys) * args.yaw_count} "
        f"({len(heights) * len(xs) * len(ys)} position-height cells × {args.yaw_count} yaws; "
        "each checks grasp and pre-grasp poses)"
    )
    print("Task workspace cells by height (at least one yaw):", [row[1] for row in sweep_rows])
    print("Dexterous workspace cells by height (all tested yaws):", [row[2] for row in sweep_rows])
    print("Episode outcomes:", {name: result["success"] for name, result in episode_results.items()})
    print(f"All PNGs and analysis data are in: {args.output_dir.resolve()}")

    backend_name = matplotlib.get_backend().lower()
    can_show = (
        not args.no_show
        and HAS_GRAPHICAL_DISPLAY
        and "agg" not in backend_name
    )
    if can_show:
        plt.show()
    else:
        print("Interactive display unavailable or disabled; PNG files were saved without showing windows.")
    for figure in figures:
        plt.close(figure)


if __name__ == "__main__":
    main()