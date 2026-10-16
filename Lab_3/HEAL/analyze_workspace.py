"""Sample and visualize the HEAL arm's kinematic reachability."""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR / "Lab3" / "robot_descriptions"
MODEL_PATH = MODEL_DIR / "arm_gripper.xml"
ARM_JOINT_NAMES = [f"joint_{index}" for index in range(1, 7)]
PAD_BODY_NAMES = ("left_pad", "right_pad")


def load_model() -> mujoco.MjModel:
    """Load the combined robot model with its relative mesh paths."""
    original_dir = Path.cwd()
    try:
        os.chdir(MODEL_DIR)
        return mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    finally:
        os.chdir(original_dir)


def inspect_model(model: mujoco.MjModel) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Print model details and return arm joint addresses and limits."""
    joint_ids = []
    lower_limits = []
    upper_limits = []

    print("\n========== MODEL INFO ==========")
    print(f"Joints: {model.njnt}")
    print(f"Degrees of freedom: {model.nv}")
    print(f"Sites: {model.nsite}")
    print(f"Actuators: {model.nu}")

    print("\n========== JOINTS ==========")
    for joint_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        joint_type = int(model.jnt_type[joint_id])
        limits = model.jnt_range[joint_id] if model.jnt_limited[joint_id] else "unlimited"
        print(
            f"{joint_id}: {name}, type={joint_type}, "
            f"qpos address={model.jnt_qposadr[joint_id]}, limits={limits}"
        )

        if name in ARM_JOINT_NAMES:
            if joint_type != mujoco.mjtJoint.mjJNT_HINGE:
                raise ValueError(f"Expected {name} to be a hinge joint")
            if not model.jnt_limited[joint_id]:
                raise ValueError(f"Expected {name} to have joint limits")
            joint_ids.append(joint_id)
            lower_limits.append(model.jnt_range[joint_id, 0])
            upper_limits.append(model.jnt_range[joint_id, 1])

    if len(joint_ids) != len(ARM_JOINT_NAMES):
        raise ValueError("The model must contain all six arm joints joint_1 through joint_6")

    print("\n========== SITES ==========")
    for site_id in range(model.nsite):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, site_id)
        print(f"{site_id}: {name}")

    print("\n========== BODIES ==========")
    for body_id in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        print(f"{body_id}: {name}")

    return joint_ids, np.asarray(lower_limits), np.asarray(upper_limits)


def sample_grasp_centers(
    model: mujoco.MjModel,
    joint_ids: list[int],
    lower_limits: np.ndarray,
    upper_limits: np.ndarray,
    sample_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample arm joint configurations and return grasp-center positions."""
    pad_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in PAD_BODY_NAMES
    ]
    if any(body_id < 0 for body_id in pad_ids):
        raise ValueError("Could not find both gripper pad bodies in the model")

    qpos_addresses = model.jnt_qposadr[joint_ids]
    rng = np.random.default_rng(seed)
    joint_samples = rng.uniform(lower_limits, upper_limits, size=(sample_count, len(joint_ids)))
    positions = np.empty((sample_count, 3), dtype=float)
    data = mujoco.MjData(model)

    for sample_id, joint_values in enumerate(joint_samples):
        data.qpos[qpos_addresses] = joint_values
        mujoco.mj_forward(model, data)
        positions[sample_id] = (data.xpos[pad_ids[0]] + data.xpos[pad_ids[1]]) / 2

    return positions, joint_samples


def plot_workspace(
    model: mujoco.MjModel,
    positions: np.ndarray,
    voxel_size: float,
    output_path: Path,
    show_plot: bool,
) -> None:
    """Plot sampled grasp centers and annotate their XYZ voxel-union domain."""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    home_pad_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in PAD_BODY_NAMES
    ]
    home_position = (data.xpos[home_pad_ids[0]] + data.xpos[home_pad_ids[1]]) / 2
    domain_minimum, domain_voxels = classify_voxels(positions, voxel_size)
    domain_maximum = domain_minimum + np.asarray(domain_voxels.shape) * voxel_size
    domain_text = (
        f"Sampled voxel-union domain ({voxel_size:g} m cells):\n"
        f"X [{domain_minimum[0]:.2f}, {domain_maximum[0]:.2f}] m   "
        f"Y [{domain_minimum[1]:.2f}, {domain_maximum[1]:.2f}] m\n"
        f"Z [{domain_minimum[2]:.2f}, {domain_maximum[2]:.2f}] m"
    )

    figure = plt.figure(figsize=(10, 8))
    axes = figure.add_subplot(111, projection="3d")
    points = axes.scatter(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        c=positions[:, 2],
        cmap="viridis",
        s=3,
        alpha=0.45,
        linewidths=0,
    )
    axes.scatter(
        *home_position,
        color="crimson",
        marker="*",
        s=140,
        label="Default grasp center",
    )
    axes.scatter(0, 0, 0, color="black", marker="^", s=65, label="Robot base")
    axes.set_xlabel("X (m)")
    axes.set_ylabel("Y (m)")
    axes.set_zlabel("Z (m)")
    axes.set_title("HEAL Arm Kinematic Reachability\nGripper grasp-center samples")
    axes.set_box_aspect(np.maximum(np.ptp(positions, axis=0), 0.05))
    axes.legend(loc="upper left")
    figure.text(0.5, 0.01, domain_text, ha="center", va="bottom", fontsize=9)
    figure.colorbar(points, ax=axes, label="Height Z (m)", shrink=0.7, pad=0.1)
    figure.tight_layout(rect=(0, 0.09, 1, 1))
    figure.savefig(output_path, dpi=180)
    print(f"\n3D workspace plot saved to: {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(figure)


def plot_reachability_map(
    positions: np.ndarray,
    output_path: Path,
    voxel_size: float,
    show_plot: bool,
) -> tuple[int, int]:
    """Plot sampled reachable voxels green and unobserved voxels red."""
    minimum, reachable = classify_voxels(positions, voxel_size)
    grid_shape = np.asarray(reachable.shape)
    maximum = minimum + grid_shape * voxel_size
    grid_indices = np.indices(tuple(grid_shape)).reshape(3, -1).T
    cell_centers = minimum + (grid_indices + 0.5) * voxel_size
    flat_reachable = reachable.ravel()
    reached_centers = cell_centers[flat_reachable]
    unreached_centers = cell_centers[~flat_reachable]
    if len(unreached_centers) > 3500:
        red_rng = np.random.default_rng(0)
        red_indices = red_rng.choice(len(unreached_centers), size=3500, replace=False)
        displayed_unreached = unreached_centers[red_indices]
    else:
        displayed_unreached = unreached_centers

    figure = plt.figure(figsize=(10, 8))
    axes = figure.add_subplot(111, projection="3d")
    if len(displayed_unreached):
        axes.scatter(
            displayed_unreached[:, 0],
            displayed_unreached[:, 1],
            displayed_unreached[:, 2],
            color="#d94b45",
            s=9,
            alpha=0.65,
            linewidths=0,
            depthshade=False,
        )
    axes.scatter(
        reached_centers[:, 0],
        reached_centers[:, 1],
        reached_centers[:, 2],
        color="#27864b",
        s=10,
        alpha=0.95,
        linewidths=0,
        depthshade=False,
    )
    axes.scatter(0, 0, 0, color="black", marker="^", s=65, label="Robot base")
    axes.set_xlabel("X (m)")
    axes.set_ylabel("Y (m)")
    axes.set_zlabel("Z (m)")
    axes.set_title(
        "Estimated HEAL Grasp-Center Reachability\n"
        f"{voxel_size:.3f} m cells; red means not reached in random sampling"
    )
    axes.set_box_aspect(np.maximum(maximum - minimum, 0.05))
    legend_items = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#27864b", markersize=8,
               label=f"Sampled reachable ({len(reached_centers)} cells)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#d94b45", markersize=8,
               label=f"Not reached in sample ({len(unreached_centers)} cells)"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="black", markersize=8,
               label="Robot base"),
    ]
    axes.legend(handles=legend_items, loc="upper left")
    figure.text(
        0.5,
        0.025,
        "Estimate only: unsampled cells are not proven unreachable; collisions are not checked.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    figure.savefig(output_path, dpi=180)
    print(f"Reachability classification plot saved to: {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(figure)

    return len(reached_centers), len(unreached_centers)


def classify_voxels(positions: np.ndarray, voxel_size: float) -> tuple[np.ndarray, np.ndarray]:
    """Mark cells containing at least one sampled grasp-center position."""
    minimum = positions.min(axis=0)
    maximum = positions.max(axis=0)
    grid_shape = np.maximum(np.ceil((maximum - minimum) / voxel_size).astype(int), 1)
    sample_indices = np.floor((positions - minimum) / voxel_size).astype(int)
    sample_indices = np.minimum(sample_indices, grid_shape - 1)
    reachable = np.zeros(tuple(grid_shape), dtype=bool)
    reachable[tuple(sample_indices.T)] = True
    return minimum, reachable


def voxel_boundary_faces(
    occupied: np.ndarray,
    minimum: np.ndarray,
    voxel_size: float,
) -> list[np.ndarray]:
    """Return quads for voxel faces exposed to a different cell class."""
    face_offsets = (
        (0, 0, ((0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1))),
        (0, 1, ((1, 0, 0), (1, 0, 1), (1, 1, 1), (1, 1, 0))),
        (1, 0, ((0, 0, 0), (0, 0, 1), (1, 0, 1), (1, 0, 0))),
        (1, 1, ((0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1))),
        (2, 0, ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0))),
        (2, 1, ((0, 0, 1), (0, 1, 1), (1, 1, 1), (1, 0, 1))),
    )
    faces = []

    for axis, side, offsets in face_offsets:
        neighbor = np.zeros_like(occupied)
        current_slice = [slice(None)] * 3
        neighbor_slice = [slice(None)] * 3
        if side == 0:
            current_slice[axis] = slice(1, None)
            neighbor_slice[axis] = slice(None, -1)
        else:
            current_slice[axis] = slice(None, -1)
            neighbor_slice[axis] = slice(1, None)
        neighbor[tuple(current_slice)] = occupied[tuple(neighbor_slice)]
        exposed = occupied & ~neighbor
        indices = np.argwhere(exposed)
        if len(indices):
            corners = indices[:, None, :] + np.asarray(offsets)[None, :, :]
            faces.extend(minimum + corners * voxel_size)

    return faces


def plot_shaded_workspace(
    positions: np.ndarray,
    output_path: Path,
    voxel_size: float,
    show_plot: bool,
) -> tuple[int, int]:
    """Save a shaded reachable/not-sampled voxel volume."""
    minimum, reachable = classify_voxels(positions, voxel_size)
    maximum = minimum + np.asarray(reachable.shape) * voxel_size
    reachable_faces = voxel_boundary_faces(reachable, minimum, voxel_size)
    bounds = np.array(
        [
            [minimum[0], minimum[1], minimum[2]],
            [maximum[0], minimum[1], minimum[2]],
            [maximum[0], maximum[1], minimum[2]],
            [minimum[0], maximum[1], minimum[2]],
            [minimum[0], minimum[1], maximum[2]],
            [maximum[0], minimum[1], maximum[2]],
            [maximum[0], maximum[1], maximum[2]],
            [minimum[0], maximum[1], maximum[2]],
        ]
    )
    bounds_faces = [
        bounds[[0, 1, 2, 3]],
        bounds[[4, 7, 6, 5]],
        bounds[[0, 4, 5, 1]],
        bounds[[1, 5, 6, 2]],
        bounds[[2, 6, 7, 3]],
        bounds[[3, 7, 4, 0]],
    ]

    figure = plt.figure(figsize=(10, 8))
    axes = figure.add_subplot(111, projection="3d")
    axes.add_collection3d(
        Poly3DCollection(
            bounds_faces,
            facecolors="#d94b45",
            edgecolors="#d94b45",
            linewidths=0.25,
            alpha=0.035,
        )
    )
    axes.add_collection3d(
        Poly3DCollection(
            reachable_faces,
            facecolors="#27864b",
            edgecolors="none",
            alpha=0.7,
        )
    )
    axes.scatter(0, 0, 0, color="black", marker="^", s=65, label="Robot base")
    axes.set_xlim(minimum[0], maximum[0])
    axes.set_ylim(minimum[1], maximum[1])
    axes.set_zlim(minimum[2], maximum[2])
    axes.set_xlabel("X (m)")
    axes.set_ylabel("Y (m)")
    axes.set_zlabel("Z (m)")
    axes.set_title(
        "Shaded HEAL Grasp-Center Reachability Estimate\n"
        f"Green is sampled reachable volume; red frame is the sample bounds ({voxel_size:.3f} m voxels)"
    )
    axes.set_box_aspect(np.maximum(maximum - minimum, 0.05))
    legend_items = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#27864b", markersize=10,
               label="Sampled reachable voxels"),
         Line2D([0], [0], color="#d94b45", linewidth=2,
             label="Bounds of sampled positions"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="black", markersize=8,
               label="Robot base"),
    ]
    axes.legend(handles=legend_items, loc="upper left")
    figure.text(
        0.5,
        0.025,
        "Red frame is not an unreachable boundary; sampling is finite and collisions are not checked.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    figure.savefig(output_path, dpi=180)
    print(f"Shaded workspace plot saved to: {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(figure)

    return int(reachable.sum()), int(reachable.size - reachable.sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--samples",
        type=int,
        default=20000,
        help="number of random arm configurations to sample (default: 20000)",
    )
    parser.add_argument("--seed", type=int, default=7, help="random seed (default: 7)")
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "workspace_reachability.png",
        help="path for the original sampled point-cloud plot",
    )
    parser.add_argument(
        "--classified-output",
        type=Path,
        default=SCRIPT_DIR / "workspace_reachability_classified.png",
        help="path for the green/red reachability classification plot",
    )
    parser.add_argument(
        "--shaded-output",
        type=Path,
        default=SCRIPT_DIR / "workspace_reachability_shaded.png",
        help="path for the shaded 3D voxel plot",
    )
    parser.add_argument(
        "--domain-output",
        type=Path,
        default=SCRIPT_DIR / "workspace_domain.npz",
        help="path for the reusable sampled XYZ voxel-union data",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.05,
        help="voxel resolution in metres for the classification plot (default: 0.05)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="save the plot without opening an interactive window",
    )
    args = parser.parse_args()

    if args.samples < 1:
        parser.error("--samples must be at least 1")
    if args.voxel_size <= 0:
        parser.error("--voxel-size must be greater than 0")

    model = load_model()
    joint_ids, lower_limits, upper_limits = inspect_model(model)
    positions, _ = sample_grasp_centers(
        model,
        joint_ids,
        lower_limits,
        upper_limits,
        args.samples,
        args.seed,
    )

    domain_minimum, domain_voxels = classify_voxels(positions, args.voxel_size)
    domain_maximum = domain_minimum + np.asarray(domain_voxels.shape) * args.voxel_size
    occupied_count = int(domain_voxels.sum())
    union_volume = occupied_count * args.voxel_size**3

    print("\n========== SAMPLED GRASP-CENTER WORKSPACE ==========")
    print(f"Configurations sampled: {len(positions)}")
    print("Minimum XYZ (m):", positions.min(axis=0))
    print("Maximum XYZ (m):", positions.max(axis=0))
    print("Extent XYZ (m):", np.ptp(positions, axis=0))
    print(f"\n========== XYZ VOXEL-UNION DOMAIN ({args.voxel_size:g} m cells) ==========")
    for axis, axis_minimum, axis_maximum in zip("XYZ", domain_minimum, domain_maximum):
        print(f"{axis}: [{axis_minimum:.4f}, {axis_maximum:.4f}] m")
    print(f"Grid shape (X, Y, Z): {domain_voxels.shape}")
    print(f"Occupied cells: {occupied_count}")
    print(f"Approximate union volume: {union_volume:.4f} m^3")
    print(
        "This is the union of sampled occupied voxels, not the exact continuous workspace; "
        "collision-free reachability is not tested."
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plot_workspace(model, positions, args.voxel_size, args.output, show_plot=not args.no_show)
    args.classified_output.parent.mkdir(parents=True, exist_ok=True)
    reachable_cells, unreached_cells = plot_reachability_map(
        positions,
        args.classified_output,
        args.voxel_size,
        show_plot=not args.no_show,
    )
    print(f"Reachability voxels: {reachable_cells} sampled, {unreached_cells} not sampled")
    args.shaded_output.parent.mkdir(parents=True, exist_ok=True)
    shaded_reachable, shaded_unreached = plot_shaded_workspace(
        positions,
        args.shaded_output,
        args.voxel_size,
        show_plot=not args.no_show,
    )
    print(f"Shaded voxels: {shaded_reachable} sampled, {shaded_unreached} not sampled")
    args.domain_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.domain_output,
        reachable_voxels=domain_voxels,
        origin_xyz=domain_minimum,
        voxel_size_m=np.asarray(args.voxel_size),
        bounds_min_xyz=domain_minimum,
        bounds_max_xyz=domain_maximum,
        sample_positions_xyz=positions,
    )
    print(f"Voxel-union domain data saved to: {args.domain_output}")


if __name__ == "__main__":
    main()