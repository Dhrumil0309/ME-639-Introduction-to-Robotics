"""
tf_broadcaster_node.py -- ME 639 HW1 Task 3 (Problem 9)
=========================================================
Production-quality ROS 2 node broadcasting space and body TF frames
demonstrating current-frame (body) vs. fixed-frame (space) rotation
composition with smooth SLERP animation and interactive terminal control.

FEATURES:
  - Continuously broadcasts 'space_frame' (fixed at origin) and 'body_frame'
    at 60 Hz relative to 'world' (or 'space_frame' as parent).
  - SLERP interpolation smoothly rotates 'body_frame' from old to new orientation
    over `anim_duration` seconds (default 1.5s) so motion animates cleanly in RViz2.
  - Background keyboard thread reads commands non-blockingly from stdin:
      * 'z 90', 'x 90', 'y -45': queue an elemental rotation
      * 'toggle' or 't': toggle between CURRENT (body) and FIXED (space) frame
      * 'current' / 'body': set mode to body-frame (post-multiply: R_new = R_old @ R_step)
      * 'fixed' / 'space': set mode to space-frame (pre-multiply: R_new = R_step @ R_old)
      * 'reset' or 'r': reset orientation to identity
      * 'help' or 'h': display command help
      * 'quit' or 'q': shutdown node cleanly
  - Also integrates with ROS 2 parameters:
      ros2 param set /hw01_tf_broadcaster compose_frame fixed
"""

import sys
import threading
import queue
import time
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from rcl_interfaces.msg import SetParametersResult


def Rx(t: float) -> np.ndarray:
    c, s = np.cos(t), np.sin(t)
    return np.array([[1.0, 0.0, 0.0],
                     [0.0,   c,  -s],
                     [0.0,   s,   c]])


def Ry(t: float) -> np.ndarray:
    c, s = np.cos(t), np.sin(t)
    return np.array([[  c, 0.0,   s],
                     [0.0, 1.0, 0.0],
                     [ -s, 0.0,   c]])


def Rz(t: float) -> np.ndarray:
    c, s = np.cos(t), np.sin(t)
    return np.array([[  c,  -s, 0.0],
                     [  s,   c, 0.0],
                     [0.0, 0.0, 1.0]])


ELEMENTARY_ROTATIONS = {"x": Rx, "y": Ry, "z": Rz}


def R_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> ROS-convention unit quaternion [x, y, z, w]."""
    R = np.asarray(R, dtype=float)
    tr = np.trace(R)
    if tr > 0.0:
        S = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S
    q = np.array([x, y, z, w], dtype=float)
    return q / np.linalg.norm(q)


def quat_xyzw_to_R(q: np.ndarray) -> np.ndarray:
    """ROS-convention unit quaternion [x, y, z, w] -> 3x3 rotation matrix."""
    x, y, z, w = q
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1.0 - s * (y * y + z * z), s * (x * y - w * z), s * (x * z + w * y)],
        [s * (x * y + w * z), 1.0 - s * (x * x + z * z), s * (y * z - w * x)],
        [s * (x * z - w * y), s * (y * z + w * x), 1.0 - s * (x * x + y * y)],
    ])


def slerp_xyzw(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two [x, y, z, w] unit quaternions.
    Handles shortest-path hemisphere check so rotation never exceeds 180 degrees.
    """
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))

    if dot < 0.0:
        q1 = -q1
        dot = -dot

    if dot > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)

    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    theta = theta_0 * t
    q_perp = (q1 - q0 * dot) / np.linalg.norm(q1 - q0 * dot)
    return q0 * np.cos(theta) + q_perp * np.sin(theta)


class Hw01TfBroadcaster(Node):
    def __init__(self):
        super().__init__("hw01_tf_broadcaster")

        # Parameters
        self.declare_parameter("compose_frame", "current")  # 'current' (body) or 'fixed' (space)
        self.declare_parameter("anim_duration", 1.5)        # seconds per SLERP animation
        self.declare_parameter("broadcast_rate", 60.0)      # Hz

        self.tf_broadcaster = TransformBroadcaster(self)

        # Orientation state
        self.R_current = np.eye(3)
        self.q_current = R_to_quat_xyzw(self.R_current)

        # Animation state
        self.anim_active = False
        self.anim_start_time = 0.0
        self.anim_q_start = self.q_current.copy()
        self.anim_q_end = self.q_current.copy()
        self.anim_duration = self.get_parameter("anim_duration").value

        # Queue for interactive commands from terminal thread
        self.cmd_queue = queue.Queue()

        # Parameter callback
        self.add_on_set_parameters_callback(self.on_set_parameters)

        # 60 Hz timer loop
        rate = self.get_parameter("broadcast_rate").value
        self.timer = self.create_timer(1.0 / rate, self.timer_callback)

        # Print welcome banner
        self.print_banner()

        # Start non-blocking stdin reader thread
        self.kbd_thread = threading.Thread(target=self._terminal_input_loop, daemon=True)
        self.kbd_thread.start()

    def print_banner(self):
        mode = self.get_parameter("compose_frame").value.upper()
        print("\n" + "=" * 70)
        print("  ME 639 HW1 Task 3: TF Broadcaster & RViz2 Interactive Demo")
        print("=" * 70)
        print(f"  Current composition mode: [{mode}] frame")
        print("  Commands available in terminal:")
        print("    <axis> <angle> : e.g. 'z 90', 'x 90', 'y -45' (rotates body)")
        print("    toggle (or t)  : switch between CURRENT (body) and FIXED (space) frame")
        print("    current (or c) : use CURRENT frame (post-multiply: R_new = R_old @ R_step)")
        print("    fixed (or f)   : use FIXED frame   (pre-multiply:  R_new = R_step @ R_old)")
        print("    reset (or r)   : reset orientation to identity")
        print("    help (or h)    : show this command help")
        print("    quit (or q)    : exit node cleanly")
        print("=" * 70 + "\n")

    def on_set_parameters(self, params):
        for param in params:
            if param.name == "compose_frame":
                val = str(param.value).strip().lower()
                if val in ("current", "body"):
                    self.get_logger().info(">> Composition mode set to CURRENT (body) frame (post-mult)")
                elif val in ("fixed", "space"):
                    self.get_logger().info(">> Composition mode set to FIXED (space) frame (pre-mult)")
                else:
                    return SetParametersResult(successful=False, reason="Must be 'current' or 'fixed'")
            elif param.name == "anim_duration":
                self.anim_duration = float(param.value)
                self.get_logger().info(f">> Animation duration set to {self.anim_duration:.2f}s")
        return SetParametersResult(successful=True)

    def _terminal_input_loop(self):
        """Runs on a background thread reading lines from sys.stdin."""
        while rclpy.ok():
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                text = line.strip()
                if text:
                    self.cmd_queue.put(text)
            except Exception:
                break

    def timer_callback(self):
        now_sec = self.get_clock().now().nanoseconds * 1e-9

        # Process pending commands from terminal
        while not self.cmd_queue.empty():
            cmd = self.cmd_queue.get_nowait()
            self.handle_command(cmd, now_sec)

        # Update animation if active
        if self.anim_active:
            elapsed = now_sec - self.anim_start_time
            alpha = min(1.0, elapsed / max(0.01, self.anim_duration))
            self.q_current = slerp_xyzw(self.anim_q_start, self.anim_q_end, alpha)

            if alpha >= 1.0:
                self.anim_active = False
                self.q_current = self.anim_q_end.copy()
                self.R_current = quat_xyzw_to_R(self.q_current)

        # Broadcast frames
        stamp = self.get_clock().now().to_msg()

        # 1. Space frame: fixed at origin (parent = world)
        self.broadcast_transform(
            parent_frame="world",
            child_frame="space_frame",
            trans=(0.0, 0.0, 0.0),
            quat=(0.0, 0.0, 0.0, 1.0),
            stamp=stamp
        )

        # 2. Body frame: animated child of space_frame (or world)
        self.broadcast_transform(
            parent_frame="space_frame",
            child_frame="body_frame",
            trans=(0.0, 0.0, 0.8),
            quat=tuple(self.q_current),
            stamp=stamp
        )

    def handle_command(self, cmd: str, now_sec: float):
        tokens = cmd.strip().split()
        if not tokens:
            return

        action = tokens[0].lower()

        if action in ("q", "quit", "exit"):
            self.get_logger().info("Shutting down hw01_tf_broadcaster...")
            rclpy.shutdown()
            return

        elif action in ("t", "toggle"):
            curr = self.get_parameter("compose_frame").value.strip().lower()
            new_mode = "fixed" if curr in ("current", "body") else "current"
            self.set_parameters([
                rclpy.parameter.Parameter("compose_frame", rclpy.Parameter.Type.STRING, new_mode)
            ])
            self.get_logger().info(f"Mode toggled to: {new_mode.upper()} frame")
            return

        elif action in ("c", "current", "body"):
            self.set_parameters([
                rclpy.parameter.Parameter("compose_frame", rclpy.Parameter.Type.STRING, "current")
            ])
            self.get_logger().info("Mode set to: CURRENT (body) frame")
            return

        elif action in ("f", "fixed", "space"):
            self.set_parameters([
                rclpy.parameter.Parameter("compose_frame", rclpy.Parameter.Type.STRING, "fixed")
            ])
            self.get_logger().info("Mode set to: FIXED (space) frame")
            return

        elif action in ("r", "reset"):
            self.trigger_rotation(np.eye(3), now_sec, label="RESET to Identity")
            return

        elif action in ("h", "help"):
            self.print_banner()
            return

        # Check for rotation command: <axis> <angle_deg>
        if action in ("x", "y", "z") and len(tokens) >= 2:
            try:
                angle_deg = float(tokens[1])
                angle_rad = np.deg2rad(angle_deg)
                R_step = ELEMENTARY_ROTATIONS[action](angle_rad)

                frame_mode = self.get_parameter("compose_frame").value.strip().lower()

                # Start from latest target orientation if currently animating
                R_base = quat_xyzw_to_R(self.anim_q_end) if self.anim_active else self.R_current

                if frame_mode in ("current", "body"):
                    # Post-multiplication (body frame)
                    R_target = R_base @ R_step
                    rule = "R_new = R_old @ R_step [POST-mult, BODY frame]"
                else:
                    # Pre-multiplication (space frame)
                    R_target = R_step @ R_base
                    rule = "R_new = R_step @ R_old [PRE-mult, SPACE frame]"

                label = f"Rot({action.upper()}, {angle_deg:+.1f}°) via {rule}"
                self.trigger_rotation(R_target, now_sec, label=label)

            except ValueError:
                self.get_logger().warn(f"Invalid angle format: '{tokens[1]}'. Example: 'z 90'")
        else:
            self.get_logger().warn(f"Unknown command: '{cmd}'. Type 'h' or 'help' for command list.")

    def trigger_rotation(self, R_target: np.ndarray, now_sec: float, label: str = ""):
        self.anim_q_start = self.q_current.copy()
        self.anim_q_end = R_to_quat_xyzw(R_target)

        # Shortest-path check
        if np.dot(self.anim_q_start, self.anim_q_end) < 0.0:
            self.anim_q_end = -self.anim_q_end

        self.anim_start_time = now_sec
        self.anim_active = True

        if label:
            self.get_logger().info(f">> {label}")
            self.print_matrix(R_target, "Target Rotation Matrix R:")

    def print_matrix(self, R: np.ndarray, title: str):
        print(f"\n  {title}")
        print("          col-x       col-y       col-z")
        for lbl, row in zip(("  row-x │", "  row-y │", "  row-z │"), R):
            vals = "   ".join(f"{v:+10.5f}" for v in row)
            print(f"  {lbl}  {vals}")
        print()

    def broadcast_transform(self, parent_frame: str, child_frame: str,
                            trans: tuple, quat: tuple, stamp):
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = parent_frame
        t.child_frame_id = child_frame
        t.transform.translation.x = float(trans[0])
        t.transform.translation.y = float(trans[1])
        t.transform.translation.z = float(trans[2])
        t.transform.rotation.x = float(quat[0])
        t.transform.rotation.y = float(quat[1])
        t.transform.rotation.z = float(quat[2])
        t.transform.rotation.w = float(quat[3])
        self.tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = Hw01TfBroadcaster()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
