"""
tf_broadcaster_node.py -- HW1 Part 2, Task 3 (optional/bonus).

STARTER CODE. Node setup, the timer loop, and TF broadcasting
plumbing are complete and working. Your job is to fill in the TODOs
so that:

  1. A "space_frame" TF frame is broadcast at the origin (fixed,
     never moves) -- this is done for you.
  2. A "body_frame" TF frame is broadcast whose orientation is
     built from the SAME rotation_sequence idea as Task 1
     (01_rotation_sandbox.py), advancing one elemental rotation
     every `step_period` seconds.
  3. A ROS parameter `compose_frame` (string: "current" or "fixed")
     controls whether each new elemental rotation is composed about
     the current body frame or the fixed space frame -- and you can
     change it live with:

         ros2 param set /hw01_tf_broadcaster compose_frame fixed

     while rviz2 is running, and SEE the difference immediately.

Run (after `colcon build` and `source install/setup.bash` from
ros_ws/):

    ros2 run hw01_tf_demo tf_broadcaster_node

Then, in another terminal:

    rviz2  # Add a TF display, set Fixed Frame to "space_frame"

This mirrors Task 1 exactly (same current-frame-vs-fixed-frame
question), just animated live via TF instead of as two separate
recordings -- which is why it's a good candidate to reuse your
compose_sequence() logic from 01_rotation_sandbox.py here, rather
than re-deriving it.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster


from rcl_interfaces.msg import SetParametersResult
import sys
import threading


def Rx(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def Ry(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def Rz(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


ELEMENTARY_ROTATIONS = {"x": Rx, "y": Ry, "z": Rz}

# Sequence matching Task 1: 90° about z, then 90° about x (plus optional 60° about y)
STEP_SEQUENCE = [
    ("z", np.deg2rad(90)),
    ("x", np.deg2rad(90)),
    ("y", np.deg2rad(60)),
]


def R_to_quat_xyzw(R):
    """3x3 rotation matrix -> ROS-convention xyzw quaternion."""
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


class Hw01TfBroadcaster(Node):
    def __init__(self):
        super().__init__("hw01_tf_broadcaster")
        self.declare_parameter("compose_frame", "current")  # "current" or "fixed"
        self.declare_parameter("step_period", 1.5)  # seconds per elemental rotation

        self.tf_broadcaster = TransformBroadcaster(self)
        self.R_body = np.eye(3)
        self.step_index = 0

        # Parameter callback to handle live changes
        self.add_on_set_parameters_callback(self.parameters_callback)

        step_period = self.get_parameter("step_period").value
        self.timer = self.create_timer(step_period, self.on_timer)
        self.get_logger().info(
            "hw01_tf_broadcaster started.\n"
            "  Current frame mode: " + self.get_parameter("compose_frame").value + "\n"
            "  Live toggle via parameter: ros2 param set /hw01_tf_broadcaster compose_frame fixed\n"
            "  Live toggle via terminal: type 't' (or 'c'/'f') and press Enter."
        )

        # Start background keyboard thread if interactive terminal is available
        if sys.stdin.isatty():
            self.kbd_thread = threading.Thread(target=self._keyboard_listener, daemon=True)
            self.kbd_thread.start()

    def parameters_callback(self, params):
        for param in params:
            if param.name == "compose_frame":
                val = param.value.strip().lower()
                if val not in ("current", "fixed", "body", "space"):
                    return SetParametersResult(successful=False, reason="Must be 'current' or 'fixed'")
                canonical = "current" if val in ("current", "body") else "fixed"
                self.get_logger().info(
                    f">> Mode switched to '{canonical.upper()}' frame! Resetting orientation to identity."
                )
                self.R_body = np.eye(3)
                self.step_index = 0
        return SetParametersResult(successful=True)

    def _keyboard_listener(self):
        while rclpy.ok():
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                cmd = line.strip().lower()
                curr = self.get_parameter("compose_frame").value
                if cmd in ("t", "toggle"):
                    new_mode = "fixed" if curr == "current" else "current"
                    self.set_parameters([rclpy.parameter.Parameter("compose_frame", rclpy.Parameter.Type.STRING, new_mode)])
                elif cmd in ("c", "current"):
                    self.set_parameters([rclpy.parameter.Parameter("compose_frame", rclpy.Parameter.Type.STRING, "current")])
                elif cmd in ("f", "fixed"):
                    self.set_parameters([rclpy.parameter.Parameter("compose_frame", rclpy.Parameter.Type.STRING, "fixed")])
                elif cmd in ("r", "reset"):
                    self.R_body = np.eye(3)
                    self.step_index = 0
                    self.get_logger().info("Reset orientation to identity.")
            except Exception:
                break

    def on_timer(self):
        # Broadcast the fixed space frame (identity at origin, never moves)
        self.broadcast_frame("world", "space_frame", np.eye(3))

        # Advance through STEP_SEQUENCE
        seq_len = len(STEP_SEQUENCE)
        step_pos = self.step_index % seq_len
        axis, angle = STEP_SEQUENCE[step_pos]
        R_step = ELEMENTARY_ROTATIONS[axis](angle)

        frame = self.get_parameter("compose_frame").value.strip().lower()
        if frame in ("current", "body"):
            # Post-multiplication (compose on the right)
            self.R_body = self.R_body @ R_step
            rule = "R_new = R_old @ R_step (Right)"
        else:
            # Pre-multiplication (compose on the left)
            self.R_body = R_step @ self.R_body
            rule = "R_new = R_step @ R_old (Left)"

        self.get_logger().info(
            f"Step {step_pos + 1}/{seq_len}: Rot({axis}, {np.rad2deg(angle):.0f}°) "
            f"about {frame.upper()} frame -> {rule}"
        )
        self.step_index += 1

        # Broadcast the moving body frame
        self.broadcast_frame("world", "body_frame", self.R_body)

    def broadcast_frame(self, parent, child, R):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = parent
        t.child_frame_id = child
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 1.0 if child == "body_frame" else 0.0
        qx, qy, qz, qw = R_to_quat_xyzw(R)
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw
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
        rclpy.shutdown()


if __name__ == "__main__":
    main()
