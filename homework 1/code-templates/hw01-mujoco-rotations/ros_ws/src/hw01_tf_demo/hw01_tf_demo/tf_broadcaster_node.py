"""
tf_broadcaster_node.py -- ME 639 HW1 Task 3 (Problem 9)
=========================================================
Production-quality ROS 2 node that integrates ROS 2 Interactive Markers
directly inside RViz2.

FEATURES:
  - Initializes an InteractiveMarkerServer under the namespace 'rotation_controls'.
  - Creates an Interactive Marker with 3 rotational controls (rings for X, Y, Z).
  - Attaches a MenuHandler with 3 right-click context menu options:
      1. "Set Mode: Current (Body) Frame"
      2. "Set Mode: Fixed (Space) Frame"
      3. "Reset to Identity"
  - Interactive Kinematic Processing:
      * Captures delta rotation dragged by user from marker feedback.
      * Computes delta rotation matrix: R_delta = R_pose @ (R_prev_pose)^T
      * In Current (Body) Frame mode:
          R_body = R_body @ R_delta (post-multiplication)
      * In Fixed (Space) Frame mode:
          R_body = R_delta @ R_body (pre-multiplication)
  - Continuously broadcasts 'space_frame' (fixed at origin) and 'body_frame'
    at 60 Hz relative to 'space_frame'.
  - Syncs the interactive marker pose to match R_body so the user sees the rings
    move together with the body frame.
"""

import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped, Pose, Quaternion
from tf2_ros import TransformBroadcaster
from rcl_interfaces.msg import SetParametersResult

from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from interactive_markers.menu_handler import MenuHandler
from visualization_msgs.msg import (
    InteractiveMarker,
    InteractiveMarkerControl,
    InteractiveMarkerFeedback,
    Marker,
)


def R_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> ROS unit quaternion [x, y, z, w]."""
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


def quat_xyzw_to_R(q) -> np.ndarray:
    """ROS unit quaternion (array-like or geometry_msgs/Quaternion) -> 3x3 rotation matrix."""
    if hasattr(q, "x"):
        x, y, z, w = q.x, q.y, q.z, q.w
    else:
        x, y, z, w = q[0], q[1], q[2], q[3]

    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1.0 - s * (y * y + z * z), s * (x * y - w * z), s * (x * z + w * y)],
        [s * (x * y + w * z), 1.0 - s * (x * x + z * z), s * (y * z - w * x)],
        [s * (x * z - w * y), s * (y * z + w * x), 1.0 - s * (x * x + y * y)],
    ])


class Hw01TfBroadcaster(Node):
    def __init__(self):
        super().__init__("hw01_tf_broadcaster")

        # Parameters
        self.declare_parameter("compose_frame", "current")  # 'current' (body) or 'fixed' (space)
        self.declare_parameter("broadcast_rate", 60.0)      # Hz

        self.tf_broadcaster = TransformBroadcaster(self)

        # Kinematic state: current 3x3 rotation matrix
        self.R_body = np.eye(3)
        self.q_body = R_to_quat_xyzw(self.R_body)

        # Track previous dragged marker orientation to compute relative deltas
        self.prev_drag_R = np.eye(3)
        self.is_dragging = False

        # Interactive Marker Server and Menu Handler
        self.server = InteractiveMarkerServer(self, "rotation_controls")
        self.menu_handler = MenuHandler()

        # Context menu items
        self.h_body_mode = self.menu_handler.insert(
            "Set Mode: Current (Body) Frame", callback=self.menu_feedback_cb
        )
        self.h_space_mode = self.menu_handler.insert(
            "Set Mode: Fixed (Space) Frame", callback=self.menu_feedback_cb
        )
        self.h_reset = self.menu_handler.insert(
            "Reset to Identity", callback=self.menu_feedback_cb
        )

        # Set initial checkmark on menu
        self.update_menu_checkmarks()

        # Create the Interactive Marker in RViz2
        self.create_interactive_marker()

        # Parameter callback
        self.add_on_set_parameters_callback(self.on_set_parameters)

        # 60 Hz timer loop for continuous TF broadcasting
        rate = self.get_parameter("broadcast_rate").value
        self.timer = self.create_timer(1.0 / rate, self.timer_callback)

        self.get_logger().info("=" * 65)
        self.get_logger().info("  ME 639 HW1 Task 3: Interactive Marker TF Broadcaster")
        self.get_logger().info(f"  Initial mode: [{self.get_parameter('compose_frame').value.upper()}] frame")
        self.get_logger().info("  Interact in RViz2:")
        self.get_logger().info("    - Drag the rotation rings (X/Y/Z) to rotate the body.")
        self.get_logger().info("    - Right-click the marker to toggle Body vs Space frame or Reset.")
        self.get_logger().info("=" * 65)

    def update_menu_checkmarks(self):
        mode = self.get_parameter("compose_frame").value.strip().lower()
        if mode in ("current", "body"):
            self.menu_handler.setCheckState(self.h_body_mode, MenuHandler.CHECKED)
            self.menu_handler.setCheckState(self.h_space_mode, MenuHandler.UNCHECKED)
        else:
            self.menu_handler.setCheckState(self.h_body_mode, MenuHandler.UNCHECKED)
            self.menu_handler.setCheckState(self.h_space_mode, MenuHandler.CHECKED)

    def on_set_parameters(self, params):
        for param in params:
            if param.name == "compose_frame":
                val = str(param.value).strip().lower()
                if val in ("current", "body"):
                    self.get_logger().info(">> Mode switched to: CURRENT (body) frame [post-multiplication]")
                elif val in ("fixed", "space"):
                    self.get_logger().info(">> Mode switched to: FIXED (space) frame [pre-multiplication]")
                else:
                    return SetParametersResult(successful=False, reason="Must be 'current' or 'fixed'")
                self.update_menu_checkmarks()
                self.menu_handler.reApply(self.server)
                self.server.applyChanges()
        return SetParametersResult(successful=True)

    def create_interactive_marker(self):
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = "space_frame"
        int_marker.name = "body_rotation_marker"
        int_marker.description = "Right-click: Menu | Drag rings: Rotate"
        int_marker.scale = 1.2

        # Position at the body origin (z = 0.8)
        int_marker.pose.position.x = 0.0
        int_marker.pose.position.y = 0.0
        int_marker.pose.position.z = 0.8

        qx, qy, qz, qw = self.q_body
        int_marker.pose.orientation.x = qx
        int_marker.pose.orientation.y = qy
        int_marker.pose.orientation.z = qz
        int_marker.pose.orientation.w = qw

        # Visual center sphere marker
        center_marker = Marker()
        center_marker.type = Marker.SPHERE
        center_marker.scale.x = 0.15
        center_marker.scale.y = 0.15
        center_marker.scale.z = 0.15
        center_marker.color.r = 0.9
        center_marker.color.g = 0.7
        center_marker.color.b = 0.1
        center_marker.color.a = 0.8

        menu_control = InteractiveMarkerControl()
        menu_control.interaction_mode = InteractiveMarkerControl.BUTTON
        menu_control.always_visible = True
        menu_control.markers.append(center_marker)
        int_marker.controls.append(menu_control)

        # 3 Rotation ring controls: X, Y, Z
        # 1. Rotate about X-axis
        ctrl_x = InteractiveMarkerControl()
        ctrl_x.name = "rotate_x"
        ctrl_x.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        ctrl_x.orientation.w = 1.0
        ctrl_x.orientation.x = 1.0
        ctrl_x.orientation.y = 0.0
        ctrl_x.orientation.z = 0.0
        int_marker.controls.append(ctrl_x)

        # 2. Rotate about Y-axis
        ctrl_y = InteractiveMarkerControl()
        ctrl_y.name = "rotate_y"
        ctrl_y.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        ctrl_y.orientation.w = 1.0
        ctrl_y.orientation.x = 0.0
        ctrl_y.orientation.y = 1.0
        ctrl_y.orientation.z = 0.0
        int_marker.controls.append(ctrl_y)

        # 3. Rotate about Z-axis
        ctrl_z = InteractiveMarkerControl()
        ctrl_z.name = "rotate_z"
        ctrl_z.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        ctrl_z.orientation.w = 1.0
        ctrl_z.orientation.x = 0.0
        ctrl_z.orientation.y = 0.0
        ctrl_z.orientation.z = 1.0
        int_marker.controls.append(ctrl_z)

        # Add to server with feedback callback
        self.server.insert(int_marker, feedback_callback=self.process_feedback)
        self.menu_handler.apply(self.server, int_marker.name)
        self.server.applyChanges()

    def menu_feedback_cb(self, feedback: InteractiveMarkerFeedback):
        entry_id = feedback.menu_entry_id

        if entry_id == self.h_body_mode:
            self.set_parameters([
                rclpy.parameter.Parameter("compose_frame", rclpy.Parameter.Type.STRING, "current")
            ])
            self.get_logger().info(">> Menu: Set Mode -> CURRENT (Body) Frame [post-multiplication]")

        elif entry_id == self.h_space_mode:
            self.set_parameters([
                rclpy.parameter.Parameter("compose_frame", rclpy.Parameter.Type.STRING, "fixed")
            ])
            self.get_logger().info(">> Menu: Set Mode -> FIXED (Space) Frame [pre-multiplication]")

        elif entry_id == self.h_reset:
            self.get_logger().info(">> Menu: Reset to Identity")
            self.R_body = np.eye(3)
            self.q_body = R_to_quat_xyzw(self.R_body)
            self.prev_drag_R = np.eye(3)
            self.sync_marker_pose()
            self.print_matrix(self.R_body, "Reset Rotation Matrix R:")

    def process_feedback(self, feedback: InteractiveMarkerFeedback):
        event_type = feedback.event_type

        # When drag starts, record the current marker orientation as baseline
        if event_type == InteractiveMarkerFeedback.MOUSE_DOWN:
            self.prev_drag_R = quat_xyzw_to_R(feedback.pose.orientation)
            self.is_dragging = True

        elif event_type == InteractiveMarkerFeedback.POSE_UPDATE:
            R_marker_now = quat_xyzw_to_R(feedback.pose.orientation)

            if not self.is_dragging:
                self.prev_drag_R = R_marker_now
                self.is_dragging = True
                return

            # Compute delta rotation: R_delta = R_marker_now @ (R_prev_drag)^T
            R_delta = R_marker_now @ self.prev_drag_R.T

            # Filter out tiny jitter
            if np.linalg.norm(R_delta - np.eye(3), ord='fro') < 1e-4:
                return

            frame_mode = self.get_parameter("compose_frame").value.strip().lower()

            if frame_mode in ("current", "body"):
                # Post-multiplication: R_new = R_old @ R_delta (body frame)
                self.R_body = self.R_body @ R_delta
            else:
                # Pre-multiplication: R_new = R_delta @ R_old (space frame)
                self.R_body = R_delta @ self.R_body

            # Re-orthogonalize R_body to prevent numerical drift
            u, _, vt = np.linalg.svd(self.R_body)
            self.R_body = u @ vt

            self.q_body = R_to_quat_xyzw(self.R_body)
            self.prev_drag_R = R_marker_now

        elif event_type == InteractiveMarkerFeedback.MOUSE_UP:
            self.is_dragging = False
            # When user releases mouse, align the marker orientation directly to R_body
            self.sync_marker_pose()
            mode = self.get_parameter("compose_frame").value.upper()
            rule = "POST-mult (Body)" if mode == "CURRENT" else "PRE-mult (Space)"
            self.print_matrix(self.R_body, f"Current R_body ({mode} Frame - {rule}):")

    def sync_marker_pose(self):
        new_pose = Pose()
        new_pose.position.x = 0.0
        new_pose.position.y = 0.0
        new_pose.position.z = 0.8
        qx, qy, qz, qw = self.q_body
        new_pose.orientation.x = qx
        new_pose.orientation.y = qy
        new_pose.orientation.z = qz
        new_pose.orientation.w = qw

        self.server.setPose("body_rotation_marker", new_pose)
        self.server.applyChanges()

    def print_matrix(self, R: np.ndarray, title: str):
        print(f"\n  {title}")
        print("          col-x       col-y       col-z")
        for lbl, row in zip(("  row-x │", "  row-y │", "  row-z │"), R):
            vals = "   ".join(f"{v:+10.5f}" for v in row)
            print(f"  {lbl}  {vals}")
        print()

    def timer_callback(self):
        stamp = self.get_clock().now().to_msg()

        # 1. Broadcast space_frame fixed at world origin
        self.broadcast_transform(
            parent_frame="world",
            child_frame="space_frame",
            trans=(0.0, 0.0, 0.0),
            quat=(0.0, 0.0, 0.0, 1.0),
            stamp=stamp
        )

        # 2. Broadcast body_frame relative to space_frame
        self.broadcast_transform(
            parent_frame="space_frame",
            child_frame="body_frame",
            trans=(0.0, 0.0, 0.8),
            quat=tuple(self.q_body),
            stamp=stamp
        )

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
