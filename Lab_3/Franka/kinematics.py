"""
kinematics.py - forward kinematics (FK), Jacobian and orientation maths.

FK  = "given joint angles q, where is the tool?"   -> tcp_pose(q)
J   = "if joints move a little (dq), how does the tool move (dx)?"  dx = J dq
IK  = the reverse of FK: "where must the joints be so the tool is HERE?"
      All IK solvers in ik_solvers.py call the functions in this file.

Quaternions: MuJoCo stores them SCALAR-FIRST  q = [w, x, y, z].
(The tutorial's derivation uses scalar-last [x, y, z, w] - same maths,
 just a different order.  Mixing the two up is the #1 bug, so be careful.)
"""
import mujoco
import numpy as np
import config as C


class Kin:
    """Small wrapper: a model + its own MjData used ONLY for kinematics.
    (We never step physics here - the IK 'imagines' joint configurations.)"""

    def __init__(self, model):
        self.m = model
        self.d = mujoco.MjData(model)
        self.qadr = np.array([model.joint(j).qposadr[0] for j in C.ARM_JOINTS])
        self.dadr = np.array([model.joint(j).dofadr[0] for j in C.ARM_JOINTS])
        self.lo = np.array([model.joint(j).range[0] for j in C.ARM_JOINTS])
        self.hi = np.array([model.joint(j).range[1] for j in C.ARM_JOINTS])
        self.site = model.site("tcp").id
        self.d.qpos[self.qadr] = C.Q_HOME
        for f in ("finger_joint1", "finger_joint2"):
            self.d.qpos[model.joint(f).qposadr[0]] = 0.04   # fingers open
        self._jp = np.zeros((3, model.nv))
        self._jr = np.zeros((3, model.nv))

    def set_q(self, q):
        """Put the 7 arm joints at q and update all link poses (FK)."""
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)   # link positions/orientations
        mujoco.mj_comPos(self.m, self.d)       # needed before mj_jacSite

    def tcp_pose(self):
        """Current TCP position (3,) and quaternion [w,x,y,z]."""
        pos = self.d.site_xpos[self.site].copy()
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, self.d.site_xmat[self.site])
        return pos, quat

    def fk(self, q):
        self.set_q(q)
        return self.tcp_pose()

    def jacobian(self):
        """6x7 geometric Jacobian of the TCP: rows 0-2 linear, 3-5 angular
        (both in the WORLD frame), columns = the 7 arm joints."""
        mujoco.mj_jacSite(self.m, self.d, self._jp, self._jr, self.site)
        return np.vstack([self._jp[:, self.dadr], self._jr[:, self.dadr]])

    def within_limits(self, q, tol=1e-6):
        return bool(np.all(q >= self.lo - tol) and np.all(q <= self.hi + tol))


# ------------------------------------------------------------ orientation
def quat_mul(a, b):
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, a, b)
    return out


def quat_conj(a):
    return np.array([a[0], -a[1], -a[2], -a[3]])   # inverse of a UNIT quaternion


def orientation_error(q_target, q_current):
    """Axis-angle error vector (3,) in the world frame, exactly as in the
    tutorial:  q_e = q_t (x) q_c^-1,  flip sign if w<0,  eps = theta * u."""
    qe = quat_mul(q_target, quat_conj(q_current))
    if qe[0] < 0:                       # hemisphere continuity (w is first!)
        qe = -qe
    w = np.clip(qe[0], -1.0, 1.0)
    theta = 2.0 * np.arccos(w)
    if theta < 1e-6:
        return np.zeros(3)
    return theta * qe[1:] / np.sin(theta / 2.0)


def pose_error(pos_t, quat_t, pos_c, quat_c):
    """6-vector [position error; orientation error] = desired 'twist' x_dot_d*dt."""
    return np.concatenate([pos_t - pos_c, orientation_error(quat_t, quat_c)])


def quat_down(yaw):
    """Gripper pointing straight down (TCP z-axis = -world z), rotated by yaw
    about the world vertical.  yaw = pi/2 is the Panda's home orientation."""
    cz, sz = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    R = Rz @ np.diag([1.0, -1.0, -1.0])
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.flatten())
    return q


def yaw_of(quat):
    """Yaw (rotation about world z) of a gripper-down quaternion."""
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, quat)
    R = R.reshape(3, 3)
    return np.arctan2(R[1, 0], R[0, 0])


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def slerp(q0, q1, t):
    """Spherical interpolation between two quaternions (0<=t<=1)."""
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    dot = np.dot(q0, q1)
    if dot < 0:
        q1, dot = -q1, -dot
    if dot > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    th = np.arccos(dot)
    return (np.sin((1 - t) * th) * q0 + np.sin(t * th) * q1) / np.sin(th)
