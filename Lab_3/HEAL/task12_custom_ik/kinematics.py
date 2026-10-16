import mujoco
import numpy as np

class HEALKinematics:
    """Kinematics wrapper for 6-DOF HEAL robotic arm with Robotiq 2F-85 gripper."""
    def __init__(self, model):
        self.m = model
        self.d = mujoco.MjData(model)
        self.arm_joints = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
        self.qadr = np.array([model.joint(j).qposadr[0] for j in self.arm_joints])
        self.dadr = np.array([model.joint(j).dofadr[0] for j in self.arm_joints])
        self.lo = np.array([model.joint(j).range[0] for j in self.arm_joints])
        self.hi = np.array([model.joint(j).range[1] for j in self.arm_joints])
        self.site = model.site("tcp").id
        self._jp = np.zeros((3, model.nv))
        self._jr = np.zeros((3, model.nv))

    def set_q(self, q):
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)

    def tcp_pose(self):
        pos = self.d.site_xpos[self.site].copy()
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, self.d.site_xmat[self.site])
        return pos, quat

    def fk(self, q):
        self.set_q(q)
        return self.tcp_pose()

    def jacobian(self):
        mujoco.mj_jacSite(self.m, self.d, self._jp, self._jr, self.site)
        return np.vstack([self._jp[:, self.dadr], self._jr[:, self.dadr]])

    def within_limits(self, q, tol=1e-5):
        return bool(np.all(q >= self.lo - tol) and np.all(q <= self.hi + tol))

def quat_mul(a, b):
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, a, b)
    return out

def quat_conj(a):
    return np.array([a[0], -a[1], -a[2], -a[3]])

def orientation_error(q_target, q_current):
    qe = quat_mul(q_target, quat_conj(q_current))
    if qe[0] < 0:
        qe = -qe
    w = np.clip(qe[0], -1.0, 1.0)
    theta = 2.0 * np.arccos(w)
    if theta < 1e-6:
        return np.zeros(3)
    return theta * qe[1:] / np.sin(theta / 2.0)

def pose_error(pos_t, quat_t, pos_c, quat_c):
    return np.concatenate([pos_t - pos_c, orientation_error(quat_t, quat_c)])
