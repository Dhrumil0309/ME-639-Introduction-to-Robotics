import time
from dataclasses import dataclass
import mujoco
import numpy as np
import quadprog
import mink
from .kinematics import HEALKinematics, pose_error

@dataclass
class IKResult:
    q: np.ndarray
    converged: bool
    iters: int
    pos_err: float
    rot_err: float
    time: float

class IterativeIK:
    name = "base"
    def __init__(self, model, max_iters=5, pos_tol=1e-3, rot_tol=1e-2, gain=0.6):
        self.kin = HEALKinematics(model)
        self.max_iters = max_iters
        self.pos_tol = pos_tol
        self.rot_tol = rot_tol
        self.gain = gain
        self.mid = 0.5 * (self.kin.lo + self.kin.hi)

    def step(self, q, e, J):
        raise NotImplementedError

    def solve(self, q0, pos_t, quat_t):
        t0 = time.perf_counter()
        q = np.array(q0, dtype=float)
        for it in range(1, self.max_iters + 1):
            self.kin.set_q(q)
            pos, quat = self.kin.tcp_pose()
            e = pose_error(pos_t, quat_t, pos, quat)
            pe, re = np.linalg.norm(e[:3]), np.linalg.norm(e[3:])
            if pe < self.pos_tol and re < self.rot_tol:
                return IKResult(q, True, it, pe, re, time.perf_counter() - t0)
            J = self.kin.jacobian()
            q = q + self.step(q, e, J)
        return IKResult(q, False, self.max_iters, pe, re, time.perf_counter() - t0)

class DLSIK(IterativeIK):
    """Closed-loop Damped Least Squares IK with null-space posture centering."""
    name = "dls"
    def __init__(self, model, damping=1e-3, null_alpha=0.1, **kwargs):
        super().__init__(model, **kwargs)
        self.damping = damping
        self.null_alpha = null_alpha

    def step(self, q, e, J):
        n_joints = len(q)
        A = J.T @ J + self.damping * np.eye(n_joints)
        Ainv_JT = np.linalg.solve(A, J.T)
        dq_task = Ainv_JT @ (self.gain * e)
        N = np.eye(n_joints) - np.linalg.pinv(J, rcond=1e-4) @ J
        dq_null = self.null_alpha * (self.mid - q)
        return dq_task + N @ dq_null

class QPIK(IterativeIK):
    """QP-based IK with joint position limits and velocity limits."""
    name = "qp"
    def __init__(self, model, damping=1e-3, dt=0.01, v_max=np.pi, margin=0.05, **kwargs):
        super().__init__(model, **kwargs)
        self.damping = damping
        self.dt = dt
        self.v_max = v_max
        self.margin = margin

    def step(self, q, e, J):
        n_joints = len(q)
        dt = self.dt
        xdot_d = self.gain * e / dt
        H = (2.0 / dt**2) * (J.T @ J + self.damping * np.eye(n_joints))
        c = -(1.0 / dt) * J.T @ xdot_d

        lb = np.maximum(self.kin.lo + self.margin - q, -self.v_max * dt)
        ub = np.minimum(self.kin.hi - self.margin - q, self.v_max * dt)
        ub = np.maximum(ub, lb)

        G = np.vstack([np.eye(n_joints), -np.eye(n_joints)])
        h = np.concatenate([ub, -lb])

        try:
            dq, *_ = quadprog.solve_qp(H, -c, -G.T, -h)
        except ValueError:
            dq = np.linalg.pinv(J, rcond=1e-4) @ (self.gain * e)
        return dq

class MinkIKSolver:
    """Baseline Mink IK tracking solver."""
    name = "mink"
    def __init__(self, model, dt=0.01):
        self.model = model
        self.dt = dt
        self.configuration = mink.Configuration(model)
        self.tcp_task = mink.FrameTask(frame_name="tcp", frame_type="site",
                                       position_cost=1.0, orientation_cost=1.0)
        self.posture_task = mink.PostureTask(model, cost=1e-2)
        d_init = mujoco.MjData(model)
        mujoco.mj_forward(model, d_init)
        self.posture_task.set_target(d_init.qpos.copy())
        self.limits = [
            mink.ConfigurationLimit(model=model),
            mink.VelocityLimit(model, {f"joint_{i}": np.pi for i in range(1, 7)}),
        ]
        self.arm_joints = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
        self.arm_qadr = np.array([model.joint(j).qposadr[0] for j in self.arm_joints])
        self.arm_dadr = np.array([model.joint(j).dofadr[0] for j in self.arm_joints])
        self.mocap_id = model.body("target_mocap").mocapid[0]

    def solve(self, q0, pos_t, quat_t, max_iters=1, data=None):
        t0 = time.perf_counter()
        q = np.array(q0, dtype=float)
        if data is None:
            data = mujoco.MjData(self.model)
        data.mocap_pos[self.mocap_id] = pos_t
        data.mocap_quat[self.mocap_id] = quat_t
        mujoco.mj_kinematics(self.model, data)
        self.tcp_task.set_target(mink.SE3.from_mocap_name(self.model, data, "target_mocap"))

        q_full = data.qpos.copy()
        q_full[self.arm_qadr] = q
        self.configuration.update(q_full)
        vel = mink.solve_ik(self.configuration, [self.tcp_task, self.posture_task],
                            self.dt, solver="daqp", damping=1e-2, limits=self.limits)
        q = q + vel[self.arm_dadr] * self.dt
        return IKResult(q, True, 1, 0.0, 0.0, time.perf_counter() - t0)
