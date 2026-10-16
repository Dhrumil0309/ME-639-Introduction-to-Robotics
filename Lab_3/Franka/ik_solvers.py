"""
ik_solvers.py - four Inverse-Kinematics solvers with ONE common interface.

    solver = make_solver("dls", ik_model)
    res = solver.solve(q_start, target_pos, target_quat)
    res.q, res.converged, res.iters, res.pos_err, res.rot_err

Because every solver has the same .solve(), pick_place.py does not care
which one it is using - this is what makes the fair comparison possible.

  mink  - off-the-shelf library (Stage 1 baseline). Internally Mink also builds
          a QP, and it supports collision-avoidance limits.
  pinv  - "method of my choice": classic Newton-Raphson with the Jacobian
          pseudo-inverse, NO damping.  Simple, but blows up near singularities.
  dls   - Damped Least Squares (closed loop) + null-space joint centring.
  qp    - the same DLS cost written as a Quadratic Program with joint-limit
          and joint-velocity constraints (solved with quadprog).

All iterative custom solvers run the same closed loop:
    repeat:  e = pose error ;  if small -> stop
             J = Jacobian   ;  dq = STEP(J, e) ;  q = q + dq
Only STEP differs.  That is the whole idea of numerical IK.
"""
import time
from dataclasses import dataclass

import mujoco
import numpy as np
import quadprog

import config as C
from kinematics import Kin, pose_error


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

    def __init__(self, model):
        self.kin = Kin(model)
        self.mid = 0.5 * (self.kin.lo + self.kin.hi)   # middle of joint ranges

    def step(self, q, e, J):            # overridden by each method
        raise NotImplementedError

    def solve(self, q0, pos_t, quat_t, max_iters=C.MAX_IK_ITERS):
        t0 = time.perf_counter()
        q = np.array(q0, dtype=float)
        for it in range(max_iters + 1):
            self.kin.set_q(q)                         # FK
            pos, quat = self.kin.tcp_pose()
            e = pose_error(pos_t, quat_t, pos, quat)  # 6-vector error
            pe, re = np.linalg.norm(e[:3]), np.linalg.norm(e[3:])
            if pe < C.POS_TOL and re < C.ROT_TOL:
                return IKResult(q, True, it, pe, re, time.perf_counter() - t0)
            if it == max_iters:
                break
            J = self.kin.jacobian()
            q = q + self.step(q, e, J)
        return IKResult(q, False, it, pe, re, time.perf_counter() - t0)


class PinvIK(IterativeIK):
    """dq = J^+ (K e).  J^+ = Moore-Penrose pseudo-inverse.
    Near a singularity a singular value sigma -> 0, 1/sigma -> huge,
    so dq explodes.  We only clip the step so the program does not crash."""
    name = "pinv"

    def step(self, q, e, J):
        dq = np.linalg.pinv(J, rcond=1e-8) @ (C.IK_GAIN * e)
        n = np.linalg.norm(dq)
        return dq if n < 0.5 else dq * 0.5 / n


class DLSIK(IterativeIK):
    """Damped least squares (Levenberg-Marquardt):
        dq = (J^T J + lam I)^-1 J^T (K e)                     <- main task
             + (I - (J^T J + lam I)^-1 J^T J) * dq_null        <- secondary task
    dq_null = alpha * (q_mid - q) pulls joints toward the middle of their
    range, WITHOUT disturbing the end-effector (it lives in the null space)."""
    name = "dls"

    def step(self, q, e, J):
        A = J.T @ J + C.DLS_LAMBDA * np.eye(7)
        Ainv_JT = np.linalg.solve(A, J.T)            # (J^T J + lam I)^-1 J^T
        dq_task = Ainv_JT @ (C.IK_GAIN * e)
        # Null-space projector.  The notes write N = I - (J^T J + lam I)^-1 J^T J.
        # With lam > 0 that N is NOT an exact null space: part of dq_null leaks
        # into the tool motion and the solver stalls ~2 mm from the target
        # (we measured this).  The exact projector I - J^+ J fixes it.
        N = np.eye(7) - np.linalg.pinv(J, rcond=1e-4) @ J
        dq_null = C.NULL_ALPHA * (self.mid - q)
        return dq_task + N @ dq_null


class QPIK(IterativeIK):
    """min_dq  1/2 dq^T H dq + c^T dq   s.t.  G dq <= h
       H = 2/dt^2 (J^T J + lam I),  c = -(1/dt) J^T x_dot_d,  x_dot_d = K e / dt
       constraints: q_min <= q + dq <= q_max  (joint limits, with a margin)
                    |dq| <= v_max * dt         (joint-velocity limits)"""
    name = "qp"

    def step(self, q, e, J):
        dt = C.QP_DT
        xdot_d = C.IK_GAIN * e / dt
        H = (2.0 / dt**2) * (J.T @ J + C.DLS_LAMBDA * np.eye(7))
        c = -(1.0 / dt) * J.T @ xdot_d
        # box bounds = joint limits intersected with velocity limits
        lb = np.maximum(self.kin.lo + C.JOINT_MARGIN - q, -C.V_MAX * dt)
        ub = np.minimum(self.kin.hi - C.JOINT_MARGIN - q, C.V_MAX * dt)
        ub = np.maximum(ub, lb)                       # keep it feasible
        G = np.vstack([np.eye(7), -np.eye(7)])        # G dq <= h
        h = np.concatenate([ub, -lb])
        # quadprog solves: min 1/2 x^T P x - a^T x  s.t.  Cm^T x >= b
        dq, *_ = quadprog.solve_qp(H, -c, -G.T, -h)
        return dq


class MinkIK:
    """Off-the-shelf IK with Mink (https://github.com/kevinzakka/mink).
    Mink: tasks (what we want) + limits (what we must respect) -> it builds
    and solves a QP each iteration and returns a joint velocity."""
    name = "mink"

    def __init__(self, model, collision=True):
        import mink
        from scene import obstacle_geoms, robot_collision_geoms
        self.mink = mink
        self.kin = Kin(model)
        self.cfg = mink.Configuration(model)
        self.task = mink.FrameTask("tcp", "site", position_cost=1.0,
                                   orientation_cost=1.0, lm_damping=1e-3)
        self.posture = mink.PostureTask(model, cost=1e-3)
        q_home = self.kin.d.qpos.copy()
        self.posture.set_target(q_home)
        self.limits = [mink.ConfigurationLimit(model),
                       mink.VelocityLimit(model, dict(zip(C.ARM_JOINTS, C.V_MAX)))]
        if collision and model.ngeom and "table_top" in [model.geom(i).name for i in range(model.ngeom)]:
            pairs = [(robot_collision_geoms(model), obstacle_geoms(model))]
            self.limits.append(mink.CollisionAvoidanceLimit(
                model, geom_pairs=pairs, minimum_distance_from_collisions=0.004,
                collision_detection_distance=0.03))
        self.dt = C.QP_DT

    def solve(self, q0, pos_t, quat_t, max_iters=C.MAX_IK_ITERS):
        mink = self.mink
        t0 = time.perf_counter()
        qfull = self.kin.d.qpos.copy()
        qfull[self.kin.qadr] = q0
        self.cfg.update(qfull)
        self.task.set_target(mink.SE3.from_rotation_and_translation(
            mink.SO3(np.asarray(quat_t, float)), np.asarray(pos_t, float)))
        tasks = [self.task, self.posture]
        for it in range(max_iters + 1):
            q = self.cfg.q[self.kin.qadr].copy()
            pos, quat = self.kin.fk(q)
            e = pose_error(pos_t, quat_t, pos, quat)
            pe, re = np.linalg.norm(e[:3]), np.linalg.norm(e[3:])
            if pe < C.POS_TOL and re < C.ROT_TOL:
                return IKResult(q, True, it, pe, re, time.perf_counter() - t0)
            if it == max_iters:
                break
            try:
                vel = mink.solve_ik(self.cfg, tasks, self.dt, solver="daqp",
                                    damping=1e-3, limits=self.limits)
            except Exception:          # QP infeasible
                break
            self.cfg.integrate_inplace(vel, self.dt)
        return IKResult(q, False, it, pe, re, time.perf_counter() - t0)


SOLVERS = {"mink": MinkIK, "pinv": PinvIK, "dls": DLSIK, "qp": QPIK}


def make_solver(name, model):
    return SOLVERS[name](model)


if __name__ == "__main__":
    # quick self-test: every solver reaches the same target from home
    from scene import build_model
    from kinematics import quat_down
    m = build_model(cube=False, table=True)
    target = (np.array([0.5, 0.1, 0.35]), quat_down(np.pi / 2 + 0.6))
    for name in SOLVERS:
        r = make_solver(name, m).solve(C.Q_HOME, *target)
        print(f"{name:5s} converged={r.converged} iters={r.iters:3d} "
              f"pos_err={r.pos_err*1000:.2f} mm rot_err={r.rot_err:.4f} "
              f"time={r.time*1000:.1f} ms")
