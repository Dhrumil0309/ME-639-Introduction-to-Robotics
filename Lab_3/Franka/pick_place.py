"""
pick_place.py - randomized pick-and-place episodes with any IK solver.

    python pick_place.py --method mink --episodes 25
    python pick_place.py --method dls  --episodes 25 --video videos/dls.mp4
    python pick_place.py --method qp   --episodes 3 --viewer      (live window)

One episode =
  1. sample the cube pose (x, y, yaw) from a seed      -> same cubes for every method
  2. PLAN: turn 6 Cartesian segments into joint waypoints with the IK solver
       home -> pre-grasp -> grasp | close | lift -> above tray -> place | open | retreat
  3. EXECUTE: send the joint waypoints to the Panda's position actuators and let
     MuJoCo physics run (the grasp is real friction, the cube can slip or drop)
  4. CHECK + LOG: success / failure reason, clearances, IK iterations, timing...

Output:  results/logs_<method>.csv  (one row per episode)
         results/failures/<method>_epXX.png  (screenshot when an episode fails)
"""
import argparse
import csv
import os
import time

import mujoco
import numpy as np

import config as C
from ik_solvers import make_solver
from kinematics import Kin, quat_down, slerp, wrap
from scene import build_model, obstacle_geoms, robot_collision_geoms

TRANSIT_Z = C.TABLE_H + max(C.TRAY_WALL_H + C.CUBE_HALF + C.SAFE_MARGIN,
                            C.CUBE_HALF + C.PREGRASP_DZ)
GRASP_Z = C.TABLE_H + C.CUBE_HALF                       # TCP at cube centre
PLACE_Z = C.TABLE_H + 0.004 + C.CUBE_HALF + 0.012      # cube 12 mm above tray floor
YAW_HOME = np.pi / 2                                    # TCP yaw at the home pose


# ======================================================================
#  helpers
# ======================================================================
def sample_cube(seed, region="designed"):
    """Same seed -> same cube pose.  That is how we compare methods fairly.
    region='designed' : the spawn rectangle chosen from the workspace analysis
    region='full'     : anywhere on the table (stress test), except on the tray"""
    rng = np.random.default_rng(seed)
    if region == "designed":
        return (rng.uniform(*C.SPAWN_X), rng.uniform(*C.SPAWN_Y), rng.uniform(*C.SPAWN_YAW))
    m = 0.03
    while True:
        x = rng.uniform(C.TABLE_X[0] + m, C.TABLE_X[1] - m)
        y = rng.uniform(C.TABLE_Y[0] + m, C.TABLE_Y[1] - m)
        far = max(abs(x - C.TRAY_POS[0]), abs(y - C.TRAY_POS[1])) > C.TRAY_INNER + 0.06
        if far:
            return (x, y, rng.uniform(*C.SPAWN_YAW))


def yaw_candidates(obj_yaw, x, y):
    """A cube looks identical every 90 deg, so 4 gripper yaws can grasp it.
    Order them so the wrist joint (joint7) stays near the middle of its range:
    predicted q7 = q7_home - (yaw - yaw_home - azimuth)."""
    az = np.arctan2(y, x)
    cands = [wrap(obj_yaw + k * np.pi / 2) for k in range(4)]
    q7 = lambda yaw: wrap(C.Q_HOME[6] - (yaw - YAW_HOME - az))
    return sorted(cands, key=lambda yw: abs(q7(yw)))


# ======================================================================
#  PLANNER: Cartesian segments -> joint waypoints, using the IK solver
# ======================================================================
class Planner:
    def __init__(self, method, naive=False):
        # naive=True switches OFF the smart choices (for the failure study):
        #   grasp yaw = cube yaw as-is (no 90-deg symmetry trick, no retries),
        #   transit only 6 cm above the table (below the tray rim!),
        #   no place-yaw alignment.
        self.naive = naive
        self.transit_z = C.TABLE_H + 0.06 if naive else TRANSIT_Z
        self.ik_model = build_model(cube=False, table=True)   # robot + table + tray
        self.solver = make_solver(method, self.ik_model)
        self.kin = Kin(self.ik_model)                          # for collision checks
        self.robot_geoms = robot_collision_geoms(self.ik_model)
        self.obstacles = obstacle_geoms(self.ik_model)
        self.fromto = np.zeros(6)

    def clearance(self, q):
        """Smallest distance (m) between gripper/wrist and table/tray for config q.
        Negative = penetration = collision."""
        self.kin.set_q(q)
        mujoco.mj_forward(self.kin.m, self.kin.d)
        best = 1.0
        for g in self.robot_geoms:
            for o in self.obstacles:
                d = mujoco.mj_geomDistance(self.kin.m, self.kin.d, g, o, 0.2, self.fromto)
                best = min(best, d)
        return best

    def segment(self, q_start, p0, r0, p1, r1, stats):
        """Straight line from pose (p0,r0) to (p1,r1), split in small steps;
        IK for every step, warm-started from the previous answer."""
        dist = np.linalg.norm(p1 - p0)
        ang = 2 * np.arccos(min(1.0, abs(np.dot(r0, r1))))
        n = max(1, int(np.ceil(dist / C.STEP_POS)), int(np.ceil(ang / C.STEP_ROT)))
        qs, q = [], q_start
        for i in range(1, n + 1):
            t = i / n
            res = self.solver.solve(q, p0 + t * (p1 - p0), slerp(r0, r1, t))
            stats["iters"].append(res.iters)
            stats["time"] += res.time
            if not res.converged:
                near = np.where((res.q - self.kin.lo < 0.05) | (self.kin.hi - res.q < 0.05))[0]
                why = f"joint{near[0] + 1} at limit" if len(near) else "no convergence"
                return None, f"ik_infeasible ({why}, err {res.pos_err*1000:.0f} mm)"
            q = res.q
            if not self.kin.within_limits(q):
                stats["jl_viol"] += 1
            qs.append(q)
        # collision check on the planned waypoints (every 3rd to save time)
        for q in qs[::3] + [qs[-1]]:
            c = self.clearance(q)
            stats["plan_clear"] = min(stats["plan_clear"], c)
            if c < 0:
                return None, "planned_collision (gripper vs table/tray)"
        return qs, None

    def plan(self, cube, retries):
        """Plan the whole episode.  If it fails, try the next grasp yaw
        (mitigation: 'alternate grasp yaw')."""
        x, y, yaw = cube
        stats = dict(iters=[], time=0.0, jl_viol=0, plan_clear=1.0, yaw_tries=0)
        tray = np.array([*C.TRAY_POS])
        reason = "no_plan"
        cands = [wrap(yaw)] if self.naive else yaw_candidates(yaw, x, y)[:retries + 1]
        for gy in cands:
            stats["yaw_tries"] += 1
            # place yaw: the multiple of 90 deg nearest to the grasp yaw.
            # The cube is held square to the fingers, so this lines its edges
            # up with the tray walls using the least wrist rotation.
            py = gy if self.naive else wrap(round(gy / (np.pi / 2)) * (np.pi / 2))
            rg, rp = quat_down(gy), quat_down(py)
            home_p, home_r = self.kin.fk(C.Q_HOME)
            P = {
                "pregrasp": np.array([x, y, GRASP_Z + C.PREGRASP_DZ]),
                "grasp": np.array([x, y, GRASP_Z]),
                "lift": np.array([x, y, self.transit_z]),
                "above_tray": np.array([*tray, self.transit_z]),
                "place": np.array([*tray, PLACE_Z]),
                "retreat": np.array([*tray, self.transit_z]),
            }
            plan = []
            q, p, r = C.Q_HOME, home_p, home_r
            ok = True
            for name, rot in [("pregrasp", rg), ("grasp", rg), ("lift", rg),
                              ("above_tray", rp), ("place", rp), ("retreat", rp)]:
                qs, err = self.segment(q, p, r, P[name], rot, stats)
                if qs is None:
                    reason, ok = f"{err} @ {name}", False
                    break
                plan.append((name, qs, P[name], rot))
                q, p, r = qs[-1], P[name], rot
            if ok:
                return plan, stats, gy, py, None
        return None, stats, None, None, reason


# ======================================================================
#  SIMULATION: execute the joint waypoints with physics
# ======================================================================
class Sim:
    def __init__(self, render=None, viewer=False):
        self.m = build_model(cube=True, table=True)
        self.d = mujoco.MjData(self.m)
        k = Kin(self.m)
        self.qadr, self.dadr = k.qadr, k.dadr
        self.tcp = self.m.site("tcp").id
        self.cube_body = self.m.body("cube").id
        self.cube_geom = self.m.geom("cube_geom").id
        self.cube_qadr = self.m.joint("cube_free").qposadr[0]
        self.f1 = self.m.joint("finger_joint1").qposadr[0]
        self.f2 = self.m.joint("finger_joint2").qposadr[0]
        self.robot_geoms = robot_collision_geoms(self.m)
        all_robot = {g for g in range(self.m.ngeom)
                     if self.m.body(self.m.geom_bodyid[g]).name.startswith(("link", "hand", "left", "right"))}
        self.all_robot = all_robot
        self.obstacles = obstacle_geoms(self.m)
        self.tray_walls = [self.m.geom(n).id for n in
                           ("tray_wall_px", "tray_wall_nx", "tray_wall_py", "tray_wall_ny")]
        self.env = set(self.obstacles) | {self.m.geom("floor").id} | \
            {self.m.geom(f"table_leg{i}").id for i in range(4)}
        self.fromto = np.zeros(6)
        self.render = render          # Recorder or None
        self.viewer = None
        if viewer:
            from mujoco import viewer as mj_viewer
            self.viewer = mj_viewer.launch_passive(self.m, self.d)

    # ------------------------------------------------------------------
    def reset(self, cube):
        x, y, yaw = cube
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[self.qadr] = C.Q_HOME
        self.d.qpos[self.f1] = self.d.qpos[self.f2] = 0.04
        self.d.qpos[self.cube_qadr:self.cube_qadr + 3] = [x, y, C.TABLE_H + C.CUBE_HALF]
        self.d.qpos[self.cube_qadr + 3:self.cube_qadr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        self.d.ctrl[:7] = C.Q_HOME
        self.d.ctrl[7] = C.GRIPPER_OPEN
        mujoco.mj_forward(self.m, self.d)
        self.flags = dict(robot_collision=None, cube_tray_contact=False,
                          min_clear_robot=1.0, min_clear_cube_tray=1.0, max_vel_ratio=0.0)
        self.carrying = False
        self.hold(0.3)

    def step(self):
        mujoco.mj_step(self.m, self.d)
        # ---- contact checks (every step) ----
        for c in self.d.contact[:self.d.ncon]:
            g1, g2 = c.geom1, c.geom2
            for a, b in ((g1, g2), (g2, g1)):
                if a in self.all_robot and b in self.env and self.flags["robot_collision"] is None:
                    self.flags["robot_collision"] = f"{self.m.body(self.m.geom_bodyid[a]).name}-{self.m.geom(b).name}"
                if a == self.cube_geom and b in self.tray_walls and self.carrying:
                    self.flags["cube_tray_contact"] = True
        # ---- clearances + joint speed (every 10 steps, distance queries are slower)
        if self.d.time % (10 * C.PHYS_DT) < C.PHYS_DT:
            for g in self.robot_geoms:
                for o in self.obstacles:
                    d = mujoco.mj_geomDistance(self.m, self.d, g, o, 0.2, self.fromto)
                    self.flags["min_clear_robot"] = min(self.flags["min_clear_robot"], d)
            if self.carrying:
                for w in self.tray_walls:
                    d = mujoco.mj_geomDistance(self.m, self.d, self.cube_geom, w, 0.3, self.fromto)
                    self.flags["min_clear_cube_tray"] = min(self.flags["min_clear_cube_tray"], d)
            v = np.abs(self.d.qvel[self.dadr]) / C.V_MAX
            self.flags["max_vel_ratio"] = max(self.flags["max_vel_ratio"], v.max())
        if self.render:
            self.render.maybe_frame(self)
        if self.viewer and self.d.time % 0.02 < C.PHYS_DT:
            self.viewer.sync()
            time.sleep(0.01)

    def hold(self, seconds):
        for _ in range(int(seconds / C.PHYS_DT)):
            self.step()

    def track(self, qs, p_goal, start_q):
        """Linearly interpolate the actuator targets through the waypoints.
        Each waypoint gets a duration from the Cartesian step size and speed,
        so a bad IK 'jump' shows up as a joint-velocity violation."""
        dt_wp = max(C.STEP_POS / C.EE_SPEED, 0.02)
        prev = np.array(start_q)
        vel_viol = 0
        for q in qs:
            if np.any(np.abs(q - prev) / dt_wp > C.V_MAX):
                vel_viol += 1
            n = max(1, int(dt_wp / C.PHYS_DT))
            for k in range(1, n + 1):
                self.d.ctrl[:7] = prev + (q - prev) * k / n
                self.step()
            prev = q
        self.hold(0.25)   # let the arm settle
        return vel_viol

    def fingers(self):
        return self.d.qpos[self.f1] + self.d.qpos[self.f2]

    def cube_pos(self):
        return self.d.xpos[self.cube_body].copy()

    def tcp_pos(self):
        return self.d.site_xpos[self.tcp].copy()


# ======================================================================
#  EPISODE
# ======================================================================
def run_episode(planner, sim, ep, seed, retries, shot_dir=None, method="",
                region="designed"):
    cube = sample_cube(seed, region)
    log = dict(episode=ep, seed=seed, cube_x=round(cube[0], 4), cube_y=round(cube[1], 4),
               cube_yaw=round(cube[2], 4))
    if sim.render:
        sim.render.new_episode(method.split("_")[0], ep, cube)   # drop the file tag
    sim.reset(cube)

    t0 = time.perf_counter()
    plan, st, gy, py, reason = planner.plan(cube, retries)
    log.update(plan_time_s=round(time.perf_counter() - t0, 4),
               ik_time_s=round(st["time"], 4),
               ik_iters_total=int(np.sum(st["iters"])) if st["iters"] else 0,
               ik_iters_mean=round(float(np.mean(st["iters"])), 2) if st["iters"] else 0,
               ik_iters_max=int(np.max(st["iters"])) if st["iters"] else 0,
               waypoints=len(st["iters"]), joint_limit_viol=st["jl_viol"],
               plan_min_clear=round(st["plan_clear"], 4), yaw_tries=st["yaw_tries"],
               grasp_yaw=None if gy is None else round(gy, 3),
               place_yaw=None if py is None else round(py, 3))

    vel_viol, fail = 0, None
    if plan is None:
        fail = reason
    else:
        q_prev = C.Q_HOME
        for name, qs, p_goal, _ in plan:
            vel_viol += sim.track(qs, p_goal, q_prev)
            q_prev = qs[-1]
            if sim.flags["robot_collision"]:
                fail = f"collision {sim.flags['robot_collision']} @ {name}"
                break
            if name == "grasp":
                sim.d.ctrl[7] = C.GRIPPER_CLOSED           # close the gripper
                sim.hold(0.8)
                if sim.fingers() < 0.02:                   # closed on nothing
                    fail = "grasp_miss (fingers closed on air)"
                    break
                sim.carrying = True
            if name == "lift" and sim.cube_pos()[2] < GRASP_Z + 0.03:
                fail = "grasp_slip (cube not lifted)"
                break
            if name in ("above_tray", "place") and \
                    np.linalg.norm(sim.cube_pos() - sim.tcp_pos()) > 0.03:
                fail = (f"cube_tray_collision (knocked off by tray rim) @ {name}"
                        if sim.flags["cube_tray_contact"] else f"dropped_in_transit @ {name}")
                break
            if name == "place":
                sim.d.ctrl[7] = C.GRIPPER_OPEN             # release
                sim.hold(0.5)
                sim.carrying = False
        if fail is None:
            sim.hold(0.6)                                  # let the cube settle
            c = sim.cube_pos()
            inside = (abs(c[0] - C.TRAY_POS[0]) < C.TRAY_INNER and
                      abs(c[1] - C.TRAY_POS[1]) < C.TRAY_INNER and
                      c[2] < C.TABLE_H + C.TRAY_WALL_H)
            if not inside:
                fail = "place_miss (cube not in tray)"
            elif sim.flags["robot_collision"]:
                fail = f"collision {sim.flags['robot_collision']} @ retreat"
    c = sim.cube_pos()
    log.update(success=fail is None, failure_reason=fail or "",
               vel_viol=vel_viol,
               max_vel_ratio=round(sim.flags["max_vel_ratio"], 3),
               min_clear_robot=round(sim.flags["min_clear_robot"], 4),
               min_clear_cube_tray=round(sim.flags["min_clear_cube_tray"], 4)
               if sim.flags["min_clear_cube_tray"] < 1 else "",
               cube_tray_contact=sim.flags["cube_tray_contact"],
               final_cube_x=round(c[0], 4), final_cube_y=round(c[1], 4),
               final_cube_z=round(c[2], 4), sim_time_s=round(sim.d.time, 2))
    if sim.render:
        sim.render.end_episode(sim, log)
    if fail and shot_dir:
        save_screenshot(sim, os.path.join(shot_dir, f"{method}_ep{ep:02d}.png"), log)
    return log


def save_screenshot(sim, path, log):
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        r = mujoco.Renderer(sim.m, 480, 640)
    except Exception as e:
        print("  (screenshot skipped:", e, ")")
        return
    r.update_scene(sim.d, "front")
    img_arr = r.render().copy()
    r.close()
    img = Image.fromarray(img_arr)
    draw = ImageDraw.Draw(img)
    text = f"FAIL ep{log['episode']}: {log['failure_reason'][:55]}"
    draw.text((10, 10), text, fill=(255, 80, 80))
    img.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="mink", choices=["mink", "pinv", "dls", "qp"])
    ap.add_argument("--episodes", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0, help="base seed (episode i uses seed+i)")
    ap.add_argument("--retries", type=int, default=3, help="alternate grasp yaws to try")
    ap.add_argument("--video", default=None, help="mp4 path to record")
    ap.add_argument("--video-seconds", type=float, default=60.0)
    ap.add_argument("--viewer", action="store_true", help="open the MuJoCo viewer")
    ap.add_argument("--region", default="designed", choices=["designed", "full"])
    ap.add_argument("--naive", action="store_true", help="switch off the smart choices")
    ap.add_argument("--tag", default="", help="suffix for the log file name")
    a = ap.parse_args()

    rec = None
    if a.video:
        from recorder import Recorder
        rec = Recorder(a.video, max_seconds=a.video_seconds)
    planner = Planner(a.method, naive=a.naive)
    sim = Sim(render=rec, viewer=a.viewer)
    os.makedirs("results", exist_ok=True)
    out = f"results/logs_{a.method}{a.tag}.csv"
    logs = []
    ep = 0
    while ep < a.episodes:                     # the 'while loop' of Section 8
        log = run_episode(planner, sim, ep, a.seed + ep, a.retries,
                          shot_dir="results/failures", method=a.method + a.tag,
                          region=a.region)
        log["method"] = a.method
        logs.append(log)
        print(f"[{a.method}] ep {ep:02d} cube=({log['cube_x']:.2f},{log['cube_y']:.2f},"
              f"{np.degrees(log['cube_yaw']):6.1f}deg) "
              f"{'SUCCESS' if log['success'] else 'FAIL: ' + log['failure_reason']}  "
              f"iters={log['ik_iters_total']} plan={log['plan_time_s']:.2f}s")
        ep += 1
        if rec and rec.full:
            rec.full = False      # keep logging, stop adding frames
            rec.recording = False
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(logs[0].keys()))
        w.writeheader()
        w.writerows(logs)
    if rec:
        rec.close()
    sr = np.mean([l["success"] for l in logs])
    print(f"\n{a.method}: success rate {sr*100:.0f}% over {len(logs)} episodes -> {out}")


if __name__ == "__main__":
    main()
