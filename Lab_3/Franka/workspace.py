"""
workspace.py - Section 5 (workspace analysis) + Section 6 (table design).

Run:   python workspace.py          (takes a few minutes, uses all CPU cores)
Makes: results/ws_fk_cloud.png      random-joint FK point cloud (all reach)
       results/ws_slices.png        task vs dexterous maps at many heights
       results/ws_area_vs_h.png     dexterous area for each table height
       results/ws_table.png         chosen table / tray / spawn region
       results/table_design.md      the table-design table with rationale
       results/workspace.npz        raw data

DEFINITIONS (for a top-down grasp, gripper z-axis pointing down):
  task workspace      : grid points the TCP can reach with AT LEAST ONE yaw
  dexterous workspace : grid points reachable with ALL 8 yaws (every 45 deg)
                        -> the cube can have ANY yaw there and we can still grab it
A point counts as reachable only if the IK converges AND respects joint limits
(we use the QP-IK because it enforces the limits).
"""
import os
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config as C
from ik_solvers import QPIK
from kinematics import quat_down, wrap
from scene import build_model

os.makedirs("results", exist_ok=True)
GRID = 0.05
XS = np.round(np.arange(-0.20, 0.951, GRID), 3)
YS = np.round(np.arange(0.0, 0.851, GRID), 3)      # y >= 0, mirrored later
YAWS = np.arange(8) * np.pi / 4
H_CANDIDATES = [0.0, 0.10, 0.20, 0.30, 0.40]
GRASP_DZ = C.CUBE_HALF          # TCP at cube centre when grasping
TRANSIT_DZ = 0.15               # pre-grasp / transit height above the table
LINK_CLEARANCE = 0.05           # elbow/wrist link origins must stay 5 cm above table
# every TCP height we test belongs to exactly one candidate table height
LEVEL_H = {round(h + GRASP_DZ, 3): h for h in H_CANDIDATES}
LEVEL_H.update({round(h + TRANSIT_DZ, 3): h for h in H_CANDIDATES})
LEVELS = sorted(LEVEL_H)


def seeds(x, y, yaw):
    """Good starting guesses: turn joint1 to face the point, set joint7 for yaw."""
    phi = np.arctan2(y, x)
    out = []
    for elbow in [(0.0, -1.57, 1.57), (0.4, -2.1, 2.4), (-0.3, -1.2, 1.0)]:
        q = C.Q_HOME.copy()
        q[0] = np.clip(phi, -2.8, 2.8)
        q[1], q[3], q[5] = elbow
        q[6] = np.clip(wrap(-0.7853 - (yaw - np.pi / 2 - q[0])), -2.8, 2.8)
        out.append(q)
    return out


ARM_BODIES = ["link3", "link4", "link5", "link6", "link7"]


def analyse_level(args):
    """Return (reach[ix, iy, iyaw], low[ix, iy, iyaw]) for one TCP height z.
    reach = IK converged inside the joint limits.
    low   = lowest arm-link origin (elbow/wrist) height in that solution;
            used later to reject configurations that would hit the table."""
    z, table_h = args
    solver = QPIK(build_model(cube=False, table=False))
    kin = solver.kin
    bodies = [kin.m.body(b).id for b in ARM_BODIES]
    reach = np.zeros((len(XS), len(YS), len(YAWS)), bool)
    low = np.full(reach.shape, -1.0)
    shoulder = np.array([0, 0, 0.333])
    rng = np.random.default_rng(0)
    for i, x in enumerate(XS):
        prev = [None] * len(YAWS)        # neighbour's solution = great seed
        for j, y in enumerate(YS):
            p = np.array([x, y, z])
            if np.linalg.norm(p - shoulder) > 0.95 or np.hypot(x, y) < 0.15:
                prev = [None] * len(YAWS)
                continue            # obviously out of reach / inside the base
            for k, yaw in enumerate(YAWS):
                tries = ([prev[k]] if prev[k] is not None else []) + seeds(x, y, yaw) \
                    + list(rng.uniform(kin.lo, kin.hi, size=(3, 7)))
                prev[k] = None
                for q0 in tries:
                    r = solver.solve(q0, p, quat_down(yaw), max_iters=200)
                    if r.converged and kin.within_limits(r.q):
                        kin.set_q(r.q)
                        h = kin.d.xpos[bodies, 2].min()
                        # keep the solution whose links are highest above the table
                        if h > low[i, j, k]:
                            low[i, j, k], prev[k] = h, r.q
                        reach[i, j, k] = True
                        if h > table_h + LINK_CLEARANCE:
                            break        # good enough, stop trying seeds
    return reach, low


def mirror(a):
    """y >= 0 data -> full -y..y map (the Panda is symmetric about the xz plane)."""
    return np.concatenate([a[:, :0:-1], a], axis=1)


def largest_rectangle(mask):
    """Largest all-True axis-aligned rectangle in a 2-D bool grid
    (classic 'maximal rectangle' / histogram algorithm). Returns (i0,i1,j0,j1)."""
    best, box = 0, None
    h = np.zeros(mask.shape[1], int)
    for i in range(mask.shape[0]):
        h = np.where(mask[i], h + 1, 0)
        stack = []
        for j in range(len(h) + 1):
            cur = h[j] if j < len(h) else 0
            start = j
            while stack and stack[-1][1] >= cur:
                start, height = stack.pop()
                area = height * (j - start)
                if area > best:
                    best, box = area, (i - height + 1, i, start, j - 1)
            stack.append((start, cur))
    return box


def fk_cloud(n=40000, seed=0):
    from kinematics import Kin
    kin = Kin(build_model(cube=False, table=False))
    rng = np.random.default_rng(seed)
    q = rng.uniform(kin.lo, kin.hi, size=(n, 7))
    return np.array([kin.fk(qi)[0] for qi in q])


def main(reuse=False):
    YF = np.round(np.concatenate([-YS[:0:-1], YS]), 3)
    if reuse and os.path.exists("results/workspace.npz"):
        D = np.load("results/workspace.npz")       # skip the slow IK sweep
        reach = {z: r for z, r in zip(LEVELS, D["reach"])}
        raw = {z: r for z, r in zip(LEVELS, D["raw"])}
        cloud = D["cloud"]
    else:
        print(f"analysing {len(LEVELS)} heights x {len(XS)}x{len(YS)} grid x "
              f"{len(YAWS)} yaws ...")
        with Pool(os.cpu_count()) as pool:
            res = pool.map(analyse_level, [(z, LEVEL_H[z]) for z in LEVELS])
        # reachable AND the arm stays clear above the table of that height
        reach = {z: mirror(r & (lw > LEVEL_H[z] + LINK_CLEARANCE))
                 for z, (r, lw) in zip(LEVELS, res)}
        raw = {z: mirror(r) for z, (r, lw) in zip(LEVELS, res)}
        cloud = fk_cloud()
    task = {z: r.any(-1) for z, r in reach.items()}
    dex = {z: r.all(-1) for z, r in reach.items()}

    # ---------------------------------------------------- Section 6: table
    xs_ok = XS >= 0.30          # keep the table clear of the robot base
    rows = []
    for H in H_CANDIDATES:
        feas = dex[round(H + GRASP_DZ, 3)] & dex[round(H + TRANSIT_DZ, 3)]
        feas &= xs_ok[:, None]
        box = largest_rectangle(feas)
        area = 0 if box is None else (box[1] - box[0] + 1) * (box[3] - box[2] + 1) * GRID**2
        rows.append((H, feas.sum() * GRID**2, area, box, feas))
        print(f"H={H:.2f}  dexterous area={feas.sum()*GRID**2:.3f} m^2  "
              f"largest rectangle={area:.3f} m^2")
    # biggest table rectangle wins; on a tie prefer the HIGHER table
    # (links stay further from the floor, and it is a real table)
    H, _, _, box, feas = max(rows, key=lambda r: (round(r[2], 3), r[0]))
    i0, i1, j0, j1 = box
    # grid points are cell CENTRES -> rectangle edges are +-GRID/2 outside them,
    # we then keep a 2.5 cm safety margin inside, i.e. edges at the centres.
    tx, ty = (XS[i0], XS[i1]), (YF[j0], YF[j1])
    print(f"chosen H={H}  table x={tx}  y={ty}")

    np.savez("results/workspace.npz", xs=XS, ys=YF, levels=LEVELS,
             reach=np.stack([reach[z] for z in LEVELS]),
             raw=np.stack([raw[z] for z in LEVELS]), cloud=cloud)
    make_plots(reach, task, dex, YF, rows, H, tx, ty)
    write_table_md(H, tx, ty, rows)


def make_plots(reach, task, dex, YF, rows, H, tx, ty):
    ext = [YF[0] - GRID / 2, YF[-1] + GRID / 2, XS[0] - GRID / 2, XS[-1] + GRID / 2]

    # 1) FK point cloud
    cloud = np.load("results/workspace.npz")["cloud"]
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(121, projection="3d")
    s = cloud[::8]
    ax.scatter(s[:, 0], s[:, 1], s[:, 2], s=0.5, c=s[:, 2], cmap="viridis")
    ax.set(xlabel="x [m]", ylabel="y [m]", zlabel="z [m]",
           title="Reachable TCP positions (random joints, any orientation)")
    ax2 = fig.add_subplot(122)
    side = cloud[np.abs(cloud[:, 1]) < 0.05]
    ax2.scatter(side[:, 0], side[:, 2], s=1, alpha=0.4)
    ax2.axhline(H, color="brown", lw=2, label=f"table top H={H} m")
    ax2.set(xlabel="x [m]", ylabel="z [m]", title="Side slice |y|<5 cm",
            aspect="equal")
    ax2.legend()
    fig.tight_layout()
    fig.savefig("results/ws_fk_cloud.png", dpi=130)
    plt.close(fig)

    # 2) task vs dexterous slices
    n = len(LEVELS)
    fig, axs = plt.subplots(2, (n + 1) // 2, figsize=(3.2 * ((n + 1) // 2), 7.5))
    for ax, z in zip(axs.flat, LEVELS):
        img = task[z].astype(int) + dex[z].astype(int)  # 0 none, 1 task, 2 dex
        ax.imshow(img, origin="lower", extent=ext, cmap="Blues", vmin=0, vmax=2)
        ax.plot(0, 0, "k^")
        kind = "grasp" if abs(z - LEVEL_H[z] - GRASP_DZ) < 1e-6 else "transit"
        ax.set_title(f"TCP z={z:.2f} ({kind}, H={LEVEL_H[z]:.1f})", fontsize=9)
        ax.set_xlabel("y [m]", fontsize=8)
        ax.set_ylabel("x [m]", fontsize=8)
    for ax in axs.flat[n:]:
        ax.axis("off")
    fig.suptitle("Top-down grasp reachability (IK inside joint limits AND arm links "
                 "5 cm above the table of that height)\nlight = task workspace (>=1 yaw), "
                 "dark = dexterous (all 8 yaws), triangle = robot base")
    fig.tight_layout()
    fig.savefig("results/ws_slices.png", dpi=120)
    plt.close(fig)

    # 3) area vs height
    fig, ax = plt.subplots(figsize=(6, 4))
    hs = [r[0] for r in rows]
    ax.plot(hs, [r[1] for r in rows], "o-", label="dexterous area (grasp & transit)")
    ax.plot(hs, [r[2] for r in rows], "s-", label="largest rectangle (table)")
    ax.axvline(H, color="r", ls="--", label=f"chosen H = {H} m")
    ax.set(xlabel="table height H [m]", ylabel="area [m$^2$]",
           title="Which table height gives the most usable area?")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("results/ws_area_vs_h.png", dpi=130)
    plt.close(fig)

    # 4) chosen table on the feasible map
    feas = [r for r in rows if r[0] == H][0][4]
    zt = round(H + GRASP_DZ, 3)
    img = task[zt].astype(int) + feas.astype(int)
    fig, ax = plt.subplots(figsize=(6, 6.5))
    ax.imshow(img, origin="lower", extent=ext, cmap="Blues", vmin=0, vmax=2)

    def rect(x, y, **kw):
        ax.add_patch(plt.Rectangle((y[0], x[0]), y[1] - y[0], x[1] - x[0],
                                   fill=False, **kw))
    rect(tx, ty, ec="saddlebrown", lw=3, label="designed table")
    rect(C.TABLE_X, C.TABLE_Y, ec="orange", lw=1.5, ls=":", label="table in config.py")
    a = C.TRAY_INNER
    rect((C.TRAY_POS[0] - a, C.TRAY_POS[0] + a), (C.TRAY_POS[1] - a, C.TRAY_POS[1] + a),
         ec="royalblue", lw=2, label="tray")
    rect(C.SPAWN_X, C.SPAWN_Y, ec="green", lw=2, ls="--", label="cube spawn region")
    ax.plot(0, 0, "k^", ms=10, label="robot base")
    ax.set(xlabel="y [m]", ylabel="x [m]",
           title=f"Table design at H={H} m\nlight=task, dark=dexterous at grasp+transit")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig("results/ws_table.png", dpi=130)
    plt.close(fig)


def write_table_md(H, tx, ty, rows):
    L, W = tx[1] - tx[0], ty[1] - ty[0]
    lines = [
        "# Table design (generated by workspace.py)\n",
        "| H candidate (m) | dexterous area (m^2) | largest rectangle (m^2) |",
        "|---|---|---|",
        *[f"| {r[0]:.2f} | {r[1]:.3f} | {r[2]:.3f} |" for r in rows],
        "",
        "| Parameter | Symbol | Value | Rationale |",
        "|---|---|---|---|",
        f"| Table length | L | {L:.2f} m (x {tx[0]:.2f} to {tx[1]:.2f}) | "
        "Largest rectangle inside the dexterous workspace along x; starts at "
        "x>=0.30 m to stay clear of the robot base |",
        f"| Table width | W | {W:.2f} m (y {ty[0]:.2f} to {ty[1]:.2f}) | "
        "Same rectangle along y; every point can be grasped at any yaw |",
        f"| Table height | H | {H:.2f} m | Height with the largest usable rectangle "
        "where BOTH the grasp height (H+0.02) and the pre-grasp/transit height "
        "(H+0.15) are dexterous -> vertical approach without hitting joint limits |",
    ]
    with open("results/table_design.md", "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    import sys
    main(reuse="--reuse" in sys.argv)
