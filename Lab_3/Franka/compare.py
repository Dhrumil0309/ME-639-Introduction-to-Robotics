"""
compare.py - Section 13: compare the IK methods on IDENTICAL episode seeds.
Run without pandas dependency, logging everything directly with csv.
"""
import argparse
import os
import subprocess
import sys
import csv
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METHODS = ["mink", "pinv", "dls", "qp"]
NAMES = {"mink": "Mink (off-the-shelf)", "pinv": "Pseudo-inverse (own choice)",
         "dls": "DLS-IK", "qp": "QP-IK"}
EXPS = {"nominal": ("", []), "stress": ("_stress", ["--region", "full"]),
        "naive": ("_naive", ["--region", "full", "--naive"])}
COL = {"mink": "#1f9bd1", "pinv": "#c94fa3", "dls": "#f08a24", "qp": "#5aa832"}
os.makedirs("results/plots", exist_ok=True)
os.makedirs("results/failures", exist_ok=True)

def run_all(n):
    for exp, (tag, extra) in EXPS.items():
        for m in METHODS:
            print(f"--- Running {exp} / {m} ({n} episodes)...")
            cmd = [sys.executable, "pick_place.py", "--method", m, "--episodes", str(n),
                   "--tag", tag] + extra
            subprocess.run(cmd, check=True)

def wilson(k, n, z=1.96):
    if n == 0:
        return 0, 0
    p = k / n
    den = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return max(0, c - h), min(1, c + h)

def category(reason):
    if not isinstance(reason, str) or reason == "":
        return "success"
    if reason.startswith("collision"):
        return "robot_collision"
    return reason.split(" (")[0].split(" @")[0]

def load():
    rows = []
    for exp, (tag, _) in EXPS.items():
        for m in METHODS:
            f = f"results/logs_{m}{tag}.csv"
            if os.path.exists(f):
                with open(f, "r") as fp:
                    reader = csv.DictReader(fp)
                    for r in reader:
                        r["experiment"] = exp
                        r["method"] = m
                        r["success_bool"] = (r["success"].lower() == "true")
                        r["category"] = category(r.get("failure_reason", ""))
                        for num_col in ["plan_time_s", "ik_iters_mean", "ik_iters_max",
                                        "min_clear_robot", "max_vel_ratio", "cube_x", "cube_y"]:
                            if num_col in r:
                                try:
                                    r[num_col] = float(r[num_col])
                                except ValueError:
                                    r[num_col] = 0.0
                        for int_col in ["cube_tray_contact", "joint_limit_viol", "vel_viol"]:
                            if int_col in r:
                                try:
                                    r[int_col] = int(float(r[int_col]))
                                except ValueError:
                                    r[int_col] = 0
                        try:
                            r["min_clear_cube_tray"] = float(r["min_clear_cube_tray"])
                        except (ValueError, KeyError):
                            r["min_clear_cube_tray"] = np.nan
                        rows.append(r)
    return rows

def summarise(rows):
    out = []
    for exp in EXPS:
        for m in METHODS:
            g = [r for r in rows if r["experiment"] == exp and r["method"] == m]
            if not g:
                continue
            k = sum(1 for r in g if r["success_bool"])
            n = len(g)
            lo, hi = wilson(k, n)
            plan_times = [r["plan_time_s"] * 1000 for r in g]
            ik_iters = [r["ik_iters_mean"] for r in g]
            ik_maxs = [r["ik_iters_max"] for r in g]
            min_clears_robot = [r["min_clear_robot"] * 1000 for r in g]
            cube_trays = [r["min_clear_cube_tray"] * 1000 for r in g if not np.isnan(r["min_clear_cube_tray"])]
            
            fails = [r["category"] for r in g if not r["success_bool"]]
            main_fail = "-"
            if fails:
                main_fail = max(set(fails), key=fails.count)

            out.append(dict(
                experiment=exp, method=m, episodes=n, success=k,
                success_rate=round(k / n, 3), ci95=f"{lo:.2f}-{hi:.2f}",
                plan_time_ms=round(float(np.mean(plan_times)), 1),
                plan_time_std=round(float(np.std(plan_times)), 1),
                ik_iters_per_waypoint=round(float(np.mean(ik_iters)), 2),
                ik_iters_max=int(max(ik_maxs)),
                min_clear_robot_mm=round(float(min(min_clears_robot)), 1),
                min_clear_cube_tray_mm=round(float(min(cube_trays)), 1) if cube_trays else 0.0,
                cube_tray_contacts=sum(r["cube_tray_contact"] for r in g),
                joint_limit_viol=sum(r["joint_limit_viol"] for r in g),
                vel_viol=sum(r["vel_viol"] for r in g),
                max_vel_ratio=round(float(max(r["max_vel_ratio"] for r in g)), 2),
                main_failure=main_fail,
            ))
    return out

def plots(rows, summary):
    # 1) Success rate per experiment
    fig, ax = plt.subplots(figsize=(8, 4))
    w = 0.2
    for i, m in enumerate(METHODS):
        rates = []
        errs_lo, errs_hi = [], []
        for exp in EXPS:
            match = next((s for s in summary if s["experiment"] == exp and s["method"] == m), None)
            if match:
                r = match["success_rate"]
                rates.append(r * 100)
                lo, hi = [float(x) for x in match["ci95"].split("-")]
                errs_lo.append((r - lo) * 100)
                errs_hi.append((hi - r) * 100)
            else:
                rates.append(0)
                errs_lo.append(0)
                errs_hi.append(0)
        ax.bar(np.arange(3) + (i - 1.5) * w, rates, w, yerr=[errs_lo, errs_hi],
               capsize=3, label=NAMES[m], color=COL[m])
    ax.set_xticks(range(3))
    ax.set_xticklabels(["nominal\n(designed region)", "stress\n(whole table)", "naive\n(smart choices OFF)"])
    ax.set_ylabel("success rate [%]")
    ax.set_ylim(0, 110)
    ax.set_title("Pick-and-place success (25 episodes each, same seeds, 95% CI)")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig("results/plots/cmp_success.png", dpi=130)
    plt.close(fig)

    # 2) IK iterations & planning time (nominal)
    nom = [r for r in rows if r["experiment"] == "nominal"]
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for ax, col, lab in [(axs[0], "ik_iters_mean", "IK iterations per waypoint"),
                         (axs[1], "plan_time_s", "planning time per episode [s]")]:
        data = [[r[col] for r in nom if r["method"] == m] for m in METHODS]
        bp = ax.boxplot(data, patch_artist=True)
        for p, m in zip(bp["boxes"], METHODS):
            p.set_facecolor(COL[m])
        ax.set_xticks(range(1, 5))
        ax.set_xticklabels([NAMES[m].split(" (")[0] for m in METHODS])
        ax.set_title(lab + " (nominal)")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("results/plots/cmp_iters_time.png", dpi=130)
    plt.close(fig)

    # 3) Failure categories (naive)
    nv = [r for r in rows if r["experiment"] == "naive"]
    cats = sorted(list(set(r["category"] for r in nv)), key=lambda c: (c != "success", c))
    cc = {"success": "#4caf50", "ik_infeasible": "#e53935", "cube_tray_collision": "#fb8c00",
          "dropped_in_transit": "#fdd835", "robot_collision": "#8e24aa", "place_miss": "#6d4c41",
          "grasp_slip": "#00897b", "grasp_miss": "#3949ab", "planned_collision": "#d81b60"}
    fig, ax = plt.subplots(figsize=(8, 4))
    bottom = np.zeros(len(METHODS))
    for c in cats:
        v = np.array([sum(1 for r in nv if r["method"] == m and r["category"] == c) for m in METHODS])
        ax.bar(range(4), v, bottom=bottom, label=c, color=cc.get(c, "grey"))
        bottom += v
    ax.set_xticks(range(4))
    ax.set_xticklabels([NAMES[m].split(" (")[0] for m in METHODS])
    ax.set_ylabel("episodes")
    ax.set_title("Outcome breakdown - naive experiment (smart choices OFF)")
    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig("results/plots/cmp_failures.png", dpi=130)
    plt.close(fig)

    # 4) Constraint violations
    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    x = np.arange(4)
    for i, exp in enumerate(["nominal", "naive"]):
        exp_sum = {s["method"]: s for s in summary if s["experiment"] == exp}
        j_viols = [exp_sum.get(m, {}).get("joint_limit_viol", 0) for m in METHODS]
        v_ratios = [exp_sum.get(m, {}).get("max_vel_ratio", 0) for m in METHODS]
        clears = [exp_sum.get(m, {}).get("min_clear_robot_mm", 0) for m in METHODS]
        axs[0].bar(x + (i - 0.5) * 0.38, j_viols, 0.38, label=exp, color=["#90caf9", "#e57373"][i])
        axs[1].bar(x + (i - 0.5) * 0.38, v_ratios, 0.38, label=exp, color=["#90caf9", "#e57373"][i])
        axs[2].bar(x + (i - 0.5) * 0.38, clears, 0.38, label=exp, color=["#90caf9", "#e57373"][i])
    axs[1].axhline(1.0, color="k", ls="--", lw=1)
    for ax, t, yl in [(axs[0], "joint-limit violations", "steps violating limit"),
                      (axs[1], "max |vel| / limit", "ratio"),
                      (axs[2], "min clearance to obstacle", "[mm]")]:
        ax.set_xticks(x)
        ax.set_xticklabels([NAMES[m].split(" (")[0] for m in METHODS], fontsize=8)
        ax.set_title(t)
        ax.set_ylabel(yl)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("results/plots/cmp_constraints.png", dpi=130)
    plt.close(fig)

    # 5) Spatial positions
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
    import config as C
    for ax, exp, title in [(axs[0], "stress", "stress (smart clearance ON)"),
                           (axs[1], "naive", "naive (straight-line, no clearance)")]:
        sub = [r for r in rows if r["experiment"] == exp]
        t = plt.Rectangle((C.TABLE_X[0], C.TABLE_Y[0]), C.TABLE_X[1]-C.TABLE_X[0],
                          C.TABLE_Y[1]-C.TABLE_Y[0], facecolor="#e0e0e0", edgecolor="k", lw=1.5)
        ax.add_patch(t)
        tr = plt.Rectangle((C.TRAY_POS[0]-C.TRAY_INNER, C.TRAY_POS[1]-C.TRAY_INNER),
                           2*C.TRAY_INNER, 2*C.TRAY_INNER, facecolor="#bbdefb", edgecolor="#1565c0", lw=2)
        ax.add_patch(tr)
        for r in sub:
            c = "#4caf50" if r["success_bool"] else "#e53935"
            m = "o" if r["success_bool"] else "x"
            ax.scatter(r["cube_x"], r["cube_y"], c=c, marker=m, s=30, alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("results/plots/cmp_positions.png", dpi=130)
    plt.close(fig)

def save_summary_files(summary):
    if not summary:
        return
    keys = summary[0].keys()
    with open("results/summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(summary)

    with open("results/summary.md", "w") as f:
        f.write("# Summary metrics (generated by compare.py)\n\n")
        for exp in EXPS:
            f.write(f"## {exp}\n\n")
            exp_rows = [s for s in summary if s["experiment"] == exp]
            if not exp_rows:
                continue
            headers = list(exp_rows[0].keys())
            f.write("| " + " | ".join(headers) + " |\n")
            f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
            for r in exp_rows:
                f.write("| " + " | ".join(str(r[h]) for h in headers) + " |\n")
            f.write("\n")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", action="store_true", help="run simulation episodes")
    p.add_argument("--episodes", type=int, default=25, help="episodes per method")
    args = p.parse_args()

    if args.run:
        run_all(args.episodes)

    rows = load()
    if not rows:
        print("No log files found in results/ to summarise.")
        return
    summary = summarise(rows)
    save_summary_files(summary)
    plots(rows, summary)
    print("\nAll experiments, tables, and comparison plots updated successfully!")

if __name__ == "__main__":
    main()
