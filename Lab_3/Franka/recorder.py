"""
recorder.py - renders the simulation to an mp4 with the TCP 'trail' drawn in.
Uses PIL and imageio_ffmpeg (independent of cv2).
"""
import os
import subprocess
import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg

COLORS = {"mink": (0.1, 0.8, 1.0), "dls": (1.0, 0.6, 0.1),
          "qp": (0.6, 1.0, 0.2), "pinv": (1.0, 0.3, 0.8)}

class Recorder:
    def __init__(self, path, max_seconds=60, fps=25, w=800, h=600, camera="front"):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        self.fps, self.w, self.h, self.cam = fps, w, h, camera
        self.max_frames = int(max_seconds * fps)
        self.frames = 0
        self.renderer = None
        self.recording, self.full = True, False
        self.trail, self.old_trails = [], []
        self.n_ok = self.n_done = 0
        self.status = ""
        self.next_t = 0.0
        self.method, self.ep, self.cube = "", 0, (0, 0, 0)
        
        ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
        self.proc = subprocess.Popen([
            ffmpeg_bin, "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{w}x{h}", "-pix_fmt", "rgb24", "-r", str(fps),
            "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-preset", "fast", path
        ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def new_episode(self, method, ep, cube):
        if self.trail:
            self.old_trails.append(self.trail[::6])
            self.old_trails = self.old_trails[-5:]
        self.trail = []
        self.method, self.ep, self.cube = method, ep, cube
        self.next_t = 0.0
        self.status = "running"

    def maybe_frame(self, sim):
        if not self.recording or sim.d.time < self.next_t:
            return
        self.next_t += 1.0 / self.fps
        if self.renderer is None:
            self.renderer = mujoco.Renderer(sim.m, self.h, self.w, max_geom=20000)
        if not self.trail or np.linalg.norm(self.trail[-1] - sim.tcp_pos()) > 0.004:
            self.trail.append(sim.tcp_pos())
        r = self.renderer
        r.update_scene(sim.d, self.cam)
        col = COLORS.get(self.method, (1, 1, 1))
        for tr, alpha, size in [(t, 0.35, 0.003) for t in self.old_trails] + \
                               [(self.trail, 1.0, 0.004)]:
            for p in tr:
                if r.scene.ngeom >= r.scene.maxgeom:
                    break
                g = r.scene.geoms[r.scene.ngeom]
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, [size, 0, 0],
                                    np.asarray(p, float), np.eye(3).flatten(),
                                    np.array([*col, alpha], np.float32))
                r.scene.ngeom += 1
        img_arr = r.render().copy()
        img = Image.fromarray(img_arr)
        draw = ImageDraw.Draw(img)
        x, y, yaw = self.cube
        lines = [f"IK Method: {self.method.upper()}   Episode: {self.ep}",
                 f"Cube: X={x:.2f} Y={y:.2f} Yaw={np.degrees(yaw):.0f} deg",
                 f"Success so far: {self.n_ok}/{self.n_done}   Status: {self.status}"]
        for i, t in enumerate(lines):
            draw.text((12, 12 + 22 * i), t, fill=(255, 255, 255))
        self.proc.stdin.write(np.array(img).tobytes())
        self.frames += 1
        if self.frames >= self.max_frames:
            self.recording, self.full = False, True

    def end_episode(self, sim, log):
        self.n_done += 1
        self.n_ok += int(log["success"])
        self.status = "SUCCESS" if log["success"] else "FAIL: " + log["failure_reason"][:40]
        if self.recording and self.renderer is not None:
            for _ in range(self.fps):
                self.r_hold(sim)

    def r_hold(self, sim):
        self.next_t = sim.d.time
        self.maybe_frame(sim)

    def close(self):
        if self.proc:
            if self.proc.stdin:
                self.proc.stdin.close()
            self.proc.wait()
        if self.renderer is not None:
            self.renderer.close()
