"""
record_3min_video.py - Records 1-minute video per IK method (Mink, DLS, QP)
and concatenates them into a single 3-minute video: Franka_3min_IK_comparison.mp4
"""
import os
import sys
import subprocess
import imageio_ffmpeg

METHODS = ["mink", "dls", "qp"]
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

def record_method(method):
    out_mp4 = f"results/video_{method}.mp4"
    print(f"\n==========================================")
    print(f"Recording 60s video for {method.upper()}...")
    print(f"==========================================")
    cmd = [
        sys.executable, "pick_place.py",
        "--method", method,
        "--episodes", "25",
        "--video", out_mp4,
        "--video-seconds", "60"
    ]
    subprocess.run(cmd, check=True)
    return out_mp4

def concat_videos(v_files, final_output):
    list_file = "results/concat_list.txt"
    with open(list_file, "w") as f:
        for vf in v_files:
            f.write(f"file '{os.path.basename(vf)}'\n")
    print(f"\nConcatenating videos into {final_output}...")
    cmd = [
        FFMPEG, "-y", "-f", "concat", "-safe", "0",
        "-i", list_file, "-c", "copy", final_output
    ]
    subprocess.run(cmd, cwd="results", check=True)
    print(f"Successfully generated final 3-minute video: {final_output}!")

def main():
    v_files = []
    for m in METHODS:
        vf = record_method(m)
        v_files.append(vf)
    final_output = "Franka_3min_IK_comparison.mp4"
    concat_videos(v_files, final_output)

if __name__ == "__main__":
    main()
