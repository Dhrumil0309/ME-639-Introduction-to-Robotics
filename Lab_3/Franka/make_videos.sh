#!/usr/bin/env bash
# make_videos.sh - records the deliverable videos (1 min per IK method).
# On Linux without a screen keep MUJOCO_GL=egl; on Windows/Mac delete that line.
export MUJOCO_GL=egl
mkdir -p videos
for m in mink dls qp pinv; do
  python pick_place.py --method $m --episodes 8 --tag _video \
         --video videos/${m}_1min.mp4 --video-seconds 60
done
# a short clip of failure cases (smart choices OFF)
python pick_place.py --method dls --episodes 4 --region full --naive --tag _video \
       --video videos/failures_naive.mp4 --video-seconds 30
# join the three required clips into one 3-minute video for submission
printf "file 'mink_1min.mp4'\nfile 'dls_1min.mp4'\nfile 'qp_1min.mp4'\n" > videos/list.txt
ffmpeg -y -loglevel error -f concat -safe 0 -i videos/list.txt -c copy videos/three_methods_3min.mp4
rm -f videos/list.txt results/logs_*_video.csv results/failures/*_video_*
echo DONE
