# Lab 1: MuJoCo Robot Simulation Challenges

Two MuJoCo challenges that visualize the **body-to-ground rotation matrix** live, next to the robot.

## Challenge 1: TurtleBot3 Waffle Pi (`lab_challenge_1.py`)
- Model: `robotis_mujoco_menagerie/robotis_tb3/scene_turtlebot3_waffle_pi.xml`
- Drive with **WASD / arrow keys**, **Space** to stop.
- Draws the fixed ground frame and the moving chassis frame (X = red, Y = green, Z = blue).
- Shows the 3x3 rotation matrix (body to ground) as stacked labels above the robot and as a fixed on-screen overlay.

## Challenge 2: Skydio X2 drone (`lab_challenge_2.py`)
- Model: `mujoco_menagerie/skydio_x2/scene.xml`
- Fully autonomous: hovers for 1 s, flies through several waypoints at different heights, then returns to the start position.
- Control: outer PD position loop, inner PD attitude and altitude controller, motor allocation through MuJoCo's actuator Jacobian.
- Live 3x3 rotation matrix (drone to world), mission status, and position shown in the viewer.

## Setup
The model folders are not included in this repo (they are large). Download them into this folder:

```bash
cd Lab_1
git clone https://github.com/google-deepmind/mujoco_menagerie.git
# also place the ROBOTIS TurtleBot3 model folder here as: robotis_mujoco_menagerie/
pip install mujoco numpy
```

## Youtube Video
## Project Demo

[![Watch the Project Demo](https://img.youtube.com/vi/VH6RUUwDnO4/0.jpg)](https://youtu.be/VH6RUUwDnO4)

**Click the image above to watch the project demonstration on YouTube.**

## Run
```bash
python lab_challenge_1.py
python lab_challenge_2.py
```
