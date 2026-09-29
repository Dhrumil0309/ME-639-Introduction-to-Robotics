import mujoco
import mujoco.viewer
import numpy as np
import os
import time

print("Which official robot do you want to simulate?")
print("1: Franka Emika Panda (7-DOF)")
print("2: Universal Robots UR5e (6-DOF)")
choice = input("Enter 1 or 2: ")

if choice == '1':
    model_path = "mujoco_menagerie/franka_emika_panda/panda.xml"
    ee_name = "hand" 
elif choice == '2':
    model_path = "mujoco_menagerie/universal_robots_ur5e/ur5e.xml"
    ee_name = "wrist_3_link" 
else:
    print("Invalid choice.")
    exit()

if not os.path.exists(model_path):
    print(f"\nERROR: Cannot find '{model_path}'.")
    exit()

# Load Model and Data
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# Find the internal ID of the end-effector so we can track it
ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ee_name)

print("\n" + "="*60)
print("FORWARD KINEMATICS VISUALIZER")
print("="*60)
print("-> Gravity is ON. Physics are active.")
print("-> Use the 'Control' menu (NOT the Joint menu) to move smoothly.")
print("-> DOUBLE-CLICK the end-effector to show its XYZ frame.")
print("="*60)
print("LIVE END-EFFECTOR COORDINATES (meters):")

# Launch in 'passive' mode so our Python code can keep running
with mujoco.viewer.launch_passive(model, data) as viewer:
    
    while viewer.is_running():
        # Step the full physics engine so gravity and motors work
        mujoco.mj_step(model, data)
        
        # Extract live X, Y, Z coordinates
        pos = data.xpos[ee_id]
        
        # Print over the same line in the terminal
        print(f"\rX: {pos[0]:.4f}  |  Y: {pos[1]:.4f}  |  Z: {pos[2]:.4f}    ", end="")
        
        viewer.sync()
        
        # Slightly faster refresh rate for smooth motor control
        time.sleep(0.02) 

print("\n\nSimulation Closed.")