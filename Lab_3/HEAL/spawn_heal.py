import mujoco
import mujoco.viewer
import time
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(current_dir, "Lab3", "robot_descriptions")
model_path = os.path.join(model_dir, "arm_gripper.xml")

def main():
    original_dir = os.getcwd()
    try:
        os.chdir(model_dir)
        model = mujoco.MjModel.from_xml_path(model_path)
    finally:
        os.chdir(original_dir)
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("Combined arm and gripper spawned. Holding posture with gravity compensation.")
        
        while viewer.is_running():
            mujoco.mj_step1(model, data)
            
            # Counteract gravity so the arm holds its starting pose perfectly
            data.qfrc_applied[:] = data.qfrc_bias
            
            mujoco.mj_step2(model, data)
            
            viewer.sync()
            time.sleep(model.opt.timestep)

if __name__ == "__main__":
    main()
