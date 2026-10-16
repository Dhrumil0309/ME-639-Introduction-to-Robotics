import mujoco
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

def check_file(filename):
    model_path = os.path.join(current_dir, "Lab3", "robot_descriptions", filename)
    print(f"\n--- Checking {filename} ---")
    try:
        model = mujoco.MjModel.from_xml_path(model_path)
        gripper_found = False
        for i in range(model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name:
                print(f"Body: {name}")
                if "gripper" in name.lower() or "finger" in name.lower() or "pad" in name.lower():
                    gripper_found = True
        
        if gripper_found:
            print("\n✅ GRIPPER IS PRESENT IN THIS FILE!")
        else:
            print("\n❌ NO GRIPPER FOUND.")
    except Exception as e:
        print(f"Error loading {filename}: {e}")

if __name__ == "__main__":
    check_file("arm_gripper.xml")
    check_file("single_arm_heal_effort_actuation_rs.xml")
    check_file("world.xml")