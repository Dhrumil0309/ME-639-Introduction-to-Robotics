import numpy as np
import roboticstoolbox as rtb

# 1. Load the Franka Emika Panda model directly from the library
# This automatically loads all the correct DH parameters and link lengths
panda = rtb.models.Panda()

# --- Interactive User Input ---
print("Enter the 7 joint angles for the Franka Panda (in degrees):")
angles_deg = []

for i in range(7):
    while True:
        try:
            angle = float(input(f"Joint {i+1} angle (degrees): "))
            angles_deg.append(angle)
            break
        except ValueError:
            print("Invalid input. Please enter a numerical value.")

# Convert degrees to radians because robotic libraries strictly use radians
angles_rad = np.radians(angles_deg)

# 2. Calculate Forward Kinematics
# The fkine() function automatically computes the final transformation matrix
T_final = panda.fkine(angles_rad)

print("\n" + "="*40)
print("FORWARD KINEMATICS (ROBOTICS TOOLBOX):")
print("="*40)

# The library outputs a beautifully formatted SE(3) transformation matrix 
# which shows both the rotation matrix and the translation (XYZ) vector.
print(T_final)

# If you want to explicitly extract just the X, Y, Z coordinates:
print("\n" + "="*40)
print("EXTRACTED END-EFFECTOR POSITION (X, Y, Z):")
print("="*40)
position = T_final.t  # .t extracts the translation vector
print(f"X: {position[0]:.4f} meters")
print(f"Y: {position[1]:.4f} meters")
print(f"Z: {position[2]:.4f} meters")