import numpy as np

def dh_transform(a, d, alpha, theta):
    """Calculates the standard 4x4 DH transformation matrix for a single joint."""
    return np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
        [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
        [0,              np.sin(alpha),                np.cos(alpha),               d],
        [0,              0,                            0,                           1] # Fixed missing row
    ])

def franka_panda_fk(q):
    """
    Computes Forward Kinematics for Franka Panda 7-DOF using standard DH parameters.
    q: list or array of 7 joint angles (in radians).
    """
    # Standard DH Parameters for Franka Emika Panda (a, d, alpha, theta)
    # Note: Distances are in meters.
    dh_params = [
        [0,       0.333,  0,        q[0]],
        [0,       0,     -np.pi/2,  q[1]],
        [0,       0.316,  np.pi/2,  q[2]],
        [0.0825,  0,      np.pi/2,  q[3]],
        [-0.0825, 0.384, -np.pi/2,  q[4]],
        [0,       0,      np.pi/2,  q[5]],
        [0.088,   0,      np.pi/2,  q[6]] 
    ]
    
    # Initialize the base transformation matrix as an Identity matrix
    T = np.eye(4)
    
    # Multiply the matrices sequentially: T = T1 * T2 * ... * T7
    for params in dh_params:
        A = dh_transform(*params)
        T = T @ A  
        
    return T

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

# Convert degrees to radians for the mathematical functions
angles_rad = np.radians(angles_deg)

# Compute the final matrix
T_final = franka_panda_fk(angles_rad)

print("\n" + "="*30)
print("FINAL TRANSFORMATION MATRIX:")
print("="*30)
print(np.round(T_final, 4))

print("\n" + "="*30)
print("END-EFFECTOR POSITION (X, Y, Z):")
print("="*30)
print(f"X: {T_final[0, 3]:.4f} meters")
print(f"Y: {T_final[1, 3]:.4f} meters")
print(f"Z: {T_final[2, 3]:.4f} meters")