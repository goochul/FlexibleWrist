from franka_easy_ik import FrankaEasyIK
from scipy.spatial.transform import Rotation as R

# Create an instance of the FrankaEasyIK class
ik = FrankaEasyIK()

print(ik.robot)

init_position = [ 0.46688291, -0.01258228,  0.40752645]
init_orientation = [-1.54192905, -0.82459033, -1.59780517]

des_position = [0.46688291, 0.03741772, 0.40852645]
des_orientation = [-1.54192905, -0.82459033, -1.59780517]

# Desired end-effector position and orientation
position = init_position  # [x, y, z]
# Convert Euler angles to quaternion
rotation = R.from_euler('xyz', init_orientation, degrees=False)  # 'xyz' specifies the order
des_quat_orientation = rotation.as_quat()  # Returns [x, y, z, w]

# Print the quaternion
print("Quaternion (x, y, z, w):", des_quat_orientation)

# orientation = [1.0, 0.0, 0.0, 0.0]  # Quaternion [x, y, z, w]

# Perform IK and retrieve joint configuration
try:
    q = ik(position, des_quat_orientation, verbose=True)
    print("Optimized Joint Positions:")
    print(q)
except Exception as e:
    print(f"Error occurred: {e}")


#[-0.0087831, 0.3709803, -0.0241358, -2.1980871, -0.0297141, 4.1597863, 0.7708481]

#[-1.11719628 -1.3341321   1.22619936 -2.82587051  2.883708    2.31870227, -0.22356839]
