import numpy as np

def dh_transform(a, d, alpha, theta):
    """
    Calculate the individual transformation matrix from DH parameters.
    """
    T = np.array([
        [np.cos(theta), -np.sin(theta), 0,           a],
        [np.sin(theta)*np.cos(alpha),  np.cos(theta)*np.cos(alpha), -np.sin(alpha), -d*np.sin(alpha)],
        [np.sin(theta)*np.sin(alpha),  np.cos(theta)*np.sin(alpha),  np.cos(alpha),  d*np.cos(alpha)],
        [0,                0,               0,                    1]
    ], dtype=float)
    return T

def forward_kinematics(joint_angles, dh_params):
    """
    Compute the forward kinematics for a given set of joint angles using DH parameters.
    """
    T_final = np.eye(4)
    for i in range(len(joint_angles)):
        a = dh_params[i, 0]
        d = dh_params[i, 1]
        alpha = dh_params[i, 2]
        theta = joint_angles[i]
        T_final = T_final @ dh_transform(a, d, alpha, theta)
    return T_final

def main():
    # Define DH parameters for the manipulator (for 7 joints + flange)
    # Each row is [a, d, alpha]
    dh_params = np.array([
        [0,       0.333,  0],       # Joint 1
        [0,       0,     -np.pi/2], # Joint 2
        [0,       0.316,  np.pi/2],  # Joint 3
        [0.0825,  0,      np.pi/2],  # Joint 4
        [-0.0825, 0.384, -np.pi/2],  # Joint 5
        [0,       0,      np.pi/2],  # Joint 6
        [0.088,   0,      np.pi/2],  # Joint 7
        [0,       0.107,  0]         # Flange (no joint angle)
    ])


# Rigid - 3mm -> 0.2557 -> 6N
    # joint_pos = [0.2603716879568429, 0.8710246234699283, 0.09491559331464389, -1.8095047967234605, 0.35325434824801843, 4.317308346533917, -0.2676839801859226]

# Rigid - 5mm -> 0.2603 -> X
    # joint_pos = [0.260246352958216, 0.8597932627679873, 0.09495501308018686, -1.8153200366326707, 0.354427433907442, 4.311771782226742, -0.27009058706378725]

# FW - 5mm -> 0.2604 -> 3N
    # joint_pos = [0.24495868246347954, 0.7845033643793414, 0.10849601777299323, -1.9869481023857858, 0.34194909233016474, 4.402225733704704, 0.5508996869844951]

# Rigid - 8mm ->  0.2507 -> 11N
    joint_pos = [0.26042601814022026, 0.8832495660605612, 0.09494363995422889, -1.8029212805176424, 0.3513636548648798, 4.322076144797115, -0.2658767068294787]

# Rigid - 8mm + End : End-effector position: [ 0.61393841 -0.18740608  0.25093243]
    # joint_pos = [-0.282516338663248, 0.9034684320516565, -0.1266899573129044, -1.759507110566509, -0.3829518516520576, 4.315862061037139, 0.22343902347074715]

# Rigid - 11mm -> 0.2476 -> 15N
    # joint_pos = [0.26066787326417723, 0.8908952793716062, 0.09416631884493447, -1.798950650665137, 0.3500657749960026, 4.32520012046846, -0.2651242760615852]

# FW - 3mm ->  0.2542 -> 6N
    # joint_pos = [0.24529865742334803, 0.7996490953187496, 0.10739062332158794, -1.9805892939209393, 0.3394619544649416, 4.409602415076757, 0.5538878960270773]


# ------ Diff Slope ----
# joint_pos = 
# FW - 45mm -> 0.2694
# [0.29438856863714497, 0.7368045069849353, 0.14336456021808536, -2.0573168467258895, 0.4192353805404943, 4.444892903442473, 0.5160285301485481]

# FW - 55mm -> 0.26448
    # joint_pos =  [0.2943614921144495, 0.7496351955637529, 0.14208269808552276, -2.0517463348256597, 0.41666481538231864, 4.451594750512129, 0.5190975804407206]

# FW - 65mm -> 0.24983827
    # joint_pos = [0.32674761713994066, 0.8036249130120485, 0.15196267319163903, -1.997130498696952, 0.44787123409119217, 4.463614442058135, 0.49853497804954666]

# FW - 70mm -> 0.24449
    # joint_pos = [0.35762041958610874, 0.8340169433386678, 0.1634229848320063, -1.950822599652529, 0.48374708059274973, 4.461776389480371, 0.4712388536213045]

# Rigid - 65mm -> 0.2497088 
    # joint_pos = [0.324457791792462, 0.8665674108505306, 0.13391879314623437, -1.846509463930537, 0.43834534968203626, 4.377006772822387, -0.26228434395358297]

# Rigid - 70mm -> 0.24469
    # joint_pos = [0.3522734289344591, 0.8951169571513091, 0.14208814977621642, -1.802834102128417, 0.47004358251320516, 4.374838868985608, -0.28766393547746144]

    # Given initial joint positions (radians) for the 7 joints (flange has no joint angle)
    initial_joint_positions = np.array(joint_pos)

    # Compute forward kinematics for the 7 joints using their DH parameters
    T = forward_kinematics(initial_joint_positions, dh_params[:7, :])
    # Apply the flange transformation (with a zero joint angle for the flange)
    T = T @ dh_transform(dh_params[7, 0], dh_params[7, 1], dh_params[7, 2], 0.0)

    print(T)

    # Extract the end-effector position from the final transformation matrix
    end_effector_position = T[0:3, 3]
    print("End-effector position:", end_effector_position)

if __name__ == "__main__":
    main()