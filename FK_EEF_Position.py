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



# Rigid - 0mm -> End-effector position: [0.58363926 0.13976081 0.30265662]
    # joint_pos = [0.2609, 0.7747, 0.1131, -1.8283, 0.3867, 4.2508, -0.2598]

# Rigid - -2mm -> End-effector position: [0.58361658 0.13975279 0.30066035]
    # joint_pos = [0.21698373768266435, 0.6939310823533847, 0.11144326795547115, -2.0084434479306004, 0.3304280088901366, 4.332134874564391, -0.19695163164288526]

# Rigid - -3mm -> End-effector position: [0.58367896 0.22112138 0.29989986]
    # joint_pos = [0.34771935232863976, 0.7638804094642561, 0.15426175992556038, -1.8615429963856076, 0.502102701115341, 4.314777325532903, -0.3276387286425575]

# Rigid - -5mm -> End-effector position: [0.58355842 0.16845577 0.29777526]
    # joint_pos = [0.2609, 0.7861, 0.1125, -1.8239, 0.3839, 4.2571, -0.2566]

# Rigid - -10mm -> End-effector position: [0.61855555 0.16848023 0.29278919]
    # joint_pos = [0.2610, 0.7976, 0.1119, -1.8192, 0.3812, 4.2633, -0.2535]   

# Rigid - -13mm -> End-effector position: [0.61855008 0.16850509 0.28982451]
    # joint_pos = [0.2611, 0.8045, 0.1115, -1.8163, 0.3796, 4.2670, -0.2517] 

# Rigid - -14mm -> End-effector position: [0.61855127 0.16850395 0.28877874]
    joint_pos = [0.2611, 0.8069, 0.1114, -1.8153, 0.3791, 4.2682, -0.2511]

# Rigid - -15mm -> End-effector position: [0.61856724 0.16845764 0.28779646]
    # joint_pos = [0.2611, 0.8092, 0.1112, -1.8143, 0.3786, 4.2694, -0.2505]

# Rigid - -15mm -> End-effector position: [0.58356616 0.16846449 0.28780274]
    # joint_pos = [0.2635, 0.7451, 0.1262, -1.9532, 0.3857, 4.3442, -0.2330]  

# Rigid - -20mm -> End-effector position: [0.61853982 0.1684269  0.28278421]
    # joint_pos = [0.2612, 0.8208, 0.1105, -1.8094, 0.3760, 4.2754, -0.2475]  

# Rigid - -25mm -> End-effector position: [0.61854072 0.16845408 0.27781509]
    # joint_pos = [0.2614, 0.8325, 0.1098, -1.8041, 0.3735, 4.2812, -0.2446]

# Rigid - -30mm -> End-effector position: [0.61854072 0.16845408 0.27781509]
    # joint_pos = [0.2616, 0.8443, 0.1091, -1.7986, 0.3711, 4.2869, -0.2418]

# Rigid - -40mm -> End-effector position: [0.61854072 0.16845408 0.27781509]
    # joint_pos = [0.2621, 0.8682, 0.1076, -1.7869, 0.3664, 4.2980, -0.2363]


# FW - -0mm -> End-effector position: [0.54363374 0.22144058 0.30277588]
    # joint_pos = [0.3519, 0.6895, 0.1798, -2.0106, 0.5197, 4.3909, 0.4279]

# FW - -0mm -> End-effector position: [0.57357688 0.16834196 0.30281881] # Front
    # joint_pos = [0.2637, 0.6915, 0.1333, -2.0045, 0.3969, 4.3439, 0.4981]

# FW - -5mm -> End-effector position: [0.57356489 0.16833813 0.29781025]
    # joint_pos = [0.2639, 0.7036, 0.1324, -2.0002, 0.3941, 4.3510, 0.5013]

# FW - -10mm -> End-effector position: [0.57355918 0.16832422 0.29277341]
    # joint_pos = [0.2641, 0.7158, 0.1315, -1.9957, 0.3914, 4.3579, 0.5044]

# FW - -15mm -> End-effector position: [0.5735545  0.16831116 0.28778069]
    # joint_pos = [0.2643, 0.7280, 0.1306, -1.9910, 0.3887, 4.3647, 0.5075]

# FW - -20mm -> End-effector position: [0.573552   0.1683476  0.28280068]
    # joint_pos = [0.2646, 0.7403, 0.1297, -1.9860, 0.3861, 4.3714, 0.5105]

# FW - -30mm -> End-effector position: [0.57354187 0.16835822 0.27280148] # Front
    # joint_pos = [0.2652, 0.7651, 0.1278, -1.9755, 0.3810, 4.3844, 0.5163]

# FW - -30mm -> End-effector position: [0.56353072 0.16835998 0.27278463] # 10mm back
    # joint_pos = [0.2664, 0.7492, 0.1322, -2.0122, 0.3843, 4.4052, 0.5210]

# FW - -40mm -> End-effector position: [0.56351145 0.16838484 0.26276639] # 10mm back
    # joint_pos = [0.2672, 0.7746, 0.1301, -2.0009, 0.3794, 4.4180, 0.5266]

# FW - -50mm -> End-effector position: [0.56344118 0.19839611 0.25272023] # 10mm back
    # joint_pos = [0.3143, 0.8230, 0.1471, -1.9361, 0.4347, 4.4207, 0.4875]

    
# FW - -40mm -> End-effector position: [0.54363395 0.22146114 0.24476718]
    # joint_pos = [0.3581, 0.8338, 0.1640, -1.9505, 0.4843, 4.4624, 0.4714]




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


'''
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

'''