clc; clear all;

% Define DH parameters for each joint
% Columns: [a, d, alpha]
dh_params = [
    0,      0.333,  0;           % Joint 1
    0,      0,     -pi/2;        % Joint 2
    0,      0.316,  pi/2;        % Joint 3
    0.0825, 0,      pi/2;        % Joint 4
   -0.0825, 0.384, -pi/2;        % Joint 5
    0,      0,      pi/2;        % Joint 6
    0.088,  0,      pi/2;        % Joint 7
    0,      0.107,  0           % Flange
];

% Initial joint values (theta) in radians
initial_joint_positions = [-0.0070, 0.3027, -0.0309, -2.5290, -0.0224, 4.4196, 0.7622];
des_joint_positions = [-0.0320, 0.3165, 0.1432, -2.5073, -0.0062, 4.4658, 0.7303];

% Compute the end-effector transformation matrix
T = forward_kinematics(initial_joint_positions, dh_params(1:7, :)); % Use only the first 7 rows for joint angles

% Compute the final transformation with the flange
T = T * dh_transform(dh_params(8, 1), dh_params(8, 2), dh_params(8, 3), 0); % Flange has no joint angle

% Extract end-effector position and orientation
end_effector_position = T(1:3, 4);
end_effector_orientation = T(1:3, 1:3);


% Compute the end-effector transformation matrix
T2 = forward_kinematics(des_joint_positions, dh_params(1:7, :)); % Use only the first 7 rows for joint angles

% Compute the final transformation with the flange
T2 = T2 * dh_transform(dh_params(8, 1), dh_params(8, 2), dh_params(8, 3), 0); % Flange has no joint angle

% Extract end-effector position and orientation
des_end_effector_position = T2(1:3, 4);
des_end_effector_orientation = T2(1:3, 1:3);


% Display results
disp('End-Effector Position:');
disp(end_effector_position);

disp('Des_End-Effector Position:');
disp(des_end_effector_position);

disp('End-Effector Orientation:');
disp(end_effector_orientation);

disp('Des_End-Effector Orientation:');
disp(des_end_effector_orientation);

% Function to calculate the transformation matrix based on DH parameters
function T = dh_transform(a, d, alpha, theta)
    T = [cos(theta), -sin(theta), 0, a;
         sin(theta)*cos(alpha), cos(theta)*cos(alpha), -sin(alpha), -d*sin(alpha);
         sin(theta)*sin(alpha), cos(theta)*sin(alpha), cos(alpha), d*cos(alpha);
         0, 0, 0, 1];
end

% Forward Kinematics to compute end-effector position and orientation
function T_final = forward_kinematics(joint_angles, dh_params)
    T_final = eye(4);
    num_joints = size(joint_angles, 2); % Ensure we only iterate over actual joints
    
    for i = 1:num_joints
        a = dh_params(i, 1);
        d = dh_params(i, 2);
        alpha = dh_params(i, 3);
        theta = joint_angles(i);
        
        T = dh_transform(a, d, alpha, theta);
        T_final = T_final * T;
    end
end