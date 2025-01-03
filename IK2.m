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

% Forward Kinematics to get the current end-effector position
T = forward_kinematics(initial_joint_positions, dh_params(1:7, :));
T = T * dh_transform(dh_params(8, 1), dh_params(8, 2), dh_params(8, 3), 0);
current_position = T(1:3, 4);

% Define the full motion (100 mm along Y-axis, in 1 mm steps)
desired_positions = repmat(current_position', 10, 1); % Replicate the current position
desired_positions(:, 2) = linspace(current_position(2), current_position(2) + 0.1, 10); % Update only the Y-axis

% Initialize variables
joint_positions = initial_joint_positions;
trajectory = current_position'; % To store the trajectory

% Iterative IK for each step
for i = 1:size(desired_positions, 1)
    desired_position = desired_positions(i, :)';

    % IK Loop for each step
    for iter = 1:10 % Max 10 iterations for each step
        T_current = forward_kinematics(joint_positions, dh_params(1:7, :));
        T_current = T_current * dh_transform(dh_params(8, 1), dh_params(8, 2), dh_params(8, 3), 0);
        current_position = T_current(1:3, 4);

        % Compute the position error (restrict to Y-axis only)
        position_error = desired_position - current_position;
        % position_error([1, 3]) = 0; % Zero out X and Z errors

        % Check convergence
        if norm(position_error) < 1e-4
            break;
        end

        % Compute Jacobian
        J = compute_jacobian(joint_positions, dh_params);
        J_translation = J(1:3, :); % Use only translational Jacobian

        % Solve for joint updates
        delta_theta = pinv(J_translation) * position_error;

        % Update joint positions
        joint_positions = joint_positions + delta_theta';
    end

    % Store the current position in the trajectory
    trajectory = [trajectory; current_position'];
end

% Plot the trajectory
figure;
plot3(trajectory(:, 1), trajectory(:, 2), trajectory(:, 3), '-o', 'LineWidth', 2);
hold on;

% Highlight the initial point in red
plot3(trajectory(1, 1), trajectory(1, 2), trajectory(1, 3), 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');

% Add labels and grid
grid on;
xlabel('X (m)');
ylabel('Y (m)');
zlabel('Z (m)');
title('End-Effector Trajectory (Strictly +Y Motion)');
legend('Trajectory', 'Initial Point');
axis equal;








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

% Compute the Jacobian
function J = compute_jacobian(joint_angles, dh_params)
    num_joints = size(joint_angles, 2);
    T = eye(4);
    z = zeros(3, num_joints);
    o = zeros(3, num_joints);
    for i = 1:num_joints
        a = dh_params(i, 1);
        d = dh_params(i, 2);
        alpha = dh_params(i, 3);
        theta = joint_angles(i);
        T = T * dh_transform(a, d, alpha, theta);
        z(:, i) = T(1:3, 3); % Z-axis of the current frame
        o(:, i) = T(1:3, 4); % Origin of the current frame
    end
    o_n = T(1:3, 4); % End-effector position
    J = zeros(6, num_joints);
    for i = 1:num_joints
        J(1:3, i) = cross(z(:, i), (o_n - o(:, i))); % Linear velocity part
        J(4:6, i) = z(:, i); % Angular velocity part
    end
end

function rpy = rotation_to_rpy(R)
    % Convert a 3x3 rotation matrix to Roll-Pitch-Yaw (RPY) angles
    % Input:
    %   R - 3x3 rotation matrix
    % Output:
    %   rpy - 1x3 vector containing [roll, pitch, yaw] in radians

    % Validate the input matrix
    if size(R, 1) ~= 3 || size(R, 2) ~= 3
        error('Input must be a 3x3 matrix.');
    end

    % Extract RPY angles from the rotation matrix
    % Pitch (y-axis rotation)
    if abs(R(3,1)) < 1
        pitch = asin(-R(3,1)); % Pitch
        roll = atan2(R(3,2), R(3,3)); % Roll
        yaw = atan2(R(2,1), R(1,1)); % Yaw
    else
        % Gimbal lock case
        yaw = 0; % Arbitrarily set yaw to 0
        if R(3,1) == -1
            pitch = pi / 2;
            roll = atan2(R(1,2), R(1,3));
        else
            pitch = -pi / 2;
            roll = atan2(-R(1,2), -R(1,3));
        end
    end

    % Return the RPY angles as a vector
    rpy = [roll, pitch, yaw];
end