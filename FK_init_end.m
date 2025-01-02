clc; clear all;

% Define joint positions
reset_joint_positions = [-0.0087831, 0.3709803, -0.0241358, -2.1980871, -0.0297141, 4.1597863, 0.7708481];
des_joint_positions = [0.0035, 0.3821, 0.0751, -2.1764, -0.1221, 4.1972, 0.7484];

% DH Parameters for Franka Emika Panda
% Columns: [a, d, alpha, theta]
DH_params = [
    0,      0.333,  0,          NaN;  % Joint 1
    0,      0,     -pi/2,       NaN;  % Joint 2
    0,      0.316,  pi/2,       NaN;  % Joint 3
    0.0825, 0,      pi/2,       NaN;  % Joint 4
   -0.0825, 0.384, -pi/2,       NaN;  % Joint 5
    0,      0,      pi/2,       NaN;  % Joint 6
    0.088,  0,      pi/2,       NaN;  % Joint 7
    0,      0.2,  0,          0     % Flange
];

% Function to calculate transformation matrix based on DH parameters
TF_matrix = @(alpha, a, d, q) ...
    [cos(q), -sin(q), 0, a;
     sin(q)*cos(alpha), cos(q)*cos(alpha), -sin(alpha), -sin(alpha)*d;
     sin(q)*sin(alpha), cos(q)*sin(alpha),  cos(alpha),  cos(alpha)*d;
     0, 0, 0, 1];

% Calculate for reset_joint_positions
[position_reset, orientation_reset] = compute_fk(reset_joint_positions, DH_params, TF_matrix);

% Calculate for des_joint_positions
[position_des, orientation_des] = compute_fk(des_joint_positions, DH_params, TF_matrix);

% Display Results
disp('End-Effector Position for Reset Joint Positions:');
disp(position_reset);

disp('End-Effector Orientation for Reset Joint Positions:');
disp(orientation_reset);

disp('End-Effector Position for Desired Joint Positions:');
disp(position_des);

disp('End-Effector Orientation for Desired Joint Positions:');
disp(orientation_des);

% Local function definition
function [position, orientation] = compute_fk(joint_positions, DH_params, TF_matrix)
    T = eye(4); % Initialize Transformation Matrix
    DH_params(1:7, 4) = joint_positions; % Update the theta column with joint positions

    for i = 1:size(DH_params, 1)
        a = DH_params(i, 1);
        d = DH_params(i, 2);
        alpha = DH_params(i, 3);
        theta = DH_params(i, 4);

        % Compute Transformation Matrix for the Current Joint
        A_i = TF_matrix(alpha, a, d, theta);

        % Accumulate Transformations
        T = T * A_i;
    end

    % Extract End-Effector Position and Orientation
    position = T(1:3, 4)'; % End-Effector Position
    orientation = T(1:3, 1:3); % End-Effector Orientation
end


