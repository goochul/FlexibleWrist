clc; clear all;

% Load the CSV file
filename = 'data/20241229/191909/joint_positions.csv'; % Replace with your actual file name
data_table = readtable(filename, 'VariableNamingRule', 'preserve');

% Extract the first column containing the joint angles
raw_joint_angles = data_table{:, 1}; % Assuming all joint angles are in the first column

% Initialize matrix to store parsed joint angles
num_samples = length(raw_joint_angles);
num_joints = 7; % Number of joints for Franka Emika Panda
joint_angles = zeros(num_samples, num_joints);

% Parse the joint angle strings
for i = 1:num_samples
    joint_angles(i, :) = str2num(raw_joint_angles{i}); %#ok<ST2NM> Parse string to numeric array
end

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

% Initialize variables to store results
end_effector_positions = zeros(num_samples, 3); % Store positions
end_effector_orientations = zeros(3, 3, num_samples); % Store orientations

% Loop through each set of joint angles
for sample_idx = 1:num_samples
    % Extract joint angles for the current sample
    current_joint_angles = joint_angles(sample_idx, :);
    
    % Update DH parameters with joint angles
    DH_params(1:7, 4) = current_joint_angles; % Update the theta column
    
    % Initialize Transformation Matrix
    T = eye(4);
    
    % Compute Forward Kinematics
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
    end_effector_positions(sample_idx, :) = T(1:3, 4)';
    end_effector_orientations(:, :, sample_idx) = T(1:3, 1:3);
end

% Display Results for the Last Sample
disp('End-Effector Position for Last Sample:');
disp(end_effector_positions(end, :));

disp('End-Effector Orientation for Last Sample:');
disp(end_effector_orientations(:, :, end));

% Extract X, Y, Z coordinates
x = end_effector_positions(:, 1); % X-coordinates
y = end_effector_positions(:, 2); % Y-coordinates
z = end_effector_positions(:, 3); % Z-coordinates

% Offset the first position value
x_offset = x - x(1);
y_offset = y - y(1);
z_offset = z - z(1);

% 2D Plot of X, Y, and Z positions (offset)
figure;
hold on;

% Plot X, Y, Z coordinates with offset
plot(x_offset, 'r', 'LineWidth', 1.5, 'DisplayName', 'X Position (Offset)');
plot(y_offset, 'g', 'LineWidth', 1.5, 'DisplayName', 'Y Position (Offset)');
plot(z_offset, 'b', 'LineWidth', 1.5, 'DisplayName', 'Z Position (Offset)');

% Labels and Legend
xlabel('Sample Points');
ylabel('Position (m, offset)');
title('Offset End-Effector Positions (X, Y, Z) Over Time');
legend('Location', 'best');
grid on;

disp('2D plot of offset end-effector positions over time generated successfully.');
