% Clear workspace and command window
clc; clear;

% Define symbolic variables
syms K1 K2

% Define parameters
alpha = 100;              % degrees
r = (180 - alpha) / 2;      % r in degrees

% Define vertical stiffness expression (symbolic)
Kv_expr = (K1 * K2 * cosd(r)^2) / (2 * (K1 * sind(r)^2 + K2));

% Convert the symbolic expression to a MATLAB function handle
Kv_fun = matlabFunction(Kv_expr, 'Vars', [K1, K2]);

% Define ranges for K1 and K2
K1_range = linspace(0.1, 20, 100);   % adjust range as needed
K2_range = linspace(0.1, 20, 100);   % adjust range as needed

% Create a meshgrid for K1 and K2
[K1_grid, K2_grid] = meshgrid(K1_range, K2_range);

% Evaluate Kv over the grid
Kv_vals = Kv_fun(K1_grid, K2_grid);

% Generate the 3D mesh plot
figure;
mesh(K1_grid, K2_grid, Kv_vals);
xlabel('K1');
ylabel('K2');
zlabel('K_{vertical}');
title('Vertical Stiffness K_{vertical} vs. K1 and K2 (alpha = 100°)');
grid on;