clc; clear; close all;

syms F1 F2  % F1, F2 are unknowns we want to solve for

% Given parameters
Fy = 1;
alpha = deg2rad(70);  % Convert 140 degrees to radians

% Define the equations (note the '== 0')
eq1 = F1*sind(130) + F2*sind(260) == 0;
eq2 = F1*cosd(130) + F2*cosd(260) - Fy*cosd(180) == 0;

% Solve the system of equations for [F1, F2]
sol = solve([eq1, eq2], [F1, F2]);

% Extract the solutions
F1_sol = double(sol.F1)
F2_sol = double(sol.F2)

% disp(F1_sol);
% 
% F1_sol*sind(70)
% F2_sol*sind(40)
% 
% F1_sol*cosd(70)
% F2_sol*cosd(40)

%%
clc; clear; close all;

syms F1 F2 alpha F% F1, F2 are unknowns we want to solve for

% Given parameters
% alpha = deg2rad(70);  % Convert 140 degrees to radians

% Define the equations (note the '== 0')
eq1 = F1*sin(alpha/2) - F2*2*sin(alpha/2)*cos(alpha/2) - F*cos(alpha/2) == 0;
eq2 = F1*cos(alpha/2) - 2*F2*(cos(alpha/2)^2 - 1) == 0;

% Solve the system of equations for [F1, F2]
sol = solve([eq1, eq2], [F1, F2]);


% Extract the solutions
F1_sol = sol.F1
F2_sol = sol.F2

% disp(F1_sol);
% 
% F1_sol*sind(70)
% F2_sol*sind(40)
% 
% F1_sol*cosd(70)
% F2_sol*cosd(40)