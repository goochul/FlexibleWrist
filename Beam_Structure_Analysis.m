clc; clear; close all;

syms F1 F2  % F1, F2 are unknowns we want to solve for

% Given parameters
Fy = 1;
alpha = deg2rad(140);  % Convert 140 degrees to radians
beta = pi-alpha/2;

% Define the equations (note the '== 0')
eq1 = F1*cos(beta) + F2*cos(2*beta) == -Fy;
eq2 = F1*sin(beta) + F2*sin(2*beta) == 0;


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

syms F3 F4  % F1, F2 are unknowns we want to solve for

Fy = 0.6840;
alpha = deg2rad(140);  % Convert 140 degrees to radians
beta = pi-alpha/2;

eq3 = F3*cos(beta)+F4*cos(alpha+beta) == -Fy;
eq4 = F3*sin(beta)+F4*sin(alpha+beta) == 0;

% Solve the system of equations for [F1, F2]
sol = solve([eq3, eq4], [F3, F4]);

% Extract the solutions
F3_sol = double(sol.F3)
F4_sol = double(sol.F4)
