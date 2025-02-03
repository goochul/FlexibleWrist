clc; clear;

% Define material and geometric parameters
E = 23;     % Young's modulus (material stiffness)
b = 1;      % Width (of the beam's cross-section)
h = 11;     % Height (of the beam's cross-section)
l = 4;      % Length of the beam

alpha = 170;  % (This is not used here, since we generate a range below)
bc = 1.25;  % Cross-beam parameter (e.g., width)
dc = 2.3;   % Cross-beam parameter (e.g., thickness)

% Compute K1 and K2 based on given formulas:
% K1 is computed as E*b*l divided by an effective length,
% where the effective length is given by h/(2*cosd(r)) and r is defined below.
% K2 is computed as E*bc*l/dc.
  
% Set the external force F
F = 20;

% Create a vector of alpha values from 180 to 100 degrees in 1° decrements.
alpha = 180:-1:100;  % Alternatively: linspace(180,100,81)

% Compute the inclination r for each alpha.
% Here, r = (180 - alpha)/2; that is, the beam is assumed to be inclined by an angle r from vertical.
r = (180 - alpha) / 2;  % in degrees

% Compute the axial stiffness of K1 beams.
% The effective length in the denominator is h/(2*cosd(r)).
K1 = E * b * l ./ ( h ./ (2 * cosd(r)) );

% Compute the stiffness of the cross-beam K2.
K2 = E * bc * l / dc;

% Compute the effective vertical stiffness Kv of the entire structure using the formula:
% Kv = (K1*K2*cos^2(r)) / (2*(K1*sin^2(r)+K2))
Kv = (K1 .* K2 .* cosd(r).^2) ./ ( 2 * (K1 .* sind(r).^2 + K2) );

% Given the external vertical load F, the total vertical displacement is:
delta = F ./ Kv;

% Plot the relationship: x-axis is alpha (in degrees) and y-axis is delta (displacement)
figure;
plot(alpha, delta, 'b-o','LineWidth',1.5);
xlabel('\alpha (deg)');
ylabel('\delta');
title('Alpha vs. \delta (Displacement) when input force is 10N');
grid on;