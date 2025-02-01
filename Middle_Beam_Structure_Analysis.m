clc; clear all

syms b l d bc lc dc alpha

%b       =   16.047;   %mm
L       =   3.5;       %mm
w       =   1.0;      %mm
h      =   7.17;        %mm

Ic      =   (w*L^3)/12; %mm^4
Ic2      =   (L*w^3)/12; %mm^4
E       =   23;      %N/mm^2

%Pcr     = (E*Ic*pi^2)/((h3)^2)
Pcr21     = (E*L*w^3*pi^2)/(12*(h)^2)


%%
syms F

alpha = 100
F0 = double(2.5*sind(alpha/2))
F1 = double(6*cosd(alpha/2)*sind(alpha/2))
F2 = double(2.5*sind(alpha/2) - cosd(alpha/2)^2)
F3 = double(2.5*sind(alpha/2))
F4 = double(2.5*sind(alpha/2) - cosd(alpha/2)^2)

%%
clc; clear all
h = 11;
bc = 1.25;
alpha = 170;
d = h/(2*sind(alpha/2))
dc = 10 - h/tand(alpha/2)

d2 = (h-bc)/(2*sind(alpha/2))
dc2 = 9.2 - 2*((h-bc)/2)/tand(alpha/2)
% dc22 = 9.2 - 2*d2*cosd(alpha/2)

%%
% clc; clear all

syms b l d bc lc dc alpha

%b       =   16.047;   %mm
L       =   3;       %mm
w       =   1.25;      %mm
h      =   dc2;        %mm

Ic      =   (w*L^3)/12; %mm^4
Ic2      =   (L*w^3)/12; %mm^4
E       =   23;      %N/mm^2

%Pcr     = (E*Ic*pi^2)/((h3)^2)
Pcrc     = (E*L*w^3*pi^2)/(12*(h)^2)

%%

syms b l d bc lc dc alpha

%b       =   16.047;   %mm
L       =   3.5;       %mm
w       =   1.0;      %mm
h      =   d2;        %mm

Ic      =   (w*L^3)/12; %mm^4
Ic2      =   (L*w^3)/12; %mm^4
E       =   23;      %N/mm^2

%Pcr     = (E*Ic*pi^2)/((h3)^2)
Pcr     = (E*L*w^3*pi^2)/(12*(h)^2)


