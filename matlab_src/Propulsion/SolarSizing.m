function [outp, re_entry] = calc_decay(ID, M, A, H, F10, Ap)
%% INPUTS
% ID = ; % identifier
% M = ;% mass 
% A = ;% area
% H = ;% altitude
% F10 = ; % Solar Radio Flux
% Ap = ; % Geomagnetic A Index


%% INITIALIZATIONS
Re = 6378000; % Earth radius
Me = 5.98*10^24; % Earth mass
G = 6.67*10^-11; % Universal constant of gravitation
T = 0; % time (days)
dT = 0.1; % time increment (days)
D9 = dT * 3600 * 24; % time increment (seconds)
H1 = 10; % height increment (km)
H2 = H; % orbit height (km)
R = Re + (H * 1000); % Orbital Radius (m)
P = 2 * pi * sqrt(R^3 / Me / G); % Period (s)

%% ITERATE
outp = [];
i = 1; 
while H > 180
    SH = (900 + 2.5 * (F10 - 70) + 1.5 * Ap) / (27 - .012 * (H - 200));
    DN = 6E-10 * exp(-(H - 175) / SH); % Atmospheric density
    dP = 3 * pi * A / M * R * DN * D9; % decrement in orbital period
    if H <= H2
        Pm = P / 60;
        MM = 1440 / Pm;
        nMM = 1440 / ((P - dP)) / 60;
        Decay = dP / dT / P * MM; % rev/day^2
        outp(i, :) = [T, H, P/60, MM, Decay];  % store values
        H2 = H2 - H1; % decrement height
        i = i+1;
    end
    P = P - dP; 
    T = T + dT; 
    R = (G * Me * P^2 / 4 / pi^2)^(1/3); % new orbital radius
    H = (R - Re) / 1000; % new altitude 
end
    
re_entry = T;

end