function DV = calc_DV(H1, H)
%% INITIALIZATIONS
H2 = H;
Re = 6378000; % Earth radius
R1 = (H1*1000) + Re; 
R2 = (H2*1000) + Re; 
grav_param = 3.99*10^5*(100^3); % m^3/s^2

%% CALCULATIONS
VC1 = sqrt(grav_param/R1);
DV = VC1 * (sqrt(R1/R2) - sqrt(2*R1/(R2*(1+(R2/R1)))));

end