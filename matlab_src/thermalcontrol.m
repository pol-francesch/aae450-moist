%battery operating temp: -10C to 50C
%cubeADCS operating temp: -10C to 60C (gen1); -25C to 80C (gen2)
%Xiphos QS7 operating temp:  -40C to +60⁰C

d = .35; %spacecraft diameter, assume spherical (m) *.35*
ab = .08; %absorbtivity (Teflon aluminum backing (2mil thickness) 
em = .66; %emissivity (Teflon aluminum backing (2mil thickness) 
alt = 350; %orbit altitude (km) *550*
P_sun = 97.3; %power utilized when in sunlight (W) *61*
P_ec = 65.22-24; %power utilized when in eclipse (W)*61*
sigma = 5.67E-8; % stefan-boltzman constant

theta = asind(6378.14/(6378.14+alt)); 
lambda=4*pi*sind(theta/2)^2; %spherical solid angle subtended by cone w half angle theta
f_e= lambda/(4*pi); %view factor from spacecraft to earth
f_s= 1 - f_e; %view factor from spacecraft to space
a_f=.21; %m^2 %a_f= pi*(d/2)^2; %front surface spacecraft area
a_tot= 0.42; %m^2 total spacecraft surface area 2Ux3Ux2U
 
Qin_s = ab*a_f*1367; %Q directed solar input
Qin_a = ab*a_tot*410*f_e; %Earth reflected solar input

Qin = P_sun+Qin_s+Qin_a; %total input

syms Tsc

Qout_e= em*sigma*a_tot*f_e*((Tsc^4)-(290^4)); %radiated output to earth
Qout_s= em*sigma*a_tot*f_s*((Tsc^4)-(4^4)); %radiated output to space
Qout = Qout_e +Qout_s; %total radiated output

Temp = solve(Qout == Qin); 
Temp_sunlight = (vpa(Temp(2))-273.15) %Spacecraft temp for Thermal Equilibrium in sunlight (C)

Temp2 = solve(Qout == P_ec);
Temp_eclipse = (vpa(Temp2(2))-273.15) %Spacecraft temp for Thermal Equilibrium in eclipse (C)

%teflon_density= 2.2 %gram/cm^3
%coatingmass= teflon_density*a_tot*10000*0.00508 %mass added to spacecraft from coating (gram) 

