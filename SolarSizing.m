%Michael Berthin
%sizes solar panels per satellite
%% Inputs
P_d = 61; %daytime power load [W], how much power system uses
T_d = 3540; %time of orbit spent in daylight [s]
P_e = P_d; %eclipse power load [W]
T_e = 2100; %time of orbit spent in eclipse [s]
X_d = 0.80; %efficiency of getting power from SA directly to loads
X_e = 0.6; %Efficiency of getting power from SA to batteries then laods
P_i = 1353.05; %Input solar power density [W/m^2], different at each orbit
n = 0.29; %solar cell efficiency, typical for multi
I_d = 0.72; %inherent degradation of solar cell, nominal value in SMAD table 21-14
theta = 23.5; %angle btw solar cell normal and the sun, use worst case (23.5 inSMAD FiresatII)

%% Calculations

P_SA = 1 / T_d * ((P_d * T_d / X_d) + (P_e * T_e) / X_e); %Required solar array output power [W]
P_o = P_i * n; %Output solar power density [W/m^2]
P_bol = P_o * I_d * cosd(theta); %beginning of life power output density [W/m^2]
L_d = (1 - 0.0275) ^ 3; %lifetime degradation of solar cells, 3 is mission lifetime in years
P_eol = P_bol * L_d; %end of life power output density [W/m^2]
A_SA = P_SA / P_eol; %required solar panel area [m^2], chosen Sparkwing has .6 m^2 area

num_panels = 1;%change depending on solar array output, look at Sparkwing fact sheet
%current values give ~137 W, use 800x750 (8 string) panel

m_panels = A_SA * 3.8; %estimated mass of array (for sparkwing) [kg]
m_mechanisms = num_panels * 0.4;
m_total = m_panels + m_mechanisms;