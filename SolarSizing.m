%Michael Berthin
%sizes solar panels per satellite

%% Notes From Research 
%(SOA_Power)
%COTS operating tmep -40 to 85 C
%BOL efficiencies p.27, ranges from 16-31%
%BOL Peak SA Power values on p.29
%Battery producttable 3-3 p.35
%Power management and dsitribution systems table 3-6 p.44
%lithium based secondary batteries most common with primary energy source
%being solar array (table 3-7)

%(SMAD, 21.2 TABLE)
%direct: X_e = 0.65 and X_d = 0.85 (shunt regulator operates in parallel to
%array and shunts current from subsytem when loads or battery charging do
%not need power, excess energy dissipated as heat)
%peak power tracking: X_e = 0.6 amd X_d = 0.80 (optimizes incidence
%sunlight angle and power generated, losses only due to efficiency)
%peak power tracking requiires power converter btw array and laods
%P_o for given junctions: Si = 202, GaAs = 253, Multi = 383
%performance degradation per yr (%): Si = 3.75, GaAs = 2.75, Tri[ple jun = 0.5
%triple junction GaAs high efficiency with high cost, silicon is lower cost where radtiation not concern
%table 21-17 for primary batteries (short missions and long term tasks like
%battery backup)
%table 21-18 for secondary batteries (rechargeable, used during eclipse)
%page 654 provides info on PPT and direct systems

%FROM POWER AND THERMAL CONTROL SLIDES LOOK AT EXAMPLE POWER PROFILE

%% Assumptions
% 1)Peak power tracking with regulation scheme
% 2)Power required during daylight and eclipse are the same
% 3)Using GaAs (gallelium arsenide) degradation (29 - 32% efficidency), use
% lower value for conservative calculation
% 4)Using secondary battery for rechargeability
% 5)Theta = 0 deg so we assume the sun is normal to SA (change)

%% Inputs
P_d = 61; %daytime power load [W], how much power system uses
T_d = ; %time of orbit spent in daylight [s]
P_e = P_d; %eclipse power load [W]
T_e = ; %time of orbit spent in eclipse [s]
X_d = 0.80; %efficiency of getting power from SA directly to loads
X_e = 0.6; %Efficiency of getting power from SA to batteries then laods
P_i = ; %Input solar power density [W/m^2], different at each orbit
n = 0.29; %solar cell efficiency
I_d = 0.185; %degradation of solar sell for GaAs (SJ)
theta = 0; %angle btw solar cell normal and the sun, use worst case (23.5 inSMAD FiresatII)

%% Calculations

P_SA = 1 / T_d * ((P_d * T_d / X_d) + (P_e * T_e) / X_e); %Required solar array output power [W]
P_o = P_i * n; %Output solar power desnity [W/m^2]
P_bol = P_o * I_d * cos(theta); %beginning of life power output density [W/m^2]
P_eol = P_bol * L_d; %end of life power output density [W/m^2]
L_d = (1 - 0.0275) ^ 3; %lifetime degration of solar cells, 3 is mission lifetime in years
A_SA = P_SA / P_EOL; %required solar panel area [m^2]

%calcualte mass as well