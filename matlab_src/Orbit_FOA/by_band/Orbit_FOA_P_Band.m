
clc
clear
close

% Script Inputs
size = 3000;
theta_Lat = 70;
low_alt = 100;
high_alt = 40000;
min_inc = 40;

% Calculate constraints on upper latitudes, inc, and alt
theta_1_lim = 60;
theta_2_lim = 60; 
theta_3_lim = 180 - theta_1_lim;
r_earth = 6378;

results = zeros(size,size);
inc_transition = zeros(1,size);

i = 1;
for inc = linspace(min_inc,90,size)
    j = 1;
    check_var = 1;
    for r_r_mag = linspace(r_earth+low_alt,r_earth+high_alt,size)
        r_r = [r_r_mag*cosd(inc) r_r_mag*sind(inc)];
        r_s = [r_earth*cosd(theta_Lat) r_earth*sind(theta_Lat)];
        r_sr = r_r - r_s;
        r_sr_mag = norm(r_sr);
        theta2 = asind(r_earth*sind(theta_Lat - inc)/r_sr_mag);
        theta3 = asind(r_r_mag*sind(theta_Lat - inc)/r_sr_mag);
        if((theta2 + theta3 + theta_Lat - inc) ~= 180)
            theta3 = 180 - theta3;
        end
        if(theta2 <= theta_2_lim && theta3 >= theta_3_lim)
            results(j,i) = 1;
            if check_var == 1
                inc_transition(i) = r_r_mag - r_earth;
                check_var = 0;
            end
        end
        j = j + 1;
    end
    i = i + 1;
end

format shortg
inc = round(linspace(min_inc,90,size),1);
alt = round(linspace(low_alt,high_alt,size),1);

%set inclination range for receiver
i_r = min_inc:0.1:90;
%antenna incidence
t_2 = theta_2_lim;

% MUOS, ORBCOMM, SWARM
const_inc = [5 45 87.5];
const_r = [35786 750 500];
r_rec = zeros(length(const_inc),length(i_r));

for i = 1:length(const_inc)
    theta4 = t_2 - i_r + const_inc(i);
    r_rec(i,:) = const_r(i)./sind(180-t_2)*sind(theta4);
end

figure
plot(i_r,r_rec(1,:),'b','LineWidth',1.2)
hold on
plot(inc,inc_transition,'r','LineWidth',1.2)
title('GEO Constellations for P/I-Band Requirements')
subtitle('Must be ABOVE red curve and BELOW constellation curves')
xlabel('Inclination [degs]')
ylabel('Altitude [km]')
legend('MUOS (P) Upper Lim','Satellite Lower Lim','Location','Southwest')
set(gca,'FontSize',12)
set(gca,'FontWeight','bold')
ylim([-5000 15000])
grid on

figure
plot(i_r,r_rec(2,:),'b',i_r,r_rec(3,:),'g','LineWidth',1.2)
hold on
plot(inc,inc_transition,'r','LineWidth',1.2)
title('LEO Constellations for P/I-Band Requirements')
subtitle('Must be ABOVE red curve and BELOW constellation curves')
xlabel('Inclination [degs]')
ylabel('Altitude [km]')
legend('ORBCOMM (I) Upper Lim','SWARM (I) Lower Lim',...
    'Satellite Lower Lim','Location','Southwest')
set(gca,'FontSize',12)
set(gca,'FontWeight','bold')
ylim([-500 2000])
grid on




