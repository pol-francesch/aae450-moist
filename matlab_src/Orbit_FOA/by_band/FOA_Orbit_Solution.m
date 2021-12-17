
clc
clear
close

% Script Inputs
size = 5000;
theta_Lat_1 = 80;
theta_Lat_2 = 70;
low_alt = 100;
high_alt = 40000;
min_inc = 40;

% Orbit Planes
orbit_2_inc = 80; 
orbit_2_alt = 400;
orbit_1_inc = 80;
orbit_1_alt = 400;

% Calculate constraints on upper latitudes, inc, and alt
theta_1_lim = 60;
theta_2_lim_P = 60;
theta_2_lim_L = 41;
theta_3_lim = 180 - theta_1_lim;
r_earth = 6378;

results_1_P = zeros(size,size);
inc_transition_1_P = zeros(1,size);
results_2_P = zeros(size,size);
inc_transition_2_P = zeros(1,size);
results_1_L = zeros(size,size);
inc_transition_1_L = zeros(1,size);
results_2_L = zeros(size,size);
inc_transition_2_L = zeros(1,size);

% Iterate through first theta_lat lim and find P- and L-band band limits
i = 1;
for inc = linspace(min_inc,90,size)
    j = 1;
    check_var_1 = 1;
    check_var_2 = 1;
    for r_r_mag = linspace(r_earth+low_alt,r_earth+high_alt,size)
        r_r = [r_r_mag*cosd(inc) r_r_mag*sind(inc)];
        r_s = [r_earth*cosd(theta_Lat_1) r_earth*sind(theta_Lat_1)];
        r_sr = r_r - r_s;
        r_sr_mag = norm(r_sr);
        theta2 = asind(r_earth*sind(theta_Lat_1 - inc)/r_sr_mag);
        theta3 = asind(r_r_mag*sind(theta_Lat_1 - inc)/r_sr_mag);
        if((theta2 + theta3 + theta_Lat_1 - inc) ~= 180)
            theta3 = 180 - theta3;
        end
        if(theta2 <= theta_2_lim_P && theta3 >= theta_3_lim)
            results_1_P(j,i) = 1;
            if check_var_1 == 1
                inc_transition_1_P(i) = r_r_mag - r_earth;
                check_var_1 = 0;
            end
        end
        if(theta2 <= theta_2_lim_L && theta3 >= theta_3_lim)
            results_1_L(j,i) = 1;
            if check_var_2 == 1
                inc_transition_1_L(i) = r_r_mag - r_earth;
                check_var_2 = 0;
            end
        end
        j = j + 1;
    end
    i = i + 1;
end

% Iterate through second theta_lat lim and find P- and L-band band limits
i = 1;
for inc = linspace(min_inc,90,size)
    j = 1;
    check_var_1 = 1;
    check_var_2 = 1;
    for r_r_mag = linspace(r_earth+low_alt,r_earth+high_alt,size)
        r_r = [r_r_mag*cosd(inc) r_r_mag*sind(inc)];
        r_s = [r_earth*cosd(theta_Lat_2) r_earth*sind(theta_Lat_2)];
        r_sr = r_r - r_s;
        r_sr_mag = norm(r_sr);
        theta2 = asind(r_earth*sind(theta_Lat_2 - inc)/r_sr_mag);
        theta3 = asind(r_r_mag*sind(theta_Lat_2 - inc)/r_sr_mag);
        if((theta2 + theta3 + theta_Lat_2 - inc) ~= 180)
            theta3 = 180 - theta3;
        end
        if(theta2 <= theta_2_lim_P && theta3 >= theta_3_lim)
            results_2_P(j,i) = 1;
            if check_var_1 == 1
                inc_transition_2_P(i) = r_r_mag - r_earth;
                check_var_1 = 0;
            end
        end
        if(theta2 <= theta_2_lim_L && theta3 >= theta_3_lim)
            results_2_L(j,i) = 1;
            if check_var_2 == 1
                inc_transition_2_L(i) = r_r_mag - r_earth;
                check_var_2 = 0;
            end
        end
        j = j + 1;
    end
    i = i + 1;
end

format shortg
inc = round(linspace(min_inc,90,size),1);
alt = round(linspace(low_alt,high_alt,size),1);
i_r = min_inc:0.1:90;

% ORBCOMM, SWARM, Iridium, MUOS, GPS, Galileo, GLONASS
const_inc = [45 87.5 86.4 5 55 56 64.8];
const_r = [750 500 780 35786 20200 23222 25508];
const_band = ['I' 'I' 'L' 'P' 'L' 'L' 'L'];
r_rec = zeros(length(const_inc),length(i_r));

for i = 1:length(const_inc)
    if (const_band(i)=='P')||(const_band(i)=='I')
        t_2 = theta_2_lim_P + 20;
    else
        t_2 = theta_2_lim_L + 20;
    end
    theta4 = t_2 - i_r + const_inc(i);
    r_rec(i,:) = const_r(i)./sind(180-t_2)*sind(theta4);
end

% Orbit Plane 2
figure
plot(i_r,r_rec(4,:),'b',i_r,r_rec(5,:),'g',...
    i_r,r_rec(6,:),'c',i_r,r_rec(7,:),'m',...
    'LineWidth',1.2)
hold on
plot(inc,inc_transition_2_P,'--r','LineWidth',1.2)
plot(inc,inc_transition_2_L,':r','LineWidth',1.2)
plot(orbit_2_inc,orbit_2_alt,'*k','LineWidth',1.5)
title('Orbital Plane 2 (inc = 60 deg, alt = 3,000 km)')
subtitle('Visibility up to 70 degs Latitude')
xlabel('Inclination [degs]')
ylabel('Altitude [km]')
legend('MUOS(P) Upper Lim (+10-deg)','GPS(L) Upper Lim','Galileo(L) Upper Lim',...
    'GLONASS(L) Upper Lim','P/I-Band Lower Lim',...
    'L-Band Lower Lim','Sat Orbit Plane 2','Location','Southwest')
set(gca,'FontSize',12)
set(gca,'FontWeight','bold')
ylim([-5000 15000])
grid on

% Orbit Plane 1
figure
plot(i_r,r_rec(1,:),'b',i_r,r_rec(2,:),'g',...
    i_r,r_rec(3,:),'c','LineWidth',1.2)
hold on
plot(inc,inc_transition_1_P,'--r','LineWidth',1.2)
plot(inc,inc_transition_1_L,':r','LineWidth',1.2)
plot(orbit_1_inc,orbit_1_alt,'*k','LineWidth',1.5)
title('Orbital Plane 1 (inc = 80 deg, alt = 300 km)')
subtitle('Visibility up to 80 degs Latitude')
xlabel('Inclination [degs]')
ylabel('Altitude [km]')
legend('ORBCOMM(I) Upper Lim','SWARM(I) Upper Lim',...
    'Iridium(L) Upper Lim','P/I-Band Lower Lim',...
    'L-Band Lower Lim','Sat Orbit Plane 1','Location','Southwest')
set(gca,'FontSize',12)
set(gca,'FontWeight','bold')
ylim([-500 2000])
grid on
