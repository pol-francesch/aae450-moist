
clc
clear
close

% Script Inputs
size = 10000;
theta_Lat = 80;
low_alt = 100;
high_alt = 80000;
min_inc = 24;

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
        if(theta2 <= 60 && theta3 >= 120)
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

%{
figure
h = heatmap(inc,alt,results);
title('Visibility of High Latitude Spectral Points for Satellite Altitude and Inclination')
xlabel('Satellite Inclination [degs]')
ylabel('Satellite Altitude [km]')
h.YDisplayData = flipud(h.YDisplayData);
set(gca,'FontSize',12)
%}

%theda_lat vs. receiv. inc
%const.
r_t = 750;
Re = 6378;

%theda 1 and 2
t_1 = 60;
t_2 = 21;

%i_r alwasy greater than i_t
i_t = 45;
i_r = i_t:0.1:90;

%gov eqns.
r_rt = r_t./sind(180 - t_2).*sind(i_r-i_t);
r_st = r_rt./sind(2*t_1).*sind(180 - 2*t_2);
t_lat = asind(r_st./r_t*sind(180-t_1))+i_t;

%set inclinations for receiver
i_r1 = 0:0.1:90;

%GPS inc, all GNSS sats had similar SMAs and inc so I just used GPS here
i_t_GPS = 55;
r_t_GPS = 20200;
%MUOS - 5 deg inc, in GEO
i_t_MUOS = 5;
r_t_MUOS = 35786;
%Iridium, 86.4 inc
i_t_Irid = 86.4;
r_t_Irid = 780;
%ORBCOMM, 45 inc
i_t_ORB = 45;
r_t_ORB = 750;
%SWARM
i_t_SWARM = 87.5;
r_t_SWARM = 500;

%antenna incidence (set as 60 for now for all cases)
t_2 = 60;

%alt. of receiver vs. inc. of receiv.
%GPS
t_4GPS = t_2 - i_r1 + i_t_GPS;
r_rGPS = r_t_GPS./sind(180-t_2)*sind(t_4GPS);
%MUOS
t_4MUOS = t_2 - i_r1 + i_t_MUOS;
r_rMUOS = r_t_MUOS./sind(180-t_2)*sind(t_4MUOS);
%Iridium
t_4Irid= t_2 - i_r1 + i_t_Irid;
r_rIrid = r_t_Irid./sind(180-t_2)*sind(t_4Irid);
%ORBCOMM
t_4ORB = t_2 - i_r1 + i_t_ORB;
r_rORB = r_t_ORB./sind(180-t_2)*sind(t_4ORB);
%SWARM
t_4SWARM = t_2 - i_r1 + i_t_SWARM;
r_rSWARM = r_t_SWARM./sind(180-t_2)*sind(t_4SWARM);


%plot for GPS, note: Earth radius is subtracted to make this altitude
figure
plot(i_r1,r_rGPS,'b',i_r1,r_rMUOS,'g','LineWidth',1.2)
hold on
plot(inc,inc_transition,'r','LineWidth',1.2)
title('Visibility of High Latitude Spectral Points for Satellite Altitude and Inclination')
subtitle('Must be ABOVE red curve and BELOW blue and green curves to meet all constraints')
xlabel('Inclination [degs]')
ylabel('Altitude [km]')
legend('GPS Lower Lim','MUOS Lower Lim','Satellite Upper Lim')
set(gca,'FontSize',12)
set(gca,'FontWeight','bold')
ylim([-5000 40000])
grid on

figure
plot(i_r1,r_rORB,'b',i_r1,r_rIrid,'g',i_r1,r_rSWARM,'c','LineWidth',1.2)
hold on
plot(inc,inc_transition,'r','LineWidth',1.2)
title('Visibility of High Latitude Spectral Points for Satellite Altitude and Inclination')
subtitle('Must be ABOVE red curve and BELOW blue and green curves to meet all constraints')
xlabel('Inclination [degs]')
ylabel('Altitude [km]')
legend('ORBCOMM Lower Lim','Iridium Lower Lim','SWARM Lower Lim','Satellite Upper Lim')
set(gca,'FontSize',12)
set(gca,'FontWeight','bold')
ylim([-500 2000])
grid on




