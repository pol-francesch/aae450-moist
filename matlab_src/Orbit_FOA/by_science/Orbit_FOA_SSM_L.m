
clc
clear
close

% Orbit Design
orbit_alt_1 = 350;
orbit_inc_1 = 80;
orbit_alt_2 = 550;
orbit_inc_2 = 63.5;
attitude = 0;
theta_Lat = 70;

% Script Inputs
size = 5000;
low_alt = 100;
high_alt = 40000;
min_inc = 40;
low_plot_alt = -100;
high_plot_alt = 4000;

% Calculate constraints on upper latitudes, inc, and alt
theta_1_lim = 60;
theta_2_spec_lim = 21 + attitude;
theta_2_direct_lim = 60;
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
        if(theta2 <= theta_2_spec_lim && theta3 >= theta_3_lim)
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
t_2 = theta_2_direct_lim;

% Iridium, GPS, Galileo, GLONASS
const_inc = [86.4 55 56 64.8];
const_r = [780 20200 23222 25508];
r_rec = zeros(length(const_inc),length(i_r));

for i = 1:length(const_inc)
    theta4 = t_2 - i_r + const_inc(i);
    r_rec(i,:) = const_r(i)./sind(180-t_2)*sind(theta4);
end

figure
plot(i_r,r_rec(1,:),'b','LineWidth',1.2)
hold on
plot(i_r,r_rec(2,:),'g','LineWidth',1.2)
plot(i_r,r_rec(3,:),'c','LineWidth',1.2)
plot(i_r,r_rec(4,:),'m','LineWidth',1.2)
plot(orbit_inc_1,orbit_alt_1,'ok','MarkerSize',10,'LineWidth',1.5)
plot(orbit_inc_2,orbit_alt_2,'sk','MarkerSize',10,'LineWidth',1.5)
plot(inc,inc_transition,'r','LineWidth',1.2)
title('Orbit Tradespace for SSM (L-Band)')
subtitle('Assumes Nadir-Pointing and Visibility up to 70 degs Latitude')
xlabel('Inclination [degs]')
ylabel('Altitude [km]')
legend('Iridium(L) Upper Lim','GPS(L) Upper Lim','Galileo(L) Upper Lim',...
    'GLONASS(L) Upper Lim',...
    'MoIST Orbit 1 (alt = 350 km, inc = 80 degs)',...
    'MoIST Orbit 2 (alt = 550 km, inc = 63.5 degs)',...
    'Lat Vis Lower Lim','Location','NorthWest')
set(gca,'FontSize',12)
set(gca,'FontWeight','bold')
set(gcf,'position',[100 100 1000 700])
ylim([low_plot_alt high_plot_alt])
grid on



