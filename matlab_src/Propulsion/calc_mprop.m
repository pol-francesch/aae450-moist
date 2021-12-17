function mprop = calc_mprop(DV, M, ISP)
%% INPUTS
g = 9.80665;

%% CALCULATIONS
mprop = M * (exp(DV/(g*ISP))-1);


end