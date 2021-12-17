[results, re_entry] = calc_decay(ID, M, A, H, F10, Ap);

i = 1; 
while i <= length(results(:,1))
    results(i,6) = calc_DV(results(i,2),H);
    results(i,7) = calc_mprop(results(i,6), M, ISP);
    results(i,8) = 365*3/results(i,1);
    results(i,9) = results(i,8)*results(i,7);
    i = i + 1;
end

