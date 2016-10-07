function tau = HOPIT_Ktau(data_point,pred_out,simu_out )
% This function calculates the Kendall's tau
% The formular is tau=(number of concordant pairs-number of discordant
% pairs)/(0.5*n*(n-1))

m = 0;
for i=2:data_point
    for j=1:(i-1)
        m = m + sign(pred_out(i) - pred_out(j)) * sign(simu_out(i) - simu_out(j));
    end
end
tau=m/(0.5*data_point*(data_point-1));

end

