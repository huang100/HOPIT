clc
clear all

%% Set up the test space of parameters
index=1;    % Row index
for b1=[1,2,4,8]
    for v=[1,3,5]
        for N=[1000,3000,5000]
            
            pars(index,:)=[b1,v,N];
            index=index+1;
            
        end
    end
end

%% Output of the test grid
save Parameters.dat pars /ascii


%% Analyze simulated results
% Load results
np=size(pars,1);
for i=1:np
    temp= eval(['load(''D:\vignette\Copy160408\HOPIT6\results\ans_',num2str(i),'.mat'')']);
    out=temp.ans;
    output(i,:)=[out.mtau,out.mrho];
end

outtable = [pars,output];
