function Simulated=HOPIT_Simulate(Pars)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The function simulates a simple HOPIT model for validation
%     
%   Inputs                      
%                               Pars.Beta                              :                   Coefficients in outcome function
%                               Pars.Gamma                             :                   Coefficients in cut-point function
%                               Pars.sigma_rp                          :                   s.d of outcome noise
%                               Pars.sigma_cp                          :                   s.d of cut-point noise
%                               Pars.sigma_vi                          :                   s.d of vignette noise
%                               Pars.data_point                        :                   Number of data points
%                               Pars.cut_point                         :                   Number of cut points                                              
%                               Pars.vignette                          :                   Number of vignettes
%   Ouputs
%
%                               Simulated.Outcome_Indep                :                   Regressors in outcome function
%                               Simulated.Outcome_Dep                  :                   Outcomes, eg. self-ratings
%                               Simulated.Cut_Indep                    :                   Regressors in cut point function
%                               Simulated.Vignette                     :                   Vignettes ratings
%                               Simulated.Outcome_Latent               :                   Latent outcomes
%                               Simulated.Cut_Point                    :                   Simulated cut points
%                               Simulated.theta                        :                   Cut point locations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Define local variables
beta=Pars.Beta;                                                             % Coefficients in outcome function
gamma=Pars.Gamma;                                                           % Coefficients in cut-point function
sigma_rp=Pars.sigma_rp;                                                     % s.d of outcome noise
%sigma_cp=Pars.sigma_cp;                                                     % s.d of cut-point noise
sigma_vi=Pars.sigma_vi;                                                     % s.d of vignette noise
sigma_mood=Pars.sigma_mood;                                                 % s.d of missing variable

n=Pars.data_point;                                                          % Number of data points
nc=Pars.cut_point;                                                          % Number of cut points                                              
kt=Pars.vignette;                                                           % Number of vignettes


%% Generate latent outcomes and cut points
% Generate independent variables
% Generate age, uniformly distributed
%age_upper=1;
%age_lower=0;
%age=age_lower+(age_upper-age_lower).*rand(n,1);
X1=(rand(n,1)-0.5)*sqrt(12);


% Generate country of residence
%country=binornd(1,0.5,n,1);
X2=(rand(n,1)-0.5)*sqrt(12);


% Generate the mood. The mood is generated to match the moment of the real
% data. We can generate it from a normal with mean to be the intercept from
% the estimated model and with variance varying according to our
% experimental design
% Note, actually, this might be the most important point in the whole
% experiment. The mean of the potential missing variable does not matter so
% much since it can be absorbed into the intercept. But, the volality of
% the missing variable (hetergeity) cannot be well signed by the intercept.
% Hence, the larger the volality of the missing variable, the worse of the
% model fit in terms of Kandell's tau and MSE of ratios. 

% A normal distribution
% The variance should be at the same level of other covariates
% mood=100.*rand(n,1);
% mood=1+randn(n,1);
%mood=sigma_mood*rand(n,1); 
X3=sigma_mood*(rand(n,1)-0.5)*sqrt(12); 


% A log-normal distribution
% m = 1;
% v = 0.1^2;
% mu = log((m^2)/sqrt(v+m^2));
% sigma = sqrt(log(v/(m^2)+1));
% mood=lognrnd(mu,sigma,n,1);


% Generate noises, normally distributed
eps_rp=sigma_rp*randn(n,1);                                                 % Noise in outcome, with sd being sigma_rp
%eps_cp=sigma_cp*randn(n,1);                                                 % Noise in cut points, with sd being sigma_cp

% eps_cp=zeros(n,1);
% for m=1:n
%     eps_cp(m)=age(m)/100*randn(1,1);  
% end

% Generate latent outcomes H*=X*beta1+Y*beta2+eps
H_star=[X1,X2]*beta+eps_rp;

% Generate cut points
Zb=zeros(n,nc+2);                                                           % Individual-specific cut points, nX(nc+2), including -Inf and Inf
Zb(:,1)=-Inf;                                                               % The first cut-point: -Inf
Zb(:,2)=[ones(n,1),X2,X3]*gamma(:,1);                                            % The second cut-point
if nc>=2
    for i=3:nc+1
        Zb(:,i)=Zb(:,i-1)+([ones(n,1),X2,X3]*gamma(:,i-1)).^2;                          % The third, fourth,...
    end
end
Zb(:,nc+2)=Inf;                                                             % The last cut-point: Inf


% Check the validity of the simulated cut points, neither too large nor
% too small
%   [min(H_star),max(H_star),median(H_star)]
%   [min(Zb(:,2)),max(Zb(:,2)),median(Zb(:,2))]
%   [min(Zb(:,3)),max(Zb(:,3)),median(Zb(:,3))]
%   [min(Zb(:,4)),max(Zb(:,4)),median(Zb(:,4))]
%   [min(Zb(:,5)),max(Zb(:,5)),median(Zb(:,5))]

%% Generate observed responses
H=ones(n,1);
for i = 2: nc+1
    H=H+(H_star>Zb(:,i));
end

%hist(H)

%% Generate vignettes, evenly distributed
% Generate Vignettes: e.g., 20,40,60, and 80 quantile of H* when vignette=4
theta=zeros(1,kt);
for i=1:kt
    theta(i)=quantile(H_star,i/(kt+1));
end

% Let one vigentte taking location zero to garantee there is always
% some variation of vigentte values no matter how the data is generated
theta((kt+1)/2)=0;

eps_vi=sigma_vi*randn(n,kt);                                                % Noise in vignette generation, a nXkt matrix
V_star=ones(n,1)*theta+eps_vi;                                              % Distribution of vignettes


%% Generate vignette ratings
V=ones(n,kt);
for j= 1:kt
    for i = 2: nc+1
        V(:,j)=V(:,j)+(V_star(:,j)>Zb(:,i));
    end
end


%    hist(V(:,1))
%    hist(V(:,2))
%    hist(V(:,3))
%    hist(V(:,4))

%% Stack simulated data
Simulated.Outcome_Indep=[X1,X2];                                              % Regressors in outcome function
Simulated.Outcome_Dep=H;                                                    % Outcomes, eg. self-ratings
Simulated.Cut_Indep=[X2,X3];                                                  % Regressors in cut point function
Simulated.Vignette=V;                                                       % Vignettes ratings
Simulated.Outcome_Latent=H_star;                                            % Latent outcomes
Simulated.Cut_Point=Zb;                                                     % Simulated cut points
Simulated.theta=theta;                                                      % Vignette locations



% %% Export the simulated data into an ASCII file
% Output = [age,edu,gender,country,mood,H,V];
% save HOPIT_Simu_Data.txt Output -ASCII
% 
% %% Plots
% % Plot distribution of outcomes
% disp('Finished simulating hierachical ordered probit data.')
% disp('The distributions of self assessment and vignettes')
% subplot(kt+1,1,1);
% hist(H)
% title('Self Asssssment')
% 
% % Plot distibution of vignettes
% for i=1:kt
% subplot(kt+1,1,i+1);
% hist(V(:,i))
% title(['Vignette ',num2str(i)])
% end

end

