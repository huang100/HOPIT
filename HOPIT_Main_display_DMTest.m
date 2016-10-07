pars= [1,1,10000];

%% Global setup
% Setup model parameters

Pars.Beta=[10,10]';                                                     % Coefficients of outcome
Pars.Gamma=[0, 10, 10;
            0, 1, 1]';                                                  % Coefficients of cut points
%Pars.Gamma=[-10, 10, 10]';   

Pars.sigma_mood=pars(1);

Pars.sigma_rp=1;                                                        % sd of outcome noise
Pars.sigma_vi=1;                                                        % sd of vignette noise
Pars.sigma_cp=1; 

Pars.vignette=pars(2);                                                  % Number of vignettes
Pars.data_point=pars(3);                                                % Number of data points generated
Pars.cut_point=2;                                                       % Number of cut points, eg. k=4 cut points, ie k+2 bins

dis = 0;

% Setup iteration paramters
Pars.n_iter=1;                                                          % Number of simulation iterations

% Error control
if Pars.cut_point ~= size(Pars.Gamma,2)
    error('Please provide the coefficients on cut point function')
end

% Setup random seed
stream = RandStream('mt19937ar','Seed',1);                              % Specify the random generator so that our results are replicable
RandStream.setGlobalStream(stream);

                                                         % converge=1 if all the following regressions converge in an iteration
%% Simulate data
SimuData= HOPIT_Simulate_DM(Pars);                                     % Simulated data
    

    
%% Direct mapping
    
    % Read data
    Data.Outcome_Indep=SimuData.Outcome_Indep;
    Data.Outcome_Dep=SimuData.Outcome_Dep;
    
    
    % Read data
    % Define dummies for vignette ratings since it is cardinal
%     Vignette=zeros(Pars.data_point,Pars.cut_point*Pars.vignette);
%     for i=1:Pars.vignette
%         for j=1:Pars.cut_point
%             Vignette(:,(i-1)*Pars.cut_point+j)=(SimuData.Vignette(:,i)==j);
%         end
%     end
%     Data.Vignette=[ones(Pars.data_point,1),Vignette]; % It is necessary to add a constant term since Vignette only take values 1-4

%     Va=0;
%     for i=1:Pars.vignette
%         Va=Va*10+SimuData.Vignette(:,i);        
%     end
%     [~,~,Vb]=unique(Va);
%     Vc=dummyvar(Vb);
%     Data.Vignette=Vc(:,any(Vc));

     Data.Vignette = [ones(Pars.data_point,1),SimuData.Cut_Indep];
    
    
    % Setup the initial guesses of parameters
    kb=size(Data.Outcome_Indep,2);                                           % Length of beta
    kt=size(Data.Vignette,2);                                                % Length of theta (ie, number of vignettes)
    kd=kt*Pars.cut_point;                                                    % Length of delta
    
    beta=zeros(kb,1);
    deta=zeros(kd,1);
    b0=[beta;deta];
    
    
    % Use a solver of constrained optimization and setup options
    options = optimset('Algorithm', 'interior-point', 'GradObj','off','GradConstr','off', 'TolX', 1e-10, 'TolFun', 1e-4,'Hessian',...
        'bfgs','display','iter');
    
    % Setup parameter bounds
     lb=-Inf*ones(kb+kd,1);
     ub=Inf*ones(kb+kd,1);
    

    
    % Begin: estimate using all data
    % Pass extra parameters to objective function
    obj=@(x) HOPIT_Directmapping(Data,x,Pars.cut_point);
    
    % Estimate
    [b,fval_g,exitflag,output,lamda,grad,hessian] = fmincon(obj,b0,[],[],[],[],lb,ub,[],options);
    if exitflag~=1
        converge=0;
        n_fail_converge=n_fail_converge+1;
        if n_fail_converge>Pars.n_iter*2
            break
        end
        continue
    end
    lik5=-fval_g;
    AIC5=2*fval_g+2*length(b);
