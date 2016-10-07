pars= [2,5,3000];

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
Pars.n_iter=50;                                                         % Number of simulation iterations

% Error control
if Pars.cut_point ~= size(Pars.Gamma,2)
    error('Please provide the coefficients on cut point function')
end

% Setup random seed
stream = RandStream('mt19937ar','Seed',1);                              % Specify the random generator so that our results are replicable
RandStream.setGlobalStream(stream);


%% We simulate, estimate and test the HOPIT model
n_converge=0;                                                           % Number of convergent iterations
n_fail_converge=0;                                                      % Number of fails of convergence
%% Simulation, estimation, comparison

while n_converge<Pars.n_iter
    
    converge=1;                                                         % converge=1 if all the following regressions converge in an iteration
    %% Simulate data
    SimuData= HOPIT_Simulate(Pars);                                     % Simulated data
    
    
    %% Estimation WHEN the model is well-specified
    % Read data
    Data.Outcome_Indep=SimuData.Outcome_Indep;
    Data.Outcome_Dep=SimuData.Outcome_Dep;
    Data.Cut_Indep=[ones(Pars.data_point,1),SimuData.Cut_Indep];
    Data.Vignette=SimuData.Vignette;
    
    
    % Setup the initial guesses of parameters
    kb=size(Data.Outcome_Indep,2);                                      % Length of beta
    kz=size(Data.Cut_Indep,2);                                          % Number of variables in cut-point functions
    kg=kz*Pars.cut_point;                                               % Length of gamma
    kt=size(Data.Vignette,2);                                           % Length of theta (ie, number of vignettes)
    
    beta=zeros(kb,1);
    gamma=zeros(kg,1);
    theta=zeros(kt,1);
    sigma_vi=1;
    sigma_cp=1;                                                         % Careful specification on the initial value of variance to guarantee convergence of algorithum
    b0=[beta;gamma;theta;sigma_vi];
    
    % Use a solver of constrained optimization and setup options
    options = optimset('Algorithm', 'interior-point', 'GradObj','on','GradConstr','off', 'TolX', 1e-10, 'TolFun', 1e-4,'Hessian',...
        'bfgs','display','iter');
    
    
    % Setup parameter bounds
    lb=[-Inf*ones(kb+kg+kt,1);0];
    ub=[Inf*ones(kb+kg+kt,1);Inf];
    
    % Begin: estimate using all data
    % Pass extra parameters to objective function
    obj=@(x) HOPIT_Likelihood_King(Data,x,Pars.cut_point);
    
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
    
    lik1=-fval_g;
    AIC1=2*fval_g+2*length(b);
    %     PE1=HOPIT_APT(SimuData,b,Pars.cut_point);
    %
    %
    %     % Eye check of convergence
    %     b;
    %
    %     % Calculate the ratio of coefficients
    %     ratio1=b(2:kb)./b(1);
    %
    %     % Calculate the correlation and Kendall's tau
    pred_out1=Data.Outcome_Indep*b(1:kb);
    rho1= corr(pred_out1,SimuData.Outcome_Latent);
    tau1=HOPIT_Ktau(Pars.data_point,pred_out1,SimuData.Outcome_Latent);
    %
    pred_cut11=Data.Cut_Indep*b(kb+1:kb+kz);
    pred_cut12=pred_cut11+exp(Data.Cut_Indep*b(kb+kz+1:kb+2*kz));
    %     pred_cut13=pred_cut12+exp(Data.Cut_Indep*b(kb+2*kz+1:kb+3*kz));
    %     pred_cut14=pred_cut13+exp(Data.Cut_Indep*b(kb+3*kz+1:kb+4*kz));
    
    % Calculate Average Partial Effect
    %b0=[Pars.Beta',reshape(Pars.Gamma,[1,kg])];
    %APT0=HOPIT_APT(Data,b0,Pars.cut_point)
    %APT1=HOPIT_APT(Data,b,Pars.cut_point)
    
    % Clear memory
    b1=b;
    clear Data
    
    %% Estimation of the misspecified model
    % Omitted variable in cut point function: mood
    
    % Read data
    Data.Outcome_Indep=SimuData.Outcome_Indep;
    Data.Outcome_Dep=SimuData.Outcome_Dep;
    Data.Cut_Indep=[ones(Pars.data_point,1),SimuData.Cut_Indep(:,1)];   % Note, Z is unavailable to the modeller. A constant is added.
    Data.Vignette=SimuData.Vignette;
    
    
    % Setup the initial guesses of parameters
    kb=size(Data.Outcome_Indep,2);                                      % Length of beta
    kz=size(Data.Cut_Indep,2);                                          % Number of variables in cut-point functions
    kg=kz*Pars.cut_point;                                               % Length of gamma
    kt=size(Data.Vignette,2);                                           % Length of theta (ie, number of vignettes)
    
    beta=zeros(kb,1);
    gamma=zeros(kg,1);
    theta=zeros(kt,1);
    sigma_vi=1;
    sigma_cp=1;                                                         % Careful specification on the initial value of variance to guarantee convergence of algorithum
    b0=[beta;gamma;theta;sigma_vi];
    
    % Use a solver of constrained optimization and setup options
    options = optimset('Algorithm', 'interior-point', 'GradObj','on','GradConstr','off', 'TolX', 1e-10, 'TolFun', 1e-4,'Hessian',...
        'bfgs','display','iter');
    
    
    % Setup parameter bounds
    lb=[-Inf*ones(kb+kg+kt,1);0];
    ub=[Inf*ones(kb+kg+kt,1);Inf];
    
    % Begin: estimate using all data
    % Pass extra parameters to objective function
    obj=@(x) HOPIT_Likelihood_King(Data,x,Pars.cut_point);
    
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
    
    lik2=-fval_g;
    AIC2=2*fval_g+2*length(b);
    %     PE2=HOPIT_APT(SimuData,b,Pars.cut_point);
    %     % Eye check of convergence
    %     b;
    %
    %
    %     % Calculate the ratio of coefficients
    %     ratio2=b(2:kb)./b(1);
    %
    %     % Calculate the correlation and Kendall's tau
    pred_out2=Data.Outcome_Indep*b(1:kb);
    rho2= corr(pred_out2,SimuData.Outcome_Latent);
    tau2=HOPIT_Ktau(Pars.data_point,pred_out2,SimuData.Outcome_Latent);
    %
    pred_cut21=Data.Cut_Indep*b(kb+1:kb+kz);
    pred_cut22=pred_cut21+exp(Data.Cut_Indep*b(kb+kz+1:kb+2*kz));
    %     pred_cut23=pred_cut22+exp(Data.Cut_Indep*b(kb+2*kz+1:kb+3*kz));
    %     pred_cut24=pred_cut23+exp(Data.Cut_Indep*b(kb+3*kz+1:kb+4*kz));
    
    %APT2=HOPIT_APT(Data,b,Pars.cut_point)
    % Clear memory
    b2=b;
    clear Data
    
    
    %% Estimation of the misspecified model
    % Redundant variable in cut point function: age
    
    % Read data
    Data.Outcome_Indep=SimuData.Outcome_Indep;
    Data.Outcome_Dep=SimuData.Outcome_Dep;
    Data.Cut_Indep=[ones(Pars.data_point,1),SimuData.Cut_Indep(:,2)];   % Note, Z is unavailable to the modeller. A constant is added.
    Data.Vignette=SimuData.Vignette;
    
    
    % Setup the initial guesses of parameters
    kb=size(Data.Outcome_Indep,2);                                      % Length of beta
    kz=size(Data.Cut_Indep,2);                                          % Number of variables in cut-point functions
    kg=kz*Pars.cut_point;                                               % Length of gamma
    kt=size(Data.Vignette,2);                                           % Length of theta (ie, number of vignettes)
    
    beta=zeros(kb,1);
    gamma=zeros(kg,1);
    theta=zeros(kt,1);
    sigma_vi=1;
    sigma_cp=1;                                                         % Careful specification on the initial value of variance to guarantee convergence of algorithum
    b0=[beta;gamma;theta;sigma_vi];
    
    % Use a solver of constrained optimization and setup options
    options = optimset('Algorithm', 'interior-point', 'GradObj','on','GradConstr','off', 'TolX', 1e-10, 'TolFun', 1e-4,'Hessian',...
        'bfgs','display','iter');
    
    
    % Setup parameter bounds
    lb=[-Inf*ones(kb+kg+kt,1);0];
    ub=[Inf*ones(kb+kg+kt,1);Inf];
    
    % Begin: estimate using all data
    % Pass extra parameters to objective function
    obj=@(x) HOPIT_Likelihood_King(Data,x,Pars.cut_point);
    
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
    
    lik3=-fval_g;
    AIC3=2*fval_g+2*length(b);
    %     PE3=HOPIT_APT(SimuData,b,Pars.cut_point);
    %     % Eye check of convergence
    %     b;
    %
    %
    %     % Calculate the ratio of coefficients
    %     ratio3=b(2:kb)./b(1);
    %
    %     % Calculate the correlation and Kendall's tau
    pred_out3=Data.Outcome_Indep*b(1:kb);
    rho3= corr(pred_out3,SimuData.Outcome_Latent);
    tau3=HOPIT_Ktau(Pars.data_point,pred_out3,SimuData.Outcome_Latent);
    %
    pred_cut31=Data.Cut_Indep*b(kb+1:kb+kz);
    pred_cut32=pred_cut31+exp(Data.Cut_Indep*b(kb+kz+1:kb+2*kz));
    %     pred_cut23=pred_cut22+exp(Data.Cut_Indep*b(kb+2*kz+1:kb+3*kz));
    %     pred_cut24=pred_cut23+exp(Data.Cut_Indep*b(kb+3*kz+1:kb+4*kz));
    
    %APT3=HOPIT_APT(Data,b,Pars.cut_point)
    % Clear memory
    b3=b;
    clear Data
    
    
        
    %% Estimation of the misspecified model
    % Redundant variable in cut point function: age
    
    % Read data
    Data.Outcome_Indep=SimuData.Outcome_Indep;
    Data.Outcome_Dep=SimuData.Outcome_Dep;
    Data.Cut_Indep=[ones(Pars.data_point,1),SimuData.Cut_Indep,SimuData.Outcome_Indep(:,1)];   % Note, Z is unavailable to the modeller. A constant is added.
    Data.Vignette=SimuData.Vignette;
    
    
    % Setup the initial guesses of parameters
    kb=size(Data.Outcome_Indep,2);                                      % Length of beta
    kz=size(Data.Cut_Indep,2);                                          % Number of variables in cut-point functions
    kg=kz*Pars.cut_point;                                               % Length of gamma
    kt=size(Data.Vignette,2);                                           % Length of theta (ie, number of vignettes)
    
    beta=zeros(kb,1);
    gamma=zeros(kg,1);
    theta=zeros(kt,1);
    sigma_vi=1;
    sigma_cp=1;                                                       % Careful specification on the initial value of variance to guarantee convergence of algorithum
    b0=[beta;gamma;theta;sigma_vi];
    
    % Use a solver of constrained optimization and setup options
    options = optimset('Algorithm', 'interior-point', 'GradObj','on','GradConstr','off', 'TolX', 1e-10, 'TolFun', 1e-4,'Hessian',...
        'bfgs','display','iter');
    
    
    % Setup parameter bounds
    lb=[-Inf*ones(kb+kg+kt,1);0];
    ub=[Inf*ones(kb+kg+kt,1);Inf];
    
    % Begin: estimate using all data
    % Pass extra parameters to objective function
    obj=@(x) HOPIT_Likelihood_King(Data,x,Pars.cut_point);
    
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
    
    lik4=-fval_g;
    AIC4=2*fval_g+2*length(b);
    %     PE3=HOPIT_APT(SimuData,b,Pars.cut_point);
    %     % Eye check of convergence
    %     b;
    %
    %
    %     % Calculate the ratio of coefficients
    %     ratio3=b(2:kb)./b(1);
    %
    %     % Calculate the correlation and Kendall's tau
    pred_out4=Data.Outcome_Indep*b(1:kb);
    rho4= corr(pred_out4,SimuData.Outcome_Latent);
    tau4=HOPIT_Ktau(Pars.data_point,pred_out4,SimuData.Outcome_Latent);
    %
    pred_cut41=Data.Cut_Indep*b(kb+1:kb+kz);
    pred_cut42=pred_cut41+exp(Data.Cut_Indep*b(kb+kz+1:kb+2*kz));
    %     pred_cut23=pred_cut22+exp(Data.Cut_Indep*b(kb+2*kz+1:kb+3*kz));
    %     pred_cut24=pred_cut23+exp(Data.Cut_Indep*b(kb+3*kz+1:kb+4*kz));
    
    %APT3=HOPIT_APT(Data,b,Pars.cut_point)
    % Clear memory
    b4=b;
    clear Data
    
    %% Estimation of the misspecified model
    % Ordered probit
    
    % Read data
    Data.Outcome_Indep=SimuData.Outcome_Indep;
    Data.Outcome_Dep=SimuData.Outcome_Dep;
    Data.Cut_Indep=ones(Pars.data_point,1);
    
    % Setup the initial guesses of parameters
    kb=size(Data.Outcome_Indep,2);                                      % Length of beta
    kz=size(Data.Cut_Indep,2);                                          % Number of variables in cut-point functions
    kg=kz*Pars.cut_point;                                               % Length of gamma
    
    beta=zeros(kb,1);
    gamma=ones(kg,1);
                                                                        % Careful specification on the initial value of variance to guarantee convergence of algorithum
    b0=[beta;gamma];
    
    % Use a solver of constrained optimization and setup options
    options = optimset('Algorithm', 'interior-point', 'GradObj','on','GradConstr','off', 'TolX', 1e-10, 'TolFun', 1e-4,'Hessian',...
        'bfgs','display','iter');
    
    
    % Setup parameter bounds
    lb=[-Inf*ones(kb+kg,1)];
    ub=[Inf*ones(kb+kg,1)];
    
    
    % Begin: estimate using all data
    % Pass extra parameters to objective function
    obj=@(x) HOPIT_Likelihood_OP(Data,x,Pars.cut_point);
    
    
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
    %     PE4=HOPIT_APT(SimuData,b,Pars.cut_point);
    %     Eye check of convergence
    %     b;
    %
    %
    %     Calculate the ratio of coefficients
    %     ratio4=b(2:kb)./b(1);
    %
    %     Calculate the correlation and Kendall's tau
    pred_out5=Data.Outcome_Indep*b(1:kb);
    rho5= corr(pred_out5,SimuData.Outcome_Latent);
    tau5=HOPIT_Ktau(Pars.data_point,pred_out5,SimuData.Outcome_Latent);
    %
    pred_cut51=Data.Cut_Indep*b(kb+1:kb+kz);
    pred_cut52=pred_cut51+exp(Data.Cut_Indep*b(kb+kz+1:kb+2*kz));
    %     pred_cut23=pred_cut22+exp(Data.Cut_Indep*b(kb+2*kz+1:kb+3*kz));
    %     pred_cut24=pred_cut23+exp(Data.Cut_Indep*b(kb+3*kz+1:kb+4*kz));
    
    %APT4=HOPIT_APT(Data,b,Pars.cut_point)
    % Clear memory
    b5=b;
    %clear Data
    
    
    
%% Direct mapping
    
    % Read data
    Data.Outcome_Indep=SimuData.Outcome_Indep;
    Data.Outcome_Dep=SimuData.Outcome_Dep;
    
    
    % Read data
    % Define dummies for vignette ratings since it is cardinal
    Vignette=zeros(Pars.data_point,Pars.cut_point*Pars.vignette);
    for i=1:Pars.vignette
        for j=1:Pars.cut_point
            Vignette(:,(i-1)*Pars.cut_point+j)=(SimuData.Vignette(:,i)==j);
        end
    end
    Data.Vignette=[ones(Pars.data_point,1),Vignette]; % It is necessary to add a constant term since Vignette only take values 1-4

%     Va=0;
%     for i=1:Pars.vignette
%         Va=Va*10+SimuData.Vignette(:,i);        
%     end
%     [~,~,Vb]=unique(Va);
%     Vc=dummyvar(Vb);
%     Data.Vignette=Vc(:,any(Vc));

%      Data.Vignette = [ones(Pars.data_point,1),SimuData.Vignette];
    
    
    % Setup the initial guesses of parameters
    kb=size(Data.Outcome_Indep,2);                                           % Length of beta
    kt=size(Data.Vignette,2);                                                % Length of theta (ie, number of vignettes)
    kd=kt*Pars.cut_point;                                                    % Length of delta
    
    beta=zeros(kb,1);
    deta=zeros(kd,1);
    b0=[beta;deta];
    
    
    % Use a solver of constrained optimization and setup options
    options = optimset('Algorithm', 'interior-point', 'GradObj','on','GradConstr','off', 'TolX', 1e-10, 'TolFun', 1e-4,'Hessian',...
        'bfgs','display','iter');
    
    % Setup parameter bounds
    lb=-Inf*ones(kb+kd,1);
    ub=Inf*ones(kb+kd,1);
    
%     lb=[-Inf*ones(kb+kd,1)];
%     ub=[Inf*ones(kb,1);Inf;0;0;0;Inf;0;0;0];
    
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
    lik6=-fval_g;
    AIC6=2*fval_g+2*length(b);
    %     PE5=HOPIT_APT(SimuData,b,Pars.cut_point);
    % End: estimate using all data
    
    %     % Eye check of convergence
    %     b
    %
    %     % Calculate the ratio of coefficients
    %     ratio5=b(2:kb)./b(1);
    %
    %     % Calculate the correlation and Kendall's tau
    pred_out6=Data.Outcome_Indep*b(1:kb);
    rho6= corr(pred_out6,SimuData.Outcome_Latent);
    tau6=HOPIT_Ktau(Pars.data_point,pred_out6,SimuData.Outcome_Latent);
    %
    pred_cut61=Data.Vignette*b(kb+1:kb+kt);
    pred_cut62=pred_cut61+exp(Data.Vignette*b(kb+kt+1:kb+2*kt));
    
    %APT5=HOPIT_APT_DM(Data,b,Pars.cut_point)
    % Clear memory
    b6=b;
    %clear Data
    
    
    %% Cross-model comparison
    if converge==1                                                              % converge==1 if all above models converge
        n_converge=n_converge+1;                                                % Convergent iterations
        AIC(n_converge,:)=[AIC1 AIC2 AIC3 AIC6 AIC4 AIC5];
        coef(:,:,n_converge)=[b1(1:2) b2(1:2) b3(1:2) b4(1:2) b5(1:2) b6(1:2)];
        lik(n_converge,:)=[lik1 lik2 lik3 lik4 lik5 lik6];
        rho(n_converge,:)=[rho1 rho2 rho3 rho4 rho5 rho6];
        tau(n_converge,:)=[tau1 tau2 tau3 tau4 tau5 tau6];
    end
    
    n_converge;
    
end

%% Indictors on model comparison
if n_converge<Pars.n_iter
    summary=NaN;
    return
end

format short
summary.mAIC=median(AIC,1);                      % Median of AICs
summary.mrho=median(rho,1);
summary.mtau=median(tau,1);
summary.mlik=median(lik,1);
summary.mcoef=median(coef,3);
summary.meanAIC=mean(AIC,1);                      % Median of AICs
summary.meanrho=mean(rho,1);
summary.meantau=mean(tau,1);
summary.meanlik=mean(lik,1);
summary.meancoef=mean(coef,3);

% Tratio=repmat((Pars.Beta(2:kb)/Pars.Beta(1)),[1,6,Pars.n_iter]);
% summary.MdRAE=median(abs((ratio-Tratio)./Tratio),3);         % Median Relative Absolute Error
% summary.MeandRAE=mean(abs((ratio-Tratio)./Tratio),3);

% RMSE=sqrt(median((ratio-Tratio).^2,3))
summary.pars=pars;

summary.AIC=AIC;                      % Median of AICs
summary.rho=rho;
summary.tau=tau;
summary.lik=lik;
summary.coef=coef;


%% Plot model fits
m=(Pars.cut_point+1);
subplot(6,m,1);
scatter(SimuData.Outcome_Latent,pred_out1)
title('Fitted outcome')
text(-80, 0.5,'M1');
subplot(6,m,2);
scatter(SimuData.Cut_Point(:,2),pred_cut11)
title('Fitted cut point-1')
subplot(6,m,3);
scatter(SimuData.Cut_Point(:,3),pred_cut12)
title('Fitted cut point-2')

subplot(6,m,4);
scatter(SimuData.Outcome_Latent,pred_out2)
text(-80, 0.5,'M2');
subplot(6,m,5);
scatter(SimuData.Cut_Point(:,2),pred_cut21)
subplot(6,m,6);
scatter(SimuData.Cut_Point(:,3),pred_cut22)

subplot(6,m,7);
scatter(SimuData.Outcome_Latent,pred_out3)
text(-80, 0.5,'M3');
subplot(6,m,8);
scatter(SimuData.Cut_Point(:,2),pred_cut31)
subplot(6,m,9);
scatter(SimuData.Cut_Point(:,3),pred_cut32)

subplot(6,m,10);
scatter(SimuData.Outcome_Latent,pred_out4)
text(-80, 0.5,'M4');
subplot(6,m,11);
scatter(SimuData.Cut_Point(:,2),pred_cut41)
subplot(6,m,12);
scatter(SimuData.Cut_Point(:,3),pred_cut42)

subplot(6,m,13);
scatter(SimuData.Outcome_Latent,pred_out5)
text(-80, 0.5,'M5');
subplot(6,m,14);
scatter(SimuData.Cut_Point(:,2),pred_cut51)
subplot(6,m,15);
scatter(SimuData.Cut_Point(:,3),pred_cut52)

subplot(6,m,16);
scatter(SimuData.Outcome_Latent,pred_out6)
text(-80, 0.5,'M5');
subplot(6,m,17);
scatter(SimuData.Cut_Point(:,2),pred_cut61)
subplot(6,m,18);
scatter(SimuData.Cut_Point(:,3),pred_cut62)


% % The descriptive statistics of simulated data
% % The plot of densities of outcome and cut points
% SimuData= HOPIT_Simulate(Pars);
% [f1,x1]=ksdensity(SimuData.Outcome_Latent);
% [f2,x2]=ksdensity(SimuData.Cut_Point(:,2));
% [f3,x3]=ksdensity(SimuData.Cut_Point(:,3));
% 
% figure;
% DEN=plotyy(x1,f1,[x2',x3'],[f2',f3'])
% 
% title('Distribution of latent health and thresholds');
% ylabel('Density');
% legend('Latent health', 'Cutoff 1', 'Cutoff 2');
% set(DEN,'xlim',[-100,100])
% 
% hold on
% 
% % Plot vignettes
% for v=1:Pars.vignette
%     plot(repmat(SimuData.theta(v),1,101),0:0.01:1,':')
% end
% 
% hold off