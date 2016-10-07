function [LLV,score] = HOPIT_Likelihood_King(Data,b,cut_point)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function produces log likelihood and score of HOPIT model
% The cut-points are specified as follows, for example with 4 cut points
%
%                           c1=-inf ---- c2 ---- c3 ---- c4 ---- c5 ---- c6=inf
%
%   Inputs:
%                           Data.Outcome_Indep                  :               Regressors in outcome function, eg. 1000X2, 1000 observations and 2 independent variables
%                           Data.Outcome_Dep                    :               Outcome, self-assessment, eg. 1000X1
%                           Data.cut_Indep                      :               Regressors in cut point function, eg. 1000X2
%                           Data.Vignette                       :               Rating on vigettes, eg. 1000X4, for 4 vignettes
%                           b                                   :               Parameters of the model
%                           cut_point                           :               Number of cut points
%
%   Outputs:
%                           LLV                                 :               Log likelihood
%                           score                               :               Score (gradient of log likelihood), a vector
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Initiation
% Import data
X=Data.Outcome_Indep;                                       % Note: if X=[], then Xb=0
H=Data.Outcome_Dep;
Z=Data.Cut_Indep;
V=Data.Vignette;


nc=cut_point;                                               % Number of cut-points
n=size(H,1);                                                % Number of data points
k=length(b);

% Keep dimensions agree, a kX1 matrix
if(size(b,1)>1)
    b=b';
end

% Calculate the size of parameters
kb=size(X,2);                                               % Length of beta
kz=size(Z,2);                                               % Number of variables in cut-point functions
kg=kz*nc;                                                   % Length of gamma
kt=size(V,2);                                               % Length of theta (ie, number of vignettes)

% Define parameters
beta=b(1:kb);                                               % Coefficients in outcome function
gamma=b(kb+1:kb+kg);                                        % Coefficients in cut-point function
theta=b(kb+kg+1:kb+kg+kt);                                  % Vignette location parameter
sigma_vi=b(kb+kg+kt+1);                                     % sd of vignettes


% Define predictors
Xb=X*beta';                                                 % Linear predictor, i.e., beta1*x1+beta2*x2..., nX1 matrix
Zb=zeros(n,nc+2);                                           % Individual-specific cut points, nX(nc+2), including -Inf and Inf
Zb(:,1)=-Inf;                                               % The first cut-point: -Inf
Zb(:,2)=Z*gamma(1:kz)';                                     % The second cut-point
if nc>=2
    for i=3:nc+1
        Zb(:,i)=Zb(:,i-1)+exp(Z*gamma(((i-2)*kz+1):((i-1)*kz))');     % The third, fourth,...
    end
end
Zb(:,nc+2)=Inf;                                             % The last cut-point: Inf


% In case X is not availalble. eg. in restricted model
if isempty(X)
    Xb=zeros(n,1);
end

nsingle = realmin('double');
% nsingle = 0; 

%% Calculate the likelihood

% Initiation (critical)
LV_All=zeros(n,1+kt);
Mills_beta=zeros(n,nc);
Mills_gamma=zeros(n,nc);
Mills_theta=zeros(n,kt);
Mills_sigma_v=zeros(n,1);

% Log likelihood of the self-assessment component

Unique_Outcome=unique(H);
npdf_1_beta=zeros(1,n);
npdf_beta=zeros(1,n);
ncdf_1_beta=zeros(1,n);
ncdf_beta=zeros(1,n);
ncdf_1_beta_alter=zeros(1,n);
ncdf_beta_alter=zeros(1,n);
for i=Unique_Outcome(1):Unique_Outcome(end)
    id=find(H==i);                                          % Tag individuals by categories
    npdf_1_beta(id)=normpdf(Zb(id,i+1)-Xb(id));
    npdf_beta(id)=normpdf(Zb(id,i)-Xb(id));
    ncdf_1_beta(id)=normcdf(Zb(id,i+1)-Xb(id));
    ncdf_beta(id)=normcdf(Zb(id,i)-Xb(id));
    ncdf_1_beta_alter(id)=normcdf(-(Zb(id,i+1)-Xb(id)));
    ncdf_beta_alter(id)=normcdf(-(Zb(id,i)-Xb(id)));
end
diff_pdf_beta=npdf_1_beta-npdf_beta;
diff_cdf_beta=max(ncdf_1_beta-ncdf_beta,ncdf_beta_alter-ncdf_1_beta_alter);
LV_All(:,1)=diff_cdf_beta';
Mills_beta(:,1)=diff_pdf_beta'./diff_cdf_beta';             % A nX1 array

if nc>=2
    for m=2:nc
        diff_pdf_beta=zeros(1,n);
        diff_cdf_beta=ones(1,n);
        for i=m:Unique_Outcome(end)                                              % It is IMPORTANT to notice iteration starts
            if i == m                                           % from m rather than Unique_Outcome(m)
                indc = 0;
            else
                indc = 1;
            end
            id=find(H==i);
            diff_pdf_beta(id)=npdf_1_beta(id)-indc*npdf_beta(id);
            diff_cdf_beta(id)=max(ncdf_1_beta(id)-ncdf_beta(id),ncdf_beta_alter(id)-ncdf_1_beta_alter(id));
        end
        Mills_beta(:,m)=diff_pdf_beta'./diff_cdf_beta'.*exp(Z*gamma(((m-1)*kz+1):(m*kz))');
    end
end


% Log likelihood of the vignette component
for i=1:kt
    Unique_Outcome=unique(V(:,i));
    npdf_1_gamma=zeros(1,n);
    npdf_gamma=zeros(1,n);
    ncdf_1_gamma=zeros(1,n);
    ncdf_gamma=zeros(1,n);
    ncdf_1_gamma_alter=zeros(1,n);
    ncdf_gamma_alter=zeros(1,n);
    diff_pdf_gamma=zeros(1,n);
    diff_cdf_gamma=ones(1,n);
    
    for j=Unique_Outcome(1):Unique_Outcome(end)
        id=find(V(:,i)==j);
        npdf_1_gamma(id)=normpdf(1/sigma_vi*(Zb(id,j+1)-theta(i)));
        npdf_gamma(id)=normpdf(1/sigma_vi*(Zb(id,j)-theta(i)));
        ncdf_1_gamma(id)=normcdf(1/sigma_vi*(Zb(id,j+1)-theta(i)));
        ncdf_gamma(id)=normcdf(1/sigma_vi*(Zb(id,j)-theta(i)));
        ncdf_1_gamma_alter(id)=normcdf(-(1/sigma_vi*(Zb(id,j+1)-theta(i))));
        ncdf_gamma_alter(id)=normcdf(-(1/sigma_vi*(Zb(id,j)-theta(i))));
    end
    diff_pdf_gamma=npdf_1_gamma-npdf_gamma;
    diff_cdf_gamma=max(ncdf_1_gamma-ncdf_gamma,ncdf_gamma_alter-ncdf_1_gamma_alter);
    LV_All(:,i+1)=diff_cdf_gamma;
    Mills_gamma(:,1) = Mills_gamma(:,1)+diff_pdf_gamma'./diff_cdf_gamma'*(1/sigma_vi);
    
    Mills_theta(:,i) = diff_pdf_gamma'./diff_cdf_gamma'*(1/sigma_vi);
    
    
    % Mills ratios:
    % [f(tau(i)-beta*x)-Indicator*f(tau(i-1)-beta*x)]/[F(tau(i)-beta*x)-F(tau(i-1)-beta*x)]*exp(gamma*z)
    % [f(tau(i)-theta)-Indicator*f(tau(i-1)-theta)]/[F(tau(i)-theta)-F(tau(i-1)-theta)]*exp(gamma*z)
    if nc>=2
        for m=2:nc
            diff_pdf_gamma=zeros(1,n);
            diff_cdf_gamma=ones(1,n);
            for j=m:Unique_Outcome(end)
                if j == m
                    indc = 0;
                else
                    indc = 1;
                end
                id=find(V(:,i)==j);
                diff_pdf_gamma(id)=npdf_1_gamma(id)-indc*npdf_gamma(id);
                diff_cdf_gamma(id)=max(ncdf_1_gamma(id)-ncdf_gamma(id),ncdf_gamma_alter(id)-ncdf_1_gamma_alter(id));
            end
            Mills_gamma(:,m) = Mills_gamma(:,m)+diff_pdf_gamma'./diff_cdf_gamma'.*exp(Z*gamma(((m-1)*kz+1):(m*kz))')*(1/sigma_vi);
        end
    end
    
    
    % Mills ratios:
    diff_pdf_rho_v=zeros(1,n);
    diff_cdf_rho_v=ones(1,n);
    for j=Unique_Outcome(1):Unique_Outcome(end)
        % We need to distinguish the first, end and middle categories since otherwise Matlab erroreously returns NaN
        if j==1
            id=find(V(:,i)==j);
            diff_pdf_rho_v(id)=npdf_1_gamma(id)'.*(Zb(id,j+1)-theta(i))*(-1/sigma_vi^2);
            diff_cdf_rho_v(id)=max(ncdf_1_gamma(id)-ncdf_gamma(id),ncdf_gamma_alter(id)-ncdf_1_gamma_alter(id))';
        elseif j==nc+1
            id=find(V(:,i)==j);
            diff_pdf_rho_v(id)=-npdf_gamma(id)'.*(Zb(id,j)-theta(i))*(-1/sigma_vi^2);
            diff_cdf_rho_v(id)=max(ncdf_1_gamma(id)-ncdf_gamma(id),ncdf_gamma_alter(id)-ncdf_1_gamma_alter(id))';
        else
            id=find(V(:,i)==j);
            diff_pdf_rho_v(id)=npdf_1_gamma(id)'.*(Zb(id,j+1)-theta(i))*(-1/sigma_vi^2)-npdf_gamma(id)'.*(Zb(id,j)-theta(i))*(-1/sigma_vi^2);
            diff_cdf_rho_v(id)=max(ncdf_1_gamma(id)-ncdf_gamma(id),ncdf_gamma_alter(id)-ncdf_1_gamma_alter(id))';
        end
    end
    Mills_sigma_v(:,1) = Mills_sigma_v(:,1)+diff_pdf_rho_v'./diff_cdf_rho_v';
end

LV=max(prod(LV_All,2),nsingle);


% Derive scores
% Calculate the derivative w.r.t beta
rep=Mills_beta(:,1)*ones(1,kb);                             % A nXkb matrix
if ~isempty(X)
    der(:,1:kb)= -rep.*X;                                      % A nXkb matrix
end

% Calculate the derivative w.r.t gamma
for m=1:nc
    rep1=Mills_beta(:,m)*ones(1,kz);                        % A nXkz matrix
    rep2=Mills_gamma(:,m)*ones(1,kz);                       % A nXkz matrix
    der(:,kb+kz*(m-1)+1:kb+kz*m)=rep1.*Z+rep2.*Z;
end

% Calculate the derivative w.r.t theta
der(:,kb+kg+1:kb+kg+kt)= -Mills_theta;


% Calculate the derivative w.r.t sigma_vi
der(:,kb+kg+kt+1)= Mills_sigma_v;

%      der(isnan(der))=Inf;

LLV=-sum(log(LV));



%% Use the score as the second output when called
if nargout > 1
    score=-sum(der);
end
end
