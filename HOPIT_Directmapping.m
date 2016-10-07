function [LLV,score] = HOPIT_Directmapping(Data,b,cut_point)
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
X=Data.Outcome_Indep;
H=Data.Outcome_Dep;
V=Data.Vignette;

nc=cut_point;                                               % Number of cut-points
n=size(H,1);                                                % Number of data points

% Keep dimensions agree, a kX1 matrix
if(size(b,1)>1)
    b=b';
end

% Calculate the size of paramters
kb=size(X,2);                                               % Length of beta
kt=size(V,2);                                               % Number of vignettes
kd=kt*nc;                                                   % Length of delta

% Define parameters
beta=b(1:kb);                                               % Coefficients in outcome function
delta=b(kb+1:kb+kd);                                        % Coefficients in cut-point function

% TR = zeros(kd);
% TR(:,1)=1;
% for i = 1:kt
%     TR(1+i*nc)


% Define indicators
Xb=X*beta';                                                 % Linear predictor, i.e., beta1*x1+beta2*x2..., nX1 matrix
Zb=zeros(n,nc+2);                                           % Individual-specific cut points, nX(nc+2), including -Inf and Inf
Zb(:,1)=-Inf;                                               % The first cut-point: -Inf
Zb(:,2)=V*delta(1:kt)';                                     % The second cut-point
if nc>=2
    for i=3:nc+1
        Zb(:,i)=Zb(:,i-1)+exp(V*delta(((i-2)*kt+1):((i-1)*kt))');     % The third, fourth,...
    end
end
Zb(:,nc+2)=Inf;                                             % The last cut-point: Inf

% In case X is not availalble. eg. in restricted model
if isempty(X)
    Xb=zeros(n,1);
end


%% Calculate the likelihood
% Initiation (critical)
Mills_beta=zeros(n,nc);

% Mills ratio: [f(tau(i)-beta*x)-f(tau(i-1)-beta*x)]/[F(tau(i)-beta*x)-F(tau(i-1)-beta*x)]
Unique_Outcome=unique(H);                                   % Sorting categories in H, ie. 1,2,3,4,5
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
LV=diff_cdf_beta';
Mills_beta(:,1)=diff_pdf_beta'./diff_cdf_beta';             % A nX1 array


% Mills ratios:
% [f(tau(i)-beta*x)-Indicator*f(tau(i-1)-beta*x)]/[F(tau(i)-beta*x)-F(tau(i-1)-beta*x)]*exp(delta*z)
% [f(tau(i)-theta)-Indicator*f(tau(i-1)-theta)]/[F(tau(i)-theta)-F(tau(i-1)-theta)]*exp(delta*z)
if nc>=2
    for m=2:nc
        diff_pdf_beta=zeros(1,n);
        diff_cdf_beta=ones(1,n);
        for i=m:Unique_Outcome(end)                             % It is IMPORTANT to notice iteration starts
            if i == m                                           % from m rather than Unique_Outcome(m)
                indc = 0;
            else
                indc = 1;
            end
            id=find(H==i);
            diff_pdf_beta(id)=npdf_1_beta(id)-indc*npdf_beta(id);
            diff_cdf_beta(id)=max(ncdf_1_beta(id)-ncdf_beta(id),ncdf_beta_alter(id)-ncdf_1_beta_alter(id));
        end
        Mills_beta(:,m)=diff_pdf_beta'./diff_cdf_beta'.*exp(V*delta(((m-1)*kt+1):(m*kt))');
    end
end

%% Derive scores
% Calculate the derivative w.r.t beta
rep=Mills_beta(:,1)*ones(1,kb);                             % A nXkb matrix
if ~isempty(X)
    der(:,1:kb)= - rep.*X;                                      % A nXkb matrix
end

% Calculate the derivative w.r.t delta
for m=1:nc
    rep1=Mills_beta(:,m)*ones(1,kt);                        % A nXkz matrix
    der(:,kb+kt*(m-1)+1:kb+kt*m)=rep1.*V;
end

LLV=-sum(log(LV));

%% Use the score as the second output when called
if nargout > 1
    score=-sum(der);
end
end
