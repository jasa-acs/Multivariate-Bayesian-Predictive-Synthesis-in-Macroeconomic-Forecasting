function [theta_K,V_K,theta_T,X_T] = mBPS(y,a_j,A_j,n_j,delta,m_0,C_0,n_0,s_0,burn_in,mcmc_iter)
% Bayesian Predictive Synthesis
%
%  Synthesis Function:      
%           y_t = x_t'\theta_t + \Nu_t
%      \theta_t = \theta_{t-1} + \Omega_t
%         \Nu_t \sim Normal(0,V_t)
%      \Omega_t \sim Normal(0,W_t)
%
%  Forecasts: 
%       x_{j,t} \sim t(a_{j,t},A_{j,t},n_{j,t})
%
%  inputs:
%    y: TxM target time series data from t=1:T
%    a_j: TxJ*M matrix of mean of agent forecast t=1:T
%    A_j: TxJ*MxJ*M matrix of covariance of agent forecast t=1:T
%    n_j: TxJ*M matrix of d.o.f. of agent forecast t=1:T
%    delta: discount rate for [state obs_var]
%    m0 --  1x(J+1)*M vector prior mean for state
%    C0 --  (J+1)*Mx(J+1)*M prior covar matrix
%    n0 --  prior d.o.f.
%    s0 --  MxM prior of obs covar
%    burn_in, mcmc_iter: number of burn-in/MCMC iterations
%  outputs:
%    theta_K: forecast sample \theta_{T+1}
%    V_k: forecast sample obs var
%    theta_T: posterior \theta_{1:T}
%    X_T: posterior sample of agent forecast
%
%  ? 2017, Kenichiro McAlinn, All rights reserved.

%% initial settings
std_var = @(x) (x+x')/2;

mcmc = burn_in+mcmc_iter;
T = size(y,1);
M = size(y,2);
J = size(a_j,3);
P = J+1;

m_t = zeros(T+1,P*M);
C_t = zeros(T+1,P*M,P*M);
V_t = zeros(T,M,M);
n_t = zeros(T,1);
h_t = zeros(T,1);
S_t = zeros(T,M,M);
a_t = zeros(T,P*M);
R_t = zeros(T,P*M,P*M);
X_t = zeros(T,M*J);
D_t = zeros(T+1,M,M);
theta_t = zeros(T,P*M);
F_t = zeros(P*M,M);
th_t = zeros(J*M,M);
theta_K = zeros(P*M,mcmc_iter);
S_K = zeros(M,M,mcmc_iter);
V_K = zeros(M,M,mcmc_iter);
R_K = zeros(P*M,P*M,mcmc_iter);
a_K = zeros(P*M,mcmc_iter);
n_T = zeros(T,mcmc_iter);
theta_1 = zeros(1,M);
ee_t = zeros(T,M);
D_T = zeros(M,M);

theta_T = zeros(T,P*M,mcmc_iter);
X_T = zeros(T,M*J,mcmc_iter);

a_J = zeros(T,J*M);
n_J = zeros(T,J);
A_J = zeros(T,J*M,J*M);

yy = reshape(repmat(1:M,M,1),[],1);
xx = reshape(repmat(1:M,M,1)',[],1);
ay = reshape(repmat(1:J:J*M,M,1),[],1)-1;
ax = reshape(repmat(1:J:J*M,M,1)',[],1)-1;

for t=1:T
    a_J(t,:) = reshape(squeeze(a_j(t,:,:))',[],1)';
    for j=1:J
        for mm=1:M*M
            A_J(t,j+ay(mm),j+ax(mm)) = squeeze(A_j(t,yy(mm),xx(mm),j));
        end
    end
    n_J(t,:) = n_j(t,:);
end

d = delta(1);
beta = delta(2);

m_t(1,:) = reshape(m_0',[],1)';
C_t(1,:,:) = C_0';
h_0 = n_0+M-1;
h = h_0';
for t=1:T+1
    D_t(t,:,:) = s_0'*h;
end
for t=1:T
    S_t(t,:,:) = s_0';
    V_t(t,:,:) = s_0';
end

for t=1:T
    X_t(t,:,:) = a_J(t,:)+randn(1,length(a_J(t,:)))*chol(std_var(squeeze(A_J(t,:,:))));
end

for t=1:T
    X_t(t,:) = reshape(squeeze(a_j(t,:,:))',[],1)';
end

%% MCMC Sampler
for i=1:mcmc
    % forward-filter
    h = h_0;
    for t=1:T
        F = X_t(t,:);
        % prior for time t
        a_t(t,:) = m_t(t,:);
        R_t(t,:,:) = squeeze(C_t(t,:,:))/d;
        R = squeeze(R_t(t,:,:));
        for m=1:M
            F_t(m*P-(P-1):m*P,m) = [1 F(:,m*J-(J-1):m*J)]';
        end
        % predict time t
        f_t = F_t'*a_t(t,:)';
        q_t = squeeze(V_t(t,:,:))+F_t'*R*F_t;
        % compute forcast error and adaptive vector
        e_t = y(t,:)-f_t';
        A_t = R*F_t*(q_t^(-1));
        h = beta*h+1;
        n = h-M+1;
        n_t(t) = n;
        h_t(t) = h;
        % posterior for time t
        m_t(t+1,:) = a_t(t,:)+(A_t*e_t')';
        C_t(t+1,:,:) = std_var(R-A_t*q_t*A_t');
    end
    % sample theta at T
    theta_t(end,:) = m_t(end,:)+randn(1,length(m_t(end,:)))*chol(std_var(squeeze(R_t(end,:,:))));
    % backward-sampler
    for t=T-1:-1:1
        % backwards mean and std
        m_star = m_t(t+1,:)+d*(theta_t(t+1,:)-a_t(t+1,:));
        C_star = squeeze(C_t(t+1,:,:))*(1-d);
        % backwards theta
        theta_t(t,:) = m_star+randn(1,length(m_star))*chol(std_var(C_star));
    end
    % sample variance
    for t=1:T
        F = X_t(t,:);
        for m=1:M
            F_t(m*P-(P-1):m*P,m) = [1 F(:,m*J-(J-1):m*J)]';
        end
        ee_t(t,:) = y(t,:)-(F_t'*theta_t(t,:)')';
        D_t(t+1,:,:) = beta*squeeze(D_t(t,:,:))+ee_t(t,:)'*ee_t(t,:);
    end
    V_t(end,:,:) = iwishrnd(std_var(squeeze(D_t(end,:,:))),n_t(end));
    for t=T-1:-1:1
        Phi_t1 = squeeze(V_t(t+1,:,:))^(-1);
        Phi_t = wishrnd(std_var(squeeze(D_t(t+1,:,:)))^(-1),(1-beta)*h_t(t));
        V_t(t,:,:) = (beta*Phi_t1+Phi_t)^(-1);
    end
    % sample X_t
    for t=1:T
        for m=1:M
            theta_1(1,m) = theta_t(t,m*P-(P-1));
            th_t(m*J-(J-1):m*J,m) = theta_t(t,1+m*P-(P-1):m*P)';
        end
        thA = th_t'*squeeze(A_J(t,:,:));
        thAth = thA*th_t;
        sigma = thA'/(squeeze(V_t(t,:,:))+thAth);
        a_star = a_J(t,:)'+sigma*(y(t,:)'-(theta_1'+th_t'*a_J(t,:)'));
        A_star = std_var(squeeze(A_J(t,:,:))-sigma*thA);
        lambda = reshape(repmat(((0.5*n_J(t,:)./randg(n_J(t,:)/2))).^(1/2),M,1)',[],1);
        X_t(t,:) = a_star'+(lambda'.*randn(1,length(a_star))*chol(std_var(A_star)));
    end
    % save parameters of interest
    if i>burn_in
        D_T =  squeeze(D_t(end,:,:))+D_T;
        n_k = beta*h_t(end)+1;
        V_K(:,:,i-burn_in) = V_t(end,:,:);
        S_K(:,:,i-burn_in) = std_var(squeeze(D_t(end,:,:)))/n_k;
        n_T(:,i-burn_in) = n_t(:,1);
        lambda = sqrt((0.5*n_t(end)/randg(n_t(end)/2)));
        theta_K(:,i-burn_in) = m_t(end,:)'+(lambda*randn(1,length(m_t(end,:)))*chol(std_var(squeeze(C_t(end,:,:))/d)))';
        theta_T(:,:,i-burn_in) = theta_t;
        X_T(:,:,i-burn_in) = X_t;
        a_K(:,i-burn_in) = m_t(end,:)';
        R_K(:,:,i-burn_in) = squeeze(C_t(end,:,:))/d;
    end
end
end