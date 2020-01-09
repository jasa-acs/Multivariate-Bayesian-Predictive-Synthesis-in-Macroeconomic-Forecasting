%% Read Data
%% Read Data
load('var1_results.mat')
load('var2_results.mat')
load('var3_results.mat')
load('var4_results.mat')
load('var5_results.mat')

len = 270-1;

data = csvread('itu.csv',1,1);
yI = data(end-len:end,:);

T = length(var1_results{1,1}(end-len:end,1));
J = 5;
M = 6;

a = zeros(T,M,J);
A = zeros(T,M,M,J);
n = zeros(T,J);

a(:,:,1) = var1_results{1,1}(end-len:end,:);
A(:,:,:,1) = var1_results{2,1}(end-len:end,:,:);
n(:,1) = var1_results{3,1}(end-len:end,1);
a(:,:,2) = var2_results{1,1}(end-len:end,:);
A(:,:,:,2) = var2_results{2,1}(end-len:end,:,:);
n(:,2) = var2_results{3,1}(end-len:end,1);
a(:,:,3) = var3_results{1,1}(end-len:end,:);
A(:,:,:,3) = var3_results{2,1}(end-len:end,:,:);
n(:,3) = var3_results{3,1}(end-len:end,1);
a(:,:,4) = var4_results{1,1}(end-len:end,:);
A(:,:,:,4) = var4_results{2,1}(end-len:end,:,:);
n(:,4) = var4_results{3,1}(end-len:end,1);
a(:,:,5) = var5_results{1,1}(end-len:end,:);
A(:,:,:,5) = var5_results{2,1}(end-len:end,:,:);
n(:,5) = var5_results{3,1}(end-len:end,1);

%% Set Prior
beta = 0.99;
delta = [0.99 beta];
m_0 = repmat([0 repmat(1/J,1,J)],M,1);
p = (J+1)*M;
C_0 = eye(p)*1;
for l=1:J+1:p
    C_0(l,l) = 0.001;
end

for l=26:30
    C_0(l,l) = 0.01;
end

n_0 = 1/(1-beta)-M+1;
s_0 = eye(M)*0.01;

%% Burn_in and MCMC
burn_in = 3000;
mcmc_iter = 5000;

%% Run BPS
TT = 90:269;
fcstT = size(TT,2);
E_p = zeros(fcstT,M,mcmc_iter);
V_p = zeros(fcstT,M,M,mcmc_iter);
error = zeros(fcstT,M,mcmc_iter);
mlike = zeros(fcstT,mcmc_iter);
mliket = zeros(fcstT,mcmc_iter);
mlikeV = zeros(fcstT,mcmc_iter);
theta_t = zeros(fcstT,J+1,M,mcmc_iter);
eltime = zeros(fcstT,1);
std_var = @(x) (x+x')/2;
E_pa = zeros(fcstT,M,mcmc_iter);
V_pa = zeros(fcstT,M,M,mcmc_iter);
V_pS = zeros(fcstT,M,M,mcmc_iter);
errora = zeros(fcstT,M,mcmc_iter);
mlikea = zeros(fcstT,mcmc_iter);
mlikeat = zeros(fcstT,mcmc_iter);
mlikeas = zeros(fcstT,mcmc_iter);

k = 1;
tic

yy = reshape(repmat(1:M,M,1),[],1);
xx = reshape(repmat(1:M,M,1)',[],1);
ay = reshape(repmat(1:J:J*M,M,1),[],1)-1;
ax = reshape(repmat(1:J:J*M,M,1)',[],1)-1;
a_t = zeros(J*M,M);
A_t = zeros(J*M,J*M);
theta_temp = zeros(J*M,M);

theta_k = cell(269,1);

for t=TT
    tic
    y = yI(1:t,:);
    a_j = a(1:t,:,:);
    A_j = A(1:t,:,:,:);
    n_j = n(1:t,:,:);
    [theta_K,V_K,theta_T,X_T] = mBPS(y,a_j,A_j,n_j,delta,m_0,C_0,n_0,s_0,burn_in,mcmc_iter);
    theta_k{k,1} = theta_K;
    for i=1:mcmc_iter
        for j=1:J
            for mm=1:M*M
                A_t(j+ay(mm),j+ax(mm)) = squeeze(A(t+1,yy(mm),xx(mm),j));
            end
        end
        for m=1:M
            a_temp = squeeze(a(t+1,:,:));
            a_t(m*(J+1)-((J+1)-1):m*(J+1),m) = [1 a_temp(m,:)];
        end
        for m=1:M
            theta_temp(m*(J)-(J-1):m*(J),m) = theta_K(m*(J+1)-(J-1):m*(J+1),i);
        end
        E_p(k,:,i) = a_t'*theta_K(:,i);
        V_p(k,:,:,i) = theta_temp'*A_t*theta_temp+squeeze(V_K(:,:,i));
        error(k,:,i) = yI(t+1,:)-squeeze(E_p(k,:,i));
        mlike(k,i) = (log(gamma(0.5*(n_j(1)+M)))-log(gamma(0.5*n_j(1)))...
            -log(pi^(M/2)*n_j(1)^(M/2)*(det(squeeze(V_p(k,:,:,i))/delta(2))^(1/2)))...
            -(0.5*(n_j(1)+M))*log(1+1/n_j(1)*(squeeze(error(k,:,i)))*(squeeze(V_p(k,:,:,i))/delta(2))^(-1)*(squeeze(error(k,:,i)))'));
     end
    k = k+1;
    toc
    eltime(k) = toc;
end

%compute MSFE
len = 180-1;
w = 1:len+1;
w = repmat(w',1,M);

er_BPS = cumsum(mean(error(end-len:end,:,:),3).^2);
mse_BPS = (er_BPS./w);

er_var2 = cumsum((yI(end-len:end,:)-var2_results{1,1}(end-len:end,1:M)).^2);
mse_var2 = (er_var2./w);
er_var1 = cumsum((yI(end-len:end,:)-var1_results{1,1}(end-len:end,1:M)).^2);
mse_var1 = (er_var1./w);
er_var3 = cumsum((yI(end-len:end,:)-var3_results{1,1}(end-len:end,1:M)).^2);
mse_var3 = (er_var3./w);
er_var4 = cumsum((yI(end-len:end,:)-var4_results{1,1}(end-len:end,1:M)).^2);
mse_var4 = (er_var4./w);
er_var5 = cumsum((yI(end-len:end,:)-var5_results{1,1}(end-len:end,1:M)).^2);
mse_var5 = (er_var5./w);

%compute LPDR
ml_BPS = mean(exp(mlike),2);
ml_BPS = cumsum(log(ml_BPS(end-len:end,:)));

ml_var1 = var1_results{4,1}(1,end-len:end);
ml_var2 = var2_results{4,1}(1,end-len:end);
ml_var4 = var4_results{4,1}(1,end-len:end);
ml_var3 = var3_results{4,1}(1,end-len:end);
ml_var5 = var5_results{4,1}(1,end-len:end);
ml_var1 = cumsum(ml_var1);
ml_var2 = cumsum(ml_var2);
ml_var4 = cumsum(ml_var4);
ml_var3 = cumsum(ml_var3);
ml_var5 = cumsum(ml_var5);