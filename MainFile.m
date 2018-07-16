 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The theory for semiparametric (adaptive) estimation of the GARCH model is reported in Drost and Klaassen (1997), Journal of Econometrics, Volume 81, pp. 193-221 
% Names of matrices used in this Matlab code follow Drost and Klaassen
% (1997). See Drost and Klaassen (1997) for more details.   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% number of repetitions for the Monte Carlo simulation:
repetitions = 2000; 

% initialize matrices for coefficients (MLE, QMLE, semiparametric adaptive
% estimators):
MLE_alpha= NaN(1,repetitions);
MLE_beta= NaN(1,repetitions);

QMLE_alpha= NaN(1,repetitions);
QMLE_beta= NaN(1,repetitions);

MC_beta_adaptive = NaN(1,repetitions);
MC_alpha_adaptive = NaN(1,repetitions);


for w = 1: repetitions

    
%% sample size for the GARCH process is set to 1000:    
N=1000;


%% generate the t5-student innovation for the GARCH process 
t5_distr = trnd(5,[N,1])';
% generate t5-student innovation with mean zero and variance 1: 
epsilon =  (t5_distr  - mean(t5_distr) )./ (sqrt(var(t5_distr)  ));


%% parameter values (GARCH volatility equation): 
beta0= 0.9;
alfa0=0.05;


%% generate the data: 
h=zeros(1,N);
y=zeros(1,N);
h(1)= 1./ (1-alfa0-beta0);
y(1)= ((h(1)).^0.5) * (epsilon(1)); 

for  i=1: (N-1);
    h(i+1)= 1+ beta0 .* h(i) + alfa0 .* ((y(i)).^2);
    y(i+1)=  ( (h(i+1)).^0.5) * (epsilon(i+1));
end


%% starting values for the estimation of parameters (maximum likelihood estimation): 
startingvalues = [0.0001;0.0001; 0.0001];
lowerbound = [  0 ;  0  ; 0];
upperbound = [  1;  1; 10];
  
%% estimate the parameters using MLE (maximum likelihood estimation)
% see the MLE_t5_NEW function for more details:
likelihood = @(x) MLE_t5_NEW(x(1),x(2),x(3), y,h );

options    = optimset('fmincon');
options    = optimset(options, 'TolFun', 1e-006);
options    = optimset(options, 'LargeScale', 'off');
options    = optimset(options, 'MaxFunEvals', 1000);
options    = optimset(options, 'MaxIter', 400);
options = optimset('maxfunevals',20000);

[PARMLE, fval] = fmincon(likelihood,startingvalues,[],[],[],[],lowerbound ,upperbound , @mycon ,options);

MLE_alpha(1,w)= PARMLE(1,1);
MLE_beta(1,w)= PARMLE(2,1);
MLE_sigma(1,w)= PARMLE(3,1);



%% estimate the parameters using QMLE (quasi maximum likelihood estimation)
% see the MLE_normal_new function for more details:
startingvalues = [0.0001;0.0001; 0.0001];
lowerbound = [  0 ;  0  ; 0];
upperbound = [  1;  1; 10];

likelihood = @(x) MLE_normal_new(x(1),x(2),x(3), y,h );

options    = optimset('fmincon');
options    = optimset(options, 'TolFun', 1e-006);
options    = optimset(options, 'LargeScale', 'off');
options    = optimset(options, 'MaxFunEvals', 1000);
options    = optimset(options, 'MaxIter', 400);
 
options = optimset('maxfunevals',20000);

[PARMLE_QMLE, fval] = fmincon(likelihood,startingvalues,[],[],[],[],lowerbound ,upperbound , @mycon ,options);

QMLE_alpha(1,w)= PARMLE_QMLE(1,1);
QMLE_beta(1,w)= PARMLE_QMLE(2,1);
QMLE_sigma(1,w)= PARMLE_QMLE(3,1);


%% PART 1 OF THE SEMIPARAMETRIC ESTIMATOR 

h_estimated= zeros(1,N);
h_estimated(1)=  1./ (1-alfa0-beta0)  ; 

for  i=1:   (N-1);

    h_estimated(i+1)= 1+ PARMLE_QMLE(1,1)  .* ((y(i)).^2) +  PARMLE_QMLE(2,1)  .* h_estimated(i) ;
    
end

estimated_chi= zeros(1,N);
estimated_chi= y./ sqrt(h_estimated) ; 

sigmahat= PARMLE_QMLE(3,1); 

estimated_error = (estimated_chi   ) ./ sigmahat  ;



%% PART 2 OF THE SEMIPARAMETRIC ESTIMATOR: the Normal kernel estimator (estimation of the error term of the GARCH model)

%  bandwith = 0.25;
bandwith = 0.40;
 
kernel = zeros(N,length(estimated_error));
for i = 1:N
    kernel(i,:) = exp(  -0.5*((estimated_error(i)-estimated_error)./bandwith).^2  )  /sqrt(2*pi);
    %kernel(i,:) = normpdf((x-epsilon(i))./bandwith,0,1);
end

sum_kernel = sum(kernel,1)/((N-1)*bandwith);




% estimation of kernel derivative: 
kernelderivative = zeros(N,length(estimated_error));
for i = 1:N
    
        kernelderivative(i,:) = - (1/sqrt(2*pi)) .* exp(  -0.5*((estimated_error(i)-estimated_error)./bandwith).^2 ).*(-((estimated_error(i)-estimated_error)./bandwith)) ;
    
end
sum_kernelderivative =  sum(kernelderivative,1)/((N-1)*(bandwith.^2));


%% PART 3 OF THE SEMIPARAMETRIC ESTIMATOR 

psi = - (  1+   estimated_error .*  (  sum_kernelderivative  ./   sum_kernel  )  ) ;
for i=1: N
psi_new(:,:,i)= psi(:,i);
end



%% PART 4 OF THE SEMIPARAMETRIC ESTIMATOR 

h_estimated; 

H(2,1,2000)= zeros; 

H(:,:,1)= [0; 0]; 

for  i=2: N ;
H(:,:,i)=  PARMLE_QMLE(2,1) .* H(:,:,i-1) + [ (y(i-1)).^2 ; h_estimated(i-1)  ] ;
end

%% PART 5 OF THE SEMIPARAMETRIC ESTIMATOR 

sigmahat = PARMLE_QMLE(3,1);
W=zeros(3,1,N);


for  i=1: N;
  W(:,:,i) =  (sigmahat.^(-1) ) *  [ (0.5.* sigmahat .* H(1,:, i)) ./   h_estimated(1,i) ;...
                                     (0.5.* sigmahat .* H(2,:, i)) ./   h_estimated(1,i) ; ...
                                        1 ] ;
                              
end
     

%% PART 6 OF THE SEMIPARAMETRIC ESTIMATOR 
% see equation (3.2) of Drost and Klaassen (1997), Journal of Econometrics,
% Volume 81, pp. 193-221 

for i=1: N
    ldot_ldotprime(:,:,i)= W(:,:,i)* psi_new(:,:,i)*psi_new(:,:,i)'* W(:,:,i)';
end
fischer_info = mean(ldot_ldotprime,3);
part2=inv(fischer_info);


%% PART 7 OF THE SEMIPARAMETRIC ESTIMATOR 

meanW= mean( W , 3);
for i=1:  N
    deltaW(:,:,i)= W(:,:,i)   - meanW;
end 

for i=1: N
    deltaW_psi(:,:,i)= deltaW(:,:,i) * psi(:,i);
end 

part3= mean(deltaW_psi, 3);
part1= [1 0 0  ; 0 1 0];
fischer_per_eff_score= part2 * part3;
add_part=  part1 *  fischer_per_eff_score ;
parameters_QMLE= PARMLE_QMLE( [1,2]  ,:);
adaptive_beta= parameters_QMLE  + add_part;


% estimated parameters of the GARCH model using semiparametric estimation:
MC_alpha_adaptive(1,w)= adaptive_beta(1,1);
MC_beta_adaptive(1,w)=adaptive_beta(2,1);


w
end

% means and standard deviation of the GARCH parameters using the ML
% estimator: 
Mean_alpha_MLE= mean(MLE_alpha,2)
Mean_beta_MLE= mean(MLE_beta,2)
Std_dev_alpha_MLE =std(MLE_alpha)
Std_dev_beta_MLE= std(MLE_beta) 
Mean_sigma_MLE= mean(MLE_sigma,2)


% means and standard deviation of the GARCH parameters using the QML (quasi maximum likelihood)
% estimator:
Mean_alphaQMLE= mean(QMLE_alpha,2)
Mean_betaQMLE= mean(QMLE_beta,2)
Mean_sigmaQMLE= mean(QMLE_sigma,2)
Std_dev_alphaQMLE=std(QMLE_alpha)
Std_dev_betaQMLE=std(QMLE_beta)
Std_dev_sigmaQMLE=std(QMLE_sigma) 




% means and standard deviation of the GARCH parameters using the
% semiparametric adaptive  estimator: 
Mean_alpha_semi_p= mean(MC_alpha_adaptive,2)
Mean_beta_semi_p= mean(MC_beta_adaptive,2)
Std_dev_alpha_semi_parametric=std(MC_alpha_adaptive)
Std_dev_beta_semi_parametric=std(MC_beta_adaptive) 


