function [ sum_lik  ] = MLE_normal_new( alfahat,betahat, sigma, y,h )


%% 
betainitial= 0.9;
alfainitial=0.05;
likelihoods= zeros(1,1000); 
h1 = ( 1./ (1-alfainitial-betainitial) ) ;
likelihoods(1) =   - 0.5 .*log(( 2.*pi )) -0.5 .*  ( (y(1)^2 ) ./ ((sigma.^2) .* h1   ) ) - log(sigma ) - 0.5.* log(h1)  ; 

%% 
T=1000;
for  j=2:T;
h(j) = (1+  betahat .* h(j-1) + alfahat .* ((y(j-1)).^2)  );
    likelihoods(j) =   - 0.5 .*log(( 2.*pi )) -0.5 .* (  (y(j)^2 ) ./ ((sigma.^2) .* h(j)   ) ) - log(sigma ) - 0.5.* log(h(j))  ; 
end

%% 
sum_lik= -sum(likelihoods) ;
 
end