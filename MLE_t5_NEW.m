  function [ sum_lik  ] = MLE_t5_NEW( alfahat,betahat, sigma,   y,h )

%%
betainitial= 0.9;
alfainitial=0.05;

%%
T=1000;
likelihoods= zeros(1,1000); 
 
h1 = (1./ (1-alfainitial-betainitial)); 
likelihoods(1) =  log(gamma(3)) - log(gamma(2.5))  - 0.5 .*log(pi) -0.5 .* log(5) - ...
                   3 .*    log(        1+ ( (     ( (y(1)^2) .* 1.6667 )  ./ ( h1 .* (sigma.^2)     )         )/5)     )    - ...
                   log(sigma ) - 0.5.* log(h1) + 0.5 .* log(1.6667)   ; 

 
%% 
T=1000;
for  j=2: T;
     h(j) = (1+  betahat .* h(j-1) + alfahat .* ((y(j-1)).^2)  ) ; 
likelihoods(j) =  log(gamma(3)) - log(gamma(2.5))  - 0.5 .*log(pi) -0.5 .* log(5) - ...
                   3 .*    log(        1+ ( (     ( (y(j)^2) .* 1.6667 )  ./ ( h(j) .* (sigma.^2)     )         )/5)     )    - ...
                   log(sigma ) - 0.5.* log(h(j)) + 0.5 .* log(1.6667)   ; 

end

%%
sum_lik= -sum(likelihoods) ;
 
end

  
  % likelihoods(j) =  log(gamma(4)) - log(gamma(3.5))  -log(sqrt(pi)) - log(sqrt(7)) - log(  (1/1.1832) .* sqrt(  (1+  betahat .* h(j-1) + alfahat .* ((y(j-1)).^2)  )   ) )  - ...
%     4 .* log(1+ (      (1./7)* ( ( (y(j)).^2)  * (1.1832.^2)  ./ (  (1+  betahat .* h(j-1) + alfahat .* ((y(j-1)).^2) ) )   ) )  );

% likelihoods(j) =  log(gamma(3)) - log(gamma(2.5))  -log(sqrt(pi)) - log(sqrt(5))  -  3 .* log(  1+ (((y(j)^2) ./  (0.6  .* h(j)))/5 )  )    - log(  sqrt(0.6 .* h(j))) ; 
% likelihoods(1) =  log(gamma(4)) - log(gamma(3.5))  -log(sqrt(pi)) - log(sqrt(7)) - log(  (1/1.1832) .*  sqrt(1./ (1-alfainitial-betainitial)) )  - ...
 %   4 .* log(1+ (      (1./7)* ( (y(1).^2)  * (1.1832.^2)   ./ (1./ (1-alfainitial-betainitial))   ) )  );
