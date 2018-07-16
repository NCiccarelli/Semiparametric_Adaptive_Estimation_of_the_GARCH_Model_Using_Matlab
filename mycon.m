function [c,ceq] = mycon(x)

% Nonlinear inequalities at x: 
c =  x(1).*  (x(3).^2)  + x(2)   - 1+ eps; 
% eps is 2^-52:  eps is the smallest number in matlab.

% Compute nonlinear equalities at x:
ceq = [] ;    
 
 
 
 
 
 
 
 
 
 
 
 
