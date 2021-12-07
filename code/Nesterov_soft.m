function X_update = Nesterov_soft(X_i,X_j, M, W, lambda, i)
V_i = X_i + (i-1)/(i+2)*(X_i-X_j);
[U,S,V]=svd(W.*M+(ones(size(W))-W).*V_i);
[m,n]=size(S);
S_lambda = zeros(m,n);
for i = 1:min(m,n)
    if S(i,i)>lambda
        S_lambda(i,i)=S(i,i)-lambda;
    else
        S_lambda(i,i)=0;
    end
end
X_update = U*S_lambda*V';