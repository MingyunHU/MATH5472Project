function X_update = PGD_soft(X, M, W, lambda, t)
[U,S,V]=svd(t*W.*M+(ones(size(W))-t*W).*X);
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