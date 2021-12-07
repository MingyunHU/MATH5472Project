function X_update = PGD_hard(X, M, W, k, t)
[U,S,V]=svd(t*W.*M+(ones(size(W))-t*W).*X);
[m,n]=size(S);
S_k = zeros(m,n);
for i = 1:k
    S_k(i,i)=S(i,i);
end
X_update = U*S_k*V';