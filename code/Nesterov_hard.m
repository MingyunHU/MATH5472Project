function X_update = Nesterov_hard(X_i,X_j, M, W, k, i)
V_i = X_i + (i-1)/(i+2)*(X_i-X_j);
[U,S,V]=svd(W.*M+(ones(size(W))-W).*V_i);
[m,n]=size(S);
S_k = zeros(m,n);
for j = 1:k
    S_k(j,j)=S(j,j);
end
X_update = U*S_k*V';