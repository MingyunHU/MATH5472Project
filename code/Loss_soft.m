function loss = Loss_soft(W, M, X, lambda)
sqrt_W = sqrt(W);
loss = norm(sqrt_W.*(M-X),'fro')/2+lambda*norm(svd(X),1);