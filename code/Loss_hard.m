function loss = Loss_hard(W, M, X)
sqrt_W = sqrt(W);
loss = norm(sqrt_W.*(M-X),'fro');