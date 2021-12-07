function loss = ALS_Loss_soft(W, M, A, B, lambda)
loss = sum(sum(W.*((M-A*B').^2)))/2;
loss = loss + lambda/2*(sum(sum(A.^2))+sum(sum(B.^2)));
[m,n]=size(M);
loss = loss/(m*n);