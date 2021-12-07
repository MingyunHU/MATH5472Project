function [A_update, B_update] = ALS_hard(W, M, A, B)
X_i = A*B';
Y_new = W.*M+(ones(size(W))-W).*X_i;
B_update = Y_new'*A/(A'*A);
X_i = A*B_update';
Y_new = W.*M+(ones(size(W))-W).*X_i;
A_update = Y_new*B_update/(B_update'*B_update);