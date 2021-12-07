function [A_update, B_update] = ALS_Nesterov_soft(W, M, A_i,A_j, B_i,B_j,i, lambda)
V_A = A_i + (i-1)/(i+2)*(A_i-A_j);
V_B = B_i + (i-1)/(i+2)*(B_i-B_j);
[A_update, B_update] = ALS_soft(W,M,V_A,V_B,lambda);