function [Z_i_new, R_i_A, R_i_B, F_i_A, F_i_B] = ALS_Anderson_hard(W, M, Z_i, R_i_1_A,R_i_1_B, F_i_1_A,F_i_1_B, maxlen, len_A)
A = Z_i(1:len_A,:);
B = Z_i(len_A+1:end,:);
Y = W.*M+(ones(size(W))-W).*(A*B');
B_update = Y'*A/(A'*A);
b_i = B(:);
f_b_i = B_update(:);
r_b_i = f_b_i-b_i;
R_i_B = [R_i_1_B, r_b_i];
F_i_B = [F_i_1_B, f_b_i];
temp2 = R_i_B'*R_i_B;
[m,n]=size(temp2);
alpha_B = linsolve(temp2,ones(m,1));
alpha_B = alpha_B/sum(alpha_B);
b_i_new = F_i_B*alpha_B;
B_i_new = reshape(b_i_new,size(B));
[m__, n__]=size(R_i_B);
if n__>maxlen
    R_i_B = R_i_B(:,2:end);
    F_i_B = F_i_B(:,2:end);
end
Y = W.*M+(ones(size(W))-W).*(A*B_update');
A_update = Y*B_update/(B_update'*B_update);
a_i = A(:);
f_a_i = A_update(:);
r_a_i = f_a_i-a_i;
R_i_A = [R_i_1_A, r_a_i];
F_i_A = [F_i_1_A, f_a_i];
temp2 = R_i_A'*R_i_A;
[m,n]=size(temp2);
alpha_A = linsolve(temp2,ones(m,1));
alpha_A = alpha_A/sum(alpha_A);
a_i_new = F_i_A*alpha_A;
A_i_new = reshape(a_i_new,size(A));
[m__, n__]=size(R_i_A);
if n__>maxlen
    R_i_A = R_i_A(:,2:end);
    F_i_A = F_i_A(:,2:end);
end
Z_i_new = [A_i_new;B_i_new];