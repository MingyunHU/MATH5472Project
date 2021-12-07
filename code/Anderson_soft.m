function [X_update,Y_i_new, R_i, F_i] = Anderson_soft(W, M, X_i, Y_i, R_i_1, F_i_1, lambda, maxlen)
temp = W.*M+(ones(size(W))-W).*X_i;
y_i = Y_i(:);
f_i = temp(:);
r_i = f_i-y_i;
R_i = [R_i_1, r_i];
F_i = [F_i_1, f_i];
temp2 = R_i'*R_i;
[m,n]=size(temp2);
alpha = linsolve(temp2,ones(m,1));
alpha = alpha/sum(alpha);
y_i_new = F_i*alpha;
Y_i_new = reshape(y_i_new,size(Y_i));
[U,S,V]=svd(Y_i_new);
[m_,n_]=size(S);
S_lambda = zeros(m_,n_);
for i = 1:min(m_,n_)
    if S(i,i)>lambda
        S_lambda(i,i)=S(i,i)-lambda;
    else
        S_lambda(i,i)=0;
    end
end
[m__, n__]=size(R_i);
if n__>maxlen
    R_i = R_i(:,2:end);
    F_i = F_i(:,2:end);
end
X_update = U*S_lambda*V';