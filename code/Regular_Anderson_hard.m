function [X_update,Y_i_new, R_i, F_i, alpha_list] = Regular_Anderson_hard(W, M, X_i, Y_i, R_i_1, F_i_1, k, maxlen, gamma, alpha_list)
[m___,n___] = size(alpha_list);
temp = W.*M+(ones(size(W))-W).*X_i;
y_i = Y_i(:);
f_i = temp(:);
r_i = f_i-y_i;
R_i = [R_i_1, r_i];
F_i = [F_i_1, f_i];
if n___<maxlen
    temp2 = R_i'*R_i;
    [m,n]=size(temp2);
    alpha = linsolve(temp2,ones(m,1));
    alpha = alpha/sum(alpha);
    [m_alpha, n_alpha]= size(alpha);
    if m_alpha == maxlen+1
        alpha_list(:,end+1)=alpha;
    end
else
    alpha_pre = mean(alpha_list(:,end-2:end),2);
    temp2 = R_i'*R_i;
    temp2 = temp2+gamma*eye(size(temp2));
    [m,n]=size(temp2);
    alpha = linsolve(temp2,ones(m,1));
    alpha = alpha/sum(alpha);
    K = temp2\(alpha_pre*ones(1,m)-ones(m,1)*alpha_pre');
    alpha = (eye(size(temp2))+gamma*K)*alpha;
    alpha_list(:,end+1)=alpha;
end
y_i_new = F_i*alpha;
Y_i_new = reshape(y_i_new,size(Y_i));
[U,S,V]=svd(Y_i_new);
[m_,n_]=size(S);
S_k = zeros(m_,n_);
for i = 1:k
    S_k(i,i)=S(i,i);
end
[m__, n__]=size(R_i);
if n__>maxlen
    R_i = R_i(:,2:end);
    F_i = F_i(:,2:end);
end
X_update = U*S_k*V';