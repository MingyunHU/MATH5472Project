n = 1000;
p = 100;
r = 70;
A = zeros(n,r);
A = normrnd(0,1,size(A));
B = zeros(p,r);
B = normrnd(0,1,size(B));
E = zeros(n,p);
E = normrnd(0,1,size(E));
W = zeros(n,p);
W = rand(size(W));
M = A*B' + E;
eps = 1e-8;
k = 50;
gamma_list = [0,0.1,1,10];
Anderson_loss = cell(1,4);
for a = 1:4
    gamma = gamma_list(a);
    Y = repmat(mean(M),n,1);
    [U,S,V]=svd(Y);
    [m_,n_]=size(S);
    S_k = zeros(m_,n_);
    for i = 1:k
        S_k(i,i)=S(i,i);
    end
    X = U*S_k*V';
    Anderson_hard_loss = [];
    iter = 0;
    R = [];
    F = [];
    alpha_hist = [];
    [X_1, Y_1, R, F,alpha_hist] = Regular_Anderson_hard(W,M,X,Y,R,F,k,3,gamma,alpha_hist);
    while Relative_change(Loss_hard(W,M,Y_1),Loss_hard(W,M,Y))>eps && iter < 200
        Anderson_hard_loss(end+1)=log10(Relative_change(Loss_hard(W,M,Y_1),Loss_hard(W,M,Y)));
        iter = iter + 1;
        [X_2, Y_2, R, F,alpha_hist] = Regular_Anderson_hard(W,M,X_1,Y_1,R,F,k,3,gamma,alpha_hist);
        X_1 = X_2;
        Y = Y_1;
        Y_1 = Y_2;
    end
    Anderson_loss(1,a)={Anderson_hard_loss};
    figure(a);
    plot(alpha_hist','LineWidth',2)
    xlabel('iteration','FontSize',14);
    ylabel('coefficient value','FontSize',14);
    ylim([-10 10]);
    yticks([-10 -5 0 5 10])
    title(strcat("¦Ã=",string(gamma)),'FontSize',14);
end
figure(5);
hold on
for i = 1:4
    plot(Anderson_loss{i},'LineWidth',2)
end
lgd = legend('¦Ã=0','¦Ã=0.1','¦Ã=1','¦Ã=10');
xlabel('iteration','FontSize',14);
ylabel('log(¦¤)','FontSize',14);
set(gcf,'color','w');
title("k=50",'FontSize',14);
ylim([-8 0]);
clear