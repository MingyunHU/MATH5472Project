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
k_list = [20,50, 70];
for a = 1:3
    k=k_list(a);
    X = zeros(n,p);
    % baseline
    PGD_hard_loss = [];
    iter = 0;
    X_1 = PGD_hard(X,M,W,k,1);
    while Relative_change(Loss_hard(W,M,X_1),Loss_hard(W,M,X))>eps && iter < 200
        PGD_hard_loss(end+1)=log10(Relative_change(Loss_hard(W,M,X_1),Loss_hard(W,M,X)));
        X = PGD_hard(X_1,M,W,k,1);
        iter = iter +1;
        PGD_hard_loss(end+1)=log10(Relative_change(Loss_hard(W,M,X),Loss_hard(W,M,X_1)));
        X_1 = PGD_hard(X,M,W,k,1);
        iter = iter +1;
    end

    % Nesterov
    X_0 = zeros(n,p);
    Nesterov_hard_loss = [];
    iter = 0;
    X_1 = zeros(n,p);
    count = 1;
    X_2 = Nesterov_hard(X_1,X_0,M,W,k,count);
    while Relative_change(Loss_hard(W,M,X_2),Loss_hard(W,M,X_1))>eps && iter < 200
        Nesterov_hard_loss(end+1)=log10(Relative_change(Loss_hard(W,M,X_2),Loss_hard(W,M,X_1)));
        count = count+1;
        X_3 = Nesterov_hard(X_2,X_1,M,W,k,count);
        iter = iter +1;
        X_0 = X_1;
        X_1 = X_2;
        X_2 = X_3;
    end

    % Anderson
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
    [X_1, Y_1, R, F] = Anderson_hard(W,M,X,Y,R,F,k,3);
    while Relative_change(Loss_hard(W,M,Y_1),Loss_hard(W,M,Y))>eps && iter < 200
        Anderson_hard_loss(end+1)=log10(Relative_change(Loss_hard(W,M,Y_1),Loss_hard(W,M,Y)));
        iter = iter + 1;
        [X_2, Y_2, R, F] = Anderson_hard(W,M,X_1,Y_1,R,F,k,3);
        X_1 = X_2;
        Y = Y_1;
        Y_1 = Y_2;
    end
    figure(a);
    plot(PGD_hard_loss,'LineWidth',2)
    hold on
    plot(Nesterov_hard_loss,'LineWidth',2)
    plot(Anderson_hard_loss,'LineWidth',2)
    lgd = legend('Baseline','Nesterov','Anderson');
    xlabel('iteration','FontSize',14);
    ylabel('log(жд)','FontSize',14);
    title(strcat("k=",string(k)),'FontSize',14);
    set(gcf,'color','w');
    ylim([-8 0]);
end