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
lambda_list = [5,30, 100];

for a = 1:3
    lambda = lambda_list(a);
    % baseline
    X = zeros(n,p);
    PGD_soft_loss = [];
    iter = 0;
    X_1 = PGD_soft(X,M,W,lambda,1);
    while Relative_change(Loss_soft(W,M,X_1,lambda),Loss_soft(W,M,X,lambda))>eps && iter < 200
        PGD_soft_loss(end+1)=log10(Relative_change(Loss_soft(W,M,X_1,lambda),Loss_soft(W,M,X,lambda)));
        X = PGD_soft(X_1,M,W,lambda,1);
        iter = iter +1;
        PGD_soft_loss(end+1)=log10(Relative_change(Loss_soft(W,M,X,lambda),Loss_soft(W,M,X_1,lambda)));
        X_1 = PGD_soft(X,M,W,lambda,1);
        iter = iter +1;
    end

    % Nesterov
    X_0 = zeros(n,p);
    Nesterov_soft_loss = [];
    iter = 0;
    X_1 = zeros(n,p);
    count = 1;
    X_2 = Nesterov_soft(X_1,X_0,M,W,lambda,count);
    while Relative_change(Loss_soft(W,M,X_2,lambda),Loss_soft(W,M,X_1,lambda))>eps && iter < 200
        Nesterov_soft_loss(end+1)=log10(Relative_change(Loss_soft(W,M,X_2,lambda),Loss_soft(W,M,X_1,lambda)));
        count = count+1;
        X_3 = Nesterov_soft(X_2,X_1,M,W,lambda,count);
        iter = iter +1;
        X_0 = X_1;
        X_1 = X_2;
        X_2 = X_3;
    end

    % Anderson
    Y = repmat(mean(M),n,1);
    [U,S,V]=svd(Y);
    [m_,n_]=size(S);
    S_lambda = zeros(m_,n_);
    for i = 1:min(m_,n_)
        if S(i,i)>lambda
            S_lambda(i,i)=S(i,i)-lambda;
        else
            S_lambda(i,i)=0;
        end
    end
    X = U*S_lambda*V';
    Anderson_soft_loss = [];
    iter = 0;
    R = [];
    F = [];
    [X_1, Y_1, R, F] = Anderson_soft(W,M,X,Y,R,F,lambda,3);
    while Relative_change(Loss_soft(W,M,Y_1,lambda),Loss_soft(W,M,Y,lambda))>eps && iter < 200
        Anderson_soft_loss(end+1)=log10(Relative_change(Loss_soft(W,M,Y_1,lambda),Loss_soft(W,M,Y,lambda)));
        iter = iter + 1;
        [X_2, Y_2, R, F] = Anderson_soft(W,M,X_1,Y_1,R,F,lambda,3);
        X_1 = X_2;
        Y = Y_1;
        Y_1 = Y_2;
    end
    figure(a);
    plot(PGD_soft_loss,'LineWidth',2)
    hold on
    plot(Nesterov_soft_loss,'LineWidth',2)
    plot(Anderson_soft_loss,'LineWidth',2)
    lgd = legend('Baseline','Nesterov','Anderson');
    xlabel('iteration','FontSize',14);
    ylabel('log(¦¤)','FontSize',14);
    title(strcat("¦Ë=",string(lambda)),'FontSize',14);
    set(gcf,'color','w');
    ylim([-8 0])
end