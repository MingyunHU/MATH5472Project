load('MovieLens.mat')
M = MovieLens;
[m,n]=size(M);
W = zeros(size(M));
for i = 1:m
    for j = 1:n
        if M(i,j)~=0
            W(i,j)=1;
        end
    end
end
rank = 70;
lambda = 40;
eps = 1e-8;
A = zeros(m,rank);
A_init = normrnd(0,1,size(A));
B = zeros(n,rank);
B_init = normrnd(0,1,size(B));
[A_sec, B_sec] = ALS_hard(W, M, A_init, B_init);

% Baseline
ALS_soft_loss = [];
iter = 0;
A_0 = A_sec;
B_0 = B_sec;
[A_1, B_1] = ALS_soft(W, M,A_0,B_0, lambda);
while Relative_change(ALS_Loss_soft(W,M,A_1,B_1,lambda),ALS_Loss_soft(W,M,A_0,B_0, lambda))>eps && iter < 200
    ALS_soft_loss(end+1)=log10(Relative_change(ALS_Loss_soft(W,M,A_1,B_1, lambda),ALS_Loss_soft(W,M,A_0,B_0, lambda)));
    [A_0, B_0] = ALS_soft(W, M,A_1,B_1, lambda);
    iter = iter +1;
    ALS_soft_loss(end+1)=log10(Relative_change(ALS_Loss_soft(W,M,A_0,B_0, lambda),ALS_Loss_soft(W,M,A_1,B_1, lambda)));
    [A_1, B_1] = ALS_soft(W, M,A_0,B_0, lambda);
    iter = iter +1;
end
plot(ALS_soft_loss,'LineWidth',2)
hold on

% Nesterov
A_0 = A_init;
B_0 = B_init;
A_1 = A_sec;
B_1 = B_sec;
ALS_Nesterov_soft_loss = [];
iter = 0;
count = 1;
[A_2,B_2] = ALS_Nesterov_soft(W,M,A_1,A_0,B_1,B_0,count, lambda);
while Relative_change(ALS_Loss_soft(W,M,A_2,B_2,lambda),ALS_Loss_soft(W,M,A_1,B_1,lambda))>eps && iter < 200
    ALS_Nesterov_soft_loss(end+1)=log10(Relative_change(ALS_Loss_soft(W,M,A_2,B_2,lambda),ALS_Loss_soft(W,M,A_1,B_1,lambda)));
    count = count+1;
    [A_0, B_0] = ALS_Nesterov_soft(W,M,A_2,A_1,B_2,B_1,count,lambda);
    iter = iter +1;
    A_1 = A_2;
    B_1 = B_2;
    A_2 = A_0;
    B_2 = B_0;
end
plot(ALS_Nesterov_soft_loss,'LineWidth',2)

% Anderson
A_0 = A_sec;
B_0 = B_sec;
Z = [A_0;B_0];
ALS_Anderson_soft_loss = [];
iter = 0;
R_A = [];
R_B = [];
F_A = [];
F_B = [];
[Z_1, R_A,R_B, F_A, F_B] = ALS_Anderson_soft(W,M,Z,R_A,R_B,F_A,F_B,3,m,lambda);
A_1 = Z_1(1:m,:);
B_1 = Z_1(m+1:end,:);
while Relative_change(ALS_Loss_soft(W,M,A_1,B_1,lambda),ALS_Loss_soft(W,M,A_0,B_0,lambda))>eps && iter < 200
    ALS_Anderson_soft_loss(end+1)=log10(Relative_change(ALS_Loss_soft(W,M,A_1,B_1,lambda),ALS_Loss_soft(W,M,A_0,B_0,lambda)));
    iter = iter + 1;
    [Z_2, R_A,R_B, F_A, F_B] = ALS_Anderson_soft(W,M,Z_1,R_A,R_B,F_A,F_B,3,m,lambda);
    Z = Z_1;
    A_0 = Z(1:m,:);
    B_0= Z(m+1:end,:);
    Z_1 = Z_2;
    A_1 = Z_1(1:m,:);
    B_1= Z_1(m+1:end,:);
end
plot(ALS_Anderson_soft_loss,'LineWidth',2)
lgd = legend('Baseline','Nesterov','Anderson');
xlabel('iterations','FontSize',14);
ylabel('log(¦¤)','FontSize',14);
title(strcat("¦Ë=",string(lambda)),'FontSize',14);
set(gcf,'color','w');
ylim([-8 0])