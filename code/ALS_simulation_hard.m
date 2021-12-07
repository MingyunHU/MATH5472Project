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
A_0 = rand(n,r);
B_0 = rand(p,r);
eps = 1e-8;

% baseline
ALS_hard_loss = [];
iter = 0;
[A_1, B_1] = ALS_hard(W, M,A_0,B_0);
while Relative_change(Loss_hard(W,M,A_1*B_1'),Loss_hard(W,M,A_0*B_0'))>eps && iter < 200
    ALS_hard_loss(end+1)=log10(Relative_change(Loss_hard(W,M,A_1*B_1'),Loss_hard(W,M,A_0*B_0')));
    [A_0, B_0] = ALS_hard(W, M,A_1,B_1);
    iter = iter +1;
    ALS_hard_loss(end+1)=log10(Relative_change(Loss_hard(W,M,A_0*B_0'),Loss_hard(W,M,A_1*B_1')));
    [A_1, B_1] = ALS_hard(W, M,A_0,B_0);
    iter = iter +1;
end
plot(ALS_hard_loss)
hold on

% Nesterov
A_0 = rand(n,r);
B_0 = rand(p,r);
A_1 = rand(n,r);
B_1 = rand(p,r);
ALS_Nesterov_hard_loss = [];
iter = 0;
count = 1;
[A_2,B_2] = ALS_Nesterov_hard(W,M,A_1,A_0,B_1,B_0,count);
while Relative_change(Loss_hard(W,M,A_2*B_2'),Loss_hard(W,M,A_1*B_1'))>eps && iter < 200
    ALS_Nesterov_hard_loss(end+1)=log10(Relative_change(Loss_hard(W,M,A_2*B_2'),Loss_hard(W,M,A_1*B_1')));
    count = count+1;
    [A_0, B_0] = ALS_Nesterov_hard(W,M,A_2,A_1,B_2,B_1,count);
    iter = iter +1;
    A_1 = A_2;
    B_1 = B_2;
    A_2 = A_0;
    B_2 = B_0;
end
plot(ALS_Nesterov_hard_loss)

% Anderson
A_0 = rand(n,r);
B_0 = rand(p,r);
Z = [A_0;B_0];
ALS_Anderson_hard_loss = [];
iter = 0;
R_A = [];
R_B = [];
F_A = [];
F_B = [];
[Z_1, R_A,R_B, F_A, F_B] = ALS_Anderson_hard(W,M,Z,R_A,R_B,F_A,F_B,3,n);
A_1 = Z_1(1:n,:);
B_1 = Z_1(n+1:end,:);
while Relative_change(Loss_hard(W,M,A_1*B_1'),Loss_hard(W,M,A_0*B_0'))>eps && iter < 200
    ALS_Anderson_hard_loss(end+1)=log10(Relative_change(Loss_hard(W,M,A_1*B_1'),Loss_hard(W,M,A_0*B_0')));
    iter = iter + 1;
    [Z_2, R_A,R_B, F_A, F_B] = ALS_Anderson_hard(W,M,Z_1,R_A,R_B,F_A,F_B,3,n);
    Z = Z_1;
    A_0 = Z(1:n,:);
    B_0= Z(n+1:end,:);
    Z_1 = Z_2;
    A_1 = Z_1(1:n,:);
    B_1= Z_1(n+1:end,:);
end
plot(ALS_Anderson_hard_loss)