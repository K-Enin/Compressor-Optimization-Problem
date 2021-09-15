% Solving Power Model given in PowerModel.m
global m margin_lower margin_upper P_beg P_end Q_beg Q_end P_at_t Q_at_t;
m = 600;
margin_upper = m/2 + 50;
margin_lower = m/2;
%P_beg = -0.9; 
%P_end = -1.8; 
Q_beg = -0.3; 
Q_end = -0.6;

file = load('P5_file');
P_at_N5 = file.P5_node;
%P_at_N5 = generate_P_at_N5();
Q_at_N5 = generate_Q_at_N5();

plot_N5 = true

if plot_N5 == true
    x_a = 0:1:(m-1);
    fig1 = figure;
    plot(x_a,P_at_N5)
    title('P at Node N5')
    xlabel('time t')
    ylabel('P (Watt)')
    fig2 = figure;
    plot(x_a,Q_at_N5)
    title('Q at Node N5')
    xlabel('time t')
    ylabel('Q (Voltage-Ampere-Reactive)')
end

solution_matrix = zeros(18,m);

fun = @PowerModel;
x0 = [0, 0.2, 0.2, 0.2, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0];

for t = 1:m
    P_at_t = P_at_N5(t);
    Q_at_t = Q_at_N5(t);
    options = optimoptions('fsolve','Algorithm','levenberg-marquardt')
    x = fsolve(fun, x0, options)
    solution_matrix(:,t) = x;
end

fig3 = figure;
x_b = 0:1:(m-1);
plot(x_b, solution_matrix(1,:))
title('Results for fsolve for P in Node N1')
xlabel('time t')
ylabel('Real Power P (per unit)')

fig4 = figure;
plot(x_b, solution_matrix(2,:),'--')
%title('Results for fsolve for Q in Node N1')
xlabel('time steps')
ylabel('Q (VAR)')

% Transform P into flow eps
a0 = 0;
a1 = -0.5;
a2 = 10;

eps = a0 + a1*solution_matrix(1,:) + a2*solution_matrix(1,:).*solution_matrix(1,:);
fig5 = figure;
plot(x_b, eps)
%title('Flow eps')
xlabel('time steps')
ylabel('Mass flow')

savdir  = '/Users/katharinaenin/Desktop/Masterarbeit/Code/Optimierungsproblem/eps_file'
save(savdir, 'eps')