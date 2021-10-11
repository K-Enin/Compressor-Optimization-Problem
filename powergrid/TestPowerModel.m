% Solving Power Model given in PowerModel.m
global m margin_lower margin_upper P_beg P_end Q_beg Q_end P_at_t Q_at_t;
m = 300;
margin_upper = m/2 + 50;
margin_lower = m/2;
P_beg = -0.9; 
P_end = -1.8; 
Q_beg = -0.3; 
Q_end = -0.6;

file = load('P5_file');
%P_at_N5 = file.P5_node;
P_at_N5 = generate_P_at_N5(); % takes the values given above
Q_at_N5 = generate_Q_at_N5();

plot_N5 = true

if plot_N5 == true
    x_a = 0:1:(m-1);
    fig1 = figure;
    plot(x_a,-P_at_N5,'LineWidth',2.0)
    title('Real Power P at Node N5', 'FontSize',14)
    xlabel('time steps', 'FontSize',14)
    ylabel('P (per unit)', 'FontSize',14)
    axis([0 300 0.7 2])
    fig2 = figure;
    plot(x_a,-Q_at_N5,'LineWidth',2.0)
    title('Reactive Power Q at Node N5', 'FontSize',14)
    xlabel('time steps', 'FontSize',14)
    ylabel('Q (per unit)', 'FontSize',14)
    axis([0 300 0.2 0.8])
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
plot(x_b, solution_matrix(1,:),'LineWidth',2.0) %non-negative
title('Results after applying fsolve: Real Power P in Node N1', 'FontSize',14)
xlabel('time steps', 'FontSize',14)
ylabel('P (per unit)','FontSize',14)
axis([0 300 0.6 1.8])

fig4 = figure;
plot(x_b, solution_matrix(2,:),'LineWidth',2.0)
title('Results after applying fsolve: Reactive Power Q in Node N1', 'FontSize',14)
xlabel('time steps', 'FontSize',14)
ylabel('Q (per unit)', 'FontSize',14)

% Transform P into flow eps
a0 = 0  %;2;
a1 = -0.5; %5;
a2 = 10; %10;

eps = a0 + a1*solution_matrix(1,:) + a2*solution_matrix(1,:).*solution_matrix(1,:);
fig5 = figure;
plot(x_b, eps,'LineWidth',2.0)
title('flow $\epsilon$','Interpreter','latex', 'FontSize',15)
xlabel('time steps', 'FontSize',14)
ylabel('mass flow ($\frac{kg}{s}$)','Interpreter','latex', 'FontSize',15)

%savdir  = '/Users/katharinaenin/Desktop/Masterarbeit/Code/Optimierungsproblem/eps_files/eps_file3'
%save(savdir, 'eps')