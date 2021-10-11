[X1,X2] = meshgrid(-2:.2:2); 
Y = (X1).^2 + (X2).^2;
%G = -2*X1 + 2*X2 - 1;
figure(1)
%grid on
hold on
surf(X1,X2,Y)
%surf(X1,X2,G)
