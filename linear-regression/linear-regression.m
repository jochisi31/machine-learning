% limpiamos la ventana de comandos
clear;
close all;
clc;

% cargamos los datos en una matriz
data = load('simple.csv');

% obtener X , y en vectores
X = data(:,1);
y = data(:,2);

% obtenemos el numero de ejemplos de nuestros datos
m = length(y);

% aplicamos feature scaling
X = (X-(sum(X)/length(X)))./(max(X)-min(X));
y = (y-(sum(y)/length(y)))./(max(y)-min(y));

% visualizamos los datos en un plot
figure;
plot(X, y, 'rx', 'MarkerSize',12);
ylabel('precio');
xlabel('tamaño');


% calcular la gradiente descendiente 
alpha = 0.2;
iteraciones = 200;
theta = zeros(2, 1);
theta(2,1)=-15;
derivada0 = 0;
derivada1 = 0;
J = zeros(iteraciones, 2);

for i=1:iteraciones
  
  costo = 0;
  
  for j=1:m
    % derivada theta0
    h = theta(1) + theta(2) * X(j);
    derivada0 = derivada0 + (h-y(j))
    derivada1 = derivada1 + (h-y(j))*X(j)
    
    costo = costo + (h-y(j)).^2
    
    endfor;
  
  %printf('costo %d \n' , costo);
  derivada0 = derivada0 / m;
  derivada1 = derivada1 / m;
  J(i, 1) = theta(2);
  J(i, 2) = costo;
%J(i, 3) = iteraciones;

  % theta0 = theta0 - alpha * derivada de J
  %theta(1) = theta(1) - (alpha * derivada0);

  % theta1 = theta1 - alpha * derivada de J
  theta(2) = theta(2) - (alpha * derivada1);
  
endfor

% visualizamos los datos de predicción de nuestro algoritmo
hold on;
X= [ones(size(X)) X];
plot(X(:,2), X*theta, '-', 'linewidth', 2);
ylabel('precio');
xlabel('tamaño');



% visualizamos como se comporta el costo con respecto a theta
figure(2);
plot(J(:, 1), J(:, 2));
ylabel('costo');
xlabel('theta');

% visualizamos como se comporta el costo con respecto al número de iteraciones
figure(3);
plot(J(:, 2));
ylabel('costo');
xlabel('iteraciones');

pause;