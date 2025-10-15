%% Hamiltonian slow-fast toy model
%%     q' = p
%% eps p' =  Jp - \nabla V(q)
%%

clear all;

%% Parameters
epsilon = 0.115^2;
epsilon = 0.14^2;
epsilon = 0.117^2;
epsilon = 0.4^2;
epsilon = 0.01;

% 4-dip potential 
% V(q) = 0.25 (A11 q11^4 + 2 A12 q1^2 q2^2 + A22 q2^4) 
%      + 0.5 (B11 q1^2 + 2 B12 q1 q2 + B22 q2^2)
A11 = 1; A12 = 0; A22 = 1; B11 = -3 ; B12 = 0; B22 = -1.5;
%A11 = 1; A12 = 0; A22 = 1; B11 = 0 ; B12 = 0; B22 = 0;
A11 = 6; A12 = 0; A22 = 0.1; B11 = -4.5; B12 = 0; B22 = -4.5; 

params = [A11, A12, A22, B11, B12, B22, epsilon];

J = [0 1; -1 0];

%% Integration of full system
%x0  = [-0.5; 0.837; 0.5; 0.3;];
% balanced initial conditions
x0(1:2) = [0, -9.8];
%x0(1:2) = [0, -2];
gradV0 = [(A11*x0(1)^3 + A12*x0(1)*x0(2)^2 + B11*x0(1) + B12*x0(2)); ...
	      (A22*x0(2)^3 + A12*x0(2)*x0(1)^2 + B22*x0(2) + B12*x0(1))];
x0(3:4) = -J*gradV0;

options = odeset('RelTol',1e-8,'AbsTol',1e-8);

Tend = 20;
dts = 0.001;

% Parent Equation
odefun = @(t,x) fullHamSys(t,x,params);
[T,X]  = ode45(odefun, [0:dts:Tend], x0, options);
q = X(:,1:2);
p = X(:,3:4);
q = q';
p = p';

%% Integration of the reduced system

% reduced Equation
x0red  = x0(1:2);
odefun = @(t,x) redHamSys(t,x,params,J);
[T,X]  = ode45(odefun, [0:dts:Tend], x0red, options);
qred = X(:,1:2);
qred = qred';

%% Balance diagnostics
gradV = [(A11*q(1,:).^3 + A12*q(1,:).*q(2,:).^2 + B11*q(1,:) + B12*q(2,:)); ...
	     (A22*q(2,:).^3 + A12*q(2,:).*q(1,:).^2 + B22*q(2,:) + B12*q(1,:))];

div = p(1:2,:) + J*gradV;
Div = diag(div'*div);

%% Plots
figure(1)
 plot(qred(1,:),qred(2,:),'r-','LineWidth',3)
 hold on
 plot(q(1,:),q(2,:),'b-','LineWidth',1)
 hold off
 xlabel('$q_1$','Fontsize',30,'Interpreter','latex'); 
 ylabel('$q_2$','Fontsize',30,'Interpreter','latex');
 set(gca,'FontSize',25)

figure(2) 
 histogram(Div,100,'Normalization','pdf');
 xlabel('$Div$','Fontsize',30,'Interpreter','latex');
 set(gca,'FontSize',25) 
