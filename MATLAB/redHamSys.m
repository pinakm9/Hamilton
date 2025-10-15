function qdot = redHamSys(t,q,params,J)

A11 = params(1);
A12 = params(2);
A22 = params(3);
B11 = params(4);
B12 = params(5);
B22 = params(6); 

gradV = [(A11*q(1)^3 + A12*q(1)*q(2,:)^2 + B11*q(1) + B12*q(2)); ...
	     (A22*q(2)^3 + A12*q(2)*q(1,:)^2 + B22*q(2) + B12*q(1))];
qdot = -J*gradV;

