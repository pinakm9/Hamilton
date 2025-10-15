function qdot = fullHamSys(t,q,params)

A11 = params(1);
A12 = params(2);
A22 = params(3);
B11 = params(4);
B12 = params(5);
B22 = params(6); 
epsilon = params(7);

qdot = [q(3);
        q(4);
        (-A11*q(1)^3-A12*q(1)*q(2)^2-B11*q(1)-B12*q(2)+q(4))/epsilon;
	    (-A22*q(2)^3-A12*q(2)*q(1)^2-B22*q(2)-B12*q(1)-q(3))/epsilon];