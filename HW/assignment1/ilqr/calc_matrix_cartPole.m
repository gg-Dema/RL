clear all;
syms x x_dot  theta theta_dot ;
syms dt u L_p L g M_p M; 


theta_2dot = ( (g*sin(theta)) - (cos(theta)) * ...
        ( (u + ( L_p*(theta_dot^2)*(sin(theta)) ) )  ...
        / (M) )...
    ) ...
    / ( L* ( (4/3) - ((M_p*(cos(theta)^2)) / M) ) );
x_2dot =  ((u + (L_p*(theta_dot^2)*sin(theta) ) - (L_p*theta_2dot*cos(theta) ) ) /(M) );
    
x_2dot = simplify(x_2dot);
theta_2dot = simplify(theta_2dot);

x_1 = x + x_dot*dt; 
x_dot_1 = x_dot + dt* x_2dot; 
theta_1 = theta + dt*theta_dot; 
theta_dot_1 =  theta_dot + dt*theta_2dot;

x_dot_1 = simplify(x_dot_1);
theta_dot_1 = simplify(theta_dot_1);

state = [x_1, x_dot_1, theta_1, theta_dot_1];
A = jacobian(state, [x, x_dot, theta, theta_dot]); 
A = simplify(A);

B = jacobian(state, u); 
B = simplify(B); 