import numpy as np
from scipy import linalg
from numpy import sin, cos
from numpy.linalg import inv

class CartPole:
    def __init__(self, env, x=None,max_linear_velocity=2, max_angular_velocity=np.pi/3):
        if x is None:
            x = np.zeros(2)
        self.env = env
        
    def getA(self, u, x=None):
        if x is None:
            x = self.x
        x, x_dot, theta, theta_dot = x
        polemass_length = self.env.polemass_length
        g = self.env.gravity
        masspole = self.env.masspole
        total_mass = self.env.total_mass
        length = self.env.length
        force = u
        dt = self.env.tau

        theta_2dot = ( (g*sin(theta))-(cos(theta))*((force
                    + ( polemass_length*(theta_dot**2)*(sin(theta)) ) )
                    / (total_mass) ) )/ ( length* ( (4/3) - ((masspole*((cos(theta)**2))) / total_mass) ) )

        A = np.eye(4)
        A[0, 1] = dt
        A[1, 1] = 1
        A[1, 2] = (polemass_length*dt*(12*(total_mass**2)*g - 12*total_mass*force*sin(2*theta) - 24*(total_mass**2)*g*(cos(theta)**2)
                                       + 16*length*(total_mass**2)*(theta_dot**2)*cos(theta) + 36*polemass_length*total_mass*(theta_dot**2)*(cos(theta)**3)
                                       - 9*polemass_length*masspole*(theta_dot**2)*(cos(theta)**5) + 9*length*(masspole**2)*(theta_dot**2)*(cos(theta)**5)
                                       + 9*total_mass*masspole*g*(cos(theta)**2) - 24*polemass_length*total_mass*(theta_dot**2)*cos(theta)
                                       - 24*length*total_mass*masspole*(theta_dot**2)*(cos(theta)**3)))/(length*total_mass*(4*total_mass - 3*masspole*(cos(theta)**2))**2)
        A[2, 2] = 1
        A[3, 2] = (3*dt*(- polemass_length*(2*(cos(theta)**2) - 1)*(theta_dot**2) + force*sin(theta) + total_mass*g*cos(theta)))\
                  /(length*(4*total_mass - 3*masspole*(cos(theta)**2))) + \
                  (18*masspole*dt*cos(theta)*sin(theta)*(polemass_length*cos(theta)*sin(theta)*(theta_dot**2) + force*cos(theta) - total_mass*g*sin(theta)))\
                  /(length*(4*total_mass - 3*masspole*(cos(theta)**2))**2)


        A[1, 3] = (2*polemass_length*dt*theta_dot*sin(theta)*(3*polemass_length*(cos(theta)**2) + 4*length*total_mass -
                                                              3*length*masspole*(cos(theta)**2)))/(length*total_mass*(4*total_mass - 3*masspole*(cos(theta)**2)))
        A[2, 3] = dt
        A[3, 3] = (2*polemass_length*dt*theta_dot*cos(theta)*sin(theta))/(length*total_mass*((masspole*(cos(theta)**2))/total_mass - 4/3)) + 1
        return A
        
    def getB(self, x=None):
        if x is None:
            x = self.x
        
        x, x_dot, theta, theta_dot = x
        polemass_length = self.env.polemass_length
        gravity = self.env.gravity
        masspole = self.env.masspole
        total_mass = self.env.total_mass
        length = self.env.length
        dt = self.env.tau

        B = np.zeros((4, 1))
        B[1] = - (dt*((polemass_length*(cos(theta)**2))/(length*total_mass*((polemass_length*(cos(theta)**2))/total_mass - 4/3)) - 1))/total_mass
        B[3] = -(3*dt*cos(theta)) / (length*(4*total_mass - 3*masspole*(cos(theta)**2)))
        return B



def lqr(A, B, T=100):
    # by prof
    K = np.zeros((1,4))
    # by my-self
    Q = np.eye(4)
    R = np.eye(1)
    P = np.zeros((4, 4))
    for t in range(T):
        K = -inv(R + B.T @ P @ B) @ B.T @ P @ A
        P = (Q + K.T @ R @ K + (A + B @ K).T @ P @ (A + B @ K))
    return K
