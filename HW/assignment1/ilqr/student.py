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
        gravity = self.env.gravity
        masspole = self.env.masspole
        total_mass = self.env.total_mass
        length = self.env.length
        force = u
        dt = self.env.tau
        A = np.eye(4)

        # A = ...

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

        B = np.zeros((4,1))

        # B = ...

        return B



def lqr(A, B, T=100):
    # by prof
    K = np.zeros((1,4))
    # by my-self
    Q = np.eye(4)  # 2 state var ??
    R = np.eye(1)  # 1 input var ??
    '''
    by reading the env code, these are the dim of x and u
    btw, thinking about the problem seems like x = [1*4]
    '''
    P = np.zeros((4, 4))
    for t in range(T):
        K = -inv(R + B.T @ P @ B) @ B.T @ P @ A
        P = (Q + K.T @ R @ K + (A + B @ K).T @ P @ (A + B @ K))


    return K
