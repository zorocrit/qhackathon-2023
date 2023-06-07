import numpy as np
from math import *

#Pauli Matrices(2X2 matrices)
def I():
    I  = np.array([[ 1, 0],
               [ 0, 1]])
    return I
def Sx():
    Sx = np.array([[ 0, 1],
               [ 1, 0]])
    return Sx
def Sy():
    Sy = np.array([[ 0,-1j],
               [1j, 0]])
    return Sy
def Sz():
    Sz = np.array([[ 1, 0],
               [ 0,-1]])
    return Sz

#Rotation operator(X-gate&Z-gate 2X2 matrices)
def Rx(theta):
    return np.matrix([[cos(theta/2),     -1j*sin(theta/2)],
                    [-1j*sin(theta/2),     cos(theta/2)]])
   
def Rz(phi):
    return np.matrix([[e**(-1j*phi/2),       0],
                     [0,                          e**(1j*phi/2)]])
    
#initial state |0> 으로 가정한 상태
def init():
    init = np.matrix([[1],[0]])
    return init

#state a|0> + b|1> 계수를 괄호안에 입력하여 density 구하는 함수
def todensity (a,b):
    UU=np.array([[a],[b]])
    D = UU@(UU.conj().T)
    return D

#target state의 density matrix
idden = np.matrix([[1/2,1/2],
                    [1/2,1/2]])


#problem(cost function)
def problem(deg):
    mc = init()*init().T                                        # |vector><vector|
    gates = np.inner(Rz(deg[1]),Rx(deg[0]))                     # Universal Gate
    #rho_measure는 계산값(측정값)
    rho_measure = gates*mc*gates.getH()                         # Gate|vector><vector|Gate
    x_m = np.trace(rho_measure*Sx())                            # Sigma X projection
    y_m = np.trace(rho_measure*Sy())                            # Sigma Y projection
    z_m = np.trace(rho_measure*Sz())                            # Sigma Z projection
    #x_id,y_id,z_id는 주어진 target state를 계산해낸 값(이론값)
    x_id = np.trace(idden*Sx())                                 # target state의 Sigma X projection
    y_id = np.trace(idden*Sy())                                 # target state의 Sigma Y projection
    z_id = np.trace(idden*Sz())                                 # target state의 Sigma Z projection
    cost = np.abs(x_m-x_id)+np.abs(y_m-y_id)+np.abs(z_m-z_id)   # 실험값과 이론값의 비교 costfunction 반환
    return cost
