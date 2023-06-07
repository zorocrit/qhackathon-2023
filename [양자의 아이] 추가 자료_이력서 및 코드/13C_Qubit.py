import numpy as np
from math import *
import math
from scipy.linalg import fractional_matrix_power
from scipy import linalg

# Define initial state of the system
irho_p = np.array([[1,0,0],[0,0,0],[0,0,0]]) #[0,0,0;0,0,0]
irho_m = np.array([[0,0,0],[0,0,0],[0,0,1]]) #[0,0,0;0,0,1]
irho_z = np.array([[0,0,0],[0,1,0],[0,0,0]]) #[0,1,0;0,0,0]
irho_mix = np.array([[1/2,0,0],[0,1/2,0],[0,0,0]]) #[1/2,0,0;0,1/2,0;0,0,0]
irho_Z = np.array([[0,0,0],[0,0,0],[0,0,1]]) #target state
irho_MIX = np.array([[1/2,0,0],[0,0,0],[0,0,1/2]])
irho = np.kron(irho_z,irho_MIX) #initial state

i = 1j #imaginary number

sx  = 1/sqrt(2)*np.array([[0, 1, 0],[1, 0, 1], [0, 1, 0]])
sy  = 1/sqrt(2)/i*np.array([[0, 1, 0], [-1, 0, 1],[0, -1, 0]])
sz  = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])

#Gellman matrix
Sx  = np.array([[0, 0, 1],[0, 0, 0], [1, 0, 0]])
Sy  = np.array([[0, 0, -i],[0, 0, 0], [i, 0, 0]])
Sz  = np.array([[1, 0, 0],[0, 0, 0], [0, 0, -1]])

# Pauli basis for 13C nuclear spin
Ix  = 1/2*np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])   
Iy  = 1/2/i*np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]])
Iz  = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])

def state_fidelity(rho_1, rho_2): #fidelity
    if np.shape(rho_1) != np.shape(rho_2):
        return 0
    else:
        sqrt_rho_1 = fractional_matrix_power(rho_1, 1 / 2)
        fidelity = np.trace(fractional_matrix_power(sqrt_rho_1 @ rho_2 @ sqrt_rho_1, 1 / 2)) ** 2
        return np.real(fidelity)
        
        
def problem(): #Nuclear(13 C) Gate operation
    rho1 = np.kron(irho_z,irho_MIX)