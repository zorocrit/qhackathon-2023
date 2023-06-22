import math

from toqito.channels import partial_trace


import numpy as np
import pandas as pd
from scipy import linalg


def operateUnitary(coeff1, coeff2, angle, param1, param2):
    """
    Args:
        coeff1 (float): ?
        coeff2 (float): ?
        angle (float): Angle parameter
        param1 (float): ?
        param2 (float): ?

    Result:
        np.ndarray: Unitary operator matrix
"""
    i = 1j

    UA = [
        [(coeff2**2 + coeff1**2 * np.cos(angle)) / (coeff1**2 + coeff2**2),
         -i * coeff1 * (np.exp(-i * param1)) * np.sin(angle) /
         np.sqrt(coeff1**2 + coeff2**2),
         ((-1 + np.cos(angle)) * coeff1 * coeff2 * (np.exp(-i * (param1 - param2)))) / (coeff1**2 + coeff2**2)],
        [-i * coeff1 * (np.exp(i * param1)) * np.sin(angle) / np.sqrt(coeff1**2 + coeff2**2),
         np.cos(angle),
         -i * coeff2 * (np.exp(i * param2)) * np.sin(angle) / np.sqrt(coeff1**2 + coeff2**2)],
        [((-1 + np.cos(angle)) * coeff1 * coeff2 * np.exp(i * (param1 - param2))) / (coeff1**2 + coeff2**2),
         -i * coeff2 * (np.exp(-i * param2)) * np.sin(angle) /
         np.sqrt(coeff1**2 + coeff2**2),
         (coeff1**2 + coeff2**2 * np.cos(angle)) / (coeff1**2 + coeff2**2)]
    ]
    return UA


i = 1j
sigmaX = 1 / np.sqrt(2) * np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
sigmaY = 1 / np.sqrt(2) / i * np.array([[0, 1, 0], [-1, 0, 1], [0, -1, 0]])
sigmaZ = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])
identity = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])


# 2levels
pSigmaX = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
msigmaX = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]])
pSigmaY = 1 / i * np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])
mSigmaY = 1 / i * np.array([[0, 0, 0], [0, 0, 1], [0, -1, 0]])
pSigmaZ = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]])


# Getting Gellmann Matrix
sigmaX4Gellmann = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])
sigmaY4Gellmann = np.array([[0, 0, -i], [0, 0, 0], [i, 0, 0]])
sigmaZ4Gellmann = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])


# Pauli
Ix = 1 / 2 * np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])
Iy = 1 / 2 / i * np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]])
Iz = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])


# Sweep Parameters
sweep = 1001
N = sweep
B = 403  # [G] magnetic field

T = 5  # sweep tau [us]
t = np.linspace(0, T, N)
n = 32  # number of pi pulses


# Single Q ms +1
U90xp = operateUnitary(1, 0, np.pi / 4, 0, 0)
U90xmp = operateUnitary(1, 0, -np.pi / 4, 0, 0)
U90yp = operateUnitary(1, 0, np.pi / 4, np.pi / 2, 0)
U90ymp = operateUnitary(1, 0, -np.pi / 4, np.pi / 2, 0)
U180xp = operateUnitary(1, 0, np.pi / 2, 0, 0)
U180xmp = operateUnitary(1, 0, -np.pi / 2, 0, 0)


# ms -1
U90xm = operateUnitary(0, 1, np.pi / 4, 0, 0)
U90xmm = operateUnitary(0, 1, -np.pi / 4, 0, 0)
U180xm = operateUnitary(0, 1, np.pi / 2, 0, 0)
U180xmm = operateUnitary(0, 1, np.pi / 2, 0, 0)

# initialize
initial_rho_p = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
initial_rho_m = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])
initial_rho_z = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
initial_rho_mix = np.array([[1 / 2, 0, 0], [0, 1 / 2, 0], [0, 0, 0]])

initial_rho_Z = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])
initial_rho_MIX = np.array([[1 / 2, 0, 0], [0, 0, 0], [0, 0, 1 / 2]])

initial_rho = np.kron(initial_rho_z, initial_rho_MIX)  # initial state

trace = [1, 1, 0, 100, 100, 1000]
vvv = [0, 0, 0, 0]
normalxyz = [0, 0, 0]

gamma_N = 2 * np.pi * 1.071e-3  # [MHz/G], 13C nuclear random dataset
# Al = 2 * np.pi * random.uniform(0.05, 0.8)  # [MHz] # A hyperfine term
# Ap = 2 * np.pi * random.uniform(0.05, 0.3)  # [MHz] # A per hyperfine term

Al = 1.1876071666049706
Ap = 1.3050296350856783

rho_0 = (np.kron(U90xp, identity)
         ) @ initial_rho @ ((np.kron(U90xp, identity)).conj().T)

Sa = []

ham = Al * np.kron(sigmaZ4Gellmann, Iz) + Ap * np.kron(sigmaZ4Gellmann,
                                                       Ix) + B * gamma_N * np.kron(identity, Iz)  # Hamiltonian
eigvals = np.linalg.eigh(ham)[0]  # diagonalizing the Hamiltonian
eigvecs = -1 * np.linalg.eigh(ham)[1]  # eigenvectors
E = np.diag(eigvals)  # exponent of eigenvalues
U_H = eigvecs.conj().T  # unitary matrix formed by eigenvectors

udf = pd.DataFrame(U_H)
udf

# Ry 90도
rho1 = np.kron(
    U90yp, identity) @ initial_rho @ (np.kron(U90yp, identity).conj().T)
df1 = pd.DataFrame(rho1)
df1

vari = [1.4120767867009363, 22.0, 0.2633900964243493, 22.0]  # 초기값
# for tau/2
U_e2 = (U_H.conj().T)@(linalg.expm(-i*E * vari[0]/2)@U_H)
# for tau
U_e = (U_H.conj().T)@(linalg.expm(-i*E * vari[0])@U_H)
# first tau/2
rho2 = U_e2@rho1@(U_e2.conj().T)
# N과 tau를 N개 생성
for k in range(1, 2*math.trunc(vari[1])):
    rho2 = U_e@np.kron(U180xp, identity) @ rho2 @ (np.kron(U180xp,
                                                           identity).conj().T) @ (U_e.conj().T)  # N & tau
rho3 = U_e2 @ np.kron(U180xp, identity) @ rho2 @ (np.kron(U180xp,
                                                          identity).conj().T) @ (U_e2.conj().T)  # last N & tau/2


# df2 = pd.DataFrame(rho2)
df2 = pd.DataFrame(rho3)
df2

# for e Rx(pi/2)
# Rx 90도
rho4 = np.kron(U90xp, identity) @ rho3 @ (np.kron(U90xp, identity).conj().T)
df4 = pd.DataFrame(rho4)
print(df4)


# for tau/2
U_e2 = (U_H.conj().T)@(linalg.expm(-i*E*vari[2]/2)@U_H)
# for tau/2
U_e = (U_H.conj().T)@(linalg.expm(-i*E*vari[2])@U_H)
# first tau/2
rho5 = U_e2@rho4@(U_e2.conj().T)
# N과 tau를 N개 생성
for k in range(1, 2*math.trunc(vari[3])):
    rho5 = U_e@np.kron(U180xp, identity) @ rho5 @ (np.kron(U180xp,
                                                           identity).conj().T) @ (U_e.conj().T)  # N & tau
rho6 = U_e2 @ np.kron(U180xp, identity) @ rho5 @ (np.kron(U180xp,
                                                          identity).conj().T) @ (U_e2.conj().T)  # last N & tau/2

df6 = pd.DataFrame(rho6)
print(df6)


# for tau/2
U_e2 = (U_H.conj().T)@(linalg.expm(-i*E * vari[0]/2)@U_H)
# for tau
U_e = (U_H.conj().T)@(linalg.expm(-i*E * vari[0])@U_H)
# first tau/2
rho7 = U_e2@rho6@(U_e2.conj().T)
# N과 tau를 N개 생성
for k in range(1, 2*math.trunc(vari[1])):
    rho7 = U_e@np.kron(U180xp, identity) @ rho7 @ (np.kron(U180xp,
                                                           identity).conj().T) @ (U_e.conj().T)  # N & tau
rho8 = U_e2 @ np.kron(U180xp, identity) @ rho7 @ (np.kron(U180xp,
                                                          identity).conj().T) @ (U_e2.conj().T)  # last N & tau/2

df8 = pd.DataFrame(rho8)
print(df8)


partial_trace(rho8, 1)
