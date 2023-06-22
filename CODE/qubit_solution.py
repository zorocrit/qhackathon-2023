import time
import random
from math import *
import math
from dataclasses import dataclass
from sys import stdout
from datetime import datetime as dt

from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import linalg, optimize
from scipy.linalg import fractional_matrix_power
from sklearn.feature_extraction.text import CountVectorizer
from mpl_toolkits.mplot3d import Axes3D

from toqito.channels import partial_trace
from qutip import *
from tqdm import tqdm, trange


@dataclass
class sweepData:
    sweep, mField, sweepTau, piPulse = [
        1001, 403, 5, 32
    ]
    tmpSweep = sweep
    space = np.linspace(0, tmpSweep, sweepTau)


@dataclass
class initialize:
    irho_p = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])  # [0,0,0;0,0,0]
    irho_m = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])  # [0,0,0;0,0,1]
    irho_z = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])  # [0,1,0;0,0,0]
    irho_mix = np.array([[1/2, 0, 0], [0, 1/2, 0], [0, 0, 0]]
                        )  # [1/2,0,0;0,1/2,0;0,0,0]
    irho_target = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])  # target state


class qubitData:
    def __init__(self) -> None:
        self.initData = initialize()

        self.i = 1j
        self.unitMatrix, self.data, self.target = [None, None, None]

        self.trace = []
        self.pDataSet = []
        self.mDataSet = []
        self.dResult = []
        self.pip = []

        self.normalxyz = [0, 0, 0]
        self.irho = np.kron(self.initData.irho_z, self.initData.irho_mix)

        # Define dimension, pauli matrices
        self.sigmaX = 1 / \
            np.sqrt(2) * np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        self.sigmaY = 1 / np.sqrt(2) / self.i * \
            np.array([[0, 1, 0], [-1, 0, 1], [0, -1, 0]])
        self.sigmaZ = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])
        self.identity = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        # Rotation matrix projected into 2 level system
        self.pSigmaX = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
        self.msigmaX = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]])
        self.pSigmaY = 1 / self.i * \
            np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])
        self.mSigmaY = 1 / self.i * \
            np.array([[0, 0, 0], [0, 0, 1], [0, -1, 0]])
        self.pSigmaZ = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]])

        # gellman matrix
        self.sigmaGX = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])
        self.sigmaGY = np.array([[0, 0, -self.i], [0, 0, 0], [self.i, 0, 0]])
        self.sigmaGZ = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])

        # pauli
        self.iX = 1 / 2 * np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])
        self.iY = 1 / 2 / self.i * np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]])
        self.iZ = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])

    def __operate__(self, B1, B2, a, D1, D2):
        i = self.i
        self.unitMatrix = [[(B2**2+B1**2*cos(a))/(B1**2+B2**2), -i*B1*(e**(-i*D1))*sin(a)/sqrt(B1**2+B2**2), ((-1+cos(a))*B1*B2*(e**(-i*(D1-D2))))/(B1**2+B2**2)],
                           [-i*B1*(e**(i*D1))*sin(a)/sqrt(B1**2+B2**2),
                            cos(a), -i*B2*(e**(i*D2))*sin(a)/sqrt(B1**2+B2**2)],
                           [((-1+cos(a))*B1*B2*e**(i*(D1-D2)))/(B1**2+B2**2), -i*B2*(e**(-i*D2))*sin(a)/sqrt(B1**2+B2**2), (B1**2+B2**2*cos(a))/(B1**2+B2**2)]]

    def __state__(self):
        if np.shape(self.data) != np.shape(self.target):
            print("Dimensions of two states do not match.")
            return 0
        else:
            sqrt_rho_1 = fractional_matrix_power(self.data, 1 / 2)
            fidelity = np.trace(fractional_matrix_power(
                sqrt_rho_1 @ self.target @ sqrt_rho_1, 1 / 2)) ** 2
            return np.real(fidelity)

    def generatePData(self):
        self.__operate__(1, 0, pi/4, 0, 0)
        self.pDataSet.append(self.unitMatrix)

        self.__operate__(1, 0, -pi/4, 0, 0)
        self.pDataSet.append(self.unitMatrix)

        self.__operate__(1, 0, pi/4, pi/2, 0)
        self.pDataSet.append(self.unitMatrix)

        self.__operate__(1, 0, -pi/4, pi/2, 0)
        self.pDataSet.append(self.unitMatrix)

        self.__operate__(1, 0, pi/2, 0, 0)
        self.pDataSet.append(self.unitMatrix)

        self.__operate__(1, 0, -pi/2, 0, 0)
        self.pDataSet.append(self.unitMatrix)
        print(self.pDataSet)

    def generateMData(self):
        self.__operate__(0, 1, pi/4, 0, 0)
        self.mDataSet.append(self.unitMatrix)

        self.__operate__(0, 1, -pi/4, 0, 0)
        self.mDataSet.append(self.unitMatrix)

        self.__operate__(0, 1, pi/2, 0, 0)
        self.mDataSet.append(self.unitMatrix)

        self.__operate__(0, 1, pi/2, 0, 0)
        self.mDataSet.append(self.unitMatrix)
        print(self.mDataSet)

    def problem(self, vari):
        gammaN = 2*pi*1.071e-3  # [MHz/G]
        Al = 2*pi * random.uniform(0.05, 0.8)  # [MHz] # A_|| hyperfine term
        Ap = 2*pi * random.uniform(0.05, 0.3)  # [MHz] # A_per hyperfine term
        I = self.identity
        init = initialize()
        trace = self.trace
        # superposition state on NV
        rho1 = (np.kron(self.pDataSet[0], self.identity))@self.irho@(
            (np.kron(self.pDataSet[0], self.identity)).conj().T)
        ham = Al*np.kron(self.sigmaZ, self.iZ) + Ap*np.kron(self.sigmaZ, self.iX) + sweepData().mField * \
            gammaN*np.kron(self.identity, self.iZ)  # Hamiltonian

        # diagonalizing the Hamiltonian
        eigvals = np.linalg.eigh(ham)[0]
        eigvecs = -1*np.linalg.eigh(ham)[1]         # eigenvectors
        E = np.diag(eigvals)                        # exponent of eigenvalues
        # unitary matrix formed by eigenvectors
        U_H = eigvecs.conj().T

        # for N Rx(pi/2)
        # for tau/2
        U_e2 = (U_H.conj().T)@(linalg.expm(-self.i*E * vari[0]/2)@U_H)
        # for tau
        U_e = (U_H.conj().T)@(linalg.expm(-self.i*E * vari[0])@U_H)
        # first tau/2
        rho2 = U_e2@rho1@(U_e2.conj().T)
        # N과 tau를 N개 생성
        for _ in range(1, 2*math.trunc(vari[1])):
            rho2 = U_e@np.kron(self.pDataSet[4], I) @ rho2 @ (np.kron(self.pDataSet[4],
                                                                      I).conj().T) @ (U_e.conj().T)  # N & tau
        rho3 = U_e2 @ np.kron(self.pDataSet[4], I) @ rho2 @ (np.kron(self.pDataSet[4],
                                                                     I).conj().T) @ (U_e2.conj().T)  # last N & tau/2
        # for e Rx(pi/2)
        # Rx 90도
        rho4 = np.kron(
            self.pDataSet[0], I)@rho3@(np.kron(self.pDataSet[0], I).conj().T)

        # for N Rz(pi/2) //이부분이 Z pulse를 다루고 있다면 N을 따로 분리해야하나?>
        # for tau/2
        U_e2 = (U_H.conj().T)@(linalg.expm(-self.i*E*vari[2]/2)@U_H)
        # for tau/2
        U_e = (U_H.conj().T)@(linalg.expm(-self.i*E*vari[2])@U_H)
        # first tau/2
        rho5 = U_e2@rho4@(U_e2.conj().T)
        # N과 tau를 N개 생성
        for _ in range(1, 2*math.trunc(vari[3])):
            rho5 = U_e@np.kron(self.pDataSet[4], I) @ rho5 @ (np.kron(self.pDataSet[4],
                                                                      I).conj().T) @ (U_e.conj().T)  # N & tau
        rho6 = U_e2 @ np.kron(self.pDataSet[4], I) @ rho5 @ (np.kron(self.pDataSet[4],
                                                                     I).conj().T) @ (U_e2.conj().T)  # last N & tau/2

        xx = (np.trace(self.iX@partial_trace(rho6, 1))).real  # for N spin
        yy = (np.trace(self.iY@partial_trace(rho6, 1))).real
        zz = (np.trace(self.iZ@partial_trace(rho6, 1))).real

        cost = 1 - self.__state__(init.irho_target,
                                  partial_trace(rho6, 1))

        if cost < trace[6]:
            trace[6] = cost
            self.normalxyz[0] = xx
            self.normalxyz[1] = yy
            self.normalxyz[2] = zz

            self.bbb[0] = vari[0]
            self.bbb[1] = vari[1]
            self.bbb[2] = vari[2]
            self.bbb[3] = vari[3]

        temp = [cost, vari[0], vari[1], vari[2], vari[3]]
        # tuples.append((cost, vari[0], vari[1], vari[2], vari[3]))
        pip.append(temp)

        return cost

    def solution(self, count: int):
        # generate datas
        self.generatePData()
        self.generatePData()
        sweeped = sweepData()
        # for e Ry(pi/2)
        pdata = self.pDataSet
        mdata = self.mDataSet

        I = self.identity
        i = self.i
        N = sweeped.tmpSweep
        t = sweeped.space
        dd = self.dResult

        irho = self.irho

        for _ in range(13):  # range 번의 실험을 진행한다.
            self.trace = [1, 1, 0, 100, 100, 1000, 100]
            self.bbb = [0, 0, 0, 0]
            bbb = self.bbb
            self.normalxyz = [0, 0, 0]
            # for making 13C nuclear random dataset
            gammaN = 2*pi*1.071e-3  # [MHz/G]
            # [MHz] # A_|| hyperfine term
            Al = 2*pi * random.uniform(0.05, 0.8)
            # [MHz] # A_per hyperfine term
            Ap = 2*pi * random.uniform(0.05, 0.3)

            # Initialization
            # superposition state on NV
            rho_0 = (np.kron(pdata[0], I)
                     )@irho@((np.kron(pdata[0], I)).conj().T)

            Sa = []

            ham = Al*np.kron(self.sigmaZ, self.iZ) + Ap*np.kron(self.sigmaZ, self.iX) + sweeped.mField * \
                gammaN*np.kron(I, self.iZ)  # Hamiltonian
            # diagonalizing the Hamiltonian 여기서부터 문제
            eigvals = np.linalg.eigh(ham)[0]
            eigvecs = -1*np.linalg.eigh(ham)[1]         # eigenvectors
            # exponent of eigenvalues
            E = np.diag(eigvals)
            # unitary matrix formed by eigenvectors
            U_H = eigvecs.conj().T

            for h in range(N):  # N개의 pulse를 생성한다.

                # free evolution unitary operator
                # for tau/2
                U_e2 = (U_H.conj().T)@(linalg.expm(-i*E*t[h]/2)@U_H)
                U_e = (U_H.conj().T)@(linalg.expm(-i*E*t[h])@U_H)  # for tau
                # first tau/2
                rho_1 = U_e2 @ rho_0 @ (U_e2.conj().T)
                for k in range(31):                                   # N과 tau를 N개 생성
                    rho_1 = U_e @ np.kron(pdata[4], I) @ rho_1 @ (np.kron(pdata[4],
                                                                          I).conj().T) @ (U_e.conj().T)  # N & tau

                rho_2 = U_e2 @ np.kron(pdata[4], I) @ rho_1 @ (np.kron(pdata[4],
                                                                       I).conj().T) @ (U_e2.conj().T)  # last N & tau/2
                rho_3 = np.kron(
                    pdata[1], I) @ rho_2 @ ((np.kron(pdata[1], I)).conj().T)    # last pi/2
                # NV state 0 population readout
                res1 = (np.trace(self.initData.irho_z @
                        partial_trace(rho_3, 2))).real
                # append to list
                Sa.append(res1)

            index = Sa.index(min(Sa))  # list에서 가장 작은 값을 가지는 index를 찾는다.
            tau = t[index]  # 그 index에 해당하는 tau를 찾는다.

            ham = Al*np.kron(self.sigmaZ, self.iZ) + Ap*np.kron(self.sigmaZ, self.iX) + sweeped.mField * \
                gammaN*np.kron(I, self.iZ)  # Hamiltonian
            eigvals = np.linalg.eigh(ham)[0]  # diagonalizing the Hamiltonian
            eigvecs = -1*np.linalg.eigh(ham)[1]  # eigenvectors
            E = np.diag(eigvals)             # exponent of eigenvalues
            U_H = eigvecs.conj().T         # unitary matrix formed by eigenvectors

            # 결과들을 저장할 list 생성
            for __ in range(count):  # 1번의 실험을 진행한다.(지역 최적화 알고리즘을 사용할 경우에 수정한다.)
                vari = [tau, 9, 0.1*tau, 9]  # 초기값
                bounds = [(0.95*tau, 1.05*tau), (1.0, 25.0),
                          (1.05*tau, 3 * tau), (1.0, 75.0)]  # boundary

                res4 = optimize.shgo(self.problem, bounds=bounds, iters=5, options={
                    'xtol': 1e-15, 'ftol': 1e-17})  # SHGO method
                # res4['x'][1] = math.floor(res4['x'][1]) #rounding
                # res4['x'][3] = math.floor(res4['x'][3]) #rounding
                bbb[1] = math.floor(bbb[1])
                bbb[3] = math.floor(bbb[3])
                dd.append([Al, Ap, self.trace[6], bbb[0], bbb[1], bbb[2], bbb[3],
                           self.normalxyz[0], self.normalxyz[1], self.normalxyz[2], res4['nfev']])
                if(self.trace[6] < 0.1):
                    count = count + 1
