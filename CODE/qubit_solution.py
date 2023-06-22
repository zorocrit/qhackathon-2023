import time
import random
from math import *
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
    sweep, tmpSweep, mField, sweepTau, piPulse = [
        1001, 1001, 403, 5, 32
    ]
    space = np.linspace(0, tmpSweep, sweepTau)


@dataclass
class initialize:
    irho_p = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])  # [0,0,0;0,0,0]
    irho_m = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])  # [0,0,0;0,0,1]
    irho_z = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])  # [0,1,0;0,0,0]
    irho_mix = np.array([[1/2, 0, 0], [0, 1/2, 0], [0, 0, 0]]
                        )  # [1/2,0,0;0,1/2,0;0,0,0]
    irho_target = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])  # target state
    trace = [1, 1, 0, 100]


class qubitData:
    def __init__(self) -> None:
        self.initData = initialize()

        self.i = 1j
        self.unitMatrix, self.data, self.target = [None, None, None]

        self.pDataSet = []
        self.mDataSet = []
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

    def __operate__(self, coeff1, coeff2, angle, param1, param2) -> None:
        self.unitMatrix = [
            [
                (coeff2**2 + coeff1**2 * np.cos(angle)) / (coeff1**2 + coeff2**2),
                -self.i * coeff1 * (np.exp(-self.i * param1)) * np.sin(angle) /
                np.sqrt(coeff1**2 + coeff2**2),
                ((-1 + np.cos(angle)) * coeff1 * coeff2 *
                 (np.exp(-self.i * (param1 - param2)))) / (coeff1**2 + coeff2**2)
            ],
            [
                -self.i * coeff1 * (np.exp(self.i * param1)) * np.sin(angle) /
                np.sqrt(coeff1**2 + coeff2**2), np.cos(angle),
                -self.i * coeff2 * (np.exp(self.i * param2)) *
                np.sin(angle) / np.sqrt(coeff1**2 + coeff2**2)
            ],
            [
                ((-1 + np.cos(angle)) * coeff1 * coeff2 *
                 np.exp(self.i * (param1 - param2))) / (coeff1**2 + coeff2**2),
                -self.i * coeff2 * (np.exp(-self.i * param2)) * np.sin(angle) /
                np.sqrt(coeff1**2 + coeff2**2),
                (coeff1**2 + coeff2**2 * np.cos(angle)) / (coeff1**2 + coeff2**2)
            ]
        ]

    def __state__(self):
        if np.shape(self.data) != np.shape(self.target):
            print("Dimensions of two states do not match.")
            return 0
        else:
            sqrt_rho_1 = fractional_matrix_power(self.data, 1 / 2)
            fidelity = np.trace(fractional_matrix_power(
                sqrt_rho_1 @ self.target @ sqrt_rho_1, 1 / 2)) ** 2
            return np.real(fidelity)

    def generateData(self):
        self.pDataSet = [
            self.__operate__(1, 0, pi/4, 0, 0),
            self.__operate__(1, 0, -pi/4, 0, 0),
            self.__operate__(1, 0, pi/4, pi/2, 0),
            self.__operate__(1, 0, -pi/4, pi/2, 0),
            self.__operate__(1, 0, pi/2, 0, 0),
            self.__operate__(1, 0, -pi/2, 0, 0)
        ]

        self.mDataSet = [
            self.__operate__(0, 1, pi/4, 0, 0),
            self.__operate__(0, 1, -pi/4, 0, 0),
            self.__operate__(0, 1, pi/2, 0, 0),
            self.__operate__(0, 1, pi/2, 0, 0)
        ]

    def solution(self, data):
        # for e Ry(pi/2)
        pdata = self.pDataSet
        mdata = self.mDataSet
        irho = self.irho

        rho1 = np.kron(
            pdata[0], self.identity)@irho@(np.kron(pdata[2], self.identity).conj().T)

        # for N Rx(pi/2)
        U_e2 = (U_H.conj().T)@(linalg.expm(-i*E * vari[0]/2)@U_H)
