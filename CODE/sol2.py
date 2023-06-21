import numpy as np
import scipy.linalg as la

# Hamiltonian
H = np.kron(np.kron(np.diag([1, -1]), np.eye(2)), np.eye(4)) + np.kron(
    np.kron(np.diag([1, -1]), np.eye(2)), np.array([[0, 1], [1, 0]]))

# initial state
p1 = 0.5 * np.kron(np.diag([1, -1]), np.eye(2)) + 0.5 * \
    np.kron(np.diag([1, -1]), np.array([[0, 1], [1, 0]]))

# parameters of the simulation
t_max = 10
dt = 0.01

# Initialize the state array
p = np.zeros((t_max + 1, 4, 4), dtype=np.complex128)
p[0] = p1

for t in range(1, t_max + 1):
    p[t] = np.tensordot(la.expm(-1j * H * dt), p[t - 1], axes=((2, 3), (0, 1)))

# partial trace of the final state
p_z = np.trace(p[t_max], axis1=0, axis2=1)


print(p_z)
