# Taken from https://github.com/qiskit-community/qiskit-community-tutorials/blob/master/algorithms/shor_algorithm.ipynb
# Taken from https://physlab.org/wp-content/uploads/2023/05/Shor_s_Algorithm_23100113_Fin.pdf

from qiskit import QuantumCircuit
from matplotlib import pyplot as plt


def c_amod15_kitaev(a, U):
    def circuit_2mod15(U):
        U.cswap(4, 3, 2)
        U.cswap(4, 2, 1)
        U.cswap(4, 1, 0)

    def circuit_7mod15(U):
        U.cswap(4, 1, 0)
        U.cswap(4, 2, 1)
        U.cswap(4, 3, 2)
        U.x(0)
        U.x(1)
        U.x(2)
        U.x(3)

    def circuit_8mod15(U):
        U.cswap(4, 1, 0)
        U.cswap(4, 2, 1)
        U.cswap(4, 3, 2)

    def circuit_11mod15(U):
        U.cswap(4, 2, 0)
        U.cswap(4, 3, 1)
        U.x(0)
        U.x(1)
        U.x(2)
        U.x(3)

    def circuit_13mod15(U):
        U.cswap(4, 3, 2)
        U.cswap(4, 2, 1)
        U.cswap(4, 1, 0)
        U.x(0)
        U.x(1)
        U.x(2)
        U.x(3)

    circuit_xmod15 = {
        2: circuit_2mod15,
        7: circuit_7mod15,
        8: circuit_8mod15,
        11: circuit_11mod15,
        13: circuit_13mod15
    }

    circuit_xmod15[a](U)


def c_amod15(a, power):
    if a not in [2, 7, 8, 11, 13]:
        raise ValueError("'a' must be 2,7,8,11 or 13")

    U = QuantumCircuit(4)
    for iteration in range(power):
        if a in [2, 13]:
            U.swap(0, 1)
            U.swap(1, 2)
            U.swap(2, 3)
        if a in [7, 8]:
            U.swap(2, 3)
            U.swap(1, 2)
            U.swap(0, 1)
        if a == 11:
            U.swap(1, 3)
            U.swap(0, 2)
        if a in [7, 11, 13]:
            for q in range(4):
                U.power(q)
    #U.draw(output='mpl', fold=-1)
    #plt.show()
    U = U.to_gate()
    U.name = "%i^%i mod 15" % (a, power)
    c_U = U.control()
    return c_U


def c_amod21(a, power):
    if a not in [4, 5, 6, 8, 13, 16]:
        raise ValueError("'a' must be 4,5,6,8,13 or 16")

    U = QuantumCircuit(5)
    for _iteration in range(power):
        if a == 4:
            U.swap(0, 4)
            U.swap(4, 2)
        if a == 5:
            U.x(0)
            U.x(2)
            U.x(4)
            U.swap(4, 2)
            U.swap(0, 4)
        if a == 8:
            U.swap(0, 3)
        if a == 13:
            if power % 2 == 1:
                U.x(2)
                U.x(3)
        if a == 16:
            U.swap(4, 2)
            U.swap(0, 4)
        if a == 6:
            if power % 2 == 1:
                U.x(0)
                U.x(1)
                U.x(2)
            else:
                U.x(1)
                U.x(2)
                U.x(3)
    # U.draw(output='mpl', fold=-1)
    # plt.show()
    U = U.to_gate()
    U.name = f"{a}^{power} mod 21"
    c_U = U.control()
    return c_U
