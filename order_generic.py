# Taken from: https://github.com/Aryaan082/Shors-Algorithm/blob/master/Replicated-Shor's-Algorithm.ipynb

import math
import pandas as pd

from utils import *
from qiskit import *
from order_quantum import QuantumPeriodFinder, fraction_to_ratio, classical_postprocessing
from fractions import Fraction
from qiskit import transpile
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram
from mod_exp import *


# Function to create QFT
def QFT(circuit, up_reg, n, with_swaps):
    i = n - 1

    while i >= 0:
        circuit.h(up_reg[i])
        j = i - 1
        while j >= 0:
            if (np.pi) / (pow(2, (i - j))) > 0:
                circuit.cp((np.pi) / (pow(2, (i - j))), up_reg[i], up_reg[j])
                j = j - 1
        i = i - 1

    if with_swaps == 1:
        i = 0
        while i < ((n - 1) / 2):
            circuit.swap(up_reg[i], up_reg[n - 1 - i])
            i = i + 1


# Function to create inverse QFT
def iQFT(circuit, up_reg, n, with_swaps):
    if with_swaps == 1:
        i = 0
        while i < ((n - 1) / 2):
            circuit.swap(up_reg[i], up_reg[n - 1 - i])
            i = i + 1

    i = 0
    while i < n:
        circuit.h(up_reg[i])
        if i != n - 1:
            j = i + 1
            y = i
            while y >= 0:
                if (np.pi) / (pow(2, (j - y))) > 0:
                    circuit.cp(- (np.pi) / (pow(2, (j - y))), up_reg[j], up_reg[y])
                    y = y - 1
        i = i + 1


# Function that calculates the array of angles to be used in the addition in Fourier Space
def getAngles(a, N):
    s = bin(int(a))[2:].zfill(N)
    angles = np.zeros([N])
    for i in range(0, N):
        for j in range(i, N):
            if s[j] == '1':
                angles[N - i - 1] += math.pow(2, -(j - i))
        angles[N - i - 1] *= np.pi
    return angles


# Creation of a doubly controlled phase gate
def ccphase(circuit, angle, ctl1, ctl2, tgt):
    circuit.cp(angle / 2, ctl1, tgt)
    circuit.cx(ctl2, ctl1)
    circuit.cp(-angle / 2, ctl1, tgt)
    circuit.cx(ctl2, ctl1)
    circuit.cp(angle / 2, ctl2, tgt)


# Creation of the circuit that performs addition by a in Fourier Space
# Can also be used for subtraction by setting the parameter inv to a value different from 0
def phiADD(circuit, q, a, N, inv):
    angle = getAngles(a, N)
    for i in range(0, N):
        if inv == 0:
            circuit.p(angle[i], q[i])
        else:
            circuit.p(-angle[i], q[i])


# Single controlled version of the phiADD circuit
def cphiADD(circuit, q, ctl, a, n, inv):
    angle = getAngles(a, n)
    for i in range(0, n):
        if inv == 0:
            circuit.cp(angle[i], ctl, q[i])
        else:
            circuit.cp(-angle[i], ctl, q[i])


# Doubly controlled version of the phiADD circuit
def ccphiADD(circuit, q, ctl1, ctl2, a, n, inv):
    angle = getAngles(a, n)
    for i in range(0, n):
        if inv == 0:
            ccphase(circuit, angle[i], ctl1, ctl2, q[i])
        else:
            ccphase(circuit, -angle[i], ctl1, ctl2, q[i])


# Circuit that implements doubly controlled modular addition
def ccphiADDmodN(circuit, q, ctl1, ctl2, aux, a, N, n):
    ccphiADD(circuit, q, ctl1, ctl2, a, n, 0)
    phiADD(circuit, q, N, n, 1)
    iQFT(circuit, q, n, 0)
    circuit.cx(q[n - 1], aux)
    QFT(circuit, q, n, 0)
    cphiADD(circuit, q, aux, N, n, 0)

    ccphiADD(circuit, q, ctl1, ctl2, a, n, 1)
    iQFT(circuit, q, n, 0)
    circuit.x(q[n - 1])
    circuit.cx(q[n - 1], aux)
    circuit.x(q[n - 1])
    QFT(circuit, q, n, 0)
    ccphiADD(circuit, q, ctl1, ctl2, a, n, 0)


# Circuit that implements the inverse of doubly controlled modular addition
def ccphiADDmodN_inv(circuit, q, ctl1, ctl2, aux, a, N, n):
    ccphiADD(circuit, q, ctl1, ctl2, a, n, 1)
    iQFT(circuit, q, n, 0)
    circuit.x(q[n - 1])
    circuit.cx(q[n - 1], aux)
    circuit.x(q[n - 1])
    QFT(circuit, q, n, 0)
    ccphiADD(circuit, q, ctl1, ctl2, a, n, 0)
    cphiADD(circuit, q, aux, N, n, 1)
    iQFT(circuit, q, n, 0)
    circuit.cx(q[n - 1], aux)
    QFT(circuit, q, n, 0)
    phiADD(circuit, q, N, n, 0)
    ccphiADD(circuit, q, ctl1, ctl2, a, n, 1)


# Circuit that implements single controlled modular multiplication
def cMULTmodN(circuit, ctl, q, aux, a, N, n):
    QFT(circuit, aux, n + 1, 0)
    for i in range(0, n):
        ccphiADDmodN(circuit, aux, q[i], ctl, aux[n + 1], (2 ** i) * a % N, N, n + 1)
    iQFT(circuit, aux, n + 1, 0)

    for i in range(0, n):
        circuit.cswap(ctl, q[i], aux[i])

    a_inv = mod_inv(a, N)
    QFT(circuit, aux, n + 1, 0)
    i = n - 1
    while i >= 0:
        ccphiADDmodN_inv(circuit, aux, q[i], ctl, aux[n + 1], math.pow(2, i) * a_inv % N, N, n + 1)
        i -= 1
    iQFT(circuit, aux, n + 1, 0)


class SemiPeriodFinder(QuantumPeriodFinder):
    def __init__(self, backend, postprocess, N):
        self.N = N
        self.n = math.ceil(math.log(self.N, 2))
        print(f'n: {self.n}')
        print('Total number of qubits used: {0}\n'.format(4 * self.n + 2))
        super().__init__(1, 1, postprocess, 1, backend)

    def create_circuit(self, a, draw):
        # Create registers
        aux = QuantumRegister(self.n + 2)
        up_reg = QuantumRegister(2 * self.n)
        down_reg = QuantumRegister(self.n)
        up_classic = ClassicalRegister(2 * self.n)
        circuit = QuantumCircuit(down_reg, up_reg, aux, up_classic)

        # Step 1
        circuit.h(up_reg)
        circuit.x(down_reg[0])
        circuit.barrier()

        # Step 2
        for i in range(0, 2 * self.n):
            cMULTmodN(circuit, up_reg[i], down_reg, aux, int(pow(a, pow(2, i))), self.N, self.n)
        circuit.barrier()

        # Step 3
        iQFT(circuit, up_reg, 2 * self.n, 1)
        circuit.barrier()

        # Step 4
        circuit.measure(up_reg, up_classic)

        self.circuit = circuit
        if draw:
            self.circuit.draw(output='mpl', fold=-1)
            plt.show()

    def simulate(self, x, N):
        # Create circuit
        self.create_circuit(x, draw=False)

        # Simulate
        transpiled = transpile(self.circuit, self.backend)
        result = self.backend.run(transpiled, shots=self.shots).result()
        classical_postprocessing(result.get_counts(), 2**(2*self.n), self.shots, self.postprocess)
