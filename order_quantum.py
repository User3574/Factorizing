# Inspired by https://physlab.org/wp-content/uploads/2023/05/Shor_s_Algorithm_23100113_Fin.pdf
# Inspired by https://arxiv.org/pdf/1804.03719.pdf
# Inspired by https://arxiv.org/abs/2103.13855
# Inspired by https://github.com/qiskit-community/qiskit-community-tutorials/blob/master/algorithms/shor_algorithm.ipynb
# Inspired by https://learning.quantum.ibm.com/course/fundamentals-of-quantum-algorithms/phase-estimation-and-factoring


import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json

from fractions import Fraction
from qiskit import Aer, transpile, execute
from qiskit.visualization import plot_histogram
from mod_exp import *
from order_classic import PeriodFinder
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister


def fraction_to_ratio(flt):
    if int(flt) == flt:
        return int(flt), 1
    flt_str = str(flt)
    flt_split = flt_str.split('.')
    numerator = int(''.join(flt_split))
    denominator = 10 ** len(flt_split[1])
    return numerator, denominator


def QFT(n):
    circuit = QuantumCircuit(n)
    for i in range(n - 1, 0, -1):
        circuit.h(i)
        for qubit in range(i):
            circuit.cp(np.pi / 2 ** (i - qubit), qubit, i)
    circuit.h(0)
    for qubit in range(n // 2):
        circuit.swap(qubit, n - qubit - 1)
    return circuit


def iQFT(n):
    qft = QFT(n)
    iqft = qft.inverse()
    return iqft.decompose()


def classical_postprocessing(counts, dividor, shots, display):
    plot_histogram({int(k, 2): v for k, v in counts.items()})
    plt.show()

    # Print results
    rows, measured_phases = [], []
    for output in counts:
        decimal = int(output, 2)
        phase = decimal / dividor
        measured_phases.append([phase, counts[output] / shots])
        rows.append([f"{decimal:>3}", f"{phase:.2f}", counts[output]])

    # Print the rows in a table
    headers = ["Register Output", "Phase", "Counts"]
    df = pd.DataFrame(rows, columns=headers)

    if display:
        print(df)

    # Convert phase to ratio (psi = s/r)
    rows = []
    for phase, prob in measured_phases:
        frac = fraction_to_ratio(phase)
        frac = Fraction(*frac).limit_denominator(10)
        rows.append([phase,
                     f"{frac.numerator}/{frac.denominator}",
                     frac.denominator,
                     prob])
    # Print as a table
    headers = ["Phase", "Fraction", "Guess for r", "Probability"]
    df = pd.DataFrame(rows, columns=headers)

    if display:
        print(df)

    df_groupby = df.groupby('Guess for r').sum()
    most_probable_r = df_groupby['Probability'].idxmax()

    if display:
        print(df_groupby["Probability"])
        print(f'Most probable r: {most_probable_r} with probability: {df_groupby["Probability"].loc[most_probable_r]}')
        p = df_groupby["Probability"].plot.bar(title="Observed Frequencies")
        p.set_ylabel("Frequency")
        p.set_xlabel("r")
        plt.show()

    # Return maximally occuring r
    return most_probable_r


class QuantumPeriodFinder(PeriodFinder):
    def __init__(self, control_register, work_register, postprocess, dividor, backend):
        super().__init__()
        self.control_register = control_register
        self.work_register = work_register
        self.postprocess = postprocess
        self.circuit = QuantumCircuit(self.control_register, self.work_register)
        self.dividor = dividor
        self.backend = backend
        self.shots = 1024

    def create_circuit(self, a, draw):
        pass

    def simulate(self, x, N):
        self.create_circuit(x, draw=True)

        # Simulate
        transpiled = transpile(self.circuit, self.backend)

        result = self.backend.run(transpiled, shots=self.shots).result()
        dicto = result.to_dict()
        with open(f'shor_{x}_{N}.json', 'w') as fp:
            json.dump(dicto, fp)
        print(result)

        counts = result.get_counts()
        return classical_postprocessing(counts, self.dividor, self.shots, self.postprocess)


class PeriodFinder15(QuantumPeriodFinder):
    def __init__(self, backend, postprocess):
        control_register = 12
        work_register = 8
        super().__init__(control_register, work_register, postprocess, 2 ** work_register, backend)

    def create_circuit(self, a, draw):
        # Step 1
        for q in range(self.work_register):
            self.circuit.h(q)
        self.circuit.x(self.work_register)
        self.circuit.barrier()

        # Step 2
        for q in range(self.work_register):
            self.circuit.append(c_amod15(a, 2 ** q), [q] + [i + self.work_register for i in range(4)])
        self.circuit.barrier()

        # Step 3
        self.circuit.append(iQFT(self.work_register), range(self.work_register))
        self.circuit.barrier()

        # Step 4
        self.circuit.measure(range(self.work_register), range(self.work_register))

        if draw:
            self.circuit.draw(output='mpl', fold=-1)
            plt.show()

    def simulate(self, x, N=15):
        super().simulate(x, N)


class PeriodFinder21(QuantumPeriodFinder):
    def __init__(self, backend, postprocess):
        control_register = 15
        work_register = 10
        super().__init__(control_register, work_register, postprocess, 2 ** work_register, backend)

    def create_circuit(self, a, draw):
        # Step 1
        for q in range(self.work_register):
            self.circuit.h(q)
        self.circuit.x(self.work_register)
        self.circuit.barrier()

        # Step 2
        for q in range(self.work_register):
            self.circuit.append(c_amod21(a, 2 ** q), [q] + [i + self.work_register for i in range(5)])
        self.circuit.barrier()

        # Step 3
        self.circuit.append(iQFT(self.work_register), range(self.work_register))
        self.circuit.barrier()

        # Step 4
        self.circuit.measure(range(self.work_register), range(self.work_register))

        if draw:
            self.circuit.draw(output='mpl', fold=-1)
            plt.show()

    def simulate(self, x, N=21):
        return super().simulate(x, N)


class PeriodFinder15Kitaev(QuantumPeriodFinder):
    def __init__(self, backend, postprocess):
        work_register = QuantumRegister(5, 'q')
        control_register = ClassicalRegister(5, 'c')
        super().__init__(control_register, work_register, postprocess, 2 ** 3, backend)

    def circuit_11_15(self):
        self.circuit.x(0)
        self.circuit.barrier()

        self.circuit.h(4)
        self.circuit.h(4)
        self.circuit.measure(4, self.control_register[0])
        self.circuit.reset(4)
        self.circuit.barrier()

        # Apply a**2 mod 15
        self.circuit.h(4)
        self.circuit.p(math.pi / 2., 4).c_if(self.control_register, 1)
        self.circuit.h(4)
        self.circuit.measure(4, 1)
        self.circuit.reset(4)
        self.circuit.barrier()

        # Apply 11 mod 15
        self.circuit.h(4)
        self.circuit.cx(4, 3)
        self.circuit.cx(4, 1)
        self.circuit.p(3. * math.pi / 4., 4).c_if(self.control_register, 3)
        self.circuit.p(math.pi / 2., 4).c_if(self.control_register, 2)
        self.circuit.p(math.pi / 4., 4).c_if(self.control_register, 1)
        self.circuit.h(4)
        self.circuit.barrier()
        self.circuit.measure(4, self.control_register[2])

    def create_circuit(self, a, draw):
        if a == 11:
            self.circuit_11_15()
        else:
            # Step 1: Initialize q[0] to |1>
            self.circuit.x(0)
            self.circuit.barrier()

            # Step 2: Apply a**4 mod 15
            self.circuit.h(4)
            self.circuit.h(4)
            self.circuit.measure(4, self.control_register[0])
            self.circuit.barrier()

            # Step 3: Apply a**2 mod 15
            self.circuit.reset(4)
            self.circuit.h(4)
            self.circuit.cx(4, 2)
            self.circuit.cx(4, 0)
            self.circuit.p(math.pi / 2., 4).c_if(self.control_register, 1)
            self.circuit.h(4)
            self.circuit.measure(4, self.control_register[1])
            self.circuit.barrier()

            # Step 4: Apply a mod 15
            self.circuit.reset(4)
            self.circuit.h(4)
            c_amod15_kitaev(a, self.circuit)
            # self.circuit.p(3. * math.pi / 4., 4).c_if(self.control_register, 3)
            self.circuit.p(math.pi / 2., 4).c_if(self.control_register, 2)
            self.circuit.p(math.pi / 4., 4).c_if(self.control_register, 1)
            self.circuit.h(4)
            self.circuit.barrier()

            # Step 5: Measure
            self.circuit.measure(4, self.control_register[2])

        if draw:
            self.circuit.draw(output='mpl', fold=-1)
            plt.show()

    def simulate(self, x, N=15):
        return super().simulate(x, N)


class PeriodFinder11_15(QuantumPeriodFinder):
    def __init__(self, backend, postprocess):
        control_register = 5
        work_register = 3
        super().__init__(control_register, work_register, postprocess, 2 ** work_register, backend)

    def create_circuit(self, a, draw):
        # Step 1
        for q in range(self.work_register):
            self.circuit.h(q)
        self.circuit.barrier()

        # Step 2 + 3
        self.circuit.cx(2, 3)
        self.circuit.cx(2, 4)
        #self.circuit.barrier()
        self.circuit.h(1)
        self.circuit.cp(np.pi/2, 1, 0)
        self.circuit.h(0)
        self.circuit.cp(np.pi/4, 1, 2)
        self.circuit.cp(np.pi/2, 0, 2)
        self.circuit.barrier()

        # Step 4
        self.circuit.measure(range(self.work_register), range(self.work_register))

        if draw:
            self.circuit.draw(output='mpl', fold=-1)
            plt.show()

    def simulate(self, x, N=15):
        super().simulate(x, N)


class PeriodFinder4_21(QuantumPeriodFinder):
    def __init__(self, backend, postprocess):
        control_register = 5
        work_register = 3
        super().__init__(control_register, work_register, postprocess, 2 ** work_register, backend)

    def create_circuit(self, a, draw):
        # Step 1
        for q in range(self.work_register):
            self.circuit.h(q)
        self.circuit.barrier()

        # Step 2
        self.circuit.cx(2, 4)
        self.circuit.barrier()
        self.circuit.cx(1, 4)
        self.circuit.cx(4, 3)
        self.circuit.ccx(1, 3, 4)
        self.circuit.cx(4, 3)
        self.circuit.barrier()
        self.circuit.x(4)
        self.circuit.ccx(0, 4, 3)
        self.circuit.x(4)
        self.circuit.cx(4, 3)
        self.circuit.ccx(0, 3, 4)
        self.circuit.cx(4, 3)
        self.circuit.barrier()

        # Step 3
        self.circuit.append(iQFT(self.work_register), range(self.work_register))
        self.circuit.barrier()

        # Step 4
        self.circuit.measure(range(self.work_register), range(self.work_register))

        if draw:
            self.circuit.draw(output='mpl', fold=-1)
            plt.show()

    def simulate(self, x, N=15):
        super().simulate(x, N)