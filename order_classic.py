# Inspired by https://quantumai.google/cirq/experiments/shor

from utils import gcd


class PeriodFinder:
    def __init__(self):
        pass

    def simulate(self, x, N):
        pass


class ClassicalPeriodFinder(PeriodFinder):
    def __init__(self):
        super().__init__()

    # Brute force period finding algorithm
    def simulate(self, x, N):
        if x < 2 or x >= N or gcd(x, N) > 1:
            raise ValueError(f"Invalid x={x} for modulus n={N}.")

        # Determine the period
        r, y = 1, x
        while y != 1:
            y = (x * y) % N
            r += 1
        return r
