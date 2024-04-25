import numpy as np

from math import sqrt
from matplotlib import pyplot as plt


# Eratosthenes prime finding (https://stackoverflow.com/questions/11619942/print-series-of-prime-numbers-in-python)
def get_primes(n):
    primes = []
    sieve = [True for i in range(n + 1)]
    for p in range(2, n + 1):
        if sieve[p]:
            primes.append(p)
            for i in range(p, n + 1, p):
                sieve[i] = False
    return primes


# Euclidean GCD
def gcd(a, b):
    return a if b == 0 else gcd(b, a % b)


# Extended Euclidean GCD
def egcd(a, b):
    x, old_x = 0, 1
    y, old_y = 1, 0

    while b != 0:
        quotient = a // b
        a, b = b, a - quotient * b
        old_x, x = x, old_x - quotient * x
        old_y, y = y, old_y - quotient * y

    return a, old_x, old_y


# Inverse Modulo
def mod_inv(a, m):
    g, x, y = egcd(a, m)
    if g != 1:
        raise Exception('modular inverse does not exist')
    else:
        return x % m


def find_factors(n):
    for i in range(2, int(sqrt(n)) + 1):
        if n % i == 0:
            # p = i => q = n/i
            return i, int(n / i)


def filter_primes_digits(digits, primes):
    primes_with_digits = []
    for prime in primes:
        if len(str(prime)) == digits:
            primes_with_digits.append(prime)
    return primes_with_digits


def plot_periodic_function(a, N):
    # Calculate the plotting data
    xvals = np.arange(N)
    yvals = [np.mod(a ** x, N) for x in xvals]

    # Plot on matplotlib
    fig, ax = plt.subplots()
    ax.plot(xvals, yvals, linewidth=1, linestyle='dotted', marker='x')
    ax.set(
        xlabel='$x$',
        ylabel=f'${a}^r$ mod ${N}$',
        title="Periodic function in Shor's Algorithm"
    )

    # Annotate
    try:
        r = yvals[1:].index(1) + 1
        plt.annotate('', xy=(0, 1), xytext=(r, 1), arrowprops=dict(arrowstyle='<->'))
        plt.annotate(f'$t={r}$', xy=(r / 3, 1.5))
        plt.show()
    except ValueError:
        print("Error while plotting: Check that a, N have no common factors")
    return xvals, yvals