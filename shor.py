import random

from utils import *


def shor(n, period_finder):
    x = random.randint(2, n)

    if gcd(x, n) != 1:
        return x, 0, gcd(x, n), n / gcd(x, n)

    r = period_finder.simulate(x, n)

    y = x ** int(r / 2)
    p = gcd(y + 1, n)
    q = gcd(y - 1, n)

    return x, r, p, q