# Inspired by: https://gist.github.com/marnix135/582c78891b29186ba4c6882a4bc62822
# Small changes made with inspiration from: https://github.com/mx0c/RSA-Implementation-in-Python/blob/master/main.py

import random

from shor import shor
from utils import *


class RSA:
    def __init__(self, max_prime=None, p=None, q=None, do_print=True):
        if max_prime:
            primes = get_primes(max_prime)
            p, q = random.sample(primes, 2)
        elif p and q:
            if p > 1 and q > 1:
                p, q = p, q
            else:
                raise Exception("Prime factors are elementary")

        self.n = p * q
        phi = (p - 1) * (q - 1)
        self.public_key = self.get_public_key(phi)
        self.private_key = self.get_private_key(phi)
        if do_print:
            print(f'Public key: {self.public_key, self.n}, Private key: {self.private_key, self.n}')

    def get_public_key(self, phi):
        while True:
            e = random.randrange(2, phi)
            if gcd(e, phi) == 1:
                return e

    def get_private_key(self, phi):
        gcd, x, y = egcd(self.public_key, phi)
        d = x + phi if x < 0 else x
        return d

    def encrypt(self, msg, public_key, n):
        if isinstance(msg, int):
            encrypted = (msg ** public_key) % n
        elif isinstance(msg, str):
            encrypted = ''
            for letter in msg:
                encrypted = encrypted + chr((ord(letter) ** public_key) % n)
        return encrypted

    def decrypt(self, msg):
        if isinstance(msg, int):
            decrypted = (msg ** self.private_key) % self.n
        elif isinstance(msg, str):
            decrypted = ''
            for letter in msg:
                decrypted = decrypted + chr((ord(letter) ** self.private_key) % self.n)
        return decrypted


class BreakRSA(RSA):
    def __init__(self, public_key, n, period_finder):
        p, q = 1, n
        while p == 1 or q == 1:
            x, r, p, q = shor(n, period_finder)

        super().__init__(max_prime=None, p=p, q=q, do_print=False)
        phi = (p - 1) * (q - 1)
        self.public_key = public_key
        self.private_key = self.get_private_key(phi)
        print(f'Found N: {n}, p: {p}, q: {q}')
        print(f'Public key: {self.public_key, self.n}, Private key: {self.private_key, self.n}')
