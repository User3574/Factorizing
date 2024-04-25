import pandas
import time
import json

from qiskit import IBMQ
from qiskit_aer import Aer, AerSimulator
from azure.quantum.qiskit import AzureQuantumProvider
from rsa import *
from order_generic import *
from order_classic import *
from order_quantum import *
from qiskit.providers.fake_provider import FakePrague
from qiskit_aer.noise import NoiseModel
from qiskit.result import Result


def run_exp_0():
    # Experiment 0: Brute-force factorization
    digits = 8
    primes = get_primes(10 ** digits)

    N = {}
    for d in range(1, digits):
        primes_with_digits = filter_primes_digits(d, primes)
        p, q = random.sample(primes_with_digits, 2)
        n = p * q
        print(f"p: {p}, q: {q}, n:{n}")
        N[n] = d

    for n, d in N.items():
        start = time.time()
        p, q = find_factors(n)
        end = time.time()
        print(f'n: {n}, digits: {d}, p: {p}, q: {q}, time: {end - start}')

    # Experiment 0.5: Produce table
    # data = {}
    # data['x'] = [2, 3, 6, 8, 10, 12, 14]
    # data['y'] = [9.059906005859375e-06, 2.86102294921875e-06,
    #              1.3113021850585938e-05, 0.00025725364685058594,
    #              0.007128238677978516, 0.043366193771362305,
    #              0.3220024108886719]
    # df = pandas.DataFrame.from_dict(data)
    # df.plot(x='x', y='y', marker='.', legend=None)
    # plt.title('Factorization of n (brute-force)')
    # plt.xlabel('Digits of n')
    # plt.ylabel('Time taken (s)')
    # plt.show()


def run_exp_1():
    # Experiment 1: Communication based on RSA
    alice, bob = RSA(max_prime=1000), RSA(max_prime=1000)
    encrypted = bob.encrypt(msg=150, public_key=alice.public_key, n=alice.n)
    decrypted = alice.decrypt(encrypted)
    print(f'Encrypted: {encrypted}, Decrypted: {decrypted}')
    encrypted = bob.encrypt(msg='secret', public_key=alice.public_key, n=alice.n)
    decrypted = alice.decrypt(encrypted)
    print(f'Encrypted: {encrypted}, Decrypted: {decrypted}')


def run_exp_2():
    # Experiment 2: Breaking RSA
    alice, bob = RSA(p=3, q=5), RSA(max_prime=1000)
    public_key, n = alice.public_key, alice.n
    attacker = BreakRSA(n=n, public_key=public_key, period_finder=ClassicalPeriodFinder())
    encrypted = bob.encrypt(msg=6, public_key=alice.public_key, n=alice.n)
    decrypted = alice.decrypt(encrypted)
    decrypted_attacker = attacker.decrypt(encrypted)
    print(f'Encrypted: {encrypted}, Decrypted: {decrypted}, Decrypted by Attacker: {decrypted_attacker}')
    plot_periodic_function(a=13, N=15)
    plt.show()


def get_ibm_provider(backend):
    # Provider - IBMQ
    # IBMQ.save_account('')
    IBMQ.load_account()
    provider = IBMQ.get_provider()
    # for p in provider.backends():
    #    print(p)
    return provider.get_backend(backend)


def get_azure_provider(backend):
    # Provider - Azure
    provider = AzureQuantumProvider(
        resource_id="",
        location="West Europe"
    )
    # print("This workspace's targets:")
    # for backend in provider.backends():
    #     print("- " + backend.name())
    # ionq.qpu.aria-1, ionq.simulator, quantinuum.sim.h1-1sc, quantinuum.qpu.h1-1, rigetti.sim.qvm, rigetti.qpu.ankaa-2
    return provider.get_backend(backend)


if __name__ == '__main__':
    # Experiment 3 - Quantum Order Finding
    backend = get_ibm_provider('ibm_brisbane')
    # backend = get_azure_provider('ionq.simulator')
    # backend = Aer.get_backend('aer_simulator')
    # backend = AerSimulator().from_backend(get_ibm_provider('ibm_brisbane'))

    # start = time.time()
    # period_finder = SemiPeriodFinder(backend=backend, postprocess=True)
    # r = period_finder.simulate(2, 21)
    # running_time = time.time() - start
    # print(f'{running_time}')

    # execute_one_by_one = True
    # shots = 1024
    # circuits = []
    # for N in [21]:
    #     for a in [4]:
    #         period_finder = SemiPeriodFinder(backend=backend, postprocess=True, N=N)
    #         period_finder.create_circuit(a, draw=False)
    #         period_finder.circuit.name = f"shor_{a}_mod_{N}_semi"
    #         if execute_one_by_one:
    #             result = backend.run(transpile(period_finder.circuit, backend), shots=shots).result()
    #             result = result.to_dict()
    #             f = open(f"shor_{a}_mod_{N}_semi", "w")
    #             # f.write(json.dumps(result))
    #             f.close()
    #         else:
    #             circuits.append(transpile(period_finder.circuit, backend))
    #     if not execute_one_by_one:
    #         backend.run(circuits, shots=shots)

    # execute_one_by_one = True
    # shots = 1024
    # circuits = []
    # for a in [4]:
    #     period_finder = PeriodFinder4_21(backend=backend, postprocess=True)
    #     period_finder.create_circuit(a, draw=False)
    #     period_finder.circuit.name = f"shor_{a}_mod21_optimized"
    #     if execute_one_by_one:
    #         result = backend.run(transpile(period_finder.circuit, backend), shots=shots)
    #     else:
    #         circuits.append(transpile(period_finder.circuit, backend))
    # if not execute_one_by_one:
    #     backend.run(circuits, shots=shots)
    #
    # execute_one_by_one = True
    # shots = 1024
    # circuits = []
    # for a in [11]:
    #     period_finder = PeriodFinder11_15(backend=backend, postprocess=True)
    #     period_finder.create_circuit(a, draw=True)
    #     period_finder.circuit.name = f"shor_{a}_mod15_optimized"
    #     if execute_one_by_one:
    #         result = backend.run(transpile(period_finder.circuit, backend), shots=shots)
    #     else:
    #         circuits.append(transpile(period_finder.circuit, backend))
    # if not execute_one_by_one:
    #     backend.run(circuits, shots=shots)

    # Submit jobs into queue (15)
    # execute_one_by_one = False
    # shots = 1024
    # circuits = []
    # for a in [2, 7, 8, 11, 13]:
    #     period_finder = PeriodFinder15(backend=backend, postprocess=True)
    #     period_finder.create_circuit(a, draw=False)
    #     period_finder.circuit.name = f"shor_{a}_mod15"
    #     if execute_one_by_one:
    #         result = backend.run(transpile(period_finder.circuit, backend), shots=shots)
    #     else:
    #         circuits.append(transpile(period_finder.circuit, backend))
    # if not execute_one_by_one:
    #     backend.run(circuits, shots=shots)
    #
    # execute_one_by_one = False
    # shots = 1024
    # circuits = []
    # for a in [2, 7, 8, 11, 13]:
    #     period_finder = PeriodFinder15Kitaev(backend=backend, postprocess=True)
    #     period_finder.create_circuit(a, draw=False)
    #     period_finder.circuit.name = f"shor_{a}_mod15_kitaev"
    #     if execute_one_by_one:
    #         result = backend.run(transpile(period_finder.circuit, backend), shots=shots)
    #     else:
    #         circuits.append(transpile(period_finder.circuit, backend))
    # if not execute_one_by_one:
    #     backend.run(circuits, shots=shots)
    #
    # Submit jobs into queue (21)
    execute_one_by_one = True
    shots = 1024
    circuits = []
    for a in [4, 5, 8, 13, 16]:
        period_finder = PeriodFinder21(backend=backend, postprocess=True)
        period_finder.create_circuit(a, draw=False)
        period_finder.circuit.name = f"shor_{a}_mod21"
        if execute_one_by_one:
            result = backend.run(transpile(period_finder.circuit, backend), shots=shots)
        else:
            circuits.append(transpile(period_finder.circuit, backend))
    if not execute_one_by_one:
        backend.run(circuits, shots=shots)

    # Retrieve job
    # job = backend.retrieve_job("crdxq9dnzrx00081w0x0")
    # result = job.result()
    # for i in range(len(job.result().results)):
    #     counts = result.get_counts(i)
    #     classical_postprocessing(counts, 2**8, 10, True)

    # Retrieve job
    # f = open('/home/user3574/PycharmProjects/master_shor/experiments/simulator/noise/21/shor_4_21_optimized.json')
    # data = json.load(f)
    # result = Result.from_dict(data)
    # for i in range(len(result.results)):
    #     counts = result.get_counts(i)
    #     classical_postprocessing(counts, 2**8, 1024, True)
    # f.close()

    # a = 4
    # period_finder = PeriodFinder21(provider, backend_name=backend, postprocess=True)
    # r = period_finder.simulate(a)
    # print(f'Period finding: N: {21}, a: {a}, r: {r}')
    # plot_periodic_function(a=a, N=21)

    # a = 2
    # period_finder = PeriodFinder15Kitaev(provider, backend_name=backend, postprocess=True)
    # r = period_finder.simulate(a)
    # print(f'Period finding: N: {15}, a: {a}, r: {r}')
    # plot_periodic_function(a=a, N=15)

    # period_finder = SemiPeriodFinder(provider=Aer, backend_name='aer_simulator', postprocess=True)
    # period_finder.simulate(4, 21)
