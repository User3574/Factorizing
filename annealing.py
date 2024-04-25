import dimod
from dwave.system import DWaveSampler, EmbeddingComposite

if __name__ == '__main__':
    Q = {(1, 1): -52, (2, 2): -52, (3, 3): -96, (4, 4): 768,
         (1, 2): 200, (1, 3): -48, (1, 4): -512,
         (2, 3):16, (2, 4): -512,
         (3, 4):128}
    offset = 196
    l, q, o = dimod.utilities.qubo_to_ising(Q)
    sampler = EmbeddingComposite(DWaveSampler())
    result = sampler.sample_qubo(Q, num_reads=1024)
    print(result)
