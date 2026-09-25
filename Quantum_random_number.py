import pennylane as pq
import numpy as np 

nbit = 8 
dev = pq.device('default.qubit', wires = nbit, shots=1)

@pq.qnode(dev)
def random_number():
    for i in range(nbit):
        pq.Hadamard(wires=i)
    return pq.sample(wires=range(nbit))

bit_array = random_number()[0]

random_integer = int("".join(map(str,bit_array)),2)

print(f"Generated Bitstring : {bit_array}")
print(f"Random Integer (0-{2**nbit - 1}): {random_integer}")