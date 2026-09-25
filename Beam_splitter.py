import pennylane as pq 
import numpy as np 

dev = pq.device('default.qubit', wires=2)

@pq.qnode(dev)
def beam_splitter(theta):
    pq.PauliX(wires=0)
    pq.SingleExcitation(2*theta, wires=[0,1])
    return pq.probs(wires=[0,1])

probabilities = beam_splitter(9*np.pi/8)

print("Probability of photon in Mode 0 (|10>):", probabilities[2])
print("Probability of photon in Mode 1 (|01>):", probabilities[1])





