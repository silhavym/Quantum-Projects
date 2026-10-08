import pennylane as pq
from pennylane import numpy as np 
import matplotlib.pyplot as plt

#-------------------------------------------------------------------------------------------------------
# Set parameters
#-------------------------------------------------------------------------------------------------------

N = 2
J_coupling = 1.0
J_competing = 0.7

n_qubits = N ** 3

#-------------------------------------------------------------------------------------------------------
# Qubit Coordinate system
#-------------------------------------------------------------------------------------------------------

def q(x, y, z):
    return x + N * (y + N * z)

#-------------------------------------------------------------------------------------------------------
# Build the Hamiltonian
#-------------------------------------------------------------------------------------------------------

coefficient = [] 
observable = []

for x in range(N):
    for y in range(N):
        for z in range(N):
            i = q(x, y, z)
            
            # Nearest neighbor interactions
            if x + 1 < N:
                j = q(x + 1, y, z)
                coefficient.append(J_coupling)
                observable.append(pq.PauliZ(i) @ pq.PauliZ(j))

            if y + 1 < N:
                j = q(x, y + 1, z)
                coefficient.append(J_coupling)
                observable.append(pq.PauliZ(i) @ pq.PauliZ(j))

            if z + 1 < N:
                j = q(x, y, z + 1)
                coefficient.append(J_coupling)
                observable.append(pq.PauliZ(i) @ pq.PauliZ(j))

            # Competing interactions
            if x + 1 < N and y + 1 < N:
                j = q(x + 1, y + 1, z)
                coefficient.append(J_competing)
                observable.append(pq.PauliZ(i) @ pq.PauliZ(j))

            if x + 1 < N and z + 1 < N:
                j = q(x + 1, y, z + 1)
                coefficient.append(J_competing)
                observable.append(pq.PauliZ(i) @ pq.PauliZ(j))

            if z + 1 < N and y + 1 < N:
                j = q(x, y + 1, z + 1)
                coefficient.append(J_competing)
                observable.append(pq.PauliZ(i) @ pq.PauliZ(j))

H = pq.Hamiltonian(coefficient, observable)

#-------------------------------------------------------------------------------------------------------
# Define the quantum device
#-------------------------------------------------------------------------------------------------------

dev = pq.device("default.qubit", wires=n_qubits)  

#-------------------------------------------------------------------------------------------------------
# Variational Circuit
#-------------------------------------------------------------------------------------------------------

depth = 2

def circuit(weights):
    for wire in range(n_qubits):
        pq.Hadamard(wires=wire)
        
    for layers in range(depth):
        for wire in range(n_qubits):
            pq.Rot(
                weights[layers, wire, 0],
                weights[layers, wire, 1],
                weights[layers, wire, 2],
                wires=wire  
            )

    # Entangle neighboring spins
    for x in range(N):
        for y in range(N):
            for z in range(N):
                i = q(x, y, z)

                if x + 1 < N: 
                    j = q(x + 1, y, z) 
                    pq.IsingZZ(0.5, wires=[i, j])

                if y + 1 < N: 
                    j = q(x, y + 1, z) 
                    pq.IsingZZ(0.5, wires=[i, j])

                if z + 1 < N: 
                    j = q(x, y, z + 1) 
                    pq.IsingZZ(0.5, wires=[i, j])

#-------------------------------------------------------------------------------------------------------
# Energy QNode
#-------------------------------------------------------------------------------------------------------

@pq.qnode(dev)
def energy_circuit(weights):
    circuit(weights)
    return pq.expval(H)

#-------------------------------------------------------------------------------------------------------
# Initial Parameters & Optimization
#-------------------------------------------------------------------------------------------------------

weights = np.array(0.01 * np.random.randn(depth, n_qubits, 3), requires_grad=True)
optimizer = pq.AdamOptimizer(stepsize=0.5)
energy_history = []

for steps in range(100):
    weights, previous_energy = optimizer.step_and_cost(energy_circuit, weights)
    energy = energy_circuit(weights)
    energy_history.append(float(energy))

    if steps % 10 == 0:
        print(f"Step {steps:3d} | Energy = {float(energy):.8f}")

print("Optimization complete")
#-------------------------------------------------------------------------------------------------------
# Final Energy & Spin Measurement (Restored)
#-------------------------------------------------------------------------------------------------------

final_energy = energy_circuit(weights)
print("Optimization complete")

@pq.qnode(dev)
def spin_measurement(weights):
     circuit(weights)
     return pq.probs(wires=range(n_qubits))

probabilities = spin_measurement(weights)

# Most probable computational-basis state
state_number = int(np.argmax(probabilities))
binary_state = format(state_number, f"0{n_qubits}b")

# Convert to +1 & -1 
spins = np.array([1 if bit == "0" else -1 for bit in binary_state])
spins = spins.reshape((N, N, N))

#-------------------------------------------------------------------------------------------------------
# Energy Convergence Plot
#-------------------------------------------------------------------------------------------------------

plt.figure(figsize=(8, 5))
plt.plot(energy_history, linewidth=2)

plt.xlabel("Optimization Step")
plt.ylabel("Energy")
plt.title("3-D Frustrated Magnet - VQE Energy")
plt.grid(True)
plt.tight_layout()
plt.show()

#-------------------------------------------------------------------------------------------------------
# 3-D configuration
#-------------------------------------------------------------------------------------------------------\

fig = plt.figure(figsize=(8,7))
ax = fig.add_subplot(111,projection ='3d')

for x in range(N):
    for y in range(N):
        for z in range(N):
            
            # Nearest neighbor interactions 
            if x + 1 < N:
                ax.plot3D([x, x + 1], [y, y], [z, z], color='black', linestyle='-', alpha=0.5, linewidth=2)
            if y + 1 < N:
                ax.plot3D([x, x], [y, y + 1], [z, z], color='black', linestyle='-', alpha=0.5, linewidth=2)
            if z + 1 < N:
                ax.plot3D([x, x], [y, y], [z, z + 1], color='black', linestyle='-', alpha=0.5, linewidth=2)
                
            # Competing interactions (dashed orange lines)
            if x + 1 < N and y + 1 < N:
                ax.plot3D([x, x + 1], [y, y + 1], [z, z], color='orange', linestyle='--', alpha=0.4, linewidth=2)
            if x + 1 < N and z + 1 < N:
                ax.plot3D([x, x + 1], [y, y], [z, z + 1], color='orange', linestyle='--', alpha=0.4, linewidth=2)
            if z + 1 < N and y + 1 < N:
                ax.plot3D([x, x], [y, y + 1], [z, z + 1], color='orange', linestyle='--', alpha=0.4, linewidth=2)


for x in range(N):
    for y in range(N):
        for z in range(N):
            spin = spins[x, y, z]
            color_choice = 'blue' if spin > 0 else 'red'
            ax.quiver(x, y, z, 0, 0, spin, length=0.35, normalize=True, color=color_choice, linewidth=2)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Final Low-Energy Spin Configuration with Lattice Bonds")

plt.tight_layout()
plt.show()