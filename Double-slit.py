import pennylane as pq
import numpy as np 
import matplotlib.pyplot as plt


dev = pq.device("default.qubit", wires = 2)

@pq.qnode(dev)
def double_slit_circuit(phase_x, phase_y):
    pq.Hadamard(wires=0)
    pq.Hadamard(wires=1)

    pq.CNOT(wires=[0,1])

    pq.PhaseShift(phase_y, wires=0)
    pq.PhaseShift(phase_x, wires=1)

    pq.Hadamard(wires=0)
    pq.Hadamard(wires=1)

    return pq.probs(wires=[0,1])

maxrange =  16 * np.pi
phase_y = np.linspace(0, maxrange, 100)
phase_x = np.linspace(0, maxrange, 100)

X, Y = np.meshgrid(phase_x, phase_y)
Z = np.zeros_like(X)


for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        probs = double_slit_circuit(X[i,j], Y[i,j])
        Z[i,j] = probs[1] + probs[3]

center_x = maxrange / 2
center_y = maxrange / 2

scale = 3.5 * np.pi 
x_env = np.sinc((X - center_x) / (np.pi * scale)) ** 2
y_env = np.sinc((Y - center_y) / (np.pi * scale)) ** 2

Z_env = Z * y_env * x_env

plt.figure(figsize =(9,8))
plt.imshow(Z_env, extent=[0, maxrange, 0, maxrange], origin='lower', cmap = 'magma', interpolation='bicubic')
plt.colorbar(label = 'Light Intensity')
plt.title('Double-Slit Interference and Diffraction pattern')
plt.xlabel('Screen positon X')
plt.ylabel('Screen postion Y')
plt.show()