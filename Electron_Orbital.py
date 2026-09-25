# Probabilty maps of electron orbitals

import numpy as np 
from scipy.special import factorial, genlaguerre, sph_harm_y
import matplotlib.pyplot as plt 

def get_orbital(n,l,m,X,Y,Z):
    radius = np.sqrt(X**2 + Y**2 + Z**2)
    radius_safe = np.where(radius == 0, 1e-10, radius)
    theta = np.arccos(Z/radius_safe)
    phi = np.arctan2(Y,X)

    rho = 2.0*radius/n
    norm_radial = np.sqrt((2.0 / n) ** 3 * factorial(n - l - 1) / (2.0 * n * factorial(n + l)))
    laguerre = genlaguerre(n - l - 1, 2 * l + 1)(rho)
    R_nl = norm_radial * np.exp(-rho / 2.0) * (rho**l) * laguerre

    Y_lm = sph_harm_y(l, m, theta, phi)

    psi = R_nl*Y_lm

    return np.abs(psi)**2

x = y = z = np.linspace(-15,15,50)
X, Y, Z = np.meshgrid(x,y,z)

orbitals = [
    (1, 0, 0, "1s"),
    (2, 0, 0, "2s"),
    (2, 1, 0, "2p_z"),
    (3, 2, 0, "3d_z^2")]

fig, axes = plt.subplots(2,2,figsize=(8,8), subplot_kw={'projection':'3d'})

for ax, (n, l, m, title) in zip(axes.flat, orbitals):
    prob = get_orbital(n,l,m,X,Y,Z)
    cutoff = .05 * prob.max()
    mask = prob > cutoff

    map = ax.scatter(X[mask], Y[mask], Z[mask], c= prob[mask], cmap = 'magma', alpha =0.25, s=8)
    ax.set_title(title, fontsize=12)
    ax.set_xlim(-15,15)
    ax.set_ylim(-15,15)
    ax.set_zlim(-15,15)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

plt.tight_layout()
plt.show()