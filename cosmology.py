import opt
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

c = 3e11

def f(H0, Omega, z):
    return 5 * np.log10(distance_modulus(H0, Omega, z)) + 25

def j(H0, Omega, z):
    jacobian = np.empty((len(z), 2))
    for i, z_i in enumerate(z):
        D_H = c / H0
        integrand = lambda z: 1 / np.sqrt(Omega_m * (1 + z) ** 3 + Omega_k * (1 + z) ** 2 + Omega_l)
        integral, _ = quad(integrand, 0, z_i)
        d_c = D_H * integral
        d_m = d_c / (1 + z_i)
        d_l = D_H * np.sqrt(1 + z_i) * integral
        mu = 5 * np.log10(d_l) + 25
        jacobian[i] = np.array([-2.5 * np.log10(np.e) * (d_m / d_l) / np.log(10), -2.5 * np.log10(np.e) * (1 - d_m / d_l) / np.log(10)])
    return j


H0, Omega = 50, 0.5

gn_params = []
lm_params = []
gn_losses = []
lm_losses = []

plt.plot(range(len(gn_result.cost)), gn_result.cost, label='Gauss-Newton')
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.legend()
plt.savefig('cost_gn.png')
plt.show()

plt.plot(range(len(lm_result.cost)), lm_result.cost, label='Levenberg-Marquardt')
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.legend()
plt.savefig('cost_lm.png')
plt.show()

if __name__=="__main__":
    with open('jla_mub.txt') as file:
        data = np.loadtxt(file)