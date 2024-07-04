import numpy as np
import matplotlib.pyplot as plt
import math

from scipy.integrate import odeint

# Definition der Differentialgleichung für Josephson-Oszillationen in einem BEC
def bec_josephson_oscillations(y, t, omega, J):

    z, phi = y
    dz_dt = -2 * J * np.sin(phi)
    dphi_dt = omega + 2 * J * z / np.sqrt(1 - z**2) * np.cos(phi)

    return [dz_dt, dphi_dt]

# Parameter
omega = 0.01  # Frequenzunterschied zwischen den beiden BECs
J = 0.5  # Kopplungsstärke

# Anfangsbedingungen
z0 = 0.0  # Initiales Populationsungleichgewicht
phi0 = 0.00*math.pi  # Initiale Phasendifferenz
initial_conditions = [z0, phi0]

# Zeitspanne
t = np.linspace(0, 10, 5000)  # Zeitarray

# Lösung der Differentialgleichung
solution = odeint(bec_josephson_oscillations, initial_conditions, t, args=(omega, J))

# Extrahieren der Lösungen
z = solution[:, 0]
phi = solution[:, 1]

# Plotten der Ergebnisse
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(t, z, label='Population Imbalance $z(t)$')
plt.xlabel('Time t [sec]')
plt.ylabel('$z(t)$')
plt.title('Josephson Oscillations in a Bose-Einstein condensate - Population Imbalance')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t, phi, label='Phase Difference $\phi(t)$', color='orange')
plt.xlabel('Time t [sec]')
plt.ylabel('$\phi(t)$ [rad]')
plt.title('Josephson-Oscillations in a Bose-Einstein condensate - Phase Difference')
plt.legend()

plt.tight_layout()
plt.show()
