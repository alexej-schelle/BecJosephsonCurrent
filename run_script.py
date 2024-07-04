# main_script.py
import os
import sys
import math
import pylab
import matplotlib.pyplot as plt
import subprocess

phi_collect_totals = []
interference_totals = []

for k in range(5):

    from TimeMeasurement import interference
    from TimeMeasurement import phi_collect

    with open("TimeMeasurement.py") as file:
        
        code = file.read()
        exec(code)

    for l in range(len(phi_collect)):

        phi_collect_totals.append(phi_collect[l])
        
    for m in range(len(interference)):

        interference_totals.append(interference[m])

    # Plot superimposed wave fields at equilibrium    

plt.figure(1)
plt.set_cmap("Blues")
plt.hist2d(phi_collect_totals, interference_totals, bins = 200, density = True)
plt.tick_params(axis='both', which='major', labelsize = 14)
plt.ylabel('$|\Psi|^2$', fontsize = 14)
plt.xlabel('$\phi$', fontsize = 14)
cbar = plt.colorbar()
cbar.ax.set_ylabel('$\Pi[Re(\Psi), Im(\Psi))]$')
plt.savefig('/Users/krealix/Desktop/KREALIX/QuantizedTime/T1_vs_T2/fig_1.png')