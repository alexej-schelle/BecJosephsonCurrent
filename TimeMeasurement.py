
############################################################################################################################################################################
#
#
# -*- coding: utf-8 -*-
#
#
############################################################################################################################################################################
#                                                                                                                                              																										  
#   Markov sampling method for calculation of two counter-propagating atom lasers at tempoeratures T1 and T2
#
#   The Python TimeMeasurement.py calculates :
#
#   - Frequency comb spectra of counter-propagating wave fields (atomic lasers) released from a magneto-optical trap
# 
#  * :
# 
#   License Copyright:  Dr. A. Schelle, Bachschmidstr. 4, 87600 Kaufbeuren 
#   License Type :      MIT license (2024)
#   License Contact:    E-Mail : alexej.schelle@gmail.com
# 
#   ** : 
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files 
#   (the "Software" bec_symmetry_breaking.py), to deal in the Software without restriction, including without limitation the rights to use, 
#   copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is 
#   furnished to do so, subject to the following conditions:
# 
#   The above copyright notice (*) and this permission notice (**) shall be included in all copies or substantial portions of the Software.
# 
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
#   WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
#
############################################################################################################################################################################

import os
import sys
import math
import random
import numpy
import numpy as np
import pylab
import matplotlib.pyplot as plt
import operator
import pandas as pd
from sklearn.linear_model import LinearRegression

# Corrective term : \delta N / N = (pi*k_B*T)/(2*\rho*g) (Is a valid approximation for large trapping potentials and high particle densities)

maxmode = 50 # Typical mode size for analysis: 500 - 2500 modes
ptn = 1000 # Typical particle number: 10^2 - 10^4
sample = 100 # Typical sample size: 100 - 200

omx = 2.0*math.pi*50.00 # Trap frequency in x direction  
omy = 2.0*math.pi*150.00 # Trap frequency in y direction 
omz = 2.0*math.pi*250.00 # Trap frequency in z direction

start_temp = 100.0 # in units of nK
rhbkb = 7.63822291E-3 # in units of nK

norm = 0.0
int_norm = 0
drop = 0
prob = 0.0
phase_0 = 0.0
z_start = start_temp
mu_start = 1.0
mu_k = 0.0

# Simulate the field atom laser

print ('Cavity depth [nK]: ', maxmode*omz*rhbkb) # Depth of the external potential - around few nK
print ('Critical temperature [nK]: ', rhbkb*pow(ptn,1.0/3.0)*pow(omx*omy*omz,1.0/3.0)/pow(1.202,1.0/3.0)) # around 1.15 mK

en_x = ['']*maxmode
en_y = ['']*maxmode
en_z = ['']*maxmode

pols_x = ['']*maxmode
pols_y = ['']*maxmode
pols_z = ['']*maxmode

mu_pols_x = ['']*maxmode
mu_pols_y = ['']*maxmode
mu_pols_z = ['']*maxmode

pols = ['']*maxmode

x = ['']*ptn
y = ['']*ptn

mu_x_1 = [] # Collect the real part of the integrated wave field
mu_y_1 = [] # Collect the imaginary part of the integrated wave field

mu_x_2 = [] # Collect the real part of the integrated wave field
mu_y_2 = [] # Collect the imaginary part of the integrated wave field

phase_dist_1 = [] # Collect the phases for distribution
phase_dist_2 = [] # Collect the phases for distribution

field_real = [] # Collect the phases for distribution
field_imag = [] # Collect the phases for distribution

phi_collect = []
interference = []

phi = 0.0

for l in range(1, sample):
          
    drop = 1
    z = ptn

    start_temp = 10.0

    mu = 0.0
    norm = 0.0

    betamu = complex(0.0,0.0)
    temp = start_temp
                  
    for k in range(1, maxmode):
        
        en_x[k] = rhbkb*k*omx/temp # Energy in x direction
        en_y[k] = rhbkb*k*omy/temp # Energy in y direction
        en_z[k] = rhbkb*k*omz/temp # Energy in z direction
     
        pols_x[k] = 1.0/(1.0-math.exp(-en_x[k])) # Complex poles in x direction
        pols_y[k] = 1.0/(1.0-math.exp(-en_y[k])) # Complex poles in y direction
        pols_z[k] = 1.0/(1.0-math.exp(-en_z[k])) # Complex poles in z direction
                    
    for k in range(1, maxmode):
    
        pols[k-1] = pols_x[maxmode-k]*pols_y[maxmode-k]*pols_z[maxmode-k]-1.0 # General poles
    
    pols[maxmode-1] = -(ptn-z_start)

    prob = complex(0.0,0.0)
    p = ['']*maxmode
    phase_0 = 0.0
   
    x = numpy.roots(pols) # Complex roots of the number conserving equation

    for k in range(0,len(x)):

        p[k] = random.uniform(0.0,1.0) # Random amplitudes - set p[k] = delta(k-k_random) for single field spectrum
        norm = norm + p[k]*p[k]*(x[k].real**2+x[k].imag**2) # Total norm

        mu_pols_x.append(x[k].real)
        mu_pols_y.append(x[k].imag)

    norm = math.sqrt(norm)
    
    for k in range(0,len(x)): # Calculate phase of the quantum field
        	
        p[k] = p[k]/norm # Random amplitudes
    	
        if (operator.gt(x[k].real**2 + x[k].imag**2,0.0)):
                
            if (operator.gt(x[k].real,0.0)):
            
                phase_0 = math.atan(x[k].imag/x[k].real)
        
            if (operator.iand(operator.lt(x[k].real,0.0),operator.ge(x[k].imag,0.0))):
            
                phase_0 = math.atan(x[k].imag/x[k].real) + math.pi
    
            if (operator.iand(operator.lt(x[k].real,0.0),operator.lt(x[k].imag,0.0))):
	            
                phase_0 = math.atan(x[k].imag/x[k].real) - math.pi
    
            if (operator.iand(operator.eq(x[k].real,0.0),operator.gt(x[k].imag,0.0))):
            
                phase_0 = 0.5*math.pi

            if (operator.iand(operator.eq(x[k].real,0.0),operator.lt(x[k].imag,0.0))):
            
                phase_0 = -0.5*math.pi

            if operator.ne(phase_0,0.0):
    	
                prob += complex(0.5*p[k]*p[k]*math.log(math.fabs(x[k].real**2 + x[k].imag**2)), p[k]*p[k]*phase_0) # Calculate transition probability
    	
            betamu = complex(prob.real, prob.imag)

            prob = math.sqrt(prob.real**2 + prob.imag**2) 
            mu += x[k]*p[k] # Random amplitudes times phases

    if (operator.gt(min((math.exp(prob))/(math.exp(mu_start)),1.00),random.uniform(0.00,1.00))): # Condition for transition to another quantum state at equilibrium

        mu_start = prob
        z_start = z
        drop = 0
	    
    if (operator.ne(drop,1)):	
		
        mu_x_1.append(1.0*mu.real) # Collect all modes without zero mode
        mu_y_1.append(1.0*mu.imag) # Collect all modes without zero mode

        mu_x_1.append(-1.0*mu.real) # Collect zero mode
        mu_y_1.append(1.0*mu.imag) # Collect zero mode
        
        if (operator.gt(mu.real,0.0)): # First case
            
            phase_0 = math.atan(mu.imag/mu.real)
        			
        if (operator.iand(operator.lt(mu.real,0.0),operator.ge(mu.imag,0.0))): # Second case
            
      	    phase_0 = math.atan(mu.imag/mu.real) + math.pi
    
        if (operator.iand(operator.lt(mu.real,0.0),operator.lt(mu.imag,0.0))): # Third case
	            
            phase_0 = math.atan(mu.imag/mu.real) - math.pi 
    
        if (operator.iand(operator.eq(mu.real,0.0),operator.gt(mu.imag,0.0))): # Fourth case
            
            phase_0 = 0.5*math.pi

        if (operator.iand(operator.eq(mu.real,0.0),operator.lt(mu.imag,0.0))): # Fifth case
            
            phase_0 = -0.5*math.pi

        if (operator.lt(math.fabs((phase_0/(1.0*0.05*math.pi))%2-1.0),0.01)):

            if (operator.gt(phase_0,0.0)):
            
                phase_dist_1.append(phase_0/math.pi - 1.0)

            if (operator.lt(phase_0,0.0)):

                phase_dist_1.append(phase_0/math.pi + 1.0)


for l in range(1, sample):
          
    drop = 1
    z = ptn

    start_temp = 10.0

    mu = 0.0
    norm = 0.0

    betamu = complex(0.0,0.0)
    temp = start_temp

    omx = 2.0*math.pi*50.00 # Trap frequency in x direction  
    omy = 2.0*math.pi*50.00 # Trap frequency in y direction 
    omz = 2.0*math.pi*50.00 # Trap frequency in z direction
                  
    for k in range(1, maxmode):
        
        en_x[k] = rhbkb*k*omx/temp # Energy in x direction
        en_y[k] = rhbkb*k*omy/temp # Energy in y direction
        en_z[k] = rhbkb*k*omz/temp # Energy in z direction
     
        pols_x[k] = 1.0/(1.0-math.exp(-en_x[k])) # Complex poles in x direction
        pols_y[k] = 1.0/(1.0-math.exp(-en_y[k])) # Complex poles in y direction
        pols_z[k] = 1.0/(1.0-math.exp(-en_z[k])) # Complex poles in z direction
                    
    for k in range(1, maxmode):
    
        pols[k-1] = pols_x[maxmode-k]*pols_y[maxmode-k]*pols_z[maxmode-k]-1.0 # General poles
    
    pols[maxmode-1] = -(ptn-z_start)

    prob = complex(0.0,0.0)
    p = ['']*maxmode
    phase_0 = 0.0
   
    x = numpy.roots(pols) # Complex roots of the number conserving equation

    for k in range(0,len(x)):

        p[k] = random.uniform(0.0,1.0) # Random amplitudes - set p[k] = delta(k-k_random) for single field spectrum
        norm = norm + p[k]*p[k]*(x[k].real**2+x[k].imag**2) # Total norm

        mu_pols_x.append(x[k].real)
        mu_pols_y.append(x[k].imag)

    norm = math.sqrt(norm)
    
    for k in range(0,len(x)): # Calculate phase of the quantum field
        	
        p[k] = p[k]/norm # Random amplitudes
    	
        if (operator.gt(x[k].real**2 + x[k].imag**2,0.0)):
                
            if (operator.gt(x[k].real,0.0)):
            
                phase_0 = math.atan(x[k].imag/x[k].real)
        
            if (operator.iand(operator.lt(x[k].real,0.0),operator.ge(x[k].imag,0.0))):
            
                phase_0 = math.atan(x[k].imag/x[k].real) + math.pi
    
            if (operator.iand(operator.lt(x[k].real,0.0),operator.lt(x[k].imag,0.0))):
	            
                phase_0 = math.atan(x[k].imag/x[k].real) - math.pi
    
            if (operator.iand(operator.eq(x[k].real,0.0),operator.gt(x[k].imag,0.0))):
            
                phase_0 = 0.5*math.pi

            if (operator.iand(operator.eq(x[k].real,0.0),operator.lt(x[k].imag,0.0))):
            
                phase_0 = -0.5*math.pi

            if operator.ne(phase_0,0.0):
    	
                prob += complex(0.5*p[k]*p[k]*math.log(math.fabs(x[k].real**2 + x[k].imag**2)), p[k]*p[k]*phase_0) # Calculate transition probability
    	
            betamu = complex(prob.real, prob.imag)

            prob = math.sqrt(prob.real**2 + prob.imag**2) 
            mu += x[k]*p[k] # Random amplitudes times phases

    if (operator.gt(min((math.exp(prob))/(math.exp(mu_start)),1.00),random.uniform(0.00,1.00))): # Condition for transition to another quantum state at equilibrium

        mu_start = prob
        z_start = z
        drop = 0
	    
    if (operator.ne(drop,1)):	
		
        mu_x_2.append(1.0*mu.real) # Collect all modes without zero mode
        mu_y_2.append(1.0*mu.imag) # Collect all modes without zero mode

        mu_x_2.append(-1.0*mu.real) # Collect zero mode
        mu_y_2.append(1.0*mu.imag) # Collect zero mode
        
        if (operator.gt(mu.real,0.0)): # First case
            
            phase_0 = math.atan(mu.imag/mu.real)
        			
        if (operator.iand(operator.lt(mu.real,0.0),operator.ge(mu.imag,0.0))): # Second case
            
      	    phase_0 = math.atan(mu.imag/mu.real) + math.pi
    
        if (operator.iand(operator.lt(mu.real,0.0),operator.lt(mu.imag,0.0))): # Third case
	            
            phase_0 = math.atan(mu.imag/mu.real) - math.pi 
    
        if (operator.iand(operator.eq(mu.real,0.0),operator.gt(mu.imag,0.0))): # Fourth case
            
            phase_0 = 0.5*math.pi

        if (operator.iand(operator.eq(mu.real,0.0),operator.lt(mu.imag,0.0))): # Fifth case
            
            phase_0 = -0.5*math.pi

        if (operator.lt(math.fabs((phase_0/(1.0*0.05*math.pi))%2-1.0),0.01)):

            if (operator.gt(phase_0,0.0)):
            
                phase_dist_2.append(phase_0/math.pi - 1.0)

            if (operator.lt(phase_0,0.0)):

                phase_dist_2.append(phase_0/math.pi + 1.0)


# Model interference measurement of two superimposed coherent atom lasers as a function of the relative phase phi

phase_sample  = 5000
var_interference = 0.0

for m in range(0, phase_sample):

    phi = random.uniform(-1.00*math.pi, 1.00*math.pi)

    print('Sample step Nr. ' + str(m))
    minimum = sample

    var_interference = 0.0
    
    for k in range(0, minimum):

        for l in range(0, minimum):

            var_interference = var_interference + complex(mu_x_1[k],mu_y_1[k])*numpy.conj(complex(mu_x_1[l],mu_y_1[l])) + complex(mu_x_2[k],mu_y_2[k])*numpy.conj(complex(mu_x_2[l],mu_y_2[l])) + complex(mu_x_1[k],mu_y_1[k])*numpy.conj(complex(mu_x_2[l],mu_y_2[l]))*complex(math.cos(phi), math.sin(phi)) + numpy.conj(complex(mu_x_1[k],mu_y_1[k]))*complex(mu_x_2[l],mu_y_2[l])*complex(math.cos(phi), -math.sin(phi))
        
            field_real.append((complex(mu_x_1[k],mu_y_1[k])*numpy.conj(complex(mu_x_1[l],mu_y_1[l])) + complex(mu_x_2[k],mu_y_2[k])*numpy.conj(complex(mu_x_2[l],mu_y_2[l])) + complex(mu_x_1[k],mu_y_1[k])*numpy.conj(complex(mu_x_2[l],mu_y_2[l]))*complex(math.cos(phi), math.sin(phi)) + numpy.conj(complex(mu_x_1[k],mu_y_1[k]))*complex(mu_x_2[l],mu_y_2[l])*complex(math.cos(phi), -math.sin(phi))).real)
            field_imag.append((complex(mu_x_1[k],mu_y_1[k])*numpy.conj(complex(mu_x_1[l],mu_y_1[l])) + complex(mu_x_2[k],mu_y_2[k])*numpy.conj(complex(mu_x_2[l],mu_y_2[l])) + complex(mu_x_1[k],mu_y_1[k])*numpy.conj(complex(mu_x_2[l],mu_y_2[l]))*complex(math.cos(phi), math.sin(phi)) + numpy.conj(complex(mu_x_1[k],mu_y_1[k]))*complex(mu_x_2[l],mu_y_2[l])*complex(math.cos(phi), -math.sin(phi))).imag)

    phi_collect.append(phi/math.pi)
    interference.append(math.fabs(var_interference.imag**2 + var_interference.real**2))

plt.figure(1)
plt.hist2d(field_real, field_imag, bins = 200, density = True)
plt.tick_params(axis='both', which='major', labelsize = 12)
plt.ylabel('$Im(\Psi)$', fontsize = 12)
plt.xlabel('$Re(\Psi)$', fontsize = 12)
plt.xlim(-1.0, 1.0)
plt.ylim(-1.0, 1.0)
cbar = plt.colorbar()
cbar.ax.set_ylabel('$\Pi[Re(\Psi), Im(\Psi))]$')
plt.savefig('/.../fig_1.png')

plt.figure(2)
plt.set_cmap("Blues")
plt.hist2d(phi_collect, interference, bins = 200, density = True)
plt.tick_params(axis='both', which='major', labelsize = 12)
plt.xlim(-1.0, 1.0)
plt.ylim(-1.0, 1.0)
plt.ylabel('$|\Psi|^2$', fontsize = 12)
plt.xlabel('$\phi$', fontsize = 12)
cbar = plt.colorbar()
cbar.ax.set_ylabel('$\Pi[Re(\Psi), Im(\Psi))]$')
plt.savefig('/.../fig_2.png')
