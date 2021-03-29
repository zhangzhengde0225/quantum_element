import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as scl
import scipy.optimize as opt
# Setup the input values
hbar=1
m=1

for a in [1.,2.]:
    z0 = 4*np.pi /2
    V0 = 10 # (z0*hbar/a)**2 /m/2
    z0 = a/hbar * np.sqrt(2*m*V0)
    # V0 = 2*(np.pi*hbar/a)**2 /m/8

    epsilon = 0.0001 # Sufficiently small but not zero to avoid devide by zero issues.
    # Setup the two equations that need to be solved.
    def opt_fun1(z,z0):
        return( np.tan(z) - np.sqrt((z0/z)**2 -1 ))
    def opt_fun2(z,z0):
        return( -1/np.tan(z) - np.sqrt((z0/z)**2 -1 ))

    result = []
    z = []
    Ex = []
    for i in range(2*int(z0/np.pi)+1):
        if i*np.pi/2 < z0:
            if (i%2 == 0):
                result.append(opt.root_scalar(opt_fun1,z0,bracket=[np.pi*i/2+epsilon,min((i+1)*np.pi/2-epsilon,z0)]))
            else:
                result.append(opt.root_scalar(opt_fun2,z0,bracket=[np.pi*i/2+epsilon,min((i+1)*np.pi/2-epsilon,z0)]))
            if not result[i].converged:
                print(result[i])
                z.append(0)
                Ex.append(0)
            else:
                z.append(result[i].root)
                Ex.append((z[i]*hbar/a)**2 /m/2 - V0)
        else:
            z.append(0)
            Ex.append(0)
    print("Potential strenght = {:7.3f}".format(V0))
    print("Widht of the well  = {:7.37}".format(a))
    print("\n Binding energies are predicted to be:")
    for i in range(len(Ex)):
        if Ex[i]<0:print(" Ex[{:1d}] = {:6.3f}".format(i,Ex[i]))
    print()