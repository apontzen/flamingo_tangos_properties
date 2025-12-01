import pynbody.units
import numpy as np


def entropy(M200m, h=0.68, z=0, OmegaM0=0.307, OmegaB0=0.048):
    G = pynbody.units.Unit("G")
    Msol = pynbody.units.Unit("Msol")
    H0 = pynbody.units.Unit("100 km s^-1 Mpc^-1") * h


    entrop = ((G*M200m*Msol)**(2,3) / 2) * ((1/400) * OmegaM0 * (8*np.pi*G)**2 / (9 * (H0)**2 * (1+z)**3 * OmegaB0**2))**(1,3)    

    return entrop.in_units("Msol^-2/3 kpc^2 km^2 s^-2")