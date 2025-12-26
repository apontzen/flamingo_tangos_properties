import pynbody.units
import numpy as np


def entropy(M200m, h=0.68, z=0, OmegaM0=0.307, OmegaB0=0.048):
    G = pynbody.units.Unit("G")
    Msol = pynbody.units.Unit("Msol")
    H0 = pynbody.units.Unit("100 km s^-1 Mpc^-1") * h


    entrop = ((G*M200m*Msol)**(2,3) / 2) * ((1/400) * OmegaM0 * (8*np.pi*G)**2 / (9 * (H0)**2 * (1+z)**3 * OmegaB0**2))**(1,3)    

    return entrop.in_units("Msol^-2/3 kpc^2 km^2 s^-2")

def velocity(M200m, h=0.68, z=0, OmegaM0=0.307):
    G = pynbody.units.Unit("G")
    Msol = pynbody.units.Unit("Msol")
    H0 = pynbody.units.Unit("100 km s^-1 Mpc^-1") * h

    vel = (100*OmegaM0)**(1./6.) * (G*M200m*Msol*H0)**(1,3) * (1+z)**(1./2.)

    return vel.in_units("km s^-1")

def temperature(M200m, h=0.68, z=0, OmegaM0=0.307, mu=0.59):
    kB = pynbody.units.Unit("k")
    mP = pynbody.units.Unit("m_p")
    vel = velocity(M200m, h=h, z=z, OmegaM0=OmegaM0) * pynbody.units.Unit("km s^-1")

    temp = mu * mP / (2 * kB) * vel**2

    return temp.in_units("K")

def virial_specific_energy(M200m, h=0.68, z=0, OmegaM0=0.307):
    vel = velocity(M200m, h=h, z=z, OmegaM0=OmegaM0)
    spec_energy = 0.5 * vel**2
    return spec_energy* pynbody.units.Unit("km^2 s^-2").in_units("erg Msol^-1")
