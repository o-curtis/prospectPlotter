import numpy as np
import astropy.units as u

def wav_to_nu(wav):
    c = 2.998*10**18 #angstroms/s
    return c/wav

def maggies_to_lsun_per_hz(Fnu,zred=0.0,d_pc = 10.0,epsilon=1e-10):
    """
    Converts luminosity in Lsun/Hz to flux density in maggies.
    
    Args:
    - Lnu (np.ndarray): Luminosity in Lsun/Hz.
    - zred (float): Redshift.
    
    Returns:
    - np.ndarray: Flux density in maggies.
    """
    pc_cgs = 3.086e18 # cm for 1 pc
    Lsun_cgs = 3.828e33 # erg s^-1
    maggie_cgs = 3631e-23 #erg s^-1 cm^-2
    
    Fnu_fixed = np.where(Fnu <= 0.0, epsilon, Fnu) # avoiding log(0)
    d_cm = pc_cgs * d_pc
    Lnu_norm = ((1 + zred)[:, None]) / (4 * np.pi)
    ln_Lnu = np.log(Fnu_fixed) + 2 * np.log(d_cm) - np.log(Lsun_cgs) + np.log(maggie_cgs) # ln maggies
    Lnu = np.exp(ln_Lnu) / Lnu_norm
    Lnu = np.where(Fnu <= 0.0, 0.0, Lnu) # setting 0 or negative luminosities to 0 flux
    return Lnu

def lsun_per_hz_to_maggies(Lnu,zred=0.0,d_pc = 10.0,epsilon=1e-10):
    """
    Converts luminosity in Lsun/Hz to flux density in maggies.
    
    Args:
    - Lnu (np.ndarray): Luminosity in Lsun/Hz.
    - zred (float): Redshift.
    
    Returns:
    - np.ndarray: Flux density in maggies.
    """
    Lnu_fixed = np.where(Lnu <= 0.0, epsilon, Lnu) # avoiding log(0)
    d_cm = pc_cgs * d_pc
    Fnu_norm = ((1 + zred)[:, None]) / (4 * np.pi)
    ln_Fnu = np.log(Lnu_fixed) - 2 * np.log(d_cm) + np.log(Lsun_cgs) - np.log(maggie_cgs) # ln maggies
    Fnu = np.exp(ln_Fnu) * Fnu_norm
    Fnu = np.where(Lnu <= 0.0, 0.0, Fnu) # setting 0 or negative luminosities to 0 flux
    return Fnu
