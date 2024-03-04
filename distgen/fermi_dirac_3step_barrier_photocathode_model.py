import numpy as np
import scipy.constants as const

from .physical_constants import unit_registry, pi, PHYSICAL_CONSTANTS

MC2 = PHYSICAL_CONSTANTS.species('electron')['rest_energy']
c = PHYSICAL_CONSTANTS['speed of light in vacuum']
kb = PHYSICAL_CONSTANTS['Boltzmann constant in eV/K']

def fermi_dirac(e, mu, t, t_cutoff=1e-3*unit_registry('K')):
    """
    Fermi-Dirac distribution
    :param e: energy to evaluate at
    :param mu: Fermi energy
    :param t: Temperature
    :param t_cutoff: Clip the temperature at this value to avoid divide by zero
    :return: value of the Fermi-Dirac distribution
    """
    
    return 1/(1 + np.exp(np.clip((e - mu)/(kb*np.clip(t, t_cutoff, None)), -256, 256)))

def fermi_dirac_3step_barrier_pdf_int(px, py, pz, fermi_energy, temp, photon_energy, workfun):
    
    cpx, cpy, cpz = c*px, c*py, c*pz

    e = (cpx**2 + cpy**2 + cpz**2)/2/MC2

    p_excite = (1 - fermi_dirac(e + photon_energy, fermi_energy, temp))*fermi_dirac(e, fermi_energy, temp)
    p_transport = cpz > 0
    p_escape = cpz**2/2/MC2 + photon_energy >= fermi_energy + workfun
    return p_excite*p_transport*p_escape


def ds_transform(px, py, pz, fermi_energy, photon_energy, workfun):

    cpx, cpy, cpz = c*px, c*py, c*pz

    a = np.sqrt(1 - photon_energy/((cpx**2 + cpy**2 + cpz**2)/2/MC2 + fermi_energy + workfun))

    return px*a, py*a, np.sqrt(cpz**2 + 2*MC2*(fermi_energy + workfun))*a/c


def ds_jacobian_factor(px, py, pz, fermi_energy, photon_energy, workfun):

    cpx, cpy, cpz = c*px, c*py, c*pz

    num = cpz*np.sqrt(cpx**2 + cpy**2 + cpz**2 + 2*MC2*(fermi_energy + workfun - photon_energy))
    den = np.sqrt((cpz**2 + 2*MC2*(fermi_energy + workfun))*(cpx**2 + cpy**2 + cpz**2 + 2*MC2*(fermi_energy + workfun)))

    return num/den


def fermi_dirac_3step_barrier_pdf(px, py, pz, photon_energy, workfun, temp, fermi_energy):
    """
    Calculates the (unnormalized) PDF for electrons emitted from a photocathode following the
    model described in [1]. Shortly, electrons are initially populated in momentum space as 
    in the Sommerfeld model. They escape over the work function barrier when their 
    longitudinal energy is high enough and lose momentum along the direction of the surface 
    normal to satisfy energy conservation. For the purposes of referencing in published 
    research, this sampling concept is the same described in [2].

    [1] Dowell, D. H., & Schmerge, J. F. (2009). Quantum efficiency and thermal emittance of 
        metal photocathodes. Physical Review Special Topics - Accelerators and Beams, 12(7). 
        https://doi.org/10.1103/PhysRevSTAB.12.074201

    [2] Pierce, C. M., Durham, D. B., Riminucci, F., Dhuey, S., Bazarov, I., Maxson, J.,
        Minor, A. M., & Filippetto, D. (2023). Experimental Characterization of Photoemission
        from Plasmonic Nanogroove Arrays. Physical Review Applied, 19(3), 034034.
        https://doi.org/10.1103/PhysRevApplied.19.034034

    :params px: x component of momentum (in eV/c)
    :params py: y component of momentum (in eV/c)
    :params pz: z component of momentum (in eV/c)
    :params photon_energy: photon energy of driving laser (in eV)
    :params workfun: photocathode work function (in eV)
    :params temp: photocathode temperature (in K)
    :params fermi_energy: photocathode Fermi energy (in eV; note, PDF should be insensitive to this value)
    :return: unnormalized density of the emitted electrons at this momentum
    """
    a = ds_jacobian_factor(px, py, pz, fermi_energy, photon_energy, workfun)

    return a*fermi_dirac_3step_barrier_pdf_int(*ds_transform(px, py, pz, fermi_energy, photon_energy, workfun), fermi_energy, temp, photon_energy, workfun)


def fermi_dirac_3step_barrier_pdf_bounds(photon_energy, workfun, temp, fermi_energy, n_tails=4):
    """
    Calculates bounding box of the non-zero part of the PDF
    :param temp: photocathode emperature (in K)
    :param photon_energy: photon energy of the driving laser (in eV)
    :param workfun: photocathode work function (in eV)
    :param n_tails: amount of Fermi tail to include (in units of kB*T)
    :return: (xl, xu), (yl, yu), (zl, zu), the lower (l) and upper (u) bounds along each axis (x, y, z)
    """
    a = np.sqrt(2*MC2*(photon_energy - workfun + kb*temp*n_tails))/c
    return [(-a, a), (-a, a), (0., a)]

def fermi_dirac_3step_barrier_pdf_bounds_quantity(photon_energy, workfun, temp, fermi_energy, n_tails=4):
    bounds = fermi_dirac_3step_barrier_pdf_bounds(photon_energy, workfun, temp, fermi_energy, n_tails=n_tails)
    return [bounds[0], bounds[1], (0*bounds[2][1], bounds[2][1])]


def fermi_dirac_3step_barrier_pdf_spherical(p, theta, photon_energy, workfun, temp, fermi_energy):
    return fermi_dirac_3step_barrier_pdf(p*np.sin(theta), 0.0*p, p*np.cos(theta), photon_energy, workfun, temp, fermi_energy)


def fermi_dirac_3step_barrier_pdf_bounds_spherical(photon_energy, workfun, temp, fermi_energy, n_tails=4):
    (_, a), _, _ = fermi_dirac_3step_barrier_pdf_bounds(photon_energy, workfun, temp, fermi_energy, n_tails=n_tails)
    return (0*a, a), (0*pi, pi/2)
