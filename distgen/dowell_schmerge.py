import numpy as np


def filled_ball(samples=1000):
    """
    Draw samples from a uniformly filled ball of unit radius.
    """
    u = np.random.normal(0, 1, samples)
    v = np.random.normal(0, 1, samples)
    w = np.random.normal(0, 1, samples)
    r = np.random.random(samples) ** (1. / 3)
    norm = (u * u + v * v + w * w) ** 0.5
    (x, y, z) = r * (u, v, w) / norm
    return np.vstack((x, y, z)).T


def sample_momentum_fermi_dirac(samples, fermi_energy, temp, temp_cutoff=1e-3, tail_cutoff=5):
    """
    Sample momentum of free electrons with Fermi-Dirac occupation function.

    :param samples: number of samples to generate
    :param fermi_energy: fermi energy of the sample in eV
    :param temp: cathode temperature in K
    :param temp_cutoff: below this temp, use step function occupation to avoid divide by zero
    :param tail_cutoff: cutoff for highest energy of the electrons in the tail (units of kB*T)
    :return: (samples, 3) ndarray of momentum electrons in eV/c
    """
    mc2 = 511e3
    kb = 8.617333262e-5  # eV/K from CODATA recommendation
    dist = filled_ball(int(samples))*(2*mc2*(fermi_energy + kb*temp*tail_cutoff))**0.5
    p_fermi = 1/(1 + np.exp((np.sum(dist**2, axis=1)/2/mc2 - fermi_energy)/(kb*temp))) if temp > temp_cutoff else 1.0
    return dist[np.random.random(size=dist.shape[0]) < p_fermi, :]


def _sample_momentum_dowell_schmerge(samples, photon_energy, workfun, fermi_energy, temp):
    """
    Internal function that draws an initial "samples" quantity of electrons for rejection sampling. Please call 
    sample_momentum from user code
    """
    vac_level = workfun + fermi_energy
    mc2 = 511e3
    dist = sample_momentum_fermi_dirac(samples, fermi_energy, temp)
    e_ext = np.sum(dist**2, axis=1)/2/mc2 - np.sum(dist[:, :2]**2, axis=1)/2/mc2 - vac_level + photon_energy
    dist = dist[e_ext > 0, :]
    dist[:, 2] = (2*mc2*(np.sum(dist**2, axis=1)/2/mc2 - np.sum(dist[:, :2]**2, axis=1)/2/mc2 - vac_level + photon_energy))**0.5
    return dist


def sample_momentum_dowell_schmerge(samples, photon_energy, workfun, fermi_energy, temp=293.15, max_loop=32_000):
    """
    Returns samples from the momentum distribution of a Dowell-Schmerge-like photocathode with
    no scattering effects. Electrons are initially populated in momentum space as in the
    Sommerfeld model. They are then accepted according to the escape condition in [1] with the
    surface normal being (0, 0, 1) in these coordinates. Rejection sampling is performed until
    "samples" samples have been drawn. For the purposes of referencing in published research,
    this is the implementation described in [2].

    [1] Dowell, D. H., & Schmerge, J. F. (2009). Quantum efficiency and thermal emittance of metal 
        photocathodes. Physical Review Special Topics - Accelerators and Beams, 12(7). 
        https://doi.org/10.1103/PhysRevSTAB.12.074201

    [2] Pierce, C. M., Durham, D. B., Riminucci, F., Dhuey, S., Bazarov, I., Maxson, J., Minor, 
        A. M., & Filippetto, D. (2023). Experimental Characterization of Photoemission from 
        Plasmonic Nanogroove Arrays. Physical Review Applied, 19(3), 034034.
        https://doi.org/10.1103/PhysRevApplied.19.034034
    
    :param samples: number of samples to generate
    :param photon_energy: photon energy of the driving laser in eV
    :param workfun: cathode workfunction in eV
    :param fermi_energy: fermi energy of the sample in eV
    :param temp: cathode temperature in K
    :param max_loop: max times to repeat the sampling before bailing out
    :return: (samples, 3) ndarray of momentum of escaped electrons (px, py, pz) in eV/c
    """
    if photon_energy <= workfun:
        raise ValueError(r'"photon_energy" must be greater than "workfun" (received photon_energy=%.2e, workfun=%.2e)' % (photon_energy, workfun))
    
    if temp < 0:
        raise ValueError(r'temp must be greater than or equal to zero (received temp=%.2e)' % temp)

    ret = np.empty((0, 3))

    for _ in range(max_loop):
        x = _sample_momentum_dowell_schmerge(samples, photon_energy, workfun, fermi_energy, temp)
        ret = np.concatenate((ret, x), axis=0)
        if ret.shape[0] >= samples:
            break
    else:
        raise ValueError(r"sampler exceeded max number of loops (here, set to %d) try increasing max_loop or looking at why so few samples are being accepted" % max_loop)

    return ret[:samples]
