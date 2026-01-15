from pmd_beamphysics import ParticleStatus
import numpy as np
from .physical_constants import PHYSICAL_CONSTANTS
from .physical_constants import unit_registry


from .tools import vprint, mean, std

"""
This class defines the container for an initial particle distribution
"""

c = PHYSICAL_CONSTANTS["speed of light in vacuum"]


class Beam:
    """
    The fundamental bunch data is stored in __dict__ with keys
        pint quantity np.array: x, px, y, py, z, pz, t, weight,
        np.array status,
        str: species,
        np.array: sx, sy, sz
    where:
        x, y, z have a base unit of meters
        px, py, pz are momenta in base units [eV/c]
        t is time in [s]
        weight is the macro-charge weight in [C], used for all statistical calulations.
        species is a proper species name: 'electron', etc.
        sx, sy, sz are the inherient spin angular momentum [meters*eV/c]
    """

    def __init__(self, **kwargs):
        self.required_inputs = ["total_charge", "n_particle", "species"]
        self.optional_inputs = []

        self.check_inputs(kwargs)

        self._q = kwargs["total_charge"]
        self._n_particle = kwargs["n_particle"]
        self._species = kwargs["species"]

        self._settable_array_keys = [
            "x",
            "px",
            "y",
            "py",
            "z",
            "pz",
            "t",
            "w",
            "theta",
            "pr",
            "ptheta",
            "xp",
            "yp",
            "thetax",
            "thetay",
            "sx",
            "sy",
            "sz",
            "s2",
        ]

    def check_inputs(self, inputs):
        allowed_params = self.optional_inputs + self.required_inputs + ["verbose"]
        for input_param in inputs:
            assert (
                input_param in allowed_params
            ), f"Incorrect param given to {self.__class__.__name__}.__init__(**kwargs): {input_param}\nAllowed params: {allowed_params}"

        # Make sure all required parameters are specified
        for req in self.required_inputs:
            assert (
                req in inputs
            ), f"Required input parameter {req} to {self.__class__.__name__}.__init__(**kwargs) was not found."

    def __getitem__(self, key):
        return getattr(self, key)

    def __add__(self, other):

        assert self.species == other.species

        total_beam = Beam(n_particle = self.n_particle + other.n_particle, 
                          total_charge = self.q + other.q,
                          species = self.species)

        for var in ['x', 'y', 'z', 'px', 'py', 'pz', 'sx', 'sy', 'sz']:
            try:
                total_beam[var] = np.concatenate( (self[var], other[var]) )
            except:
                print('Could not add together data for variable ', var)
                
        N = len(total_beam.x)
        total_beam.w = np.full((N,), 1 / N) * unit_registry("dimensionless")
        
        return total_beam

    @property
    def species(self):
        return self._species

    @property
    def n_particle(self):
        return self._n_particle

    @property
    def q(self):
        return self._q

    # Cylindrical coordinates
    @property
    def r(self):
        return np.sqrt(self.x**2 + self.y**2)

    @r.setter
    def r(self, r):
        theta = getattr(self, "theta")
        self.x = r * np.cos(theta)
        self.y = r * np.sin(theta)

    @property
    def theta(self):
        return np.arctan2(self.y, self.x)

    @theta.setter
    def theta(self, theta):
        r = getattr(self, theta)
        self.x = r * np.cos(theta)
        self.y = r * np.sin(theta)

    @property
    def pr(self):
        return self.px * np.cos(self.theta) + self.py * np.sin(self.theta)

    @pr.setter
    def pr(self, pr):
        ptheta = getattr(self, "ptheta")
        theta = getattr(self, "theta")
        self.px = pr * np.cos(theta) - ptheta * np.sin(theta)
        self.py = pr * np.sin(theta) + ptheta * np.cos(theta)

    @property
    def ptheta(self):
        return -self.px * np.sin(self.theta) + self.py * np.cos(self.theta)

    @ptheta.setter
    def ptheta(self, ptheta):
        pr = getattr(self, "pr")
        theta = getattr(self, "theta")
        self.px = pr * np.cos(theta) - ptheta * np.sin(theta)
        self.py = pr * np.sin(theta) + ptheta * np.cos(theta)

    # Transverse Derivatives and Angles
    @property
    def xp(self):
        return self.px / (self.pz.to(str(self.px.units)))

    @xp.setter
    def xp(self, xp):
        self.px = xp * self.pz

    @property
    def thetax(self):
        return np.arctan2(self.px, self.pz.to(str(self.px.units)))

    @thetax.setter
    def thetax(self, thetax):
        self.px = np.tan(thetax) * self.pz

    @property
    def yp(self):
        return self.py / (self.pz.to(str(self.py.units)))

    @yp.setter
    def yp(self, yp):
        self.py = yp * self.pz

    @property
    def thetay(self):
        return np.arctan2(self.py, self.pz.to(str(self.py.units)))

    @thetay.setter
    def thetay(self, thetay):
        self.py = np.tan(thetay) * self.pz

    # Relativistic quantities:
    @property
    def p(self):
        """Total momemtum"""
        return np.sqrt(self.px**2 + self.py**2 + self.pz**2)

    @property
    def mc2(self):
        return PHYSICAL_CONSTANTS.species(self.species)["mc2"].to("eV")

    @property
    def species_mass(self):
        return PHYSICAL_CONSTANTS.species(self.species)["mass"]

    @property
    def species_charge(self):
        return PHYSICAL_CONSTANTS.species(self.species)["charge"]

    @property
    def energy(self):
        return np.sqrt((c * self.p) ** 2 + self.mc2**2).to("eV")

    @property
    def gamma(self):
        return self.energy / self.mc2

    @property
    def kinetic_energy(self):
        return self.energy - self.mc2  # self.mc2*(self.gamma-1)

    @property
    def beta(self):
        return (
            c * self.p / self.energy
        ).to_reduced_units()  # np.sqrt( 1 - 1/self.gamma**2 )

    @property
    def beta_x(self):
        """vx/c"""
        return (
            c * self.px / self.energy
        ).to_reduced_units()  # (self.px/self.species_mass/c/self.gamma).to_reduced_units()

    @property
    def beta_y(self):
        """vy/c"""
        return (
            c * self.py / self.energy
        ).to_reduced_units()  # (self.py/self.species_mass/c/self.gamma).to_reduced_units()

    @property
    def beta_z(self):
        """vz/c"""
        return (
            c * self.pz / self.energy
        ).to_reduced_units()  # (self.pz/self.species_mass/c/self.gamma).to_reduced_units()

    @property
    def gamma_beta_x(self):
        """gamma * vx/c"""
        return (self.gamma * self.beta_x).to_reduced_units()

    @property
    def gamma_beta_y(self):
        """gamma * vy/c"""
        return (self.gamma * self.beta_y).to_reduced_units()

    @property
    def gamma_beta_z(self):
        """gamma * vz/c"""
        return (self.gamma * self.beta_z).to_reduced_units()

    # Statistical quantities
    def avg(self, var, desired_units=None):
        avgv = mean(getattr(self, var), getattr(self, "w"))
        if desired_units:
            avgv.ito(desired_units)

        return avgv

    def std(self, var, desired_units=None):
        stdv = std(getattr(self, var), getattr(self, "w"))
        if desired_units:
            stdv.ito(desired_units)

        return stdv

    def delta(self, key):
        """Attribute (array) relative to its mean"""
        return getattr(self, key) - self.avg(key)

    # Twiss parameters
    def Beta(self, var):
        varx = self.std(var) ** 2
        eps = self.emitt(var, "geometric")

        return varx / eps

    def Alpha(self, var):
        x = getattr(self, var)
        x0 = self.avg(var)

        p = getattr(self, f"{var}p")
        p0 = self.avg(f"{var}p")

        xp = mean((x - x0) * (p - p0), self.w)
        eps = self.emitt(var, "geometric")

        return -xp / eps

    def Gamma(self, var):
        varp = std(getattr(self, f"{var}p"), getattr(self, "w")) ** 2
        eps = self.emitt(var, "geometric")

        return varp / eps

    def emitt(self, var, units="normalized"):
        x = getattr(self, var)
        mc = self.species_mass * c

        if units == "normalized":
            p = (getattr(self, f"p{var}") / mc).to_reduced_units()

        elif units == "geometric":
            p = getattr(self, f"{var}p").to_reduced_units()
        else:
            raise ValueError(f"unknown emittance type: {units}")

        x0 = mean(x, getattr(self, "w"))
        p0 = mean(p, getattr(self, "w"))

        x2 = std(x, getattr(self, "w")) ** 2
        p2 = std(p, getattr(self, "w")) ** 2
        xp = mean((x - x0) * (p - p0), getattr(self, "w"))

        return np.sqrt(x2 * p2 - xp**2)

    def twiss(self, var):
        return (self.Beta(var), self.Alpha(var), self.emitt(var, "geometric"))

    # Spin
    @property
    def s2(self):
        return self.sx**2 + self.sy**2 + self.sz**2

    @property
    def spin_polarization(self):
        hbar = PHYSICAL_CONSTANTS["reduced Planck constant"].to("nm * eV/c")

        if self.species in ["electron", "positron", "tau", "muon"]:
            Sz = hbar / 2
        else:
            raise ValueError(f"Unsupported particles species: {self.species}")

        return (
            np.sqrt(self.avg("sx") ** 2 + self.avg("sy") ** 2 + self.avg("sz") ** 2)
            / Sz
        )

        # norm = np.linalg.norm(ehat)
        # if norm == 0:
        #   raise ValueError('Spin polarization direction must not be zero.')

        # ehat = ehat / norm

        # spin_projection = self.sx*ehat[0] + self.sy*ehat[1] + self.sz*ehat[2]

        # return np.sum(self['w'] * spin_projection) / np.sum( np.abs(self['w']*spin_projection) )

    @property
    def g_factor(self):
        return np.full(
            (self._n_particle,), PHYSICAL_CONSTANTS.species_g_factor(self.species)
        ) * unit_registry("dimensionless")

    # Set functiontality
    def __setitem__(self, key, value):
        if key in self._settable_array_keys:
            setattr(self, key, value)
        else:
            raise ValueError(f"Beam: quantity {key} is not settable.")

    def print_stats(self):
        """
        Prints averages and standard deviations of the beam variables.
        """

        stat_vars = {
            "x": "mm",
            "y": "mm",
            "z": "mm",
            "px": "eV/c",
            "py": "eV/c",
            "pz": "eV/c",
            "t": "ps",
        }

        vprint("\nBeam stats:", True, 0, True)
        for x, unit in stat_vars.items():
            vprint(
                f"avg_{x} = {self.avg(x).to(unit):G~P}, sigma_{x} = {self.std(x).to(unit):G~P}",
                True,
                1,
                True,
            )

    def data(self, status=ParticleStatus.ALIVE):
        """
        Converts to fixed units and returns a dict of data.

        See function Sbeam_data
        """
        return beam_data(self, status)


def beam_data(beam, status):
    """
    Converts all units to standard units and strips them as a data dict with:
        str: species
        int: n_particle
        np.array: x, px, y, py, z, pz, t, status, weight
    where:
        x, y, z are positions in units of [m]
        px, py, pz are momenta in units of [eV/c]
        t is time in [s]
        status = 1
        weight is the macro-charge weight in [C]


    """
    # assert species == 'electron' # TODO: add more species

    # number of lines in file
    n_particle = beam["n_particle"]
    total_charge = (beam.q.to("C")).magnitude
    species = beam.species

    # weight
    weight = np.abs(
        (beam["w"].magnitude) * total_charge
    )  # Weight should be macrocharge in C

    # Status
    status_array = np.full(n_particle, int(status))  # Status == 1 means live

    # standard units and types
    names = ["x", "y", "z", "px", "py", "pz", "t"]
    units = ["m", "m", "m", "eV/c", "eV/c", "eV/c", "s"]

    data = {
        "n_particle": n_particle,
        "species": species,
        "weight": weight,
        "status": status_array,
    }

    for name, unit in zip(names, units):
        data[name] = (beam[name].to(unit)).magnitude

    return data
