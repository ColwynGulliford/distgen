"""
Defines common/useful physical constants.  
"""

import math
import warnings
from pint import UnitRegistry, Quantity

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    Quantity([])

unit_registry = UnitRegistry()
unit_registry.setup_matplotlib()

import scipy.constants

# Constants come from scipy.constants
# Check http://docs.scipy.org/doc/scipy/reference/constants.html for a complete constants listing.

# physical_constants
#c = scipy.constants.c * unit_registry.parse_expression('m/s')                                    # Speed of light [m/s]
#e = scipy.constants.e * unit_registry.parse_expression('coulomb')                                # Fundamental Charge Unit [C]
#qe = -e                                                                                          # Charge on electron [C]
#me = scipy.constants.m_e * unit_registry.kg                                                      # Mass of electron [kg]
#MeC2 = scipy.constants.value('electron mass energy equivalent in MeV')*1e6*unit_registry('eV')    # Electron rest energy [eV]
#h = scipy.constants.value('Planck constant in eV/Hz')*unit_registry('eV/Hz')
#hc = h*c
#kb = scipy.constants.value('Boltzmann constant in eV/K')*unit_registry('eV/K')                   # Boltzmann Constant
#ge = scipy.constants.value('electron g factor')*unit_registry('dimensionless')                   # Lande g-factor for electron

# Proton
#mp = scipy.constants.m_p * unit_registry.kg
#MpC2 = scipy.constants.value('proton mass energy equivalent')*1e6
#qp = abs(qe)
#gp = scipy.constants.value('proton g factor')*unit_registry('dimensionless')

# Register gamma*beta units for electrons
#unit_registry.define(f'GB = {MeC2.magnitude} * eV/c')

# Mathematical constants
pi = math.pi*unit_registry("rad")

def is_quantity(q):
    return isinstance(q, unit_registry.Quantity)

class PhysicalConstants():

    def __init__(self):
        
        self._species_charge = {'electron': -1,
                        'proton': +1,
                        'positron': +1, 
                        'muon': -1,
                        'neutron': 0, 
                        'tau': -1,
        }

        self._pi = math.pi*unit_registry("rad")
        #h = scipy.constants.value('Planck constant in eV/Hz')*unit_registry('eV/Hz')
#hc = h*c
#kb = scipy.constants.value('Boltzmann constant in eV/K')*unit_registry('eV/K')                   # Boltzmann Constant

    def __getitem__(self, key):
        return scipy.constants.value(key)*unit_registry(scipy.constants.unit(key))

    def species(self, key):
        
        in_scipy = len([res for res in  scipy.constants.find(f'{key} mass') if res == f'{key} mass'])==1
        
        if(in_scipy):

            return {'charge': self.species_charge(key)*self['elementary charge'],
                    'mass': self[f'{key} mass'],
                    'rest_energy': self.species_rest_energy(key),
                    'g_factor': self._species_charge[key]*self[f'{key} g factor']
                   }

        else:
            raise ValueError(f'Unsupported particle species: {key}')

    def species_rest_energy(self, species):
        return (scipy.constants.value(f'{species} mass energy equivalent in MeV')*unit_registry('MeV')).to('eV')

    def species_charge(self, species):
        return self._species_charge[species]

    @property
    def species_list(self):
        return list(self._species_charge.keys())

    @property
    def pi(self):
        return self._pi

PHYSICAL_CONSTANTS = PhysicalConstants()
        
