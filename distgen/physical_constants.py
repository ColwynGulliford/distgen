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
c = scipy.constants.c * unit_registry.parse_expression('m/s')                                    # Speed of light
e = scipy.constants.e * unit_registry.parse_expression('coulomb')                                # Fundamental Charge Unit
qe = -e                                                                                          # Charge on electron 
me = scipy.constants.m_e * unit_registry.kg                                                      # Mass of electron
MC2 = scipy.constants.value('electron mass energy equivalent in MeV')*1e6*unit_registry('eV')    # Electron rest mass

# Register gamma*beta units for electrons
unit_registry.define(f'GB = {MC2.magnitude} * eV/c')

# Mathematical constants
pi = math.pi*unit_registry("rad")

def is_quantity(q):
    return isinstance(q, unit_registry.Quantity)


