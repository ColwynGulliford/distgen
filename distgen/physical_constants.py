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

# Register gamma*beta units for electrons
unit_registry.define('GB = 510998.946 * eV/c')

# physical_constants
c = 299792458 * unit_registry.parse_expression("m/s")
e = 1.602176634e-19 * unit_registry.parse_expression("coulomb")     # Fundamental Charge Unit
qe = -e                                                             # Charge on electron 
me = 9.1093837015e-31 * unit_registry.kg                            # Mass of electron
MC2 = (me*c*c).to(unit_registry.electron_volt)                      # Electron rest mass

# Mathematical constants
pi = math.pi*unit_registry("rad")
