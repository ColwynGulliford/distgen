"""
Defines common/useful physical constants.  
"""

import math
import numpy as np
import warnings
from pint import UnitRegistry, Quantity

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    Quantity([])

unit_registry = UnitRegistry()
unit_registry.setup_matplotlib()

import scipy.constants

# Mathematical constants
pi = math.pi*unit_registry("rad")

def is_quantity(q):
    return isinstance(q, unit_registry.Quantity)

class PhysicalConstants():

    def __init__(self):
        
        self._supported_species = ['electron',
                                   'photon',
                                   'positron',
                                   'proton',
                                   'muon',
                                   'neutron',
                                   'tau']

        self._pi = math.pi*unit_registry("rad")

        self.construct_species_data()

    def __getitem__(self, key):
        return scipy.constants.value(key)*unit_registry(scipy.constants.unit(key))

    def construct_species_data(self):

        self._species_data = {}

        charges = {'electron': -1,
                   'photon': 0,
                   'positron': +1,
                   'proton': +1,
                   'muon': -1,
                   'neutron': 0,
                   'tau': -1 }

        extra_species = {
            'photon': {
                'charge': 0*self['elementary charge'],
                'mass': 0*self['electron mass'],
                'mc2': 0.0*unit_registry('eV'),
                'g_factor': 0*self[f'electron g factor']
            },
            'positron': {
                'charge': self['elementary charge'],
                'mass': self['electron mass'],
                'mc2': self['electron mass energy equivalent in MeV'].to('eV'),
                'g_factor': -self[f'electron g factor']
            },
            'tau': {
                'charge': -self['elementary charge'],
                'mass': self['tau mass'],
                'mc2': self['tau mass energy equivalent in MeV'].to('eV'),
                'g_factor': +self[f'electron g factor']
            }
                
            
            
        }   

        for species in self._supported_species:

            if(species not in charges):
                raise ValueError(f'Species charge for {species} not defined.')

            if(species in extra_species):
                self._species_data[species] = extra_species[species].copy()
            
            elif(self.species_in_scipy(species)):
                self._species_data[species] = {'charge': charges[species]*self['elementary charge'],
                                               'mass': self[f'{species} mass'],
                                               'mc2': self[f'{species} mass energy equivalent in MeV'].to('eV'),
                                               'g_factor': self[f'{species} g factor']}

            

            else:
                raise ValueError(f'Missing species data for {species}.')        

    def species_in_scipy(self, key):
        return len([res for res in  scipy.constants.find(f'{key} mass') if res == f'{key} mass'])==1

    def species(self, key):
        
        if(key in self._species_data):
            return self._species_data[key]
        else:
            raise ValueError(f'Unsupported particle species: {key}')

    def species_rest_energy(self, species):
        return self._species_data[species]['rest_energy'] 

    def species_charge(self, species):
        return self.species_data[species]['charge']

    @property
    def species_list(self):
        return list(self._species_data.keys())

    @property
    def pi(self):
        return self._pi

PHYSICAL_CONSTANTS = PhysicalConstants()
        
