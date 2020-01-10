import numpy as np

from .tools import vprint, mean, std

"""
This class defines the container for an initial particle distribution 
"""

class Beam():

    def __init__(self,n,q,species="electron"):
        """
        Initializes a beam class object with n particles and bunch charge q
        """
        self.n = n
        self.q = q
        self.species = species
        self.params = {}

    def __getitem__(self,var):
        """
        Allows direct get access via brackets to the params dictionary with key var.
        """
        return self.params[var]
    
    def __setitem__(self,var,item):
        """
        Allows direct set access via brackets to the params dictionary with key var and value item.
        """
        if(var in self.params.keys()):
            self.params[var]=item
        else:
            raise ValueError("Beam object has no parameter "+var)

    def avg(self,var,desired_units=None):

        if(desired_units is None):
            return mean(self.params[var],self.params["w"])
        else:
            avgvar = mean(self.params[var],self.params["w"])
            return avgvar.to(desired_units)
  
    def std(self,var,desired_units=None):
        
        if(desired_units is None):
            return std(self.params[var],self.params["w"])   
        else:
            stdvar = std(self.params[var],self.params["w"]) 
            return stdvar.to(desired_units)

        print("BOOF")

    def data(self):
        """
        Converts to fixed units and returns a dict of data.
        
        See function Sbeam_data
        """
        return beam_data(self)
    
        
    def print_stats(self):
        """
        Prints averages and standard deviations of the beam variables.
        """  

        stat_exceptions=["w"]

        vprint("\nBeam stats:",True,0,True)
        for x in self.params:
            if(x not in stat_exceptions):
                vprint("avg_"+x+ " = {:0.3f~P}".format(mean(self.params[x], self.params["w"])) +", sigma_"+x+" = {:0.3f~P}".format(std(self.params[x],weights=self.params["w"])), True, 1, True)
            

            
            
            
            
            
def beam_data(beam):
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
    #assert species == 'electron' # TODO: add more species
    
    # number of lines in file
    n_particle = beam.n
    total_charge = (beam.q.to('C')).magnitude
    species = beam.species
    
    # weight
    weight = (beam['w'].magnitude) * total_charge # Weight should be macrocharge in C
    
    # Status
    status = np.full(n_particle, 1) # Status == 1 means live
    
    # standard units and types
    names = ['x', 'y', 'z', 'px',   'py',   'pz',   't']
    units = ['m', 'm', 'm', 'eV/c', 'eV/c', 'eV/c', 's']
    
    data = {'n_particle':n_particle,
            'species':species,
            'weight':weight,
            'status':status
    }
    
    for name, unit in zip(names, units):
        data[name] = (beam[name].to(unit)).magnitude
    
    return data            