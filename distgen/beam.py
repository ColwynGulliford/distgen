import numpy as np

from .tools import vprint, mean, std

"""
This class defines the container for an initial particle distribution 
"""

class Beam():

    params = {}

    def __init__(self,n,q,species="electron"):
        """
        Initializes a beam class object with n particles and bunch charge q
        """
        self.n = n
        self.q = q
        self.species = species

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

    def print_stats(self):
        """
        Prints averages and standard deviations of the beam variables.
        """  

        stat_exceptions=["w"]

        vprint("\nBeam stats:",True,0,True)
        for x in self.params:
            if(x not in stat_exceptions):
                vprint("avg_"+x+ " = {:0.3f~P}".format(mean(self.params[x], self.params["w"])) +", sigma_"+x+" = {:0.3f~P}".format(std(self.params[x],weights=self.params["w"])), True, 1, True)
            
