import numpy as np

from .tools import vprint

"""
This class defines the container for an initial particle distribution 
"""

class beam():
  
    n = 0
    q = 0

    params = {}

    def __init__(self,n,q):
        """
        Initializes a beam class object with n particles and bunch charge q
        """
        self.n = n
        self.q = q

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

    def print_stats(self):
        """
        Prints averages and standard deviations of the beam variables.
        """
        vprint("Beam stats:",True,0,True)
        for x in self.params:
            vprint("avg_"+x+ " = {:0.3f~P}".format(np.mean(self.params[x])) +", sigma_"+x+" = {:0.3f~P}".format(np.std(self.params[x])), True, 1, True)
            
