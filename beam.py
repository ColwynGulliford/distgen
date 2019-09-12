import numpy as np

from tools import vprint

class beam():
  
    n = 0
    q = 0

    params = {}

    def __init__(self,n,q):

        self.n = n
        self.q = q

    def __getitem__(self,var):
        return self.params[var]
    
    def __setitem__(self,var,item):

        if(var in self.params.keys()):
            self.params[var]=item
        else:
            raise ValueError("Beam object has no parameter "+var)

    def print_stats(self):
        vprint("Beam stats:",True,0,True)
       
        for x in self.params:
            vprint("avg_"+x+ " = {:0.3f~P}".format(np.mean(self.params[x])) +", sigma_"+x+" = {:0.3f~P}".format(np.std(self.params[x])), True, 1, True)
            
