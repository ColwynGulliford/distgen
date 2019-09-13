from .tools import vprint, stopwatch, is_floatable, is_unit_str

import time
import os
from collections import OrderedDict as odict

class reader():
  
    verbose = 0
    unit_register = []


    # File data
    filename = None
    flines = []

    params = odict()

    def __init__(self,filename,verbose,ureg):

        self.filename=filename
        self.verbose=verbose
        self.unit_register = ureg

    def read(self):
    
        if(self.filename is None):
            raise ValueError("PyDist::reader: No input file specified!")

        if(not os.path.exists(self.filename)):
            raise ValueError("PyDist::reader: input file doesn't exist!")
                

        watch = stopwatch()
        watch.start()
        vprint("Reading file '"+self.filename+"'...",self.verbose>0,0,False)    
       
        f = open(self.filename,'r')
        for line in f:
            self.flines.append(line)
        
        f.close()
        watch.stop()
        vprint("done. Time Ellapsed: "+watch.print(),self.verbose>0,0,True) 

        watch.start()
        vprint("Parsing data...",self.verbose>0,0,False)   
        count = 1 
        for line in self.flines:
            line = line.split("#")[0].strip()
            if(line!=""):
                tokens = line.split()
                if(len(tokens)>=2):
                    if(tokens[0] not in self.params.keys()):
                        self.params[tokens[0]]=tokens[1:]
                else:
                    raise ValueError("Parameter "+tokens[0]+" on line " + str(count) + " had no associated values.")
 
        watch.stop()
        vprint("done. Time Ellapsed: "+watch.print(),self.verbose>0,0,True)

        if(self.verbose>=2):
            for param in self.params:
                print(param+": ",self.params[param])

    def get_params(self):
        return self.params
            
    def check_for_parameter(self,name):

        if(name in self.params):
            return True
        else:
            return False

    def get_parameter(self,name):

        if(name in self.params):
            return self.params[name]
        else:
            print("Could not find parameter "+name+"in parameter data.")



 

