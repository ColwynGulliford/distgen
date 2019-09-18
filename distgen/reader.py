from .tools import vprint, stopwatch, is_floatable, is_unit_str

import time
import os
from collections import OrderedDict as odict
import json
        
class reader():

    def __init__(self,filename,verbose):

        self.filename=filename
        self.verbose=verbose

        self.flines = []
        self.params = {}
        
    def read(self):
    
        if(self.filename is None):
            raise ValueError("PyDist::reader: No input file specified!")

        if(not os.path.exists(self.filename)):
            raise ValueError("PyDist::reader: input file doesn't exist!")
                

        watch = stopwatch()
        watch.start()
        vprint("Reading file '"+self.filename+"'...",self.verbose>0,0,False)    
       
        f = open(self.filename,'r')
        try:
            # Try loading as a json
            params = json.load(f) 
            
        except:
            
            for line in f:
                self.flines.append(line)
            
        f.close()
        watch.stop()
        vprint("done. Time Ellapsed: "+watch.print(),self.verbose>0,0,True) 

        self.params=params
        return params
        
    def reset(self,filename,verbose):
        self.__init__(filename,verbose)
        
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



 

