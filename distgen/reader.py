from .tools import vprint, StopWatch, is_floatable, is_unit_str

import time
import os
from collections import OrderedDict as odict
import json
        
"""
This class handles input file reading and currently supports reading json files. 
Coming soon ascii files.
    
"""
class Reader():

    def __init__(self,file_name,verbose=0):

        """
        The class init file takes in a file name (string) 
        and a verbose level (int) controlling the amount of text output to the console
        """

        self.file_name=file_name   # input file name (str)
        self.verbose=verbose       # verbose setting (int)

        self.file_lines = []       # text lines for a text based input file
        self.params = {}           # output parameter structure based on input data
        
    def read(self):
        """
        Reads the file set in the initialization of the class
        """

        if(self.file_name is None):
            raise ValueError("PyDist::reader: No input file specified!")

        if(not os.path.exists(self.file_name)):
            raise ValueError("PyDist::reader: input file doesn't exist!")
                
        # Get a stop watch for timeing the file read
        watch = StopWatch()
        watch.start()
        vprint("Reading file '"+self.file_name+"'...",self.verbose>0,0,False)    
       
        # Open file
        file_handle = open(self.file_name,'r')
        try:
            # Try loading as a json
            params = json.load(file_handle) 
            
        except:
            # If not, read the file assuming ascii format
            for line in file_handle:
                self.file_lines.append(line)

            params="File type not supported"  # ASCII parsing isn't supported yet
            
        file_handle.close()
        watch.stop()
        vprint("done. Time Ellapsed: "+watch.print(),self.verbose>0,0,True) 

        self.params=params
        return params
        
    def reset(self,filename,verbose):
        """
        Resets the initialization parameters for the class
        """
        self.__init__(filename,verbose)
        
    def get_params(self):
        """
        Return the pointer to the input file dictionary
        """
        return self.params
            
    def check_for_parameter(self,name):
        """
	Query if a parameter "name" (str) is in the stored parameter dictionary
        """
        if(name in self.params):
            return True
        else:
            return False

    def get_parameter(self,name):
        """
	Return the value for a the key "name" (str) in the stored parameter dictionary
        """
        if(name in self.params):
            return self.params[name]
        else:
            print("Could not find parameter "+name+"in parameter data.")



 

