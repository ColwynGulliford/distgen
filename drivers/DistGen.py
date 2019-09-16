#!/usr/bin/env python

from optparse import OptionParser

from distgen.physical_constants import *
from distgen.tools import stopwatch, vprint
from distgen.reader import reader
from distgen.writers import get_writer
from distgen.generator import generator

def run_distgen(inputs=None,outputfile=None,output_type="gpt",verbose=0):

    watch = stopwatch()
    watch.start()
    vprint("**************************************************",verbose>0,0,True)
    vprint( "             Dist Generator v 0.0",verbose>0,True,True)
    vprint("**************************************************",verbose>0,0,True)
    
    f=open(inputs,'r')
    print(f.readline())
    f.close()
    
    if(isinstance(inputs,str)):
        # Read input file
        par = reader(inputs,verbose,unit_registry)
        par.read()
        params = par.get_params()
    elif(isinstance(inputs,dict)):
        params = inputs
    else:
        raise ValueError("Unsupported input parameter: "+str(type(inputs)))  
        
    print(params)    
        
    # Make distribution
    gen = generator(verbose)
    gen.parse_input(params)
    beam,outfile = gen.get_beam()
    
    f=open(inputs,'r')
    print(f.readline())
    f.close()
    
    if(outputfile is not None):
        
        # Print distribution to file
        file_writer = get_writer(output,outfile)
        file_writer.write(beam,verbose,params)

    # Print beam stats
    if(verbose>0):
        beam.print_stats()

    watch.stop()
    vprint("\nTotal time ellapsed: "+watch.print()+".",verbose>0,0,True)

    return beam

def main():

    """
    Main function for running distgen as a single execution program
    """

    #---------------------------------------------------------------------------------------
    # Parse Input Content and Run
    #---------------------------------------------------------------------------------------
    parser = OptionParser()
    parser.add_option("-f", "--file", dest="filename", default=None, 
                      help="write report to FILE", metavar="FILE")
    parser.add_option("-v",dest="verbose", default=0,help="Print short status messages to stdout")
    parser.add_option("-o",dest="output", default="gpt",help="Particle output type (default gpt)")
    parser.add_option("-p",dest="plots_on",default=False,action = "store_true")

    (options, args) = parser.parse_args()

    inputfile = options.filename
    verbose = int(options.verbose)  
    output = options.output    
    #plots_on = options.plots_on

    run_distgen(inputs=inputfile,output_type=output,verbose=verbose)
    #---------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------- 
#   This allows the main function to be at the beginning of the file
# ---------------------------------------------------------------------------- 
if __name__ == '__main__':
    main()
