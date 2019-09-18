#!/usr/bin/env python

from distgen.tools import stopwatch, vprint
from distgen.reader import reader
from distgen.writers import get_writer
from distgen.generator import generator

def run_distgen(inputs=None,outputfile=None,output_type="gpt",verbose=0):

    watch = stopwatch()
    watch.start()
    vprint("**************************************************",verbose>0,0,True)
    vprint( "             Dist Generator v 1.0",verbose>0,True,True)
    vprint("**************************************************",verbose>0,0,True)

    if(isinstance(inputs,str)):
        # Read input file
        par = reader(inputs,verbose=verbose)
        params = par.read()
    elif(isinstance(inputs,dict)):
        params = inputs
    else:
        raise ValueError("Unsupported input parameter: "+str(type(inputs)))    
        
    # Make distribution
    gen = generator(verbose)
    gen.parse_input(params)
    beam,outfile = gen.get_beam()
    
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
