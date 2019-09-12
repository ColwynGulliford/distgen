#!/usr/bin/env python

from optparse import OptionParser

from physical_constants import *
from tools import stopwatch, vprint
from reader import reader
from writers import get_writer
from generator import generator
from plot import plot_beam

def main():

    #---------------------------------------------------------------------------------------
    # Parse Input Content
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
    plots_on = options.plots_on
    #---------------------------------------------------------------------------------------

    watch = stopwatch()
    watch.start()
    vprint("**************************************************",verbose>0,0,True)
    vprint( "           PyDist Generator v 0.0",verbose>0,True,True)
    vprint("**************************************************",verbose>0,0,True)

    # Read input file
    par = reader(inputfile,verbose,unit_registry)
    par.read()
    params = par.get_params()

    # Make distribution
    gen = generator(verbose)
    gen.parse_input(params)
    beam,outfile = gen.get_beam()

    # Print beam stats
    if(verbose>0):
        beam.print_stats()

    # Print distribution to file
    file_writer = get_writer(output,outfile)
    file_writer.write(beam,verbose,params)

    watch.stop()
    vprint("\nTotal time ellapsed: "+watch.print()+".",verbose>0,0,True)

    # Plot particles if desired
    if(plots_on):
        plot_beam(beam)
   

# ---------------------------------------------------------------------------- 
#   This allows the main function to be at the beginning of the file
# ---------------------------------------------------------------------------- 
if __name__ == '__main__':
    main()
