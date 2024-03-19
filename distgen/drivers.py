#!/usr/bin/env python

from distgen.tools import vprint
from distgen.writers import writer
from distgen.generator import Generator

import copy


#def run_distgen(inputs=None,outputfile=None,output_type="gpt",verbose=0):

#    watch = StopWatch()
#    watch.start()
#    vprint("**************************************************",verbose>0,0,True)
#    vprint( "             Dist Generator v 1.0",verbose>0,True,True)
#    vprint("**************************************************",verbose>0,0,True)

#    if(isinstance(inputs,str)):
        # Read input file
#        par = Reader(inputs,verbose=verbose)
#        params = par.read()
#    elif(isinstance(inputs,dict)):
#        params = inputs
#    else:
#        raise ValueError("Unsupported input parameter: "+str(type(inputs)))    
        
    # Make distribution
#    gen = Generator(verbose)
#    gen.parse_input(params)
#    beam = gen.beam()
    
    # Print distribution to file
#    writer(params["output"]["type"],beam,params["output"]["file"],verbose,params)

    # Print beam stats
#    if(verbose>0):
#        beam.print_stats()

#    watch.stop()
#    vprint("\nTotal time ellapsed: "+watch.print()+".",verbose>0,0,True)

#    return beam

def run_distgen(
    settings={},
    inputs='distgen.json',
    verbose=0):
    """
    Driver routine to generate a beam accordng to inputs (json or dict)
    
    Settings can contain modifications to inputs, with nested keys separated by :. 
    
    Example:
        beam=distgen.drivers.run_distgen(
            settings = {'beam:params:total_charge:value': 456,
                        'output:type':'astra',
                        'output:file':'astra_particles.dat'},
            input = 'gunb_gaussian.json',
            verbose=True)
    
    """

    if(isinstance(inputs, dict)):
        inputs = copy.deepcopy(inputs)
    
    # Make distribution
    gen = Generator(inputs, verbose=verbose)

    #gen._input = update_nested_dict(gen._input, settings, verbose=verbose)

    for k,v in settings.items():
        vprint(f"Replacing parameter {k} with value {v}.", verbose>0, 0, True)
        gen[k] = v
    
    beam = gen.beam()

    # Write to file
    if 'file' in gen['output']:
        writer(gen['output']['type'], beam, gen['output']['file'],verbose)

    # Print beam stats
    if(verbose>0):
        beam.print_stats()

    return beam
