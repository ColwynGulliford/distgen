#!/usr/bin/env python

from distgen.tools import StopWatch, vprint
from distgen.reader import Reader
from distgen.writers import writer
from distgen.generator import Generator

def set_param(param_struct, keystring, val, sep=':'):
    """
    Set a value inside nested dicts using a key string. 
    Example:
        set_distgen_param(
        {d['key1']['key2']['key3']},
        'key1:key2:key3', 999
        )
        is equivalent to:
        d['key1']['key2']['key3'] = 999
    
    """
    keys = keystring.split(':')
    d = param_struct
    # Go through nested dicts
    for k in keys[0:-1]:
        d = d[k]
    final_key = keys[-1]
    # Set
    if final_key in d:
        d[final_key] = val
    else:
        print(f'Error: keystring {keystring} key does not exist:', final_key)

def get_param(param_struct, keystring, sep=':'):
    """
    
    """
    keys = keystring.split(':')
    d = param_struct
    # Go through nested dicts
    for k in keys:
        d = d[k]
    return d

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
    
    
    beam=distgen.drivers.run_distgen(
        settings = {'beam:params:total_charge:value': 456,
                    'output:type':'astra',
                    'output:file':'astra_particles.dat'},
        inputs = 'gunb_gaussian.json',
        verbose=True)
    
    """
    
    # Get basic inputs
    if(isinstance(inputs,str)):
        # Read input file
        par = Reader(inputs, verbose=verbose)
        params = par.read()
    elif(isinstance(inputs, dict)):
        params = inputs
    else:
        raise ValueError("Unsupported input parameter: "+str(type(inputs)))    
    
    # Replace these with custom settings
    for key, value in settings.items():    
        set_param(params, key, value)
        # Check
        if verbose:
            print('replaced: ', key, 'with:', get_param(params, key))
    
    # Make distribution
    gen = Generator(verbose)
    gen.parse_input(params)
    beam = gen.beam()
    
    # Write to file
    if 'file' in params['output']:
        writer(params['output']['type'], beam, params['output']['file'],verbose, params)
    # Print beam stats
    if(verbose>0):
        beam.print_stats()

    return beam
