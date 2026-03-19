#!/usr/bin/env python

from distgen.tools import vprint
from distgen.writers import writer
from distgen.generator import Generator

import copy

def run_distgen(settings=None, inputs="distgen.json", verbose=0):
    if settings is None:
        settings = {}
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

    if isinstance(inputs, dict):
        inputs = copy.deepcopy(inputs)

    # Make distribution
    gen = Generator(inputs, verbose=verbose)

    for k, v in settings.items():
        vprint(f"Replacing parameter {k} with value {v}.", verbose > 0, 0, True)
        gen[k] = v

    beam = gen.beam()

    # Write to file
    if "file" in gen["output"]:
        writer(gen["output"]["type"], beam, gen["output"]["file"], verbose)

    # Print beam stats
    if verbose > 0:
        beam.print_stats()

    return beam
