from .tools import vprint, StopWatch, mean
from .physical_constants import  *

import numpy as np
import subprocess
import os
from collections import OrderedDict as odict

def get_species_charge(species):

    if(species=="electron"):
        return qe
    else:
        raise ValueError("get_species_charge: Species '"+species+" is not supported.")

def writer(output_format,beam,outfile,verbose=0,params=None):

    if(output_format=="gpt"):

        write_gpt(beam,outfile,verbose=verbose,params=params)

    elif(output_format=="astra"):
        write_astra(beam,outfile,verbose=verbose,params=params)


def write_gpt(beam,outfile,verbose=0,params=None):  

        watch = StopWatch()

        # Format particles
        gpt_units={"x":"m", "y":"m", "z":"m","px":"GB","py":"GB","pz":"GB","t":"s"}

        qspecies = get_species_charge(beam.species)
        qspecies.ito("coulomb")
        qs = np.full((beam.n,),1.0)*qspecies
        qbunch = beam.q.to("coulomb")

        watch.start()
        
        assert beam.species == 'electron' # TODO: add more species

        nspecies = np.abs(qbunch.magnitude/qspecies.magnitude)
        nmacro = nspecies*beam["w"]    #np.full((beam.n,),1)*np.abs( (beam.q.to("coulomb")).magnitude/beam.n/qspecies.magnitude)

        vprint("Printing "+str(beam.n)+" particles to '"+outfile+"': ",verbose>0,0,False)
        
        # Scale parameters to GPT units
        for var in gpt_units:
            beam[var].ito(gpt_units[var])

        headers = odict( {"x":"x", "y":"y", "z":"z", "px":"GBx",  "py":"GBy", "pz":"GBz", "t":"t", "q":"q", "nmacro":"nmacro"} )
        header = '   '.join(headers.values())

        data = np.zeros( (len(beam["x"]),len(headers)) )
        for index, var in enumerate(headers):
            if(var=="q"):
                data[:,index]=qs.magnitude
            elif(var=="nmacro"):
                data[:,index]=nmacro.magnitude
            else:
                data[:,index] = beam[var].magnitude
        np.savetxt(outfile,data,header=header,comments='')

        if("asci2gdf_binary" in params):
            gdfwatch = stopwatch()
            gdfwatch.start()
            vprint("Converting file to GDF: ",verbose>0,1,False)
            if(".txt"==self.outfile[-4:]):
                gdffile = outfile[:-4]+".gdf"
            else:
                gdffile = outfile+".gdf"

            try:
                os.system(params["asci2gdf_binary"][0]+" -o "+gdffile+" "+self.outfile)
                 #subprocess.call([params["asci2gdf_binary"][0], "-o ",gdffile, self.outfile],shell=True)
                
                subprocess.call(["rm",self.outfile])
                gdfwatch.stop() 
            except Exception as ex:
                print("Error occured while converting ascii to gdf file: ")
                print(str(ex))

            gdfwatch.stop()
            vprint("done. Time ellapsed: "+gdfwatch.print()+".",verbose>0,0,True)

        watch.stop() 
        vprint("...done. Time ellapsed: "+watch.print()+".",verbose>0 and "asci2gdf_binary" in params,0,True)
        vprint("done. Time ellapsed: "+watch.print()+".",verbose>0 and not ("asci2gdf_binary" in params),0,True)


def write_astra(beam,
                outfile,
                verbose=False,
                params=None,
                species='electron',
                probe=True):
    """
    Writes Astra style particles from a beam.
    
    For now, the species must be electrons. 
    
    If probe, the six standard probe particles will be written. 
    """
    watch = StopWatch()
    watch.start()

    vprint("Printing "+str(beam.n)+" particles to '"+outfile+"': ",verbose>0,0,False)

    assert species == 'electron' # TODO: add more species
    
    # number of lines in file
    size = beam.n + 1 # Allow one for reference particle
    i_start = 1 # Start for data particles
    if probe:
        # Add six probe particles, according to the manual
        size += 6
        i_start += 6
    
    # macro charge for each particle
    q_macro = beam.q.to('nC').magnitude / beam.n
    
    qs = np.full((beam.n,), 1.0)*q_macro*unit_registry("nanocoulomb")

    # Astra units and types
    units = ['m', 'm', 'm', 'eV/c', 'eV/c', 'eV/c', 'ns', 'nC']
    names = ['x', 'y', 'z', 'px', 'py', 'pz', 't', 'q', 'index', 'status']
    types = 8*[np.float] + 2*[np.int8]
    # Convert to these units in place
    for i in range(8):
        name = names[i]
        if(name=="q"):
            qs.ito(units[i])
        else:
            beam[name].ito(units[i])
    
    # Reference particle
    ref_particle = {'q':0}
    sigma = {}
    for k in ['x', 'y', 'z', 'px', 'py', 'pz', 't']:
        ref_particle[k] = np.mean(beam[k]).magnitude
        sigma[k] =  np.std(beam[k]).magnitude
        
    # Make structured array
    dtype = np.dtype(list(zip(names, types)))
    data = np.zeros(size, dtype=dtype)
    for k in ['x', 'y', 'z', 'px', 'py', 'pz', 't']:
        data[k][i_start:] = beam[k].magnitude
    
    # Set these to be the same
    data['q'] = q_macro    
    data['index'] = 1    # electron
    data['status'] = -1  # Particle at cathode
    
    # Subtract off reference z, pz, t
    for k in ['z', 'pz', 't']:
        data[k] -= ref_particle[k]
        
    # Put ref particle in first position
    for k in ref_particle:
        data[k][0] = ref_particle[k]
    
    # Optional: probes, according to the manual
    if probe:
        data[1]['x'] = 0.5*sigma['x'];data[1]['t'] =  0.5*sigma['t']
        data[2]['y'] = 0.5*sigma['y'];data[2]['t'] = -0.5*sigma['t']
        data[3]['x'] = 1.0*sigma['x'];data[3]['t'] =  sigma['t']
        data[4]['y'] = 1.0*sigma['y'];data[4]['t'] = -sigma['t']
        data[5]['x'] = 1.5*sigma['x'];data[5]['t'] =  1.5*sigma['t']
        data[6]['y'] = 1.5*sigma['y'];data[6]['t'] = -1.5*sigma['t']        
        data[1:7]['status'] = -3
        data[1:7]['pz'] = 0 #? This is what the Astra Generator does
    
    # Save in the 'high_res = T' format
    np.savetxt(outfile, data, fmt = ' '.join(8*['%20.12e']+2*['%4i']))
    watch.stop() 
    vprint("done. Time ellapsed: "+watch.print()+".",verbose>0,0,True)

#def write_astra(beam,outfile,verbose=0,params=None):  

#        watch = StopWatch()

        # Format particles
#        astra_units={"x":"m", "y":"m", "z":"m","px":"eV/c","py":"eV/c","pz":"eV/c","t":"ns","q":"nC"}
 
#        qspecies = get_species_charge(beam.species)
#        qspecies.ito("nanocoulomb")

#        watch.start()
#        qs = (np.full( (beam.n,), beam.q.to("nanocoulomb")/beam.n))*np.sign(qspecies)*unit_registry("nanocoulomb")
     
#        particle_index = 1;
#        particle_status = -1;
#                
#        vprint("Printing "+str(beam.n)+" particles to '"+outfile+"': ",verbose>0,0,False)
        
#        # Scale parameters to ASTRA units
#        ref_particle = {}
#        for var in astra_units:
#            if(var in beam.params.keys()):
#                beam[var].ito(astra_units[var])
#                ref_particle[var]=beam.avg(var)

#        diff_vars = ["t","z","pz"]
        
#        data = np.zeros( (len(beam["x"]),len(astra_units.keys())+2) )
#        for index, var in enumerate(astra_units.keys()):
#            if(var=="q"):
#                data[:,index] = qs.magnitude
#            else:
#                if(var in diff_vars):
#                    data[:,index] = beam[var].magnitude - ref_particle[var]
#                else:
#                    data[:,index] = beam[var].magnitude
        
#        ref_particle = [ref_particle["x"].magnitude,
#                        ref_particle["y"].magnitude,
#                        ref_particle["z"].magnitude,
#                        ref_particle["px"].magnitude,
#                        ref_particle["py"].magnitude,
#                        ref_particle["pz"].magnitude,
#                        ref_particle["t"].magnitude,
#                        qs[0].magnitude, 1,-1]


#        data[:,-2]=particle_index
#        data[:,-1]=particle_status
#        data[0,:] = ref_particle
        
#        np.savetxt(outfile,data)    
            
#        watch.stop() 
#        vprint("done. Time ellapsed: "+watch.print()+".",verbose>0,0,True)




