from tools import vprint
from tools import stopwatch
from physical_constants import  *
import numpy as np
import subprocess
import os
from collections import OrderedDict as odict

def get_writer(output,outfile):

    if(output=="gpt"):
        return gpt_writer(outfile)
    elif(output=="astra"):
        return astra_writer(outfile)
    else:
        raise ValueError("Output format "+str(output)+" is not currently supported.")

class writer():

    output = None
    def __init__(self,outfile):

        self.outfile = outfile
        
    def write(self,dist,verbose,params):

        print("Undefined writer")

class gpt_writer(writer):

    def __init__(self,outfile):
        super().__init__(outfile)

    def write(self,beam,verbose,params):  

        watch = stopwatch()

        # Format particles
        gpt_units={"x":"m", "y":"m", "z":"m","px":"GB","py":"GB","pz":"GB","t":"s"}
 
        watch.start()
        beam.params["q"] = (np.full( (beam.n,), 1.0)*qe).to("coulomb")
        beam.params["nmacro"] = np.full( (beam.n,), np.abs(beam.q.to("coulomb")/qe/beam.n).magnitude )*unit_registry("dimensionless")

        vprint("\nPrinting particles to '"+self.outfile+"'...",verbose>0,0,True)
        
        # Scale parameters to GPT units
        for var in gpt_units:
            beam[var].ito(gpt_units[var])

        headers = odict( {"x":"x", "y":"y", "z":"z", "px":"GBx",  "py":"GBy", "pz":"GBz", "t":"t", "q":"q", "nmacro":"nmacro"} )
        header = '   '.join(headers.values())

        data = np.zeros( (len(beam["x"]),len(headers)) )
        for index, var in enumerate(headers):
            data[:,index] = beam[var].magnitude
        np.savetxt(self.outfile,data,header=header)
 
        if("asci2gdf_binary" in params):
            gdfwatch = stopwatch()
            gdfwatch.start()
            vprint("Converting file to GDF: ",verbose>0,1,False)
            if(".txt"==self.outfile[-4:]):
                gdffile = self.outfile[:-4]+".gdf"
            else:
                gdffile = self.outfile+".gdf"

            try:
                os.system(params["asci2gdf_binary"][0]+" -o "+gdffile+" "+self.outfile)
                #subprocess.call([params["asci2gdf_binary"][0], "-o ",gdffile, self.outfile],shell=True)
                
                subprocess.call(["rm",self.outfile])
                gdfwatch.stop() 
            except Exception as ex:
                print("Error occured while converting ascii to gdf file: ")
                print(str(ex))

            #gdfwatch.stop()
            vprint("done. Time ellapsed: "+gdfwatch.print()+".",verbose>0,0,True)

        watch.stop() 
        vprint("...done. Time ellapsed: "+watch.print()+".",verbose>0,0,True)

class astra_writer(writer):

    def __init__(self,outfile):
        super().__init__(outfile)

    def write(self,beam,verbose,params):  

        watch = stopwatch()

        # Format particles
        astra_units={"x":"m", "y":"m", "z":"m","px":"eV/c","py":"eV/c","pz":"eV/c","t":"ns"}
 
        watch.start()
        qs = (np.full( (beam.n,), beam.n)*qe).to("nanocoulomb")
        particle_index = 1;
        particle_status = -1;

        vprint("\nPrinting particles to '"+self.outfile+"': ",verbose>0,0,False)
        f = open(self.outfile,'w')

        sp = "  "

        # Scale parameters to ASTRA units
        for var in astra_units:
            beam[var].ito(astra_units[var])

        pz0 = np.mean(beam["pz"])
        
        line = "0.0"+sp+"0.0"+sp+"0.0"+sp+"0.0"+sp+"0.0"+sp+str(pz0.magnitude)+sp+"0.0"+sp+str(qs[0].magnitude)+sp+str(particle_index)+sp+str(particle_status)+"\n"
        f.write(line)

        for ii in range(len(beam["x"])-1):
            line = (
                    str(beam["x"][ii].magnitude)  + sp + 
                    str(beam["y"][ii].magnitude)  + sp + 
                    str(beam["z"][ii].magnitude)  + sp + 
                    str(beam["px"][ii].magnitude) + sp + 
                    str(beam["py"][ii].magnitude) +  sp + 
                    str(beam["pz"][ii].magnitude-pz0.magnitude) + sp +
                    str(-beam["t"][ii].magnitude) + sp +
                    str(qs[ii].magnitude) + sp + str(particle_index) + str(particle_status) +
                    "\n")
            f.write(line)

        f.close()
        watch.stop() 
        vprint("...done. Time ellapsed: "+watch.print()+".",verbose>0,0,True)


