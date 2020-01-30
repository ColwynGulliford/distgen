from .physical_constants import *
from .beam import Beam
from .transforms import set_avg_and_std
from .tools import *
from .dist import *
from collections import OrderedDict as odic
import numpy as np

"""
This class defines the main run engine object for distgen and is responsible for
1. Parsing the input data dictionary passed from a Reader object
2. Check the input for internal consistency
3. Collect the parameters for distributions requested in the params dictionary 
4. Form a the Beam object and populated the particle phase space coordinates
"""
class Generator:

    def __init__(self, params=None, verbose=0):
        """
        The class initialization takes in a verbose level for controlling text output to the user
        """
        self.verbose = verbose 
    
        if params:
            self.parse_input(params)
            
    
    def parse_input(self,params):
        """
        Parse the input structure passed from a Reader object.  
        The structure is then converted to an easier form for use in populating the Beam object.
        """
        params = self.convert_params(params)  # Conversion of the input dictionary
        self.input_params = params            # Saving the converted dictionary to the Generator object
        self.check_input_consistency(params)  # Check that the result is logically sound 

    def check_input_consistency(self,params):
        ''' Perform consistency checks on the user input data'''
        
        if( ("r_dist" in params) or ("x_dist" in params) or ("xy_dist" in params) ):
            assert_with_message( ("r_dist" in params)^("x_dist" in params)^("xy_dist" in params),"User must specify only one transverse distribution.")
        if( ("r_dist" in params) or ("y_dist" in params) or ("xy_dist" in params) ):
            assert_with_message( ("r_dist" in params)^("y_dist" in params)^("xy_dist" in params),"User must specify r dist OR y dist NOT BOTH.")
        
        if(params["beam"]["start_type"] == "cathode"):

            vprint("Ignoring user specified px distribution for cathode start.",self.verbose>0 and "px_dist" in params,0,True )
            vprint("Ignoring user specified py distribution for cathode start.",self.verbose>0 and "py_dist" in params,0,True )
            vprint("Ignoring user specified pz distribution for cathode start.",self.verbose>0 and "pz_dist" in params,0,True )
            assert_with_message("MTE" in params["beam"]["params"],"User must specify the MTE for cathode start.") 

            # Handle momentum distribution for cathode
            MTE = self.input_params["beam"]["params"]["MTE"]
            sigma_pxyz = (np.sqrt( (MTE/MC2).to_reduced_units() )*unit_registry("GB")).to("eV/c")
            self.input_params["px_dist"]={"type":"g","params":{"sigma_px":sigma_pxyz}}
            self.input_params["py_dist"]={"type":"g","params":{"sigma_py":sigma_pxyz}}
            self.input_params["pz_dist"]={"type":"g","params":{"sigma_pz":sigma_pxyz}}
                
    def convert_params(self,all_params):
        
        cparams = {}
        for key in all_params:
            cparams[key]=self.get_dist_params(key,all_params)
            
        return cparams
        
    def get_dist_params(self,dname,all_params):
        
        dparams = {}
        for key in all_params[dname].keys():
            
            if(key=="params"): # make physical quantity
                params = {}
                for p in all_params[dname]["params"]:
                    if(isinstance(all_params[dname]["params"][p],dict) and
                       "value" in all_params[dname]["params"][p] and 
                       "units" in all_params[dname]["params"][p]):
                        params[p]=all_params[dname]["params"][p]["value"]*unit_registry(all_params[dname]["params"][p]["units"])
                    else:
                        params[p]=all_params[dname]["params"][p]
                dparams["params"]=params
                
            else: # Copy over
                dparams[key]=all_params[dname][key]
                
        return dparams
                
    def beam(self):
    
        watch = StopWatch()
        watch.start()
    
        verbose = self.verbose
        outputfile = []
        
        beam_params = self.input_params["beam"]
        out_params = self.input_params["output"]

        dist_params = {}
        for p in self.input_params:
            if("_dist" in p):
                var = p[:-5]
                dist_params[var]=self.input_params[p]
        
        vprint("Distribution format: "+out_params["type"],self.verbose>0,0,True)
        
        N = int(beam_params["particle_count"])
        bdist = Beam(N, beam_params["params"]["total_charge"])
        
        if("file" in out_params):
            outfile = out_params["file"]
        else:
            outfile = "test.out.txt"
            vprint("Warning: no output file specified, defaulting to "+outfile+".",verbose>0,1,True)
        vprint("Output file: "+outfile,verbose>0,0,True)
        
        vprint("\nCreating beam distribution....",verbose>0,0,True)
        vprint("Beam starting from: cathode.",verbose>0,1,True)
        vprint("Total charge: {:0.3f~P}".format(bdist.q)+".",verbose>0,1,True)
        vprint("Number of macroparticles: "+str(N)+".",verbose>0,1,True)
        
        bdist.params["x"] = np.full((N,), 0.0)*unit_registry("meter")
        bdist.params["y"] = np.full((N,), 0.0)*unit_registry("meter")
        bdist.params["z"] = np.full((N,), 0.0)*unit_registry("meter")
        bdist.params["px"]= np.full((N,), 0.0)*unit_registry("eV/c")
        bdist.params["py"]= np.full((N,), 0.0)*unit_registry("eV/c")
        bdist.params["pz"]= np.full((N,), 0.0)*unit_registry("eV/c")
        bdist.params["t"] = np.full((N,), 0.0)*unit_registry("s")
        bdist.params["w"] = np.full((N,), 1/bdist.n)*unit_registry("dimensionless")

        avgs = odic()
        avgs["x"] = 0*unit_registry("meter")
        avgs["y"] = 0*unit_registry("meter")
        avgs["z"] = 0*unit_registry("meter")
        avgs["px"]= 0*unit_registry("eV/c")
        avgs["py"]= 0*unit_registry("eV/c")
        avgs["pz"]= 0*unit_registry("eV/c")
        avgs["t"] = 0*unit_registry("s")

        stds = odic()
        stds["x"] = 0*unit_registry("meter")
        stds["y"] = 0*unit_registry("meter")
        stds["z"] = 0*unit_registry("meter")
        stds["px"]= 0*unit_registry("eV/c")
        stds["py"]= 0*unit_registry("eV/c")
        stds["pz"]= 0*unit_registry("eV/c")
        stds["t"] = 0*unit_registry("s")
        
        # Get number of populations:
        npop = 0
        for param in self.input_params:

            if("_dist" in param):
                vstr = param[:-5]
                if(vstr in ["r","x","y","z","px","py","pz","t","theta"]):
                    npop = npop + 1
                elif(vstr in ["xy"]):
                    npop = npop + 2
            
        rgen = RandGen()
        shape = ( N, npop )
        if(beam_params["rand_type"]=="hammersley"):
            rns = rgen.rand(shape, sequence="hammersley",params={"burnin":-1,"primes":()})
        else:
            rns = rgen.rand(shape)
        
        count = 0
            
        # Do radial dist first if requested
        if("r" in dist_params):
                
            r="r"
            vprint("r distribution: ",verbose>0,1,False)  
                
            # Get distribution
            dist = get_dist(r,dist_params[r]["type"],dist_params[r]["params"],verbose=verbose)      
            rs = dist.cdfinv(rns[count,:]*unit_registry("dimensionless") )       # Sample to get beam coordinates

            count = count + 1

            if("theta" not in dist_params):

                vprint("Assuming cylindrical symmetry...",verbose>0,2,True)
                    
                # Sample to get beam coordinates
                params = {"min_theta":0*unit_registry("rad"),"max_theta":2*pi}
                ths=(Uniform("theta",**params)).cdfinv(rns[-1,:]*unit_registry("dimensionless"))        
   
                avgr=0*unit_registry("m")

                if("sigma_xy" in dist_params[r]["params"]):
                    rrms= math.sqrt(2)*dist_params[r]["params"]["sigma_xy"]
                elif("sigma_xy" in beam_params["params"]):
                    rrms= math.sqrt(2)*beam_params["params"]["sigma_xy"]
                else:
                    rrms = dist.rms()

                avgCos = 0
                avgSin = 0
                avgCos2 = 0.5
                avgSin2 = 0.5
                   
            else:
                count = count+1
                dist_params.pop("theta")
  
            bdist.params["x"]=rs*np.cos(ths)
            bdist.params["y"]=rs*np.sin(ths)

            avgs["x"] = avgr*avgCos
            avgs["y"] = avgr*avgSin

            stds["x"] = rrms*np.sqrt(avgCos2)
            stds["y"] = rrms*np.sqrt(avgSin2)   

            # remove r from list of distributions to sample
            dist_params.pop("r")
            #self.dist_params.pop("x",None)
            #self.dist_params.pop("y",None)
           
        # Do 2D distributions
        if("xy" in dist_params):

            vprint("xy distribution: ",verbose>0,1,False) 
            dist = get_dist("xy",dist_params["xy"]["type"],dist_params["xy"]["params"],verbose=0)
            bdist["x"],bdist["y"] = dist.cdfinv(rns[count:count+2,:]*unit_registry("dimensionless"))
            count = count + 2
            dist_params.pop("xy")

            stds["x"]=bdist.std("x")
            stds["y"]=bdist.std("y")
        
        # Do all other specified single coordinate dists   
        for x in dist_params.keys():

            vprint(x+" distribution: ",verbose>0,1,False)   
            dist = get_dist(x,dist_params[x]["type"],dist_params[x]["params"],verbose=verbose)      # Get distribution
            bdist[x]=dist.cdfinv(rns[count,:]*unit_registry("dimensionless"))               # Sample to get beam coordinates
              
            # Fix up the avg and std so they are exactly what user asked for
            if("avg_"+x in dist_params[x]["params"]):
                avgs[x]=dist_params[x]["params"]["avg_"+x]
            else:
                avgs[x] = dist.avg()

            if("sigma_"+x in dist_params[x]["params"]):
                stds[x] = dist_params[x]["params"]["sigma_"+x]
            else:
                stds[x] = dist.std()
               
            count=count+1
        
        # Allow user to overite the distribution moments if desired
        for x in ["x","y","t"]:
            if("avg_"+x in beam_params["params"]):
                avgx = beam_params["params"]["avg_"+x] 
                if(x in avgs and avgx!=avgs[x]):
                    vprint("Overwriting distribution avg "+x+" with user defined value",verbose>0,1,True)
                    avgs[x] = avgx
            if("sigma_"+x in beam_params["params"]):
                stdx = beam_params["params"]["sigma_"+x]
                if(x in stds and stdx!=stds[x]):
                    vprint("Overwriting distribution sigma "+x+" with user defined value",verbose>0,1,True)
                stds[x] = stdx                 

        # Shift and scale coordinates to undo sampling error
        for x in avgs:

            vprint("Scaling sigma_"+x+" -> {:0.3f~P}".format(stds[x]),verbose>0 and bdist[x].std()!=stds[x],1,True)
            vprint("Shifting avg_"+x+" -> {:0.3f~P}".format(avgs[x]),verbose>0 and bdist[x].mean()!=avgs[x],1,True)
            bdist = set_avg_and_std(bdist,x,avgs[x],stds[x])


            #avgi = bdist.avg(x)
            #stdi = bdist.std(x)
            #avgf = avgs[x]
            #stdf = stds[x]

            #vprint("Scaling sigma_"+x+" -> {:0.3f~P}".format(stdf),verbose>0 and stdi!=stdf,1,True)

            # Scale and center each coordinate
            #if(stdi.magnitude>0):
                #bdist[x] = (avgf + (stdf/stdi)*(bdist[x] - avgi)).to(avgi.units)
            #    bdist[x] = ((stdf/stdi)*(bdist[x] - avgi)).to(avgi.units)
            #else:
            #    #bdist[x] = (avgf + (bdist[x] - avgi)).to(avgi.units)
            #    bdist[x] = (bdist[x] - avgi).to(avgi.units)
        
        # Perform any coordinate rotations before shifting to final average locations
        if("rotate_xy" in beam_params["params"]):
            angle = beam_params["params"]["rotate_xy"]
            C = np.cos(angle)
            S = np.sin(angle)
            
            x =  C*bdist["x"]-S*bdist["y"]
            y = +S*bdist["x"]+C*bdist["y"]
            
            bdist["x"]=x
            bdist["y"]=y
        
        for x in avgs:
            bdist[x] = avgs[x] + bdist[x]
        
        if(beam_params["start_type"]=="cathode"):

            bdist["pz"]=np.abs(bdist["pz"])   # Only take forward hemisphere 
            vprint("Cathode start: fixing pz momenta to forward hemisphere",verbose>0,1,True)
            vprint("avg_pz -> {:0.3f~P}".format(bdist.avg("pz"))+", sigma_pz -> {:0.3f~P}".format(bdist.std("pz")),verbose>0,2,True)

        else:
            raise ValueError("Beam start '"+beam_params["start_type"]+"' is not supported!")
        
        watch.stop()
        vprint("...done. Time Ellapsed: "+watch.print()+".\n",verbose>0,0,True)
        return bdist



