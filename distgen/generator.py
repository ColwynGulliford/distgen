from .physical_constants import *
from .beam import beam
from .tools import *
from .dist import *
from collections import OrderedDict as odic
import numpy as np
from matplotlib import pyplot as plt

#import seaborn

class generator():

    verbose = 0    
    beam_params = {}
    dist_params = odic()

    npop = 0

    supported_dists = ['r','theta','x','y','z','px','py','pz','t','r','E','crystals',"file","xy"]

    def __init__(self,verbose):
        self.verbose = verbose 
        
    def parse_input(self,params):

        self.input_params = params
        self.check_input_consistency(params)
        self.set_beam_params(params)

        for p in params:
            if("_dist"==p[-5:]):
                x = p[:-5]
                
                dtype = params[p][0]
                
                if(not (params["start_type"][0]=="cathode" and x in ["px","py","pz"])):
                    self.dist_params[x]=self.set_dist_params(x,dtype,params)
                
                   
        #self.check_input_consistency(self.dist_params)

    def check_input_consistency(self,params):
        ''' Perform consistency checks on the user input data'''
 
        assert_with_message( ("r_dist" in params)^("x_dist" in params)^("xy_dist" in params),"User must specify only one transverse distribution.")
        assert_with_message( ("r_dist" in params)^("y_dist" in params)^("xy_dist" in params),"User must specify r dist OR y dist NOT BOTH.")
        
        if(params["start_type"][0] == "cathode"):
            #assert_with_message("px_dist" not in params,"If cathode start, do not specify px distribution.")
            #assert_with_message("py_dist" not in params,"If cathode start, do not specify py distribution.")
            #assert_with_message("pz_dist" not in params,"If cathode start, do not specify pz distribution.")
            vprint("Ignoring user specified px distribution for cathode start.",self.verbose>0 and "px_dist" in params,0,True )
            vprint("Ignoring user specified py distribution for cathode start.",self.verbose>0 and "py_dist" in params,0,True )
            vprint("Ignoring user specified pz distribution for cathode start.",self.verbose>0 and "pz_dist" in params,0,True )
            assert_with_message("MTE" in params,"User must specify the MTE for cathode start.") 


    def set_dist_params(self,x,dist,params):

        preqs,popts = self.get_dist_params_list(x,dist,params)

        ps = {"type":dist}  # list of parameters supplied by user + defaults 

        # Look for required arguements in user supplied list
        for preq in preqs:
            #print(x,preq)
            if(preq == "file"):
                if(x!="xy"):
                    ps[preq] = {"file":params[x+"_dist"][1],"units":get_unit_str(params[x+"_dist"][2])}
                elif(x=="xy"):
                    ps[preq] = {"file":params[x+"_dist"][1]}
            elif(preq in params.keys()):
                ps[preq]=self.parse_physical_param(params[preq])

        # Check that all requirements were found
        for preq in preqs:
            if(preq not in ps.keys()):
                raise ValueError("Required dist parameter '"+preq+"' was not supplied!")

        # Look for optional user values:
        for popt in popts:
            if(popt in params.keys()):
                ps[popt]=self.parse_physical_param(params[popt])

        return ps

    def get_dist_params_list(self,x,dist,params):

        if(dist=="uniform" or dist=="u"):
            self.npop = self.npop + 1
            return (["min_"+x,"max_"+x],[])
        if(dist=="gaussian" or dist=="g"):
            self.npop = self.npop + 1
            return (["sigma_"+x],["avg_"+x])
        if(dist=="rg" or dist=="radial_gaussian"):
            self.npop = self.npop + 1
            return (["sigma_xy"],["sigma_x","sigma_y"])
        if(dist=="truncated_gaussian" or dist=="tg"):
            self.npop = self.npop + 1
            return (["truncation_fraction","pinhole_size"],[])
        if(dist=="crystals"):
            self.npop = self.npop + 1
            return ([p for p in params if ("crystal_length_" in p or "crystal_angle_" in p)],["avg_t"])
        if(dist=="file" and x=="xy"):
            self.npop = self.npop + 2
            return (["file"],["avg_x","avg_y"])
        elif(dist=="file"):
            self.npop = self.npop + 2
            return (["file"],["avg_"+x])
        
        else:
            raise ValueError("Unkown distribution type: "+dist)

    def is_physical_param(ps):

        if( (len(ps)==2 and is_floatable(ps[0]) and is_unit_str(ps[1])) or (len(ps)==1 and is_floatable(ps[0])) ):
            return True
        return False

    def parse_physical_param(self,ps):

        if(len(ps)==2 and is_floatable(ps[0]) and is_unit_str(ps[1])):
            return float(ps[0])*unit_registry(ps[1][1:-1])
        elif(len(ps)==1 and is_floatable(ps[0])):
            return float(ps[0])*unit_registry("dimensionless")  
        else:
            raise ValueError("Could not parse '"+str(ps)+"' to physical parameter.")
        
    def set_beam_params(self,params):
 
        reqs = ["total_charge","particle_count","rand_type","start_type"]
        for req in reqs: 
            if(req not in params.keys()):
                raise ValueError("Required beam parameter "+req+"not specified!")
        
        self.beam_params["total_charge"]=self.parse_physical_param(params["total_charge"])
        self.beam_params["particle_count"]=int(params["particle_count"][0])
        self.beam_params["rand_type"]=params["rand_type"][0]
        self.beam_params["start_type"]=params["start_type"][0]
        self.beam_params["output_format"]=params["output_format"][0]

        for req in reqs: 
            if(req not in self.beam_params.keys()):
                raise ValueError("Required beam parameter "+req+"not specified!")

    def get_beam(self):
    
        verbose = self.verbose

        #vprint("--------------------------------------------------",self.verbose>0,0,True)
        vprint("Distribution format: "+self.beam_params["output_format"],self.verbose>0,0,True)
        #vprint("--------------------------------------------------",self.verbose>0,0,True)

        N = int(self.beam_params["particle_count"])
        bdist = beam(N,  self.beam_params["total_charge"])

        if("dist_name" in self.input_params):
            outfile = self.input_params["dist_name"][0]
        else:
            outfile = "test.out.txt"
            vprint("Warning: no output file specified, defaulting to "+outfile+".",verbose>0,1,True)
        vprint("Output file: "+outfile,verbose>0,0,True)

        watch = stopwatch()
        watch.start()

        vprint("\nCreating beam distribution....",verbose>0,0,True)
        vprint("Beam starting from: cathode.",verbose>0,1,True)
        vprint("Total charge: {:0.3f~P}".format(bdist.q)+".",verbose>0,1,True)
        vprint("Number of macroparticles: "+str(N)+".",verbose>0,1,True)
             
        if(self.beam_params["start_type"]=="cathode"):

            bdist.params["x"] = np.full((N,), 0.0)*unit_registry("meter")
            bdist.params["y"] = np.full((N,), 0.0)*unit_registry("meter")
            bdist.params["z"] = np.full((N,), 0.0)*unit_registry("meter")
            bdist.params["px"]= np.full((N,), 0.0)*unit_registry("eV/c")
            bdist.params["py"]= np.full((N,), 0.0)*unit_registry("eV/c")
            bdist.params["pz"]= np.full((N,), 0.0)*unit_registry("eV/c")
            bdist.params["t"] = np.full((N,), 0.0)*unit_registry("s")

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

            # Handle momentum distribution for cathode
            MTE = self.parse_physical_param(self.input_params["MTE"])
            sigma_pxyz = (np.sqrt( (MTE/MC2).to_reduced_units() )*unit_registry("GB")).to("eV/c")
            self.dist_params["px"]={"type":"g","sigma_px":sigma_pxyz}
            self.dist_params["py"]={"type":"g","sigma_py":sigma_pxyz}
            self.dist_params["pz"]={"type":"g","sigma_pz":sigma_pxyz}

            rgen = randgen()
            shape = ( N, self.npop+3 )

            if(self.beam_params["rand_type"]=="hammersley"):
                rns = rgen.rand(shape, sequence="hammersley",params={"burnin":-1,"primes":()})
            else:
                rns = rgen.rand(shape)

            count = 0

            # Do radial dist first if requested
            if("r" in self.dist_params):
                r="r"
                vprint("r distribution: ",verbose>0,1,False)   
                dist = self.get_dist(r,self.dist_params[r])      # Get distribution

                rs = dist.cdfinv(rns[count,:])        # Sample to get beam coordinates

                count = count + 1

                if("theta" not in self.dist_params):

                    vprint("Assuming cylindrical symmetry...",verbose>0,2,True)
                    ths=(uniform(0*unit_registry("rad"), 2*pi)).cdfinv(rns[-1,:])        # Sample to get beam coordinates
   
                    avgr=0*unit_registry("m")

                    if("sigma_xy" in self.dist_params[r]):
                        rrms= math.sqrt(2)*self.dist_params[r]["sigma_xy"]
                    else:
                        rrms = dist.rms()
                        print(rrms)

                    avgCos = 0
                    avgSin = 0
                    avgCos2 = 0.5
                    avgSin2 = 0.5
                   
                else:
                    count = count+1
                    self.dist_params.pop("theta")
  
                bdist.params["x"]=rs*np.cos(ths)
                bdist.params["y"]=rs*np.sin(ths)

                avgs["x"] = avgr*avgCos
                avgs["y"] = avgr*avgSin

                stds["x"] = rrms*np.sqrt(avgCos2)
                stds["y"] = rrms*np.sqrt(avgSin2)       

                # remove r,x,y from list of distributions to sample
                self.dist_params.pop("r")
                #self.dist_params.pop("x",None)
                #self.dist_params.pop("y",None)

            # Do 2D distributions
            if("xy" in self.dist_params):
                vprint("xy distribution: ",verbose>0,1,False) 
                dist = self.get_dist("xy",self.dist_params["xy"])
                bdist["x"],bdist["y"] = dist.cdfinv(rns[count:count+2,:]*unit_registry("dimensionless"))
                count = count + 2
                self.dist_params.pop("xy")

                stds["x"]=bdist["x"].std()
                stds["y"]=bdist["y"].std()

                #dist.test_sampling()

            # Do all other specified single coordinate dists   
            for x in self.dist_params.keys():

                vprint(x+" distribution: ",verbose>0,1,False)   
                dist = self.get_dist(x,self.dist_params[x])      # Get distribution
                bdist[x]=dist.cdfinv(rns[count,:])               # Sample to get beam coordinates
              
                # Fix up the avg and std so they are exactly what user asked for
                if("avg_"+x in self.dist_params[x]):
                    avgs[x]=self.dist_params[x]["avg_"+x]
                else:
                    avgs[x] = dist.avg()

                if("sigma_"+x in self.dist_params[x]):
                    stds[x] = self.dist_params[x]["sigma_"+x]
                else:
                    stds[x] = dist.std()
               
                count=count+1

            # Allow user to overite the distribution moments if desired
            for x in ["x","y","t"]:
                if("avg_"+x in self.input_params):
                    avgx = self.parse_physical_param(self.input_params["avg_"+x]) 
                    if(x in avgs and avgx!=avgs[x]):
                        vprint("Overwriting distribution avg "+x+" with user defined value",verbose>0,1,True)
                    avgs[x] = avgx
                if("sigma_"+x in self.input_params):
                    stdx = self.parse_physical_param(self.input_params["sigma_"+x]) 
                    if(x in stds and stdx!=stds[x]):
                        vprint("Overwriting distribution sigma "+x+" with user defined value",verbose>0,1,True)
                    stds[x] = stdx                 

            # Shift and scale coordinates to undo sampling error
            for x in avgs:

                avgi = np.mean(bdist[x])
                stdi = np.std(bdist[x])
                avgf = avgs[x]
                stdf = stds[x]

                
           
                if(stdi.magnitude>0):
                    bdist[x] = (avgf + (stdf/stdi)*(bdist[x] - avgi)).to(avgi.units)
                else:
                    bdist[x] = (avgf + (bdist[x] - avgi)).to(avgi.units)

            bdist["pz"]=np.abs(bdist["pz"])   # Only take forward hemisphere 
            vprint("Cathode start: fixing pz momenta to forward hemisphere",verbose>0,1,True)
            vprint("avg_pz -> {:0.3f~P}".format(np.mean(bdist["pz"]))+", sigma_pz -> {:0.3f~P}".format(np.std(bdist["pz"])),verbose>0,2,True)

        else:
            raise ValueError("Beam start '"+self.beam_params["start_type"]+"' is not supported!")

        watch.stop()
        vprint("...done. Time Ellapsed: "+watch.print()+".\n",verbose>0,0,True)

        return bdist,outfile

    def get_dist(self,x,dparams):

        dtype = dparams["type"]
        dist=None

        if(dtype=="u" or dtype=="uniform"):
            vprint("uniform",self.verbose>0,0,True)
            vprint("min_"+x+" = {:0.3f~P}".format(dparams["min_"+x])+", max_"+x+" = {:0.3f~P}".format(dparams["max_"+x]),self.verbose>0,2,True)
            dist = uniform(dparams["min_"+x],dparams["max_"+x],xstr=x)

        elif(dtype=="g" or dtype=="gaussian"):

            vprint("Gaussian",self.verbose>0,0,True)
            if("avg_"+x not in dparams):
                dparams["avg_"+x] = 0*dparams["sigma_"+x].units
            vprint("avg_"+x+" = {:0.3f~P}".format(dparams["avg_"+x])+", sigma_"+x+" = {:0.3f~P}".format(dparams["sigma_"+x]),self.verbose>0,2,True)
            dist = norm(dparams["avg_"+x],dparams["sigma_"+x],xstr=x)

        elif(dtype=="crystals"):

            vprint("crystal temporal laser shaping",self.verbose>0,0,True)
            lengths = [dparams[dp] for dp in dparams if "crystal_length" in dp]
            angles  = [dparams[dp] for dp in dparams if "crystal_angle" in dp]
            dist = temporal_laser_pulse_stacking(lengths,angles,verbose=self.verbose)

        elif( (dtype=="rg" or dtype=="radial_gaussian") and x=="r"):

            vprint("radial Gaussian",self.verbose>0,0,True)
            vprint("sigma_xy = {:0.3f~P}".format(dparams["sigma_xy"]),self.verbose>0,2,True)
            dist = normrad(dparams["sigma_xy"])

        elif( (dtype=="tg" or dtype=="truncated_radial_gaussian") and x=="r"):

            vprint("radial Gaussian",self.verbose>0,0,True)
            vprint("f = {:0.3f~P}".format(dparams["truncation_fraction"]),self.verbose>0,2,False)
            vprint(", pinhole size = {:0.3f~P}".format(dparams["pinhole_size"]),self.verbose>0,0,True)
            dist = normrad_trunc(dparams["pinhole_size"]/2.0, dparams["truncation_fraction"]) 

        elif(dtype == "file" and x == "r"):
           
            vprint("radial distribution file: '"+dparams["file"]["file"]+"' ["+dparams["file"]["units"]+"]",self.verbose>0,0,True)
            dist = radfile(dparams["file"]["file"],units=dparams["file"]["units"])

        elif(dtype == "file" and x == "xy"):

            vprint("xy distribution file: '"+dparams["file"]["file"],self.verbose>0,0,True)
            dist = file2d(dparams["file"]["file"])

        else:
            raise ValueError("Distribution type '"+dtype+"' is not supported.")

        return dist

    def dist_is_supported(self,x):
    
        if(x in self.supported_dists):
           return 
        else:
           raise ValueError("Distribution type "+x+" is not supported")


