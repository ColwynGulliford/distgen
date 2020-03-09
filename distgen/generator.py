from .physical_constants import *
from .beam import Beam
from .transforms import set_avg_and_std, transform
from .tools import *
from .dist import *
from collections import OrderedDict as odic
from pmd_beamphysics import ParticleGroup
import numpy as np
import h5py
import yaml
import copy

"""
This class defines the main run engine object for distgen and is responsible for
1. Parsing the input data dictionary passed from a Reader object
2. Check the input for internal consistency
3. Collect the parameters for distributions requested in the params dictionary 
4. Form a the Beam object and populated the particle phase space coordinates
"""
class Generator:

    def __init__(self, input=None, verbose=0):
        """
        The class initialization takes in a verbose level for controlling text output to the user
        """
        self.verbose = verbose 
    
        self.input = input

        # This will be set with .run()
        self.particles = None

        if input:
            self.parse_input(input)
            self.configure()
            
    
    def parse_input(self, input):
        """
        Parse the input structure passed from a Reader object.  
        The structure is then converted to an easier form for use in populating the Beam object.
        
        YAML or JSON is accepted if params is a filename (str)
        """
        if isinstance(input, str):
            if os.path.exists(os.path.expandvars(input)):
                # Try file
                input = os.path.expandvars(input)
                input = yaml.safe_load(open(input))
            else:
                #Try raw string
                input = yaml.safe_load(input)
        self.input = input

    def configure(self):

        self.params = copy.deepcopy(self.input)         # Copy the input dictionary
        convert_params(self.params)                     # Conversion of the input dictionary using tools.convert_params
        self.check_input_consistency(self.params)       # Check that the result is logically sound 
        
    def check_input_consistency(self, params):

        ''' Perform consistency checks on the user input data'''

        # Make sure all required top level params are present
        required_params = ['generator','beam']
        for rp in required_params:
            assert rp in params, 'Required generator parameter ' + rp + ' not found.'

        # Check that only allowed params present at top level
        allowed_params = required_params + ['output','transforms']
        for p in params:
            assert p in allowed_params or '_dist'==p[-5:], 'Unexpected distgen input parameter: ' + p[-5:]
        
        # Check consistency of transverse coordinate definitions
        if( ("r_dist" in params) or ("x_dist" in params) or ("xy_dist" in params) ):
            assert_with_message( ("r_dist" in params)^("x_dist" in params)^("xy_dist" in params),"User must specify only one transverse distribution.")
        if( ("r_dist" in params) or ("y_dist" in params) or ("xy_dist" in params) ):
            assert_with_message( ("r_dist" in params)^("y_dist" in params)^("xy_dist" in params),"User must specify r dist OR y dist NOT BOTH.")

        if(params["generator"]["start"]['type'] == "cathode"):

            vprint("Ignoring user specified z distribution for cathode start.", self.verbose>0 and "z_dist" in params,0,True )
            vprint("Ignoring user specified px distribution for cathode start.", self.verbose>0 and "px_dist" in params,0,True )
            vprint("Ignoring user specified py distribution for cathode start.", self.verbose>0 and "py_dist" in params,0,True )
            vprint("Ignoring user specified pz distribution for cathode start.", self.verbose>0 and "pz_dist" in params,0,True )
            assert "MTE" in params['generator']['start']['params'], "User must specify the MTE for cathode start." 

            # Handle momentum distribution for cathode
            MTE = self.params['generator']['start']['params']["MTE"]
            sigma_pxyz = (np.sqrt( (MTE/MC2).to_reduced_units() )*unit_registry("GB")).to("eV/c")
            self.params["px_dist"]={"type":"g","params":{"sigma_px":sigma_pxyz}}
            self.params["py_dist"]={"type":"g","params":{"sigma_py":sigma_pxyz}}
            self.params["pz_dist"]={"type":"g","params":{"sigma_pz":sigma_pxyz}}

        elif(params["generator"]["start"]['type']=='time'):

            vprint("Ignoring user specified t distribution for time start.", self.verbose>0 and "t_dist" in params, 0, True)
            params.pop('t_dist')

        if('output' in self.params):
            out_params = self.params["output"]
            for op in out_params:
                assert op in ['file','type'], 'Unexpected output parameter specified: '+op
        else:
            self.params['output'] = {"type":None}

    def __getitem__(self, varstr):
         return get_nested_dict(self.input, varstr, sep=':', prefix='distgen')

    def __setitem__(self, varstr, val):
        return set_nested_dict(self.input, varstr, val, sep=':', prefix='distgen')
   
    def beam(self):
    
        self.configure()
    
        watch = StopWatch()
        watch.start()
    
        verbose = self.verbose
        outputfile = []
        
        beam_params = self.params["beam"]

        if('transforms' in self.params):
            transforms = self.params['transforms']
        else:
            transforms = None

        dist_params = {}
        for p in self.params:
            if("_dist" in p):
                var = p[:-5]
                dist_params[var]=self.params[p]

        for var in dist_params:
            for p in dist_params[var]:
                assert p in ['type','params'],'Unexpected distribution parameter:' + var+'_dist:'+p
        
        vprint("Distribution format: "+str(self.params['output']["type"]), self.verbose>0, 0, True)

        N = int(self.params['generator']['rand']['count'])
        bdist = Beam(**beam_params['params'])
        
        if("file" in self.params['output']):
            outfile = self.params['output']["file"]
        else:
            outfile = "None"
            vprint("Warning: no output file specified, defaulting to "+outfile+".", verbose>0, 1, True)
        vprint("Output file: "+outfile, verbose>0, 0, True)
        
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
        bdist.params["w"] = np.full((N,), 1/N)*unit_registry("dimensionless")

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
        for param in self.params:

            if("_dist" in param):
                vstr = param[:-5]
                if(vstr in ["r","x","y","z","px","py","pz","t","theta"]):
                    npop = npop + 1
                elif(vstr in ["xy"]):
                    npop = npop + 2
            
        rgen = RandGen()
        shape = ( N, npop )
        if(self.params['generator']['rand']['type']=="hammersley"):
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

        # Apply any user desired coordinate transformations
        if(transforms):
            for t,T in transforms.items():
                vprint('Applying user defined transform "'+t+'"...',verbose>0,1,True)
                bdist = transform(bdist, T['type'], T['variables'], **T['params'])
        
        # Handle any start type specific settings
        if(self.params['generator']['start']['type']=="cathode"):

            bdist["pz"]=np.abs(bdist["pz"])   # Only take forward hemisphere 
            vprint("Cathode start: fixing pz momenta to forward hemisphere",verbose>0,1,True)
            vprint("avg_pz -> {:0.3f~P}".format(bdist.avg("pz"))+", sigma_pz -> {:0.3f~P}".format(bdist.std("pz")),verbose>0,2,True)

        elif(self.params['generator']['start']['type']=='time'):
            
            if('tstart' in self.params['generator']['start']['params']):
                tstart = self.params['generator']['start']['params']['tstart']
    
            else:
                tstart = 0*unit_registry('sec')


            vprint('Time start: fixing all particle time values to start time: {:0.3f~P}'.format(tstart), verbose>0, 1, True);
            bdist = set_avg_and_std(bdist,'t',tstart,0.0*unit_registry('sec'))

        else:
            raise ValueError("Beam start '"+beam_params["start_type"]+"' is not supported!")
        
        watch.stop()
        vprint("...done. Time Ellapsed: "+watch.print()+".\n",verbose>0,0,True)
        return bdist
    
    
    def run(self):
        beam = self.beam()
        self.particles = ParticleGroup(data=beam.data())
        
        vprint(f'Created particles in .particles: {self.particles}',self.verbose>0,1,False) 
    
    
    def fingerprint(self):
        """
        Data fingerprint using the input. 
        """
        return fingerprint(self.input)    
    
    def archive(self, h5=None):
        """
        Archive all data to an h5 handle or filename.
        
        If no file is given, a file based on the fingerprint will be created.
        
        """
        if not h5:
            h5 = 'distgen_'+self.fingerprint()+'.h5'
         
        if isinstance(h5, str):
            g = h5py.File(h5, 'w')
            self.vprint(f'Archiving to file {h5}')
        else:
            g = h5
        
        # Initial particles
        if self.particles:
            self.initial_particles.write(g, name='particles')
        
        # Input as flattened dict
        g2 = g.create_group('input')
        d = flatten_dict(self.input)
        for k, v in d.items():
            g2.attrs[k] = v
        
              
        
        return h5    
    
    def __repr__(self):
        s = '<disgten.Generator with input: \n'
        return s+yaml.dump(self.input)+'\n>'

    def check_inputs(self,params):

        # Make sure user isn't passing the wrong parameters:
        allowed_params = self.optional_params + self.required_params + ['verbose']
        for param in params:
            assert param in allowed_params, 'Incorrect param given to '+self.__class__.__name__+ '.__init__(**kwargs): '+param+'\nAllowed params: '+str(allowed_params)

        # Make sure all required parameters are specified
        for req in self.required_params:
            assert req in params, 'Required input parameter '+req+' to '+self.__class__.__name__+'.__init__(**kwargs) was not found.'

