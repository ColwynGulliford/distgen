from .physical_constants import *
from .beam import Beam
from .transforms import set_avg_and_std, transform, set_avg
from .tools import *
from .dist import *
from collections import OrderedDict as odic
from pmd_beamphysics import ParticleGroup, pmd_init
from . import archive

import warnings

import numpy as np
import h5py
import yaml
import copy


class Generator:

    """
    This class defines the main run engine object for distgen and is responsible for
    1. Parsing the input data dictionary passed from a Reader object
    2. Check the input for internal consistency
    3. Collect the parameters for distributions requested in the params dictionary 
    4. Form a the Beam object and populated the particle phase space coordinates
    """

    def __init__(self, input=None, verbose=0):
        """
        The class initialization takes in a verbose level for controlling text output to the user
        """
        self.verbose = verbose 
    
        self.input = input
        
        # This will be set with .beam()
        self.rands = None

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
                with open(input) as fid:
                    input = yaml.safe_load(fid)
            else:
                #Try raw string
                input = yaml.safe_load(input)
                assert isinstance(input, dict), f'ERROR: parsing unsuccessful, could not read {input}'

        self.input = input

    def configure(self):

        """ Configures the generator for creating a 6d particle distribution:
        1. Copies the input dictionary read in from a file or passed directly
        2. Converts physical quantities to PINT quantities in the params dictionary
        3. Runs consistency checks on the resulting params dict
        """

        self.params = copy.deepcopy(self.input)         # Copy the input dictionary
        convert_params(self.params)                     # Conversion of the input dictionary using tools.convert_params
        self.check_input_consistency(self.params)       # Check that the result is logically sound 
        
    def check_input_consistency(self, params):

        ''' Perform consistency checks on the user input data'''

        # Make sure all required top level params are present
        required_params = ['n_particle', 'random_type', 'total_charge','start']
        for rp in required_params:
            assert rp in params, 'Required generator parameter ' + rp + ' not found.'

        # Check that only allowed params present at top level
        allowed_params = required_params + ['output','transforms']
        for p in params:
            #assert p in allowed_params or '_dist'==p[-5:], 'Unexpected distgen input parameter: ' + p[-5:]
            assert p in allowed_params or p.endswith('_dist'), 'Unexpected distgen input parameter: ' + p
        
        assert params['n_particle']>0, 'User must speficy n_particle must > 0.'

        # Check consistency of transverse coordinate definitions
        if( ("r_dist" in params) or ("x_dist" in params) or ("xy_dist" in params) ):
            assert ("r_dist" in params)^("x_dist" in params)^("xy_dist" in params),"User must specify only one transverse distribution."
        if( ("r_dist" in params) or ("y_dist" in params) or ("xy_dist" in params) ):
            assert  ("r_dist" in params)^("y_dist" in params)^("xy_dist" in params),"User must specify r dist OR y dist NOT BOTH."

        if(params['start']['type'] == "cathode"):

            vprint("Ignoring user specified z distribution for cathode start.", self.verbose>0 and "z_dist" in params,0,True )
            vprint("Ignoring user specified px distribution for cathode start.", self.verbose>0 and "px_dist" in params,0,True )
            vprint("Ignoring user specified py distribution for cathode start.", self.verbose>0 and "py_dist" in params,0,True )
            vprint("Ignoring user specified pz distribution for cathode start.", self.verbose>0 and "pz_dist" in params,0,True )
            assert "MTE" in params['start'], "User must specify the MTE for cathode start." 

            # Handle momentum distribution for cathode
            MTE = self.params['start']["MTE"]
            sigma_pxyz = (np.sqrt( (MTE/MC2).to_reduced_units() )*unit_registry("GB")).to("eV/c")

            self.params["px_dist"]={"type":"g","sigma_px":sigma_pxyz}
            self.params["py_dist"]={"type":"g","sigma_py":sigma_pxyz}
            self.params["pz_dist"]={"type":"g","sigma_pz":sigma_pxyz}

        elif(params['start']['type']=='time'):

            vprint("Ignoring user specified t distribution for time start.", self.verbose>0 and "t_dist" in params, 0, True)
            if('t_dist' in params):
                warnings.warn('Ignoring user specified t distribution for time start.')
                self.params.pop('t_dist')

        if('output' in self.params):
            out_params = self.params["output"]
            for op in out_params:
                assert op in ['file','type'], f'Unexpected output parameter specified: {op}'
        else:
            self.params['output'] = {"type":None}

    def __getitem__(self, varstr):
         return get_nested_dict(self.input, varstr, sep=':', prefix='distgen')

    def __setitem__(self, varstr, val):
        return set_nested_dict(self.input, varstr, val, sep=':', prefix='distgen')
        

    def get_dist_params(self):

        """ Loops through the input params dict and collects all distribution definitions """
        
        dist_vars = [p.replace('_dist','') for p in self.params if(p.endswith('_dist')) ]
        dist_params = {p.replace('_dist',''):self.params[p] for p in self.params if(p.endswith('_dist'))}

        if('r' in dist_vars and 'theta' not in dist_vars):
            vprint("Assuming cylindrical symmetry...",self.verbose>0,1,True)
            dist_params['theta']={'type':'ut','min_theta':0*unit_registry('rad'),'max_theta':2*pi}

        if(self.params['start']['type']=='time' and 't_dist' in self.params):
            raise ValueError('Error: t_dist should not be set for time start')

        return dist_params


    def get_rands(self, variables):

        """ Gets random numbers [0,1] for the coordinatess in variables 
        using either the Hammersley sequence or rand """
 
        specials = ['xy']
        self.rands = {var:None for var in variables if var not in specials}

        if('xy' in variables):
            self.rands['x']=None
            self.rands['y']=None

        elif('r' in variables and 'theta' not in variables):
            self.rands['theta']=None

        n_coordinate = len(self.rands.keys())
        n_particle = int(self.params['n_particle'])
        shape = ( n_coordinate, n_particle )
        
        if(n_coordinate>0):
            rns = random_generator(shape, sequence=self.params['random_type'])
        
        for ii, key in enumerate(self.rands.keys()):
            if(len(rns.shape)>1):
                self.rands[key] = rns[ii,:]*unit_registry('dimensionless')
            else:
                self.rands[key] = rns[:]*unit_registry('dimensionless')

        var_list = list(self.rands.keys())
        for ii, vii in enumerate(var_list[:-1]):
            viip1 = var_list[ii+1]
            assert (not np.array_equal(self.rands[vii].magnitude, self.rands[viip1].magnitude)) or n_particle==1, f'Error: coordinate probalitiies for {vii} and {viip1} are the same!'
            
            # These lines can be used to check for unwanted correlations
            #v0 = self.rands[vii].magnitude-self.rands[vii].magnitude.mean()
            #v1 = self.rands[viip1].magnitude-self.rands[viip1].magnitude.mean()
            #print( np.mean(v0*v1) )
             

    def beam(self):

        """ Creates a 6d particle distribution and returns it in a distgen.beam class """

        watch = StopWatch()
        watch.start()
    
        self.configure()

        verbose = self.verbose
        outputfile = []
        
        beam_params = {'total_charge':self.params['total_charge'], 'n_particle':self.params['n_particle']}

        if('transforms' in self.params):
            transforms = self.params['transforms']
        else:
            transforms = None

        #dist_params = {p.replace('_dist',''):self.params[p] for p in self.params if(p.endswith('_dist')) }        
        #self.get_rands()

        vprint(f'Distribution format: {self.params["output"]["type"]}', self.verbose>0, 0, True)

        N = int(self.params['n_particle'])
        bdist = Beam(**beam_params)
        
        if("file" in self.params['output']):
            outfile = self.params['output']["file"]
        else:
            outfile = "None"
            vprint(f'Warning: no output file specified, defaulting to "{outfile}".', verbose>0, 1, True)
        vprint(f'Output file: {outfile}', verbose>0, 0, True)

        vprint('\nCreating beam distribution....',verbose>0,0,True)
        vprint(f"Beam starting from: {self.input['start']['type']}",verbose>0,1,True)
        vprint(f'Total charge: {bdist.q:G~P}.',verbose>0,1,True)
        vprint(f'Number of macroparticles: {N}.',verbose>0,1,True)

        units = {'x':'m', 'y':'m', 'z':'m', 'px':'eV/c', 'py':'eV/c', 'pz':'eV/c', 't':'s'}

        # Initialize coordinates to zero       
        for var, unit in units.items():
            bdist[var] = np.full(N, 0.0)*unit_registry(units[var])

        bdist["w"] = np.full((N,), 1/N)*unit_registry("dimensionless")

        avgs = {var:0*unit_registry(units[var]) for var in units}
        stds = {var:0*unit_registry(units[var]) for var in units}

        dist_params = self.get_dist_params()   # Get the relevant dist params, setting defaults as needed, and samples random number generator
        self.get_rands(list(dist_params.keys()))

        # Do radial dist first if requested
        if('r' in dist_params and 'theta' in dist_params):

            vprint('r distribution: ',verbose>0, 1, False)  
                
            # Get r distribution
            rdist = get_dist('r', dist_params['r'], verbose=verbose)      

            if(rdist.rms()>0):
                r = rdist.cdfinv(self.rands['r'])       # Sample to get beam coordinates
                    
            # Sample to get beam coordinates
            vprint('theta distribution: ', verbose>0, 1, False)
            theta_dist = get_dist('theta', dist_params['theta'], verbose=verbose)  
            theta = theta_dist.cdfinv(self.rands['theta'])

            rrms = rdist.rms()
            avgr = rdist.avg()

            avgCos = 0
            avgSin = 0
            avgCos2 = 0.5
            avgSin2 = 0.5
            
            bdist['x']=r*np.cos(theta)
            bdist['y']=r*np.sin(theta)

            avgs['x'] = avgr*avgCos
            avgs['y'] = avgr*avgSin

            stds['x'] = rrms*np.sqrt(avgCos2)
            stds['y'] = rrms*np.sqrt(avgSin2)   

            # remove r, theta from list of distributions to sample
            del dist_params['r']
            del dist_params['theta']

        # Do 2D distributions
        if("xy" in dist_params):

            vprint('xy distribution: ', verbose>0, 1, False) 
            dist = get_dist('xy', dist_params['xy'], verbose=verbose)
            bdist['x'], bdist['y'] = dist.cdfinv(self.rands['x'], self.rands['y'])

            dist_params.pop('xy')

            stds['x']=bdist.std('x')
            stds['y']=bdist.std('y')
        
        # Do all other specified single coordinate dists   
        for x in dist_params.keys():

            vprint(x+" distribution: ",verbose>0,1,False)   
            dist = get_dist(x, dist_params[x], verbose=verbose)      # Get distribution

            if(dist.std()>0):

                # Only reach here if the distribution has > 0 size
                bdist[x]=dist.cdfinv(self.rands[x])                      # Sample to get beam coordinates

                # Fix up the avg and std so they are exactly what user asked for
                if("avg_"+x in dist_params[x]):
                    avgs[x]=dist_params[x]["avg_"+x]
                else:
                    avgs[x] = dist.avg()

                stds[x] = dist.std()
                #if("sigma_"+x in dist_params[x]):
                #    stds[x] = dist_params[x]["sigma_"+x]
                #else:
                    #stds[x] = dist.std()
                    #print(x, stds[x])

        # Shift and scale coordinates to undo sampling error
        for x in avgs:

            vprint(f'Shifting avg_{x} = {bdist.avg(x):G~P} -> {avgs[x]:G~P}', verbose>0 and bdist[x].mean()!=avgs[x],1,True)
            vprint(f'Scaling sigma_{x} = {bdist.std(x):G~P} -> {stds[x]:G~P}',verbose>0 and bdist[x].std() !=stds[x],1,True)

            #bdist = transform(bdist, {'type':f'set_avg_and_std {x}', 'avg_'+x:avgs[x],'sigma_'+x:stds[x], 'verbose':0}) 
            bdist = set_avg_and_std(bdist, **{'variables':x, 'avg_'+x:avgs[x],'sigma_'+x:stds[x], 'verbose':0})
        
        # Handle any start type specific settings
        if(self.params['start']['type']=="cathode"):

            bdist['pz']=np.abs(bdist['pz'])   # Only take forward hemisphere 
            vprint('Cathode start: fixing pz momenta to forward hemisphere',verbose>0,1,True)
            vprint(f'avg_pz -> {bdist.avg("pz"):G~P}, sigma_pz -> {bdist.std("pz"):G~P}',verbose>0,2,True)

        elif(self.params['start']['type']=='time'):
            
            if('tstart' in self.params['start']):
                tstart = self.params['start']['tstart']
    
            else:
                vprint("Time start: no start time specified, defaulting to 0 sec.",verbose>0,1,True)
                tstart = 0*unit_registry('sec')

            vprint(f'Time start: fixing all particle time values to start time: {tstart:G~P}.', verbose>0, 1, True);
            bdist = set_avg(bdist,**{'variables':'t','avg_t':0.0*unit_registry('sec'), 'verbose':verbose>0})

        else:
            raise ValueError(f'Beam start type "{self.params["start"]["type"]}" is not supported!')
        
        # Apply any user desired coordinate transformations
        if(transforms):

            # Check if the user supplied the transform order, otherwise just go through the dictionary
            if('order' in transforms):
                order = transforms['order']
                if(not isinstance(order, list)):
                    raise ValueError('Transform "order" key must be associated a list of transform IDs')
                del transforms['order']
            else:
                order = transforms.keys()

            
            for name in order:

                T = transforms[name]
                T['verbose']=verbose>0
                vprint(f'Applying user supplied transform: "{name}" = {T["type"]}...', verbose>0, 1, True)
                bdist = transform(bdist, T)

        watch.stop()
        vprint(f'...done. Time Ellapsed: {watch.print()}.\n',verbose>0,0,True)
        return bdist
    
    
    def run(self):
        """ Runs the generator.beam function stores the partice in 
        an openPMD-beamphysics ParticleGroup in self.particles """
        beam = self.beam()
        self.particles = ParticleGroup(data=beam.data())
        vprint(f'Created particles in .particles: \n   {self.particles}', self.verbose>0,1,False) 
    
    
    def fingerprint(self):
        """
        Data fingerprint using the input. 
        """
        return fingerprint(self.input)    
    
    
    def load_archive(self, h5=None):
        """
        Loads input and output from archived h5 file.
        
        
        
        
        See: Generator.archive
        
        
        """
        if isinstance(h5, str):
            g = h5py.File(h5, 'r')
            
            glist = archive.find_distgen_archives(g)
            n = len(glist)
            if n == 0:
                # legacy: try top level
                message = 'legacy'
            elif n == 1:
                gname = glist[0]
                message = f'group {gname} from'
                g = g[gname]
            else:
                raise ValueError(f'Multiple archives found in file {h5}: {glist}')
            
            vprint(f'Reading {message} archive file {h5}', self.verbose>0,1,False)             
        else:
            g = h5            
            
            vprint(f'Reading Distgen archive file {h5}', self.verbose>0,1,False) 

        self.input = archive.read_input_h5(g['input']) 
        
        if 'particles' in g:
            self.particles = ParticleGroup(g['particles'])
        else:
            vprint(f'No particles found.', self.verbose>0,1,False) 


    def archive(self, h5=None):
        """
        Archive all data to an h5 handle or filename.
        
        If no file is given, a file based on the fingerprint will be created.
        
        """
        if not h5:
            h5 = 'distgen_'+self.fingerprint()+'.h5'
            
        if isinstance(h5, str):
            g = h5py.File(h5, 'w')    
            # Proper openPMD init
            pmd_init(g, basePath='/', particlesPath='particles/')
            g.attrs['software'] = np.string_('distgen') # makes a fixed string
            #TODO: add version: g.attrs('version') = np.string_(__version__) 

        else:
            g = h5
        
        # Init
        archive.distgen_init(g)
        
        # Input
        archive.write_input_h5(g, self.input, name='input')
        
        # Particles
        if self.particles:
            self.particles.write(g, name='particles')
        

        return h5    
    
    def __repr__(self):
        s = '<disgten.Generator with input: \n'
        return s+yaml.dump(self.input)+'\n>'

    def check_inputs(self, params):

        """ Checks the params sent to the generator only contain allowed inputs """

        # Make sure user isn't passing the wrong parameters:
        allowed_params = self.optional_params + self.required_params + ['verbose']
        for param in params:
            assert param in allowed_params, 'Incorrect param given to '+self.__class__.__name__+ '.__init__(**kwargs): '+param+'\nAllowed params: '+str(allowed_params)

        # Make sure all required parameters are specified
        for req in self.required_params:
            assert req in params, 'Required input parameter '+req+' to '+self.__class__.__name__+'.__init__(**kwargs) was not found.'







