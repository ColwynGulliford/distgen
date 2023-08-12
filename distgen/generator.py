




#from .dist import get_dist
#from .dist import random_generator

#from .tools import convert_params
#from .tools import full_path

#from .tools import StopWatch
#from .tools import vprint
#from .tools import update_quantity


#from .transforms import set_avg
#from .transforms import set_avg_and_std
#from .transforms import transform


#from .physical_constants import pi
#from .physical_constants import MC2
#from .physical_constants import unit_registry



import warnings

import numpy as np
import h5py
import yaml
import copy
import os

from lume.base import Base
from lume.tools import full_path


from . import archive

from .beam import Beam

from concurrent.futures import ProcessPoolExecutor

from .dist import get_dist
from .dist import random_generator

#from .parallelization import set_up_generators

from .parsing import convert_input_quantities
from .parsing import convert_quantities_to_user_input
from .parsing import expand_input_filepaths
from .parsing import update_quantity

from .physical_constants import c
from .physical_constants import is_quantity
from .physical_constants import MC2
from .physical_constants import pi
from .physical_constants import unit_registry

from pmd_beamphysics import ParticleGroup, ParticleStatus, pmd_init
from pmd_beamphysics.particles import join_particle_groups

from .tools import get_nested_dict
from .tools import is_key_in_nested_dict
from .tools import set_nested_dict
from .tools import StopWatch
from .tools import update_nested_dict
from .tools import vprint


from .transforms import set_avg
from .transforms import set_avg_and_std
from .transforms import transform

from pprint import pprint

class Generator(Base):

    """
    This class defines the main run engine object for distgen and is responsible for
    1. Parsing the input data dictionary passed from a Reader object
    2. Check the input for internal consistency
    3. Collect the parameters for distributions requested in the params dictionary
    4. Form a the Beam object and populated the particle phase space coordinates
    """

    def __init__(self, *args, **kwargs):
        """
        The class initialization takes in a verbose level for controlling text output to the user
        """
        super().__init__(*args, **kwargs)
        
        # This will be set by 
        self._input = None   # The parsed and most up-to-date configuration of input for the generator

        # This will be set with .beam()
        self.rands = None

        # This will be set with .run()
        self.particles = None

        if self.input_file:
            self.parse_input(self.input_file)
            self.configure()

    def parse_input(self, input):
        """
        Parse the input structure passed from a Reader object.
        The structure is then converted to an easier form for use in populating the Beam object.

        YAML or JSON is accepted if params is a filename (str)

        Relative paths for input 'file' keys will be expanded.
        """
        if isinstance(input, str):
            if os.path.exists(full_path(input)):
                # File
                filename = full_path(input)
                with open(filename) as fid:
                    input = yaml.safe_load(fid)
                # Fill any 'file' keys
                expand_input_filepaths(input, root=os.path.split(filename)[0], ignore_keys=['output'])


            else:
                #Try raw string
                input = yaml.safe_load(input)
                assert isinstance(input, dict), f'ERROR: parsing unsuccessful, could not read {input}'
                expand_input_filepaths(input)
                
        else: expand_input_filepaths(input)

        input = convert_input_quantities(input)
        self.check_input_consistency(input)
        self._input = input

    @property
    def input(self):        
        # User should see the generator input structure in user input notation  
        return convert_quantities_to_user_input(copy.deepcopy(self._input)) 
    
    #@input.setter
    #def input(self, input):   
    #    # When setting the input dictionary, convert user input notation to internal format    
    #    self._input = convert_input_quantities(input)
    #    self.check_input_consistency()
    
    def __repr__(self):
        s = '<disgten.Generator with input: \n'
        return s+yaml.dump(self.input)+'\n>'
        
    def __getitem__(self, varstr):
        
        if(varstr.endswith(':value')):
            
            pstr = varstr.replace(':value', '')
            var = get_nested_dict(self._input, pstr, sep=':', prefix='distgen')
            
            if(is_quantity(var)):
                return var.magnitude
            else:
                return var
            
        elif(varstr.endswith(':units')):
            
            pstr = varstr.replace(':units', '')
            var = get_nested_dict(self._input, pstr, sep=':', prefix='distgen')
            
            if(is_quantity(var)):
                return str(var.units)
            else:
                return var 
        else:
            var = get_nested_dict(self._input, varstr, sep=':', prefix='distgen')
            
            if(is_quantity(var)):
                return {'value':var.magnitude, 'units':str(var.units)}
            
            elif(isinstance(var, dict)):
                return convert_quantities_to_user_input(var, in_place=False)
            
            else:
                return var

    def __setitem__(self, varstr, val):
        
        params = copy.deepcopy(self._input)
        
        if(isinstance(val, dict)):
            val = convert_input_quantities(val, in_place=False)
            
        pstr = varstr.replace(':value', '').replace(':units', '')
            
        if(not is_key_in_nested_dict(params, pstr, sep=':', prefix='distgen')):
            params = update_nested_dict(params, {varstr:val}, verbose=False, create_new=True)
        
        if(varstr.endswith(':value') or varstr.endswith(':units')):

            var = get_nested_dict(params, pstr, sep=':', prefix='distgen')
            set_nested_dict(params, pstr, update_quantity(var, val), sep=':', prefix='distgen')
            
        else:
            var = get_nested_dict(params, varstr, sep=':', prefix='distgen')

            if(is_quantity(var)):
                set_nested_dict(params, varstr, update_quantity(var, val), sep=':', prefix='distgen')
        
            else:
                set_nested_dict(params, varstr, val, sep=':', prefix='distgen')
                
        self.check_input_consistency(params)  # Raises if something is wrong
        self._input = params                  # Accept changes into the Generator input state
    
    def check_input_consistency(self, params):

        """
        Perform consistency/sanity checks on new user input data
        """

        # Make sure all required top level params are present
        required_params = ['n_particle', 'total_charge']
        for rp in required_params:
            assert rp in params, 'Required generator parameter ' + rp + ' not found.'

        # Check that only allowed params present at top level
        allowed_params = required_params + ['output', 'transforms', 'start', 'random_seed', 'random_type', 'random']
        for p in params:
            #assert p in allowed_params or '_dist'==p[-5:], 'Unexpected distgen input parameter: ' + p[-5:]
            assert p in allowed_params or p.endswith('_dist'), 'Unexpected distgen input parameter: ' + p

        assert params['n_particle']>0, 'User must speficy n_particle must > 0.'

        if(isinstance(params['n_particle'], float)):
            params['n_particle']=int(params['n_particle'])
            warnings.warn('Input variable n_particle was a float, expected int.')
        
        assert isinstance(params['n_particle'], int), f'Invalid type for n_particle parameter: {type(params["n_particle"])}'
        
        # Check consistency of transverse coordinate definitions
        if( ("r_dist" in params) and ("x_dist" in params or "xy_dist" in params) ):
            raise ValueError('Multiple/Inconsistent transverse distribution specification.')
        if( ("r_dist" in params) and ("y_dist" in params or "xy_dist" in params) ):
            raise ValueError('Multiple/Inconsistent transverse distribution specification.')

        if('output' in params):
            out_params = params["output"]
            for op in out_params:
                assert op in ['file','type'], f'Unexpected output parameter specified: {op}'
        else:
            params['output'] = {"type":None}
            
        if('transforms' not in params):
            params['transforms']=None
            
        if(params['start']['type'] == "cathode"):

            for d in ['z_dist', 'px_dist', 'py_dist', 'pz_dist']:
                if(d in params):
                    vprint(f"Ignoring user specified {d} distribution for cathode start.", self.verbose>0, 0, True)
                    params.pop(d)

            #assert "MTE" in params['start'], "User must specify the MTE for cathode start."
            
            if('p_dist' in params or 'KE_dist' in params):
                pass

            else: 
                assert "MTE" in params['start'], "User must specify the MTE for cathode start if momentum/energy distribution not specified."

        elif(params['start']['type']=='time'):

            vprint("Ignoring user specified t distribution for time start.", self.verbose>0 and "t_dist" in params, 0, True)
            if('t_dist' in params):
                warnings.warn('Ignoring user specified t distribution for time start.')
                params.pop('t_dist')   

    def configure(self):
        pass
    
    
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

        #self.input = archive.read_input_h5(g['input'])

        if 'particles' in g:
            self.particles = ParticleGroup(g['particles'])
            self.output = self.particles
        else:
            vprint('No particles found.', self.verbose>0,1,False)

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
        #archive.write_input_h5(g, self.input, name='input')

        # Particles
        if self.particles:
            self.particles.write(g, name='particles')
        return h5
    
    def get_dist_params(self, params):

        """ Loops through the input params dict and collects all distribution definitions """

        #params = self._input
        
        dist_vars = [p.replace('_dist','') for p in params if(p.endswith('_dist')) ]
        dist_params = {p.replace('_dist',''):params[p] for p in params if(p.endswith('_dist'))}
        
        #pprint(dist_params)
        
        if('r' in dist_vars and 'theta' not in dist_vars):
            vprint("Assuming cylindrical symmetry...",self.verbose>0,1,True)
            dist_params['theta']={'type':'ut','min_theta':0*unit_registry('rad'),'max_theta':2*pi}
            
        if("p" in dist_params or "KE" in dist_params):
        
            dist_params['azimuthal_angle']={'type':'ut','min_theta':0*unit_registry('rad'),'max_theta':2*pi}
            dist_params['polar_angle']={'type':'up','min_phi':0*unit_registry('rad'),'max_phi':pi}

        if(params['start']['type']=='time' and 't_dist' in params):
            raise ValueError('Error: t_dist should not be set for time start')

        return dist_params

    def get_rands(self, variables):

        """ Gets random numbers [0,1] for the coordinatess in variables
        using either the Hammersley sequence or rand """
        
        params = self._input

        specials = ['xy']
        self.rands = {var:None for var in variables if var not in specials}

        if('xy' in variables):
            self.rands['x']=None
            self.rands['y']=None

        elif('r' in variables and 'theta' not in variables):
            self.rands['theta']=None

        n_coordinate = len(self.rands.keys())
        n_particle = int(params['n_particle'])
        shape = ( n_coordinate, n_particle )

        random = {}

        if(n_coordinate>0):
            
            if('random' in params):
                random = params['random']
                
            elif('random_type' in params):
                random['type'] = params['random_type']
                
                if('random_seed' in params):
                    random['seed']=params['random_seed']
            
            rns = random_generator(shape, random['type'], **random)

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
        
        params = copy.deepcopy(self._input)
        
        if(params['start']['type'] == "cathode"):

            if('p_dist' not in params and 'KE_dist' not in params):
                
                new_method=False
                
                # Handle momentum distribution for cathode
                if('MTE' in params['start']):
                    MTE = params['start']["MTE"]
                    sigma_p = (np.sqrt( (MTE/MC2).to_reduced_units() )*unit_registry("GB")).to("eV/c")
                
                if(new_method): 
                    params['p_dist'] = {"type":"mb", 'scale_p': sigma_p}
                else: 
                    params["px_dist"]={"type":"g", "sigma_px":sigma_p}
                    params["py_dist"]={"type":"g", "sigma_py":sigma_p}
                    params["pz_dist"]={"type":"g", "sigma_pz":sigma_p}
                    
            
                

            #assert "MTE" in params['start'], "User must specify the MTE for cathode start."

            # Handle momentum distribution for cathode
            #MTE = params['start']["MTE"]
            #sigma_pxyz = (np.sqrt( (MTE/MC2).to_reduced_units() )*unit_registry("GB")).to("eV/c")

            #params["px_dist"]={"type":"g","sigma_px":sigma_pxyz}
            #params["py_dist"]={"type":"g","sigma_py":sigma_pxyz}
            #params["pz_dist"]={"type":"g","sigma_pz":sigma_pxyz}

        elif(params['start']['type']=='time'):

            vprint("Ignoring user specified t distribution for time start.", self.verbose>0 and "t_dist" in params, 0, True)
            if('t_dist' in params):
                warnings.warn('Ignoring user specified t distribution for time start.')
                params.pop('t_dist')    

                
        verbose = self.verbose
        #outputfile = []

        beam_params = {'total_charge':params['total_charge'], 'n_particle':params['n_particle']}

        if('transforms' in params):
            transforms = params['transforms']
        else:
            transforms = None

        #dist_params = {p.replace('_dist',''):self.params[p] for p in self.params if(p.endswith('_dist')) }
        #self.get_rands()

        vprint(f'Distribution format: {params["output"]["type"]}', verbose>0, 0, True)

        N = int(params['n_particle'])
        bdist = Beam(**beam_params)

        if("file" in params['output']):
            outfile = params['output']["file"]
        else:
            outfile = "None"
            vprint(f'Warning: no output file specified, defaulting to "{outfile}".', verbose>0, 1, True)
        vprint(f'Output file: {outfile}', verbose>0, 0, True)

        vprint('\nCreating beam distribution....',verbose>0,0,True)
        vprint(f"Beam starting from: {params['start']['type']}",verbose>0,1,True)
        vprint(f'Total charge: {bdist.q:G~P}.',verbose>0,1,True)
        vprint(f'Number of macroparticles: {N}.',verbose>0,1,True)

        units = {'x':'m', 'y':'m', 'z':'m', 'px':'eV/c', 'py':'eV/c', 'pz':'eV/c', 't':'s'}

        # Initialize coordinates to zero
        for var, unit in units.items():
            bdist[var] = np.full(N, 0.0)*unit_registry(units[var])

        bdist["w"] = np.full((N,), 1/N)*unit_registry("dimensionless")

        avgs = {var:0*unit_registry(units[var]) for var in units}
        stds = {var:0*unit_registry(units[var]) for var in units}

        dist_params = self.get_dist_params(params)   # Get the relevant dist params, setting defaults as needed, and samples random number generator
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

            # These should be filled by the theta dist?
            avgCos = 0
            avgSin = 0
            avgCos2 = 0.5
            avgSin2 = 0.5

            bdist['x']=r*np.cos(theta)
            bdist['y']=r*np.sin(theta)

            avgs['x'] = avgr*avgCos
            avgs['y'] = avgr*avgSin

            stds['x'] = rrms*np.sqrt(avgCos2)   # Should check that average needs subtracting?
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

            avgs['x']=bdist.avg('x')
            avgs['y']=bdist.avg('y')

            stds['x']=bdist.std('x')
            stds['y']=bdist.std('y')
            
        if("p" in dist_params or 'KE' in dist_params):# or "E_dist" in dist_params or "KE_dist"):
            
            if("p" in dist_params):
                vprint('p distribution: ', verbose>0, 1, False)

                # Get p distribution
                pdist = get_dist('p', dist_params['p'], verbose=verbose)

                if(pdist.rms()>0):
                    p = pdist.cdfinv(self.rands['p'])       # Sample to get beam coordinates
                    
                avgp = pdist.avg()
                prms = pdist.rms()
                    
            elif('KE' in dist_params):
                
                vprint('KE distribution: ', verbose>0, 1, False)

                # Get p distribution
                KEdist = get_dist('KE', dist_params['KE'], verbose=verbose)
                
                if(KEdist.rms()>0):
                    KE = KEdist.cdfinv(self.rands['KE'])       # Sample to get beam coordinates       
                    p = np.sqrt( (KE+MC2)**2 - MC2**2 )/c
                    
                    avgp = np.mean(p)
                    stdp = np.std(p)
                    prms = np.sqrt(stdp**2 + avgp**2)
                
            # Sample to get beam coordinates
            vprint('azimuthal angle distribution: ', verbose>0, 1, False)
            theta_dist = get_dist('azimuthal_angle', dist_params['azimuthal_angle'], verbose=verbose)
            theta = theta_dist.cdfinv(self.rands['azimuthal_angle'])
            
            vprint('polar angle distribution: ', verbose>0, 1, False)
            phi_dist = get_dist('polar_angle', dist_params['polar_angle'], verbose=verbose)
            phi = phi_dist.cdfinv(self.rands['polar_angle'])
            
            bdist['px']=p*np.cos(theta)*np.sin(phi)
            bdist['py']=p*np.sin(theta)*np.sin(phi)
            bdist['pz']=p*np.cos(phi)

            avgCosTheta = theta_dist.avgCos()
            avgSinTheta = theta_dist.avgSin()
            
            avgCos2Theta = theta_dist.avgCos2()
            avgSin2Theta = theta_dist.avgCos2()
            
            avgCosPhi = phi_dist.avgCos()
            avgSinPhi = phi_dist.avgSin()
            
            avgCos2Phi = phi_dist.avgCos2()
            avgSin2Phi = phi_dist.avgSin2()
            
            avgs['px'] = avgp * avgCosTheta * avgSinPhi
            avgs['py'] = avgp * avgSinTheta * avgSinPhi
            avgs['pz'] = avgp * avgCosPhi
            
            pxrms2 = prms**2 * avgCos2Theta * avgSin2Phi  
            pyrms2 = prms**2 * avgSin2Theta * avgSin2Phi 
            pzrms2 = prms**2 * avgCos2Phi 
            
            stds['px'] = np.sqrt( pxrms2 - avgs['px']**2 )
            stds['py'] = np.sqrt( pxrms2 - avgs['py']**2 )
            stds['pz'] = np.sqrt( pxrms2 - avgs['pz']**2 )
            
            if('p' in dist_params):
                dist_params.pop('p')
            elif('KE' in dist_params):
                dist_params.pop('KE')
                
            dist_params.pop('azimuthal_angle')
            dist_params.pop('polar_angle')
       
        #pprint(dist_params)
    
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
        if(params['start']['type']=="cathode"):

            bdist['pz']=np.abs(bdist['pz'])   # Only take forward hemisphere
            vprint('Cathode start: fixing pz momenta to forward hemisphere',verbose>0,1,True)
            vprint(f'avg_pz -> {bdist.avg("pz"):G~P}, sigma_pz -> {bdist.std("pz"):G~P}',verbose>0,2,True)

        elif(params['start']['type']=='time'):

            if('tstart' in params['start']):
                tstart = params['start']['tstart']

            else:
                vprint("Time start: no start time specified, defaulting to 0 sec.",verbose>0,1,True)
                tstart = 0*unit_registry('sec')

            vprint(f'Time start: fixing all particle time values to start time: {tstart:G~P}.', verbose>0, 1, True);
            bdist = set_avg(bdist,**{'variables':'t','avg_t':0.0*unit_registry('sec'), 'verbose':verbose>0})

        elif(params['start']['type']=='free'):
            pass

        else:
            raise ValueError(f'Beam start type "{params["start"]["type"]}" is not supported!')

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
        vprint(f'...done. Time Elapsed: {watch.print()}.\n',verbose>0,0,True)
        return bdist

    def run(self, max_workers=1, executor=None):
        """ Runs the generator.beam function stores the partice in
        an openPMD-beamphysics ParticleGroup in self.particles """

        if(max_workers==1):
            # Default run, no parallelization
            if self._input is not None:
                beam = self.beam()
            
                if self._input['start']['type'] == "cathode":
                    status = ParticleStatus.CATHODE
                else:
                    status = ParticleStatus.ALIVE
                    
                self.particles = ParticleGroup(data=beam.data(status=status))
                self.output = self.particles
                vprint(f'Created particles in .particles: \n   {self.particles}', self.verbose > 0, 1, True)
            else:
                print('No input data specified.')
    
            #return self.output

        else:

            vprint(f'Creating particles in parallel with {max_workers} workers', self.verbose > 0, 0, True)
            if(executor is None):
        
                executor = ProcessPoolExecutor()
                executor.max_workers = max_workers

            vprint(f'Setting up workers...', self.verbose > 0, 1, False)
            generators = set_up_worker_generators(self, n_gen=max_workers)
            inputs = [gen.input for gen in generators]
            vprint(f'done.', self.verbose > 0, 0, True)
            
            # Run
            vprint(f'Executing worker tasks...', self.verbose > 0, 1, False)
            with executor as p:
                ps = list(p.map(worker_func, inputs))
            vprint(f'done', self.verbose > 0, 0, True)

            vprint(f'Collecting beamlets...', self.verbose > 0, 1, False)
            
            #P = ps[0]
            #for Pi in ps[1:]: P = P + Pi
            #vprint(f'done', self.verbose > 0, 0, True)

            #data = {k:np.hstack([pg[k] for pg in ps]) for k in ps[0].data.keys() if k not in ['species']}
            #data['species'] = ps[0].species
            
            self.particles = join_particle_groups(*ps)
            self.output = self.particles
            vprint(f'Created particles in .particles: \n   {self.particles}', self.verbose > 0, 1, True)

        return self.output
    
    
def worker_func(inputs):

    G = Generator(inputs)

    return G.run()


def set_up_worker_generators(G, n_gen=1):
    
    inputs = G.input

    if('random_type' in inputs):
        inputs['random'] = {'type': inputs['random_type']}
        del inputs['random_type']

        G = Generator(inputs)
        
    generators = [G.copy() for ii in range(n_gen)]

    for ii, gen in enumerate(generators):

        n_particle_subset = int( G['n_particle'] / n_gen )
        
        gen['n_particle'] = n_particle_subset
        gen['total_charge:value'] = G['total_charge:value'] / n_gen

        if(gen['random']['type']=='hammersley'):
            gen['random:burnin'] = ii*n_particle_subset

    return generators     



        

    
   