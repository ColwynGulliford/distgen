import numpy as np
from .physical_constants import unit_registry, pi, MC2
import functools

from .tools import vprint, mean, std

"""
This class defines the container for an initial particle distribution 
"""

class Beam():

    """
    The fundamental bunch data is stored in __dict__ with keys
        pint quantity np.array: x, px, y, py, z, pz, t, weight, 
        np.array status, 
        str: species
    where:
        x, y, z have a base unit of meters 
        px, py, pz are momenta in base units [eV/c]
        t is time in [s]
        weight is the macro-charge weight in [C], used for all statistical calulations.
        species is a proper species name: 'electron', etc. 
    """

    def __init__(self, **kwargs):

        self.required_inputs = ['total_charge', 'n_particle']
        self.optional_inputs = ['species']

        self.check_inputs(kwargs)

        self._q = kwargs['total_charge']
        self._n_particle = kwargs['n_particle']
        self._species ='electron'   # TODO

        self._settable_array_keys = ['x', 'px', 'y', 'py', 'z', 'pz', 't', 'w', 'theta', 'pr', 'ptheta', 'xp', 'yp', 'thetax', 'thetay'] 

    def check_inputs(self,inputs):

        allowed_params = self.optional_inputs + self.required_inputs + ['verbose']
        for input_param in inputs:
            assert input_param in allowed_params, f'Incorrect param given to {self.__class__.__name__}.__init__(**kwargs): {input_param}\nAllowed params: {allowed_params}'

        # Make sure all required parameters are specified
        for req in self.required_inputs:
            assert req in inputs, f'Required input parameter {req} to {self.__class__.__name__}.__init__(**kwargs) was not found.'

    def __getitem__(self, key):
        return getattr(self, key) 

    @property
    def n_particle(self):
        return self._n_particle

    @property
    def q(self):
        return self._q

    @property
    def species(self):
        return self._species

    # Cylindrical coordinates
    @property
    def r(self):
        return np.sqrt( self.x**2 + self.y**2 )

    @r.setter
    def r(self, r):
        self.x = r*np.cos(self.theta)
        self.y = r*np.sin(self.theta)

    @property
    def theta(self):
        return np.arctan2(self.y, self.x)

    @theta.setter
    def theta(self,theta):
        self.x = r*np.cos(self.theta)
        self.y = r*np.sin(self.theta) 

    @property
    def pr(self):
        return self.px*np.cos(self.theta) + self.py*np.sin(self.theta)

    @pr.setter
    def pr(self,pr):
        self.px=pr*np.cos(self.theta)-self.ptheta*np.sin(self.theta)
        self.py=pr*np.sin(self.theta)+self.ptheta*np.cos(self.theta)

    @property
    def ptheta(self):
        return -self.px*np.sin(self.theta) + self.py*np.cos(self.theta)

    @ptheta.setter
    def ptheta(self,ptheta):
        self.px=self.pr*np.cos(self.theta)-ptheta*np.sin(self.theta)
        self.py=self.pr*np.sin(self.theta)+ptheta*np.cos(self.theta)


    # Transverse Derivatives and Angles
    @property
    def xp(self):
        return self.px/( self.pz.to(str(self.px.units)) )

    @xp.setter
    def xp(self, xp):
        self.px = xp*self.pz

    @property
    def thetax(self):
        return np.arctan2( self.px, self.pz.to(str(self.px.units)) )

    @thetax.setter
    def thetax(self, thetax):
        self.px = np.tan(thetax)*self.pz

    @property
    def yp(self):
        return self.py/( self.pz.to(str(self.py.units)) )

    @yp.setter
    def yp(self, yp):
        self.py = yp*self.pz

    @property
    def thetay(self):
        return np.arctan2( self.py, self.pz.to(str(self.py.units)) )

    @thetay.setter
    def thetay(self, thetay):
        self.py = np.tan(thetay)*self.pz


    # Relativistic quantities:
    @property
    def p(self):
        """Total momemtum"""
        return np.sqrt(self.px**2 + self.py**2 + self.pz**2)

    @property
    def beta_x(self):
        """vx/c"""
        return self.px.to('GB')/self.gamma

    @property
    def beta_y(self):
        """vy/c"""
        return self.py.to('GB')/self.gamma

    @property
    def beta_z(self):
        """vz/c"""
        return self.pz.to('GB')/self.gamma

    @property
    def gamma(self):
        return np.sqrt(1+self.p.to('GB')**2)

    @property
    def kinetic_energy(self):
        return MC2*(self.gamma-1)

    @property
    def energy(self):
        return MC2*self.gamma

    # Statistical quantities
    def avg(self,var,desired_units=None):

        avgv = mean(getattr(self, var), getattr(self, 'w'))
        if(desired_units):
            avgv.ito(desired_units)

        return avgv
  
    def std(self,var,desired_units=None):
        
        stdv = std(getattr(self, var), getattr(self, 'w'))
        if(desired_units):
            stdv.ito(desired_units)

        return stdv

    def delta(self, key):
        """Attribute (array) relative to its mean"""
        return getattr(self, key) - self.avg(key)

    # Twiss parameters
    def Beta(self, var):

        varx = self.std(var)**2
        eps = self.emitt(var, 'geometric')

        return varx/eps
    
    def Alpha(self, var):

        x = getattr(self, var)
        x0 = mean(x, getattr(self, 'w'))

        p = getattr(self, f'{var}p')
        p0 = mean(p, getattr(self, 'w'))

        xp = mean( (x-x0)*(p-p0), getattr(self, 'w'))
        eps = self.emitt(var,'geometric')

        return -xp/eps

    def Gamma(self, var):

        varp = std(getattr(self, f'{var}p'), self.params['w'])**2
        eps = self.emitt(var, 'geometric')

        return varp/eps        

    def emitt(self, var, units='normalized'):

        x = getattr(self, var)

        if(units == 'normalized'):
            p = getattr(self, f'p{var}').to('GB')

        elif(units == 'geometric'): 
            p = getattr(self, f'{var}p')

        x0 = mean(x, getattr(self, 'w'))
        p0 = mean(p, getattr(self, 'w'))

        x2 = std(x, getattr(self, 'w'))**2
        p2 = std(p, getattr(self, 'w'))**2
        xp = mean( (x-x0)*(p-p0), getattr(self, 'w') )

        return np.sqrt( x2*p2 - xp**2 )
   
    def twiss(self,var):
        return (self.Beta(var), self.Alpha(var), self.emitt(var,'geometric'))
    

    # Set functiontality
    def __setitem__(self, key, value):
        if(key in self._settable_array_keys):
            setattr(self, key, value)
        else:
            raise ValueError(f'Beam: quantity {key} is not settable.')
 
    def print_stats(self):
        """
        Prints averages and standard deviations of the beam variables.
        """  

        stat_vars={'x':'mm',  'y':'mm', 'z':'mm', 'px':'eV/c', 'py':'eV/c', 'pz':'eV/c', 't':'ps'}

        vprint("\nBeam stats:",True,0,True)
        for x, unit in stat_vars.items():
            vprint(f'avg_{x} = {self.avg(x).to(unit):G~P}, sigma_{x} = {self.std(x).to(unit):G~P}', True, 1, True)
            

    def data(self):
        """
        Converts to fixed units and returns a dict of data.
        
        See function Sbeam_data
        """
        return beam_data(self)
   
'''
class Beam_old():

    """
    The fundamental bunch data is stored in __dict__ with keys
        pint quantity np.array: x, px, y, py, z, pz, t, weight, 
        np.array status, 
        str: species
    where:
        x, y, z have a base unit of meters 
        px, py, pz are momenta in base units [eV/c]
        t is time in [s]
        weight is the macro-charge weight in [C], used for all statistical calulations.
        species is a proper species name: 'electron', etc. 
    """

    def __init__(self,  **kwargs):
        """
        Initializes a beam class object with n particles and bunch charge q
        """
        
        self.required_inputs = ['total_charge']
        self.optional_inputs = ['species']

        self.check_inputs(kwargs)

        self.q = kwargs['total_charge']
        self.species = 'electron'   # <- Hard coded for now...

        self.params = {}

        #self.__dict__ ={}

    def check_inputs(self,inputs):

        allowed_params = self.optional_inputs + self.required_inputs + ['verbose']
        for input_param in inputs:
            assert input_param in allowed_params, 'Incorrect param given to '+self.__class__.__name__+ '.__init__(**kwargs): '+input_param+'\nAllowed params: '+str(allowed_params)

        # Make sure all required parameters are specified
        for req in self.required_inputs:
            assert req in inputs, 'Required input parameter '+req+' to '+self.__class__.__name__+'.__init__(**kwargs) was not found.'

    def __getitem__(self, var):

        """
        Allows direct get access via brackets to the params dictionary with key var.
        """
        if(var in self.params):
            return self.params[var]

        elif(var in ['r','theta','pr','ptheta','xp','yp','thetax','thetay','n']):
            return getattr(self,'get_'+var)()

        else:
            return None
    
    def __setitem__(self, var, item):

        """
        Allows direct set access via brackets to the params dictionary with key var and value item.
        """
        if(var in self.params.keys()):
            self.params[var]=item
        elif(var in ['r','theta','pr','ptheta','xp','yp','thetax','thetay']):  # Special variables that are functions of the basic coordinates
            getattr(self,'set_'+var)(item)

        else:
            raise ValueError("Beam object has no parameter "+var)

    def avg(self,var,desired_units=None):

        if(desired_units is None):
            return mean(self.params[var],self.params["w"])
        else:
            avgvar = mean(self.params[var],self.params["w"])
            return avgvar.to(desired_units)
  
    def std(self,var,desired_units=None):
        
        if(desired_units is None):
            return std(self.params[var],self.params["w"])   
        else:
            stdvar = std(self.params[var],self.params["w"]) 
            return stdvar.to(desired_units)

    def beta(self,var):

        varx = self.std(var)**2
        eps = self.emitt(var,'geometric')

        return varx/eps
    
    def alpha(self,var):

        pvar = var+'p'
        x = self.params[var]
        x0 = mean(x,self.params['w'])

        p = self.__getitem__(pvar)
        p0 = mean(p,self.params['w'])

        xp = mean( (x-x0)*(p-p0), self.params['w'])
        eps = self.emitt(var,'geometric')

        return -xp/eps

    def gamma(self,var):

        p = self.__getitem__('xp')
        varp = std(p,self.params['w'])
        eps = self.emitt(var,'geometric')

        return varp/eps        

    def emitt(self,var,units='normalized'):

        if(units == 'normalized'):

            x = self.params[var]
            p = self.params['p'+var].to('GB')

        elif(units == 'geometric'): 

            x = self.params[var]    
            p = self.__getitem__(var+'p')

        x0 = mean(x,self.params['w'])
        p0 = mean(p,self.params['w'])

        x2 = std(x,self.params['w'])**2
        p2 = std(p,self.params['w'])**2
        xp = mean( (x-x0)*(p-p0), self.params['w'] )

        return np.sqrt( x2*p2 - xp**2 )
   
    def twiss(self,var):
        return (self.beta(var), self.alpha(var), self.emitt(var,'geometric'))

    def data(self):
        """
        Converts to fixed units and returns a dict of data.
        
        See function Sbeam_data
        """
        return beam_data(self)
    
    def print_stats(self):
        """
        Prints averages and standard deviations of the beam variables.
        """  

        stat_exceptions=["w"]

        vprint("\nBeam stats:",True,0,True)
        for x in self.params:
            if(x not in stat_exceptions):
                vprint("avg_"+x+ " = {:0.3f~P}".format(mean(self.params[x], self.params["w"])) +", sigma_"+x+" = {:0.3f~P}".format(std(self.params[x],weights=self.params["w"])), True, 1, True)
            
    def get_n(self):

        ns = [len(self.params[p]) for p in self.params]
        assert len(np.unique(ns))==1, 'Length of coordinate vectors in ' + self.__class__.__name__ + ' are not the same.  Error in creating beam has occured'
        return ns[0]
        
    def get_q(self):
        return beam.q

    def get_q(self):
        return self.params['w']*self.q

    def get_r(self):
        return np.sqrt( self.params['x']**2 + self.params['y']**2 )

    def set_r(self,r):
        theta = self.get_theta()
        self.params['x']=r*np.cos(theta)
        self.params['y']=r*np.sin(theta)

    def get_theta(self):
        return np.arctan2(self.params['y'],self.params['x'])

    def set_theta(self,theta):
        r = self.get_r()
        self.params['x']=r*np.cos(theta)
        self.params['y']=r*np.sin(theta)

    def get_pr(self):
        theta = self.get_theta()
        return self.params['px']*np.cos(theta) + self.params['py']*np.sin(theta)

    def set_pr(self,pr):
        
        theta = self.get_theta()
        ptheta = self.get_ptheta()

        self.params['px']=pr*np.cos(theta)-ptheta*np.sin(theta)
        self.params['py']=pr*np.sin(theta)+ptheta*np.cos(theta)

    def get_ptheta(self):
        theta = self.get_theta()
        return -self.params['px']*np.sin(theta) + self.params['py']*np.cos(theta)

    def set_ptheta(self,ptheta):

        theta = self.get_theta()
        pr = self.get_pr()

        self.params['px']=pr*np.cos(theta)-ptheta*np.sin(theta)
        self.params['py']=pr*np.sin(theta)+ptheta*np.cos(theta)

    # Transverse Derivatives and Angles

    def get_dvdz(self,vstr):
        pstr='p'+vstr
        return self.params[pstr]/( self.params['pz'].to(str(self.params[pstr].units)) )

    def get_xp(self):
        return self.get_dvdz('x')
  
    def get_yp(self):
        return self.get_dvdz('y')

    def set_dvdz(self,vstr,dvdz):
        pstr='p'+vstr
        self.params[pstr] = dvdz * (self.params['pz'].to(str(self.params[pstr].units)) )

    def set_xp(self,xp):
        self.set_dvdz('x',xp)

    def set_yp(self,yp):
        self.set_dvdz('y',yp)

    def get_thetax(self):
        return np.arctan2( self.params['px'],  self.params['pz'].to(str(self.params['px'].units)))

    def set_thetax(self,thetax):
        self.params['px'] =  self.params['pz']*np.tan(thetax)

    def get_thetay(self):
        return np.arctan2( self.params['pz'].to(str(self.params['py'].units)) , self.params['py'])

    def set_thetay(self,thetay):
        self.params['py'] =  self.params['pz']*np.tan(thetay)

    def get_E(self):
        pass

'''
     
'''            
def beam_data_old(beam):
    """
    Converts all units to standard units and strips them as a data dict with:
        str: species
        int: n_particle
        np.array: x, px, y, py, z, pz, t, status, weight
    where:
        x, y, z are positions in units of [m]
        px, py, pz are momenta in units of [eV/c]
        t is time in [s]
        status = 1 
        weight is the macro-charge weight in [C]
        

    """
    #assert species == 'electron' # TODO: add more species
    
    # number of lines in file
    n_particle = beam['n']
    total_charge = (beam.q.to('C')).magnitude
    species = beam.species
    
    # weight
    weight = (beam['w'].magnitude) * total_charge # Weight should be macrocharge in C
    
    # Status
    status = np.full(n_particle, 1) # Status == 1 means live
    
    # standard units and types
    names = ['x', 'y', 'z', 'px',   'py',   'pz',   't']
    units = ['m', 'm', 'm', 'eV/c', 'eV/c', 'eV/c', 's']
    
    data = {'n_particle':n_particle,
            'species':species,
            'weight':weight,
            'status':status
    }
    
    for name, unit in zip(names, units):
        data[name] = (beam[name].to(unit)).magnitude
    
    return data            
'''

def beam_data(beam):
    """
    Converts all units to standard units and strips them as a data dict with:
        str: species
        int: n_particle
        np.array: x, px, y, py, z, pz, t, status, weight
    where:
        x, y, z are positions in units of [m]
        px, py, pz are momenta in units of [eV/c]
        t is time in [s]
        status = 1 
        weight is the macro-charge weight in [C]
        

    """
    #assert species == 'electron' # TODO: add more species
    
    # number of lines in file
    n_particle = beam['n_particle']
    total_charge = (beam.q.to('C')).magnitude
    species = beam.species
    
    # weight
    weight = (beam['w'].magnitude) * total_charge # Weight should be macrocharge in C
    
    # Status
    status = np.full(n_particle, 1) # Status == 1 means live
    
    # standard units and types
    names = ['x', 'y', 'z', 'px',   'py',   'pz',   't']
    units = ['m', 'm', 'm', 'eV/c', 'eV/c', 'eV/c', 's']
    
    data = {'n_particle':n_particle,
            'species':species,
            'weight':weight,
            'status':status
    }
    
    for name, unit in zip(names, units):
        data[name] = (beam[name].to(unit)).magnitude
    
    return data            





