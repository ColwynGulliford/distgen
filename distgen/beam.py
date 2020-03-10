import numpy as np
from .physical_constants import *
import functools

from .tools import vprint, mean, std

"""
This class defines the container for an initial particle distribution 
"""

class Beam():

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


# Functions for converting between x,y <-> r,theta
def xy_to_r(x,y):
    return np.sqrt( x**2 + y**2 )     
            
def xy_to_theta(x,y):
    return np.arctan2(y,x)

def rtheta_to_x(r,theta):
    return r*np.cos(theta)

def rtheta_to_y(r,theta):
    return r*np.sin(theta)  

# Functions for converting px,py <-> pr,ptheta
def xypxpy_to_pr(x,y,px,py):
    theta = xy_to_theta(x,y)
    return px*np.cos(theta) + py*np.sin(theta)

def xypxpy_to_ptheta(x,y,px,py):
    theta = xy_to_theta(x,y)
    return -px*np.sin(theta) + py*np.cos(theta)

def xyprptheta_to_px(x,y,pr,ptheta):
    theta = xy_to_theta(x,y)
    return pr*np.cos(theta) - ptheta*np.sin(theta)

def xyprptheta_to_py(x,y,pr,ptheta):
    theta = xy_to_theta(x,y)
    return pr*np.sin(theta) + ptheta*np.cos(theta)

     
            
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







