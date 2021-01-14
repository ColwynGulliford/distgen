from pint import Quantity
from .physical_constants import unit_registry 

import time
import numpy as np
import scipy.integrate 
import scipy.special

import json
from hashlib import blake2b
import datetime
import os


# HELPER FUNCTIONS:

def full_path(path):
    """
    Helper function to expand enviromental variables and return the absolute path
    """
    return os.path.abspath(os.path.expandvars(path))


def vprint(out_str,verbose,indent_number,new_line):

    """Defines verbose printing used for output:
    Inputs: out_str = string to be printed to screen, 
            verbose = boolean to turn printing on/off, 
            indent_number = how many indentations go before output_str, 
            new_line = boolean to print on newline or not
    """

    indent="   "
    total_indent = ""

    for x in range(0,indent_number):
        total_indent = total_indent + indent

    if(verbose):
        if(new_line):
            print(total_indent+out_str,end="\n")
        else:
            print(total_indent+out_str,end="")

def is_floatable(value):

    """Check if an object can be cast to a float, return true if so, false if not"""

    try:
        float(value)
        return True
    except:
        return False

def is_unit_str(ustr):

   """Check if a string defines a unit"""
   ustr = ustr.strip()
   if(len(ustr)>=2 and ustr[0]=="[" and ustr[-1]=="]"):
       return True
   else: 
       return False

def get_unit_str(ustr):

    """Parse a string that defines a unit"""
    if(is_unit_str(ustr)):
        return ustr[1:-1]
    else:
        raise ValueError("Could not recover units string from "+ustr)

class StopWatch():

    """
    Defines an object that can be used to time sections of code
    """
    ureg = unit_registry

    def __init__(self):

       self.tstart = time.time() * self.ureg.second
       self.tstop = time.time() * self.ureg.second

    def start(self):
        """ Starts the stop watch """
        self.tstart = time.time() * self.ureg.second

    def stop(self):
        """ Stops the stop watch """
        self.tstop = time.time() * self.ureg.second

    def print(self):
        """ Output time ellapsed on stop watch """
        dt = self.tstop - self.tstart
        return f'{dt.to_compact():G~P}'


#--------------------------------------------------------------
# Statistical operations:
#--------------------------------------------------------------
def mean(x, weights=None):
    """ Wraps numpy.mean"""
    if(weights is None):
        return np.mean(x)
    else:
        return np.sum(x*weights)

def std(x, weights=None):
    """Wraps numpy.std"""
    if(weights is None):
        return np.std(x)
    else:
        return np.sqrt(np.sum( weights*(x-mean(x,weights))**2 ) )
   
 
#--------------------------------------------------------------
# Numerical integration routines
#--------------------------------------------------------------
@unit_registry.wraps('=A*B', ('=A', '=B'))
def trapz(f, x):
    """ 
    Numerically integrates f(x) using trapezoid method.
    """
    return np.trapz(f, x)


@unit_registry.wraps('=A*B', ('=B', '=A'))
def cumtrapz(f, x):
    """
    Numerically integrates f(x) using trapezoid method cummulatively
    """
    return scipy.integrate.cumtrapz(f, x, initial=0)


@unit_registry.wraps('=A*A*B', ('=B', '=A'))
def rectint(f, x):
 
    """
    Computes integral[ f(x) dx ] ~ sum[ (x(i+1) - x(i))*f(i) ]
    """
    uxstr = str(x.units)
    ufstr = str(f.units)    

    xb = np.zeros(len(x)+1)
    xb[1:-1] = (x[1:]+x[:-1])/2.0

    dxL = xb[1]-x[0]
    dxR = xb[-2]-x[-1]

    xb[0] = x[0]-dxL
    xb[-1]= x[-1]+dxR

    xb = xb*unit_registry(str(uxstr))
    return np.sum( (xb[1:]-xb[:-1])*f )


@unit_registry.wraps('=A*A*B', ('=B', '=A'))
def cumrectint(f,x):  
    """
    Computes cummulative integral[ f(x) dx ] ~ sum[ (x(i+1) - x(i))*f(i) ]
    """
    xb = np.zeros(len(x)+1)
    xb[1:-1] = centers(x)

    dxL = xb[1]-x[0];    dxR = xb[-2]-x[-1]
    xb[0] = x[0]-dxL;    xb[-1]= x[-1]+dxR

    crint = np.zeros(len(xb))
    crint[1:] = np.cumsum((xb[1:]-xb[:-1])*f)
    return crint


@unit_registry.wraps('=A*A*B', ('=B', '=A'))
def radint(f, r):
    """
    Computes the integral[r*f(r) dr] ~ sum[ 0.5( r(i+1)^2 - r(i)^2 )*f(r(i)) ]
    """
    r_bins = centers(r)
    rs = np.zeros( (len(r_bins)+2,) )
    rs[1:-1] = r_bins
    rs[0] = r[0]; rs[-1] = r[-1]

    return np.sum( 0.5*(rs[1:]**2 - rs[:-1]**2)*f )


@unit_registry.wraps(('=A*A*B','=A'), ('=B','=A'))
def radcumint(f, r):
    """
    Defines cumulative radial integration with the rdr jacobian
    Inputs: r, f(r) with units, returns int( f(r) * r dr)_0^r
    """
    r_bins = centers(r)
    rs = np.zeros( (len(r_bins)+2,) )
    rs[1:-1] = r_bins
    rs[0]  = r[0]
    rs[-1] = r[-1] 

    rcint = np.zeros(len(rs))
    rcint[1:] = np.cumsum(0.5*(rs[1:]**2-rs[:-1]**2)*f)

    return (rcint, rs)


#--------------------------------------------------------------
# Interpolation routines
#--------------------------------------------------------------
@unit_registry.wraps('=B', ('=A','=A','=B'))
def interp(x, xp, fp):
    """
    1d interpolation of [xp,f(xp)] @ x
    """
    return np.interp(x, xp, fp)


@unit_registry.wraps('=A', ('=A', '=A', None))
def linspace(x1, x2, N):
    """
    Returns array of N values on interval [x1, x2]
    """
    return np.linspace(x1, x2, N)


def centers(x):
    """
    Compute the center points in array x
    """
    return (x[1:] + x[:-1]) / 2


def nearest_neighbor(array, values):
    """
    find the nearest neighbor index in array for each value in values
    """
    array = array.magnitude
    values = values.magnitude
    return np.abs(np.subtract.outer(array, values)).argmin(0)


#--------------------------------------------------------------
# Misc Numpy
#--------------------------------------------------------------
@unit_registry.wraps('=A', ('=A','=A'))
def concatenate(x1, x2):
    return np.concatenate((x1,x2))



#--------------------------------------------------------------
# Histogramming routines
#--------------------------------------------------------------
def weights(func):

    """
    Wrapper function to default the weight variable to histogram functions
    """
    def wrapper(*args, **kwargs):
        
        if('weights' not in kwargs or kwargs['weights'] is None):
            kwargs['weights'] = np.full(args[0].shape, 1/len(args[0]) )

        return func(*args, **kwargs)

    return wrapper 


@weights
@unit_registry.wraps( ('', '=A'), ('=A', '=B', None))
def histogram(x, weights=None, nbins=100):
    """"Wraps the numpy histogram function to include units"""
    return np.histogram(x, weights=weights, bins=nbins)


@weights
@unit_registry.wraps( ('', '=A'), ('=A', '=B', None))
def radial_histogram(r, weights=None, nbins=1000):

    """ Performs histogramming of the varibale r using non-equally space bins """
    r2 = r*r
    dr2 = (max(r2)-min(r2))/(nbins-2);
    r2_edges = np.linspace(min(r2), max(r2) + 0.5*dr2, nbins);
    dr2 = r2_edges[1]-r2_edges[0]
    edges = np.sqrt(r2_edges)
    
    which_bins = np.digitize(r2, r2_edges)-1
    minlength = r2_edges.size-1
    hist = np.bincount(which_bins, weights=weights, minlength=minlength)/(np.pi*dr2)

    return (hist, edges)


#--------------------------------------------------------------
# Special Scipy functions
#--------------------------------------------------------------
@unit_registry.check('[]')
def erf(x):
    return scipy.special.erf(x.magnitude)*unit_registry('dimensionless')

@unit_registry.check('[]')
def erfinv(x):
    return scipy.special.erfinv(x.magnitude)*unit_registry('dimensionless')

@unit_registry.check('[]')
def gamma(x):
    return scipy.special.gamma(x.magnitude)*unit_registry('dimensionless')


# Misc
def zeros(shape, units):
    """ Wraps numpy.zeros for use with units """
    return np.zeros(shape)*units


def get_vars(varstr):
    """Gets 2d variable labels from a single string"""
    variables = ['x', 'y', 'z', 'px', 'py', 'pz', 't']

    all_2d_vars = {}
    for var1 in variables:
        for var2 in variables:
            if(varstr == f'{var1}{var2}'):
                return [var1,var2]

#--------------------------------------------------------------
# File reading
#--------------------------------------------------------------
def read_2d_file(filename):

    """ Reads in a 2D image file """

    xs=0
    ys=0
    Pxy=0
    
    with open(filename,'r') as f:
        header1 = f.readline().split()
        header2 = f.readline().split()

    xstr = header1[0]
    delta_x = float(header1[1])*unit_registry(get_unit_str(header1[3]))
    avg_x = float(header1[2])*unit_registry(get_unit_str(header1[3]))

    ystr = header2[0]
    delta_y = float(header2[1])*unit_registry(get_unit_str(header2[3]))
    avg_y = float(header2[2])*unit_registry(get_unit_str(header2[3]))

    Pxy = np.loadtxt(filename,skiprows=2)*unit_registry("1/"+str(avg_x.units)+"/"+str(avg_y.units))

    xs = avg_x + linspace(-delta_x/2.0,+delta_x/2.0,Pxy.shape[1])
    ys = avg_y + linspace(-delta_y/2.0,+delta_y/2.0,Pxy.shape[0])

    return (xs, ys, Pxy, xstr, ystr)


#--------------------------------------------------------------
# Nested Dict Functions
#--------------------------------------------------------------
def flatten_dict(dd, sep=':', prefix=''):
    """
    Flattens a nested dict into a single dict, with keys concatenated with sep.
    
    Similar to pandas.io.json.json_normalize
    
    Example:
        A dict of dicts:
            dd = {'a':{'x':1}, 'b':{'d':{'y':3}}}
            flatten_dict(dd, prefix='Z')
        Returns: {'Z:a:x': 1, 'Z:b:d:y': 3}
    
    """
    return { prefix + sep + k if prefix else k : v
             for kk, vv in dd.items()
             for k, v in flatten_dict(vv, sep, kk).items()
             } if isinstance(dd, dict) else { prefix : dd }

def unflatten_dict(d, sep=':', prefix=''):
    """
    Inverse of flatten_dict. Forms a nested dict.
    """
    dd = {}
    for kk, vv in d.items():
        if kk.startswith(prefix+sep):
            kk=kk[len(prefix+sep):]
            
        klist = kk.split(sep)
        d1 = dd
        for k in klist[0:-1]:
            if k not in d1:
                d1[k] = {}
            d1 = d1[k]
        
        d1[klist[-1]] = vv
    return dd

#def update_nested_dict(d, settings, verbose=False, create_new):
#    """
#    Updates a nested dict with flattened settings
#    """
#    flat_params = flatten_dict(d)

#    for key, value in settings.items():
#        if verbose:
#            if key in flat_params:
#                print(f'Replacing param {key} with value {value}')
#            else:
#                print(f'New param {key} with value {value}')

        
#       flat_params[key] = value
#        
#    new_dict = unflatten_dict(flat_params)
#    
#    return new_dict

def update_nested_dict(d, settings, verbose=False, create_new=True):
    """
    Updates a nested dict with flattened settings
    """
    flat_params = flatten_dict(d)

    for key, value in settings.items():
    
        if(key in flat_params or create_new):
            flat_params[key] = value

        if(verbose and key in flat_params):
            print(f'Replacing param {key} with value {value}')

        elif(verbose and create_new):
            print(f'New param {key} with value {value}')

        elif(verbose): 
            print(f'Skipping param {key}')
        
    new_dict = unflatten_dict(flat_params)
    
    return new_dict


def set_nested_dict(dd, flatkey, val, sep=':', prefix='', verbose=0):
    """
    Set a value inside nested dicts using a key string. 
    Example:
        dd = {'key1':{'key2':{'key3':9}}}
        set_nested_dict(dd, 'P:key1:key2:key3', 4, prefix='P')
        
        will set dd in place as:
            {'key1': {'key2': {'key3': 4}}}
        
    
    """
    if flatkey.startswith(prefix+sep):
        flatkey=flatkey[len(prefix+sep):]    
    keys = flatkey.split(':')
    d = dd
    # Go through nested dicts
    for k in keys[0:-1]:
        d = d[k]
    final_key = keys[-1]
    # Set
    if final_key in d:
        d[final_key] = val
        return True
    else:
        if(verbose>0):
            print(f'Error: flat key {flatkey} key does not exist:', final_key)
        return False

def get_nested_dict(dd, flatkey, sep=':', prefix='distgen'):
    """
    Gets the value in a nested dict from a flattened key.
    See: flatten_dict
    """
    if flatkey.startswith(prefix+sep):
        flatkey=flatkey[len(prefix+sep):]
    keys = flatkey.split(':')
    d = dd
    # Go through nested dicts
    for k in keys:
        d = d[k]
    return d

def is_quantity(d):
    """ Checks if a dict can be converted to a quantity with units """
    if(isinstance(d,dict) and len(d.keys())==2 and "value" in d and "units" in d):
        return True
    else:
        return False

def dict_to_quantity(qd):
    
    """ Converts a dict to quantity with units """

    assert is_quantity(qd), 'Could not convert dictionary to quantity: '+str(qd)
 
    if(isinstance(qd['value'], float) or is_floatable(qd['value'])):
        return float(qd['value'])*unit_registry(qd['units'])
    else:
        return np.array(qd['value'])*unit_registry(qd['units'])

def list_to_dict(ll):
    assert isinstance(ll, list), 'input to list_to_dict must be a list'
    return {index:ll[index] for index in range(len(ll))}
        
def convert_list_params(d):
    """ Converts elements in a list to quantities with units where appropriate """
    for ii,v in enumerate(d):
        if(is_quantity(v)):
            d[ii]=dict_to_quantity(v)
        elif(isinstance(v,dict)):
            convert_params(v)

def convert_params(d): 

    """ Converts a nested dictionary to quantities with units where appropriate """
    for k, v in d.items():
        if(is_quantity(v)):
            d[k]=dict_to_quantity(v)
        elif isinstance(v, list):
            convert_list_params(v)
        elif isinstance(v, dict) or isinstance(v, list):
            convert_params(v)           

class NpEncoder(json.JSONEncoder):
    """
    See: https://stackoverflow.com/questions/50916422/python-typeerror-object-of-type-int64-is-not-json-serializable/50916741
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def fingerprint(keyed_data, digest_size=16):
    """
    Creates a cryptographic fingerprint from keyed data. 
    Used JSON dumps to form strings, and the blake2b algorithm to hash.
    
    """
    h = blake2b(digest_size=16)
    for key in sorted(keyed_data.keys()):
        val = keyed_data[key]
        s = json.dumps(val, sort_keys=True, cls=NpEncoder).encode()
        h.update(s)
    return h.hexdigest()  


"""UTC to ISO 8601 with Local TimeZone information without microsecond"""
def isotime():
    return datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc).astimezone().replace(microsecond=0).isoformat()    
    



