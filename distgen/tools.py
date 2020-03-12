from .physical_constants import unit_registry 
import time
import numpy as np
from scipy.integrate import cumtrapz as scipy_cumtrapz 
from scipy.special import erf as sci_erf
from scipy.special import erfinv as sci_erfinv
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
    except ValueError:
        return False

def is_unit_str(ustr):

   ustr = ustr.strip()
   if(len(ustr)>=2 and ustr[0]=="[" and ustr[-1]=="]"):
       return True
   else: 
       return False

def get_unit_str(ustr):

    if(is_unit_str(ustr)):
        return ustr[1:-1]
    else:
        raise ValueError("Could not recover units string from "+ustr)

def assert_with_message(bool_val,msg):

    """Assert that boolean condition bool_val is true, raise value error with msg if false"""
    if(not bool_val):
        raise ValueError(msg)

class StopWatch():

    ureg = unit_registry

    def __init__(self):

       self.tstart = time.time() * self.ureg.second
       self.tstop = time.time() * self.ureg.second

    def start(self):
        self.tstart = time.time() * self.ureg.second

    def stop(self):
        self.tstop = time.time() * self.ureg.second

    def print(self):
        dt = self.tstop - self.tstart
        return "{0:.2f}".format(dt.to_compact())

# statistical operations:
def mean(x,weights=None):

    if(weights is None):
        return np.mean(x)
    else:
        return np.sum(x*weights)

def std(x,weights=None):

    if(weights is None):
        return np.std(x)
    else:
        return np.sqrt(np.sum( weights*(x-mean(x,weights))**2 ) )
    
# Numerical integration routines
def trapz(f,q):

    """ Wraps the numpy trapz function for arrays carrying units. Numerically integrates f=f(q) using trapezoid method.
    Inputs: quantity arrays q, f(q).
    Returns: integral( f(q) dq) as a float with units [result] = [f][q]
    """
    uqstr = str(q.units)
    ufstr = str(f.units)
    return np.trapz(f.magnitude,q.magnitude)*unit_registry.parse_expression(uqstr+"*"+ufstr)

def cumtrapz(f,q,initial=None):
    
    """ Wraps the scipy cumtrapz function for arrays carrying units. Numerically integrates f=f(q) using trapezoid method.
    Inputs: quantity arrays q, f(q), initial value for cumulative array
    Returns: cumulative integral( f(q) dq) as an array with units [result] = [f][q]
    """
    uqstr = str(q.units)
    ufstr = str(f.units)

    return scipy_cumtrapz(f.magnitude,q.magnitude,initial=initial)*unit_registry.parse_expression(uqstr+"*"+ufstr)

def rectint(f,x):
 
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

def cumrectint(f,x,initial):

    uxstr = str(x.units)
    ufstr = str(f.units)    

    xb = np.zeros(len(x)+1)
    xb[1:-1] = (x[1:]+x[:-1])/2.0

    dxL = xb[1]-x[0]
    dxR = xb[-2]-x[-1]

    xb[0] = x[0]-dxL
    xb[-1]= x[-1]+dxR

    crint = np.zeros(len(xb))
    crint[1:] = np.cumsum((xb[1:]-xb[:-1])*f)
    return crint

def interp(x,xs,fs):
    """
    Wraps the numpy interp function for arrays carrying units. 
    Inputs: quantity arrays xs, f(xs) and points to interpolate f at.
    Returns: f(x), based on interpolation of xs,fs
    """
    x.ito(xs.units)
    return np.interp(x.magnitude,xs.magnitude,fs.magnitude)*unit_registry(str(fs.units))

def linspace(x1,x2,N):
    return np.linspace(x1.to(x2.units).magnitude, x2.magnitude, N)*unit_registry(str(x2.units))
    
def radint(f,r):
 
    # for integrating function with rdr as the jacobian: int(r*f(r)*dr)
    urstr = str(r.units)
    ufstr = str(f.units)
    rbins = ((r.magnitude)[:-1]+(r.magnitude)[1:])/2.0
    rbins = np.insert(rbins,0,(r.magnitude)[0])
    rbins = np.append(rbins,(r.magnitude)[-1])

    return np.sum(0.5*(rbins[1:]**2-rbins[:-1]**2)*f.magnitude)*unit_registry(urstr+"*"+urstr+"*"+ufstr)

def radcumint(f,r,initial=None):
    """
    Defines cumulative radial integration with the rdr jacobian
    Inputs: r, f(r) with units, returns int( f(r) * r dr)_-inf^r
    """
    urstr = str(r.units)
    ufstr = str(f.units)
    rbins = ((r.magnitude)[:-1]+(r.magnitude)[1:])/2.0
    rbins = np.insert(rbins,0,(r.magnitude)[0])
    rbins = np.append(rbins,(r.magnitude)[-1])    

    rcint = np.zeros(len(rbins))
    rcint[1:] = np.cumsum(0.5*(rbins[1:]**2-rbins[:-1]**2)*f.magnitude)

    return (rcint*unit_registry(urstr+"*"+urstr+"*"+ufstr),rbins*unit_registry(urstr))

def histogram(x,bins=100):

    xmag = x.magnitude
    hist,edges = np.histogram(xmag,bins=bins)
    return (hist,edges)

def erf(x):

    return sci_erf(x.magnitude)*unit_registry('dimensionless')

def erfinv(x):
    return sci_erfinv(x.magnitude)*unit_registry('dimensionless')

# File reading

def read_2d_file(filename):

    xs=0
    ys=0
    Pxy=0
    
    f = open(filename,'r')

    header1 = f.readline().split()
    header2 = f.readline().split()

    xstr = header1[0]
    delta_x = float(header1[1])*unit_registry(get_unit_str(header1[3]))
    avg_x = float(header1[2])*unit_registry(get_unit_str(header1[3]))

    ystr = header2[0]
    delta_y = float(header2[1])*unit_registry(get_unit_str(header2[3]))
    avg_y = float(header2[2])*unit_registry(get_unit_str(header2[3]))

    f.close()

    Pxy = np.loadtxt(filename,skiprows=2)*unit_registry("1/"+str(avg_x.units)+"/"+str(avg_y.units))

    xs = avg_x + linspace(-delta_x/2.0,+delta_x/2.0,Pxy.shape[1])
    ys = avg_y + linspace(-delta_y/2.0,+delta_y/2.0,Pxy.shape[0])

    return (xs,ys,Pxy,xstr,ystr)

def nearest_neighbor(array,values):
    """
    find the nearest neighbor index in array for each value in values
    """
    array = array.magnitude
    values = values.magnitude
    return np.abs(np.subtract.outer(array, values)).argmin(0)




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

def update_nested_dict(d, settings, verbose=False):
    """
    Updates a nested dict with flattened settings
    """
    flat_params = flatten_dict(d)

    for key, value in settings.items():
        if verbose:
            if key in flat_params:
                print(f'Replacing param {key} with value {value}')
            else:
                print(f'New param {key} with value {value}')
        flat_params[key] = value
        
    new_dict = unflatten_dict(flat_params)
    
    return new_dict









def set_nested_dict(dd, flatkey, val, sep=':', prefix=''):
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
    else:
        print(f'Error: flat key {flatkey} key does not exist:', final_key)

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

#def convert_params(d):
#    
#    for k, v in d.items():
#        if(k=='params' and isinstance(v,dict)):
#            params = {}
#            for p in v.keys():
#                if(isinstance(v[p],dict) and 'value' in v[p] and 'units' in v[p]):
#                    params[p]=v[p]["value"]*unit_registry(v[p]["units"])
#                else:
#                    params[p]=v[p]
#            d[k]=params
#
#        elif isinstance(v, dict):
#            convert_params(v)

def is_quantity(d):

    if(isinstance(d,dict) and len(d.keys())==2 and "value" in d and "units" in d):
        return True
    else:
        return False

def dict_to_quantity(qd):

    assert is_quantity(qd), 'Could not convert dictionary to quantity: '+str(qd)
    return qd['value']*unit_registry(qd['units'])
        
def convert_params(d):

    for k, v in d.items():
        if(is_quantity(v)):
            d[k]=dict_to_quantity(v)
        elif isinstance(v, dict):
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
    for key in keyed_data:
        val = keyed_data[key]
        s = json.dumps(val, sort_keys=True, cls=NpEncoder).encode()
        h.update(s)
    return h.hexdigest()  


"""UTC to ISO 8601 with Local TimeZone information without microsecond"""
def isotime():
    return datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc).astimezone().replace(microsecond=0).isoformat()    
    



