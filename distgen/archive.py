import numpy as np

from .tools import isotime, flatten_dict, unflatten_dict
from . import _version
__version__ = _version.get_versions()['version']




def fstr(s):
    """
    Makes a fixed string for h5 files
    """
    return np.string_(s)




def distgen_init(h5, version=__version__):
    """
    Set basic information to an open h5 handle
    
    """
    
    d = {
        'dataType':'distgen',
        'software':'distgen',
        'version':version,
        'date':isotime()     
    }
    for k,v in d.items():
        h5.attrs[k] = fstr(v)
        
        
#----------------------------        
# Searching archives

def is_distgen_archive(h5, key='dataType', value=fstr('distgen')):
    """
    Checks if an h5 handle is a lume-astra archive
    """
    return key in h5.attrs and h5.attrs[key]==value
            
      
def find_distgen_archives(h5):
    """
    Searches one level deep
    """
    if is_distgen_archive(h5):
        return ['./']
    else:
        return [g for g in h5 if is_distgen_archive(h5[g])]             
    
    
    
#----------------------------        
# input
def write_input_h5(h5, distgen_input, name='input'):
    """
    
    Writes distgen input to h5 as a flattened dict. 
    
    distgen_input is a nested dict.
    
    See: read_input_h5
    """
    # Input as flattened dict
    g = h5.create_group(name)
    d = flatten_dict(distgen_input)
    for k, v in d.items():
        g.attrs[k] = v
            
def read_input_h5(h5):
    """
    Reads distgen input from h5. Unflattens the input dict
    
    See: write_input_h5
    
    
    Note: h5py returns numbers as numpy types. This will cast to native python types, as in:
    https://stackoverflow.com/questions/9452775/converting-numpy-dtypes-to-native-python-types/11389998#11389998
    """  
    d = {}
    for k in h5.attrs:
        v = h5.attrs[k]
        
        if isinstance(v, str):
            pass
        elif np.isscalar(v):
            # convert numpy types to native python types
            v = v.item()
        else:
            v = list(v)
        
    
        d[k] = v
    
    return unflatten_dict(d)