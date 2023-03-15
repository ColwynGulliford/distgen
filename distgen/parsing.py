import copy
import numpy as np
import os

from .physical_constants import unit_registry 
from .physical_constants import is_quantity

def is_floatable(value):

    """Check if an object can be cast to a float, return true if so, false if not"""

    try:
        float(value)
        return True
    except:
        return False

def expand_input_filepaths(input_dict, root=None, ignore_keys=[]):
    """
    Searches for 'file' keys with relative path values in the input dict,
    and fills the value with an absolute path based on root.
    If no root is given, os.path.getcwd() is used.
    """

    if not root:
        root = os.getcwd()
    else:
        root = os.path.abspath(root)

    for k1, v1 in input_dict.items():

        if k1 in ignore_keys:
            continue
        if not isinstance(v1, dict):
            continue
        if 'file' in v1:
            filepath = v1['file']
            #print(filepath)
            if not os.path.isabs(filepath):
                fnew = os.path.join(root, filepath)
                #print(fnew)
                #assert os.path.exists(fnew), f'{fnew} does not exist'

                v1['file'] = fnew

        # Recursively search
        expand_input_filepaths(v1, root=root, ignore_keys=ignore_keys)
        
        
def is_unit(u):
    
    if(isinstance(u, str)):
        u = u.replace('1', '2')  # '2 mm' is not a unit, but '1 mm' is, so handle this oddity
    try: 
        unit_registry.Unit(u)
        return True
    except:
        return False
        
def is_quantizable(d):
    
    """ Checks if a dict can be converted to a quantity with units """
    if(isinstance(d, dict) and len(d.keys())==2 and "value" in d and "units" in d):
        return True
    
    elif(isinstance(d, str)):
        
        if(is_unit(d)):
            return False
            
        try: 
            q = unit_registry(d)
            if(isinstance(q, unit_registry.Quantity)):
                return True
            else:
                return False
        except:
            return False

    else:
        return False    

def dict_to_quantity(qd):
    
    """ Converts a dict to quantity with units """

    assert is_quantizable(qd), 'Could not convert dictionary to quantity: '+str(qd)
 
    if(isinstance(qd['value'], float) or is_floatable(qd['value'])):
        return float(qd['value'])*unit_registry(qd['units'])
    else:
        return np.array(qd['value'])*unit_registry(qd['units'])
    
def parse_quantity(q):
    
    if(is_quantizable(q)):
    
        if(isinstance(q, str)): 
            return unit_registry.Quantity(q)
        elif(isinstance(q, dict)):
            return dict_to_quantity(q)
    else:
        raise ValueError(f'Could not parse object into a quantity: {q}')
        
def update_quantity_in_dict(k, d, new_val): 
    d[k] = update_quantity(d[k], new_val)
    
def update_quantity(x, new_val):
    
    Q_ = unit_registry.Quantity
    
    if(is_floatable(new_val) or isinstance(new_val, np.ndarray)):
        x = Q_(new_val, x.units)
    
    elif(isinstance(new_val, str)):
        x = Q_(x.magnitude, unit_registry.parse_expression(new_val))
        
    elif(isinstance(new_val, unit_registry.Quantity)):
        return new_val
        
    else:
        raise ValueError('Unsupported input value for setting quantity!')
        
    return x

def list_to_dict(ll):
    assert isinstance(ll, list), 'input to list_to_dict must be a list'
    return {index:ll[index] for index in range(len(ll))}
        
def convert_list_quantities(d):
    """ Converts elements in a list to quantities with units where appropriate """
    for ii,v in enumerate(d):
        
        if(is_quantizable(v)):
            d[ii]=parse_quantity(v)
        elif(isinstance(v,dict)):
            convert_input_quantities(v)
            
            
def convert_list_quantities_to_user_input(d):
    
    """ Converts quantities in a list to user input where appropriate """
    for ii,v in enumerate(d):
        
        if(is_quantity(v)):
            d[ii]={'value':v.magnitude, 'units':str(v.units)}
        elif(isinstance(v,dict)):
            convert_input_quantities(v)

def convert_input_quantities(d, in_place=True): 
    
    if(not in_place):
        d = copy.deepcopy(d)

    """ Converts a nested dictionary to quantities with units where appropriate """
    for k, v in d.items():
    
        if(is_quantizable(v)):
            d[k]=parse_quantity(v)
        elif isinstance(v, list):
            convert_list_quantities(v)
        elif isinstance(v, dict) or isinstance(v, list):
            convert_input_quantities(v) 
            
    return d
            
def convert_quantities_to_user_input(d, in_place=True):
    
    if(not in_place):
        d = copy.deepcopy(d)
    
    """ Converts a nested dictionary of quantities with units to user input format """
    for k, v in d.items():
        
        if(is_quantity(v)):
            d[k]={'value':v.magnitude, 'units':str(v.units)}
        elif isinstance(v, list):
            convert_list_quantities_to_user_input(v)
        elif isinstance(v, dict) or isinstance(v, list):
            convert_quantities_to_user_input(v) 
    
    return d
            
