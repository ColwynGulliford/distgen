from .physical_constants import unit_registry
from .tools import dict_to_quantity
from .tools import vprint
import numpy as np


ALLOWED_VARIABLES = ['x','y','z','t','r','theta','px','py','pz','pr','ptheta','xp','yp']

def get_variables(varstr):

   varstr=varstr.strip()
   variables = varstr.split(':')
   for variable in variables:
       assert variable in ALLOWED_VARIABLES, 'transforms::get_variables -> variable '+variable+' is not supported.'
   return variables
   
def get_origin(beam,varstr,origin):

    if(origin=='centroid'):
        o = beam[varstr].mean()
    elif(origin is None):
        o = 0*unit_registry(str(beam[varstr].units))
    else:
        o = origin

    return o

def check_inputs(params, required_params, optional_params, n_variables, name):

    assert 'variables' in params, 'All transforms colon separated "variables" string specifying which coordinates to transform.'
    assert n_variables==len(params['variables'].split(':')), name+' function requires '+str(n_variables)+'.'

    # Make sure user isn't passing the wrong parameters:
    allowed_params = optional_params + required_params + ['variables', 'type', 'verbose']
    for param in params:
        assert param in allowed_params, 'Incorrect param given to '+name+': '+param+'\nAllowed params: '+str(allowed_params)

    # Make sure all required parameters are specified
    for req in required_params:
        assert req in params, 'Required input parameter '+req+' to '+name+'.__init__(**kwargs) was not found.'

    for opt in optional_params:
        if(opt not in params):
            params[opt]=None

    if('verbose' not in params):
        params['verbose']=False

# Single variable transforms:

def translate(beam, **params):

    check_inputs(params, ['delta'], [], 1, 'translate(beam, **kwargs)')  
    var = params['variables']
    delta = params['delta'] 
    vprint(f'Translating {var} by {"{:g~P}".format(delta)}', params['verbose'], 2, True)
    beam[var] = delta + beam[var]
    

    return beam


def set_avg(beam, **params):

    var = params['variables']
    check_inputs(params, ['avg_'+var], [], 1, 'set_avg(beam, **kwargs)')  
    new_avg = params['avg_'+var] 
    beam[var] = new_avg + (beam[var]-beam[var].mean())
    vprint(f'Setting avg {var} -> {"{:g~P}".format(new_avg)}', params['verbose'], 2, True)

    return beam

def scale(beam, **params):

    check_inputs(params, ['scale'], ['fix_average'], 1, 'scale(beam, **kwargs)')  
    var = params['variables']
    scale = params['scale']
    fix_average = params['fix_average']

    if(isinstance(scale,float) or isinstance(scale,int)):
        scale = float(scale)*unit_registry('dimensionless')

    avg = beam[var].mean()
    if(fix_average):
        beam[var] = avg + scale*(beam[var]-avg)
    else:
        beam[var] = scale*beam[var]

    return beam

def set_std(beam, **params):

    var = params['variables'] 
    check_inputs(params, ['sigma_'+var], [], 1, 'set_std(beam, **kwargs)')  
    new_std = params['sigma_'+var]
    vprint(f'Setting avg {var} -> {"{:g~P}".format(new_std)}', params['verbose'], 2, True)
    old_std = beam[var].std()
    if(old_std.magnitude>0):
        beam = scale(beam, **{'variables':var,'scale':new_std/old_std, 'fix_average':True})

    return beam

def set_avg_and_std(beam, **params):

    var = params['variables'] 
    check_inputs(params, ['sigma_'+var, 'avg_'+var], [], 1, 'set_avg_and_std(beam, **kwargs)') 
    beam = set_std(beam,**{'variables':var, 'sigma_'+var:params['sigma_'+var]})
    beam = set_avg(beam,**{'variables':var, 'avg_'+var:params['avg_'+var]})
    return beam


# 2 variable transforms:
def rotate2d(beam, **params): #variables, angle, origin=None):

    check_inputs(params, ['angle'], ['origin'], 2, 'rotate2d(beam, **kwargs)') 
    variables = params['variables']
    angle = params['angle']
    origin = params['origin']
   
    if(isinstance(variables,str) and len(variables.split(":"))==2):
        
        var1,var2=variables.split(':')

    C = np.cos(angle)
    S = np.sin(angle)

    v1 = beam[var1]
    v2 = beam[var2]

    if(origin=='centroid'):
        o1 = v1.mean()
        o2 = v2.mean()
 
    elif(origin is None):
        o1 = 0*unit_registry(str(v1.units))
        o2 = 0*unit_registry(str(v1.units))

    else:
        o1 = origin[0]
        o2 = origin[1]

    beam[var1] =  o1 + C*(v1-o1) - S*(v2-o2)
    beam[var2] =  o2 + S*(v1-o1) + C*(v2-o2)

    return beam


def shear(beam, **params):

    check_inputs(params, ['shear_coefficient'], ['origin'], 2, 'shear(beam, **kwargs)') 
    variables = params['variables']
    shear_coefficient = params['shear_coefficient']
    origin = params['origin'] 

    if(isinstance(variables,str) and len(variables.split(":"))==2):
        var1,var2=variables.split(':')

    if(origin=='centroid'):
        o1 = beam[var1].mean()
        #o2 = v2.mean()
 
    elif(origin is None):
        o1 = 0*unit_registry(str(beam[var1].units))
        #o2 = 0*unit_registry(str(v1.units))

    else:
        o1 = origin[0]
        #o2 = origin[1]

    beam[var2] = beam[var2] + shear_coefficient*(beam[var1]-o1)
    return beam


def polynomial(beam, **params):#variables, polynomial_coefficients, origin=None, zero_dependent_var=False):

    check_inputs(params, ['coefficients'], ['origin','zero_dependent_var'], 2, 'polynomial(beam, **kwargs)') 
    variables = params['variables']
    coefficients = params['coefficients']
    variables = get_variables(variables)
    zero_dependent_var = params['zero_dependent_var']
    origin = params['origin']

    v1 = beam[variables[0]]

    units = beam[variables[1]].units

    if(zero_dependent_var):
        v2 = np.zeros(beam[variables[1]].shape)*unit_registry( str(units)  )
    else:
        v2 = beam[variables[1]]
   
    origin = get_origin(beam,variables[0],origin)

    for n, coefficient in enumerate(coefficients):
        v2 = v2 + coefficient*np.power(v1-origin,n)

    beam[variables[1]] = v2
    return beam

def cosine(beam, **params):#variables, amplitude, phase, omega, zero_dependent_var=False):

    check_inputs(params, ['amplitude','phase','omega'], ['zero_dependent_var'], 2, 'cosine(beam, **kwargs)') 
    variables = params['variables']
    amplitude = params['amplitude']
    phase = params['phase']
    omega = params['omega']
    zero_dependent_var = params['zero_dependent_var']

    variables = get_variables(variables)

    v1 = beam[variables[0]]

    units = beam[variables[1]].units

    if(zero_dependent_var):
        v2 = np.zeros(beam[variables[1]].shape)*unit_registry( str(units)  )
    else:
        v2 = beam[variables[1]]

    beam[variables[1]] = v2 + amplitude*np.cos( omega*v1 + phase )
    return beam


def matrix2d(beam, variables, m11, m12, m21, m22):

   variables = get_variables(variables)
   v1 = beam[variables[0]]
   v2 = beam[variables[1]]

   beam[variables[0]] = m11*v1 + m12*v2
   beam[variables[1]] = m21*v1 + m22*v2

   return beam


def magnetize(beam, **params):

    check_inputs(params, ['magnetization'], [], 2, 'magnetize(beam, **kwargs)') 
    variables=params['variables'] 

    if(variables=='r:ptheta'):

        sigx = beam['x'].std()
        sigy = beam['y'].std()
  
        magnetization = params['magnetization']
        sparams = {'type':'shear','variables':'r:ptheta','shear_coefficient': -magnetization/sigx/sigx }        

        return shear(beam, **sparams) 

    else:
        raise ValueError('transforms::magnetize(beam, **kwargs) -> variables "'+variables+'" not supported!.')

def set_twiss(beam, **params): #plane, beta, alpha, eps):

    check_inputs(params, ['beta','alpha','eps'], [], 1, 'set_twiss(beam, **kwargs)') 
    plane = params['variables']
    beta = params['beta']
    alpha = params['alpha']
    eps = params['eps']

    if(plane not in ['x','y']):
        raise ValueError('set_twiss -> unsupported twiss plane: '+plane)

    xstr = plane
    pstr = xstr+'p'

    x0 = beam[xstr]
    p0 = beam[pstr]

    avg_x0 = x0.mean()
    beam[xstr]=beam[xstr]-avg_x0

    avg_p0 = p0.mean()
    beam[pstr]=beam[pstr]-avg_p0

    beta0,alpha0,eps0 = beam.twiss(xstr)

    m11 = (np.sqrt(beta*eps/beta0/eps0)).to_base_units()
    m12 = (0*unit_registry(str(beam[xstr].units)+'/'+str(beam[pstr].units))).to_base_units()
    m21 = (( (alpha0-alpha)/np.sqrt(beta*beta0) )*np.sqrt(eps/eps0)).to_base_units()
    m22 = (np.sqrt(beta0/beta)*np.sqrt(eps/eps0)).to_base_units()

    beam = matrix2d(beam, xstr+':'+pstr, m11, m12, m21, m22)
    beam[xstr] = avg_x0 + beam[xstr]
    beam[pstr] = avg_p0 + beam[pstr]

    return beam

def transform(beam, desc, varstr, **kwargs):
    variables = get_variables(varstr)
    transform_fun = globals()[desc]
    return transform_fun(beam, **kwargs)


