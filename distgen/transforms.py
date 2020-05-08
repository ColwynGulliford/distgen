from .physical_constants import unit_registry
from .tools import dict_to_quantity
from .tools import vprint, mean
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
        o = beam.avg(varstr)
    elif(origin is None):
        o = 0*beam[varstr].units
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
    vprint(f'Translating {var} by {delta:g~P}.', params['verbose'], 2, True)
    beam[var] = delta + beam[var]
    
    return beam


def set_avg(beam, **params):

    var = params['variables']
    check_inputs(params, ['avg_'+var], [], 1, 'set_avg(beam, **kwargs)')  
    new_avg = params['avg_'+var] 
    beam[var] = new_avg + (beam[var]-beam[var].mean())
    vprint(f'Setting avg_{var} -> {new_avg:G~P}.', params['verbose'], 2, True)

    return beam

def scale(beam, **params):

    check_inputs(params, ['scale'], ['fix_average'], 1, 'scale(beam, **kwargs)')  
    var = params['variables']
    scale = params['scale']
    fix_average = params['fix_average']

    if(isinstance(scale,float) or isinstance(scale,int)):
        scale = float(scale)*unit_registry('dimensionless')

    if(fix_average):
        avg = beam[var].mean()
        beam[var] = avg + scale*(beam[var]-avg)
        vprint(f'Scaling {var} by {scale:G~P} holding avg_{var} = {avg:G~P} constant.', params['verbose'], 2, True)
    else:
        beam[var] = scale*beam[var]
        vprint(f'Scaling {var} by {scale:G~P}.', params['verbose'], 2, True)

    return beam

def set_std(beam, **params):

    var = params['variables'] 
    check_inputs(params, ['sigma_'+var], [], 1, 'set_std(beam, **kwargs)')  
    new_std = params['sigma_'+var]
    vprint(f'Setting sigma_{var} -> {new_std:G~P}', params['verbose'], 2, True)
    old_std = beam[var].std()
    if(old_std.magnitude>0):
        beam = scale(beam, **{'variables':var,'scale':new_std/old_std, 'fix_average':True})

    return beam

def set_stdxy(beam, **params):

    var = params['variables'] 
    check_inputs(params, ['sigma_xy'], [], 2, 'set_stdxy(beam, **kwargs)') 
   
    beam = set_std(beam, **{'variables':'x', 'sigma_x':params['sigma_xy']})
    beam = set_std(beam, **{'variables':'y', 'sigma_y':params['sigma_xy']})

    return beam

def set_avg_and_std(beam, **params):

    var = params['variables'] 
    check_inputs(params, ['sigma_'+var, 'avg_'+var], [], 1, 'set_avg_and_std(beam, **kwargs)') 
    beam = set_std(beam, **{'variables':var, 'sigma_'+var:params['sigma_'+var]})
    beam = set_avg(beam, **{'variables':var, 'avg_'+var:params['avg_'+var]})
    vprint(f'Setting avg_{var} -> {beam.avg(var):G~P} and sigma_{var} -> {beam.std(var):G~P}', params['verbose'], 2, True)
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
        vprint(f'Rotating {var1}-{var2} by {angle.to("deg"):G~P} around {var1} and {var2} centroid.', params['verbose'], 2, True) 

    elif(origin is None):
        o1 = 0*unit_registry(str(v1.units))
        o2 = 0*unit_registry(str(v1.units))
        vprint(f'Rotating {var1}-{var2} by {angle.to("deg"):G~P}.', params['verbose'], 2, True) 

    else:
        o1 = origin[0]
        o2 = origin[1]
        vprint(f'Rotating {var1}-{var2} by {angle.to("deg"):G~P} around {var1} = {o1:G~P} and {var2} = {o2:G~P}.', params['verbose'], 2, True) 

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
        o1 = beam.avg(var1)
        vprint(f'Shearing {var1} into {var2} around {var1} = {o1:G~P} with shear coefficient {shear_coefficient:G~P}.', params['verbose'], 2, True) 

 
    elif(origin is None):
        o1 = 0*beam[var1].units
        #o2 = 0*unit_registry(str(v1.units))
        vprint(f'Shearing {var1} into {var2} with shear coefficient {shear_coefficient:G~P}.', params['verbose'], 2, True) 


    else:
        o1 = origin[0]
        #o2 = origin[1]

    v1 = beam[var1]
    v2 = beam[var2]

    beam[var2] = v2 + shear_coefficient*(v1-o1)

    
    return beam


def polynomial(beam, **params):#variables, polynomial_coefficients, origin=None, zero_dependent_var=False):

    check_inputs(params, ['coefficients'], ['origin','zero_dependent_var'], 2, 'polynomial(beam, **kwargs)') 
    variables = params['variables']
    coefficients = params['coefficients']
    variables = get_variables(variables)
    zero_dependent_var = params['zero_dependent_var']
    origin = params['origin']

    v1 = beam[variables[0]]

    if(zero_dependent_var):
        v2 = np.zeros(beam[variables[1]].shape)*unit_registry(beam[variables[1]].units)
    else:
        v2 = beam[variables[1]]
   
    origin = get_origin(beam, variables[0], origin)

    vprint(f'Applying polynomial p({variables[0]} -> {variables[1]}) around {variables[0]} = {origin:G~P}, with coefficients:', params['verbose'], 2, True) 

    for n, coefficient in enumerate(coefficients):
        vprint(f'c{n} = {coefficient.to_reduced_units():G~P},', params['verbose'], 3, True)
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

    vprint(f'Applying cosine function: {variables[1]}) -> {variables[1]} + A*cos(w*{variables[0]} + phi), with:', params['verbose'], 2, True) 
    vprint(f'amplitude = {amplitude:G~P}, omega = {omega:G~P}, and phase = {phase:G~P}.', params['verbose'], 3, True) 

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

        sigx = beam.std('x')
        sigy = beam.std('y')
    
        magnetization = params['magnetization']
        sparams = {'type':'shear','variables':'r:ptheta','shear_coefficient': -magnetization/sigx/sigx, 'verbose':params['verbose'] }        

        return shear(beam, **sparams) 

    else:
        raise ValueError('transforms::magnetize(beam, **kwargs) -> variables "'+variables+'" not supported!.')

def set_twiss(beam, **params): #plane, beta, alpha, eps):

    check_inputs(params, ['beta','alpha','emittance'], [], 1, 'set_twiss(beam, **kwargs)') 
    plane = params['variables']
    beta = params['beta']
    alpha = params['alpha']
    eps = params['emittance']

    vprint(f'Setting beta_{plane} -> {beta:G~P}, alpha_{plane} -> {alpha:G~P}, and emittance_{plane} -> {eps:G~P}.', params['verbose'], 2, True) 

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

    beta0, alpha0, eps0 = beam.twiss(xstr)

    assert beta0>0, f'Error in set_twiss: initial beta = {beta0} was <=0, the initial distribution must have finite size to use this transform.'
    assert eps0>0, f'Error in set_twiss: initial emit = {eps0} was <=0, the initial distribution must have finite size to use this transform.'
    assert beta>0, f'Error in set_twiss: final beta = {beta} was <=0, the final distribution must have finite size to use this transform.'

    m11 = (np.sqrt(beta*eps/beta0/eps0)).to_base_units()
    m12 = (0*unit_registry(str(beam[xstr].units)+'/'+str(beam[pstr].units))).to_base_units()
    m21 = (( (alpha0-alpha)/np.sqrt(beta*beta0) )*np.sqrt(eps/eps0)).to_base_units()
    m22 = (np.sqrt(beta0/beta)*np.sqrt(eps/eps0)).to_base_units()

    beam = matrix2d(beam, xstr+':'+pstr, m11, m12, m21, m22)
    beam[xstr] = avg_x0 + beam[xstr]
    beam[pstr] = avg_p0 + beam[pstr]

    return beam

def transform(beam, T):

    desc = T['type']
    tokens = desc.split(' ')
    transfunc = tokens[0]
    variables = tokens[1]
    
    T['variables']=variables

    variables = get_variables(variables)
    transform_fun = globals()[transfunc]
    return transform_fun(beam, **T)


