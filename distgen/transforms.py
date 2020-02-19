from .physical_constants import unit_registry
import numpy as np

ALLOWED_VARIABLES = ['x','y','z','t','r','theta','px','py','pz','pr','ptheta']

def get_variables(varstr):

   varstr=varstr.strip()
   variables = varstr.split(':')
   for variable in variables:
       assert variable in ALLOWED_VARIABLES, 'transforms::get_variables -> variable '+variable+' is not supported.'
   return variables
   
# Single variable transforms:

def translate(beam, var, delta):
    beam[var] = delta + beam[var]
    return beam

def set_avg(beam, var, new_avg):
    beam[var] = new_avg + (beam[var]-beam[var].mean())
    return beam

def scale(beam, var, scale, fix_average=False):

    if(isinstance(scale,float) or isinstance(scale,int)):
        scale = float(scale)*unit_registry('dimensionless')

    avg = beam[var].mean()
    if(fix_average):
        beam[var] = avg + scale*(beam[var]-avg)
    else:
        beam[var] = scale*beam[var]

    return beam

def set_avg_and_std(beam, var, new_avg, new_std):

    old_std = beam[var].std()
    if(old_std.magnitude>0):
        beam = scale(beam, var, new_std/old_std, fix_average=True)

    beam = set_avg(beam, var, new_avg)
    return beam

# 2 variable transforms:
def rotate2d(beam, variables, angle, origin=None):

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

def sheer(beam, variables, sheer_coefficient, origin=None):

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

    beam[var2] = beam[var2] + sheer_coefficient*(beam[var1]-o1)
    return beam

def magnetize(beam, variables, magnetization):

    if(variables=='r:ptheta'):

        sigx = beam['x'].std()
        sigy = beam['y'].std()
  
        return sheer(beam, variables, -magnetization/sigx/sigx ) 


def transform(beam, desc, varstr, **kwargs):
    variables = get_variables(varstr)
    transform_fun = globals()[desc]
    return transform_fun(beam,varstr,**kwargs)


