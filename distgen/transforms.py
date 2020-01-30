from .physical_constants import unit_registry
import numpy as np


def translate(beam, var, new_avg):
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

    beam = translate(beam, var, new_avg)
    return beam

# 2 variable transforms
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


#def polynomial(beam, variables, coefficients):

#    if(isinstance(variables,str) and len(variables.split(":"))==2):
#        var1,var2=variables.split(':')

#    y = np.zeros(beam[var2].shape)*unit_registry(beam[var2].units)

#    for ii,coeff in coefficients:
#        if(ii==0):
#            y = coeff
#        else:
#            y = y + coeff*np.pow(y,ii)

#    beam[var2]=y
        


def transform(func):
    @functools.wraps(func)
    def transform_wrapper(*args,**kwargs):
        func(*args, **kwargs)
        return func(*args, **kwargs)
    return transform_wrapper
