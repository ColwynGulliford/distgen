from .physical_constants import unit_registry 
import time
import numpy as np
from scipy.integrate import cumtrapz as scipy_cumtrapz 

# HELPER FUNCTIONS:
def vprint(out_str,verbose,indent_number,new_line):

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
    if(not bool_val):
        raise ValueError(msg)

class stopwatch():

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

def trapz(f,q):

    uqstr = str(q.units)
    ufstr = str(f.units)
    return np.trapz(f.magnitude,q.magnitude)*unit_registry.parse_expression(uqstr+"*"+ufstr)

def cumtrapz(f,q,initial=None):

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

    # for integrating function with rdr as the jacobian: int(r*f(r)*dr)
    urstr = str(r.units)
    ufstr = str(f.units)
    rbins = ((r.magnitude)[:-1]+(r.magnitude)[1:])/2.0
    rbins = np.insert(rbins,0,(r.magnitude)[0])
    rbins = np.append(rbins,(r.magnitude)[-1])    

    rcint = np.zeros(len(rbins))
    rcint[1:] = np.cumsum(0.5*(rbins[1:]**2-rbins[:-1]**2)*f.magnitude)

    return (rcint*unit_registry(urstr+"*"+urstr+"*"+ufstr),rbins*unit_registry(urstr))

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
    array = array.magnitude
    values = values.magnitude
    return np.abs(np.subtract.outer(array, values)).argmin(0)


