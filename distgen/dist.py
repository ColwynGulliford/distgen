"""
Defines the random number generator class as well all distribution function objects.
"""


#import chaospy
# Replaced by:
from .hammersley import create_hammersley_samples
from .physical_constants import *
from .tools import *

import numpy as np
import numpy.matlib as mlib
#import scipy
import math

from matplotlib import pyplot as plt


class RandGen():

    """
    Defines object responsible for providing random numbers
    """
        
    def rand(self,shape,sequence=None,params=None):
        """
        Method for obtaining random numbers on the unit interval
        Inputs: shape is a tuple describing the shape of the output array of random numbers
                sequence is a str describing what type of random sequence to draw from, 
                defaults to rand(), or user can request Hammersley
                params is a dict for supply extra params (sequence parameters)
        """
    
        if(sequence is None):
            return np.random.random(shape)

        elif(sequence=="hammersley"):
            N = shape[0] 
            dim = shape[1]
            if(params is None):
                #return np.squeeze(chaospy.distributions.sampler.sequences.hammersley.create_hammersley_samples(N, dim=dim, burnin=-1, primes=()))
                return np.squeeze(create_hammersley_samples(N, dim=dim, burnin=-1, primes=()))
            else:
                #return np.squeeze(chaospy.distributions.sampler.sequences.hammersley.create_hammersley_samples(N, dim=dim, burnin=params["burnin"], primes=params["primes"]))
                return np.squeeze(create_hammersley_samples(N, dim=dim, burnin=params["burnin"], primes=params["primes"]))
        else:
            raise ValueError("Sequence: "+str(sequence)+" is not supported")


def get_dist(var,dtype,params=None,verbose=0):
    
    if(dtype=="uniform" or dtype=="u"):
        dist = Uniform(var,verbose=verbose,**params)
    elif(dtype=="gaussian" or dtype=="g"):
        dist = Norm(var,verbose=verbose,**params)
    elif(dtype=="file1d"):
        dist = File1d(var,verbose=verbose,**params)
    elif(dtype=='tukey'):
        dist = Tukey(var,verbose=verbose,**params)
    elif((dtype=="radial_uniform" or dtype=="ru") and var=="r"):
        dist = UniformRad(verbose=verbose,**params)
    elif((dtype=="radial_gaussian" or dtype=="rg") and var=="r"):
        dist = NormRad(verbose=verbose,**params)
    elif(dtype=="radfile" and var=="r"):
        dist = RadFile(verbose=verbose,**params)
    elif((dtype=="radial_truncated_gaussian" or dtype=="rtg") and var=="r"):
        dist = NormRadTrunc(verbose=verbose,**params)
    elif(dtype=="radial_tukey"):
        dist = TukeyRad(verbose=verbose,**params)
    elif(dtype=="file2d"):
        dist = File2d("x","y",verbose=verbose,**params)
    elif(dtype=="crystals"):
        dist = TemporalLaserPulseStacking(verbose=verbose,**params)
    else:
        raise ValueError("Distribution type '"+dtype+"' is not supported.")
            
    return dist
    
class Dist1d():

    """
    Defines the base class for 1 dimensional distribution functions.  
    Assumes user will pass in [x,f(x)] as the pdf. 
    Numerically intergates to find the cdf and to sample the distribution.  
    Methods should be overloaded for specific derived classes, particularly if
    the distribution allows analytic treatment.
    """

    xstr = ""
    xs = []    # x pts
    Px = []    # Probability Distribution Function Px(x)
    Cx = []    # Cumulative Disrtirbution Functoin Cx(x)
    
    rgen = RandGen()
    
    def __init__(self,xs,Px,xstr="x"):

        self.xs = xs
        self.Px = Px
        self.xstr = xstr

        norm = trapz(self.Px,self.xs)
        if(norm<=0):
            raise ValueError("Normalization of PDF was <= 0")

        self.Px = self.Px/norm
        self.Cx = cumtrapz(self.Px,self.xs,initial=0)
    
    def get_x_pts(self,n):
        return linspace(self.xs[0],self.xs[-1],n)

    def pdf(self,x):
        """"
        Evaluates the pdf at the user supplied points in x
        """
        return interp(x,self.xs,self.Px)
 
    def cdf(self,x):
        """"
        Evaluates the cdf at the user supplied points in x
        """
        return interp(x,self.xs,self.Cx)

    def cdfinv(self,rns):
        """
        Evaluates the inverse of the cdf at probabilities rns
        """
        return interp(rns,self.Cx,self.xs)

    def sample(self,N,sequence=None,params=None):
        """
        Generate coordinates by sampling the underlying pdf
        """
        return self.cdfinv(self.rgen.rand((N,1),sequence,params)*unit_registry("dimensionless"))

    def plot_pdf(self,n=1000):
        x=self.get_x_pts(n)
        plt.figure()
        plt.plot(x,self.pdf(x))
        plt.xlabel(self.xstr)
        plt.ylabel("PDF("+self.xstr+")")

    def plot_cdf(self,n=1000):
        x=self.get_x_pts(n)
        plt.figure()
        plt.plot(x,self.cdf(x))
        plt.xlabel(self.xstr)
        plt.ylabel("CDF("+self.xstr+")")

    def avg(self):
        """
        Defines the 1st moment of the pdf, defaults to using trapz integration
        """
        return trapz(self.xs*self.Px,self.xs)
  
    def rms(self):
        """
        Defines the rms of the pdf, defaults to using trapz integration
        """
        return np.sqrt(trapz(self.xs*self.xs*self.Px,self.xs))

    def std(self):
        """
        Defines the sqrt of the variance of the pdf, defaults to using trapz integration
        """
        avg = self.avg()
        rms = self.rms()
        return np.sqrt(rms*rms - avg*avg)

    def test_sampling(self):

        xs=self.sample(100000,sequence="hammersley")
        x = self.get_x_pts(1000)
        pdf = self.pdf(x)

        rho,edges = histogram(xs,bins=100)
        xhist = (edges[1:] + edges[:-1]) / 2
        rho = rho/np.trapz(rho,xhist)

        avgx = xs.mean()
        avgx_str = "{:0.3f~P}".format(avgx)
        stdx = xs.std()
        stdx_str = "{:0.3f~P}".format(stdx)

        davgx = self.avg()
        dstdx = self.std()
        davgx_str = "{:0.3f~P}".format(davgx)
        dstdx_str = "{:0.3f~P}".format(dstdx)       

        plt.figure()
        plt.plot(x, pdf, xhist, rho, 'or')
        
        if(isinstance(x,unit_registry.Quantity)):
            plt.xlabel(self.xstr+" ({:~P}".format(x.units)+")")
        else:
            plt.xlabel(self.xstr)
  
        if(isinstance(x,unit_registry.Quantity)):
            plt.ylabel("PDF("+self.xstr+") ({:~P}".format(pdf.units)+")")
        else:
            plt.ylabel("PDF("+self.xstr+")")
        plt.title("Sample stats: <"+self.xstr+"> = "+avgx_str+", $\sigma_{x}$ = "+stdx_str+"\nDist. stats: <"+self.xstr+"> = "+davgx_str+", $\sigma_{x}$ = "+dstdx_str)
        plt.legend(["PDF","Hist. of Sampling"])
    
class Uniform(Dist1d):

    def __init__(self,var,verbose=0,**kwargs):
        
        self.xstr = var
        
        minstr = "min_"+var
        if(minstr in kwargs.keys()):
            self.xL = kwargs[minstr]
        else:
            raise ValueError("Uniform dist required parameter "+minstr+" not found in input parameters.")
            
        maxstr = "max_"+var
        if(maxstr in kwargs.keys()):
            self.xR = kwargs[maxstr]
        else:
            raise ValueError("Uniform dist required parameter "+maxstr+" not found in input parameters.")
        
        vprint("uniform",verbose>0,0,True)
        vprint(minstr+" = {:0.3f~P}".format(self.xL)+", "+maxstr+" = {:0.3f~P}".format(self.xR),verbose>0,2,True)
   
    def get_x_pts(self,n):
        f = 0.2
        dx = f*np.abs(self.avg())
        return np.linspace(self.xL-dx,self.xR+dx,n)

    def pdf(self,x):
        #print(self.xL,self.xR,x)
        nonzero = (x >= self.xL) & (x <= self.xR)
        res = np.zeros(len(x))
        res[nonzero]=1/(self.xR-self.xL)
        res = res*unit_registry('1/'+str(self.xL.units))
        return res

    def cdf(self,x):
        nonzero = (x >= self.xL) & (x <= self.xR)
        res = np.zeros(len(x))
        res[nonzero]=(x[nonzero]-self.xL)/(self.xR-self.xL)
        return res

    def cdfinv(self,rns):
        return (self.xR-self.xL)*rns + self.xL

    def avg(self):
        return 0.5*(self.xR+self.xL)

    def std(self):
        return (self.xR-self.xL)/np.sqrt(12) 
  
    def rms(self):
        avg=self.avg()
        std=self.std()
        return np.sqrt(std*std + avg*avg)
    
class Norm(Dist1d):

    def __init__(self,var,verbose=0,**kwargs):
        
        self.xstr = var
        
        sigmastr = "sigma_"+var
        if(sigmastr in kwargs.keys()):
            self.sigma = kwargs[sigmastr]
        else:
            raise ValueError("Norm dist required parameter "+sigmastr+" not found in input parameters.")
            
        avgstr = "avg_"+var
        if(avgstr in kwargs.keys()):
            self.mu = kwargs[avgstr]
        else:
            self.mu = 0*unit_registry(str(self.sigma.units))

        vprint("Gaussian",verbose>0,0,True)
        vprint("avg_"+var+" = {:0.3f~P}".format(self.mu)+", sigma_"+var+" = {:0.3f~P}".format(self.sigma),verbose>0,2,True)
            
    def get_x_pts(self,n):
        return self.mu + linspace(-5*self.sigma,+5*self.sigma,1000)

    def pdf(self,x):        
        return (1/np.sqrt(2*math.pi*self.sigma*self.sigma))*np.exp( -(x-self.mu)**2/2.0/self.sigma**2 ) 

    def cdf(self,x):
        theta = (x-self.mu)/self.sigma/math.sqrt(2)
        print(erf)
        return 0.5*(1+erf( (x-self.mu)/self.sigma/math.sqrt(2) ) )

    def cdfinv(self,rns):
        return self.mu + math.sqrt(2)*self.sigma*erfinv((2*rns-1))

    def avg(self):
        return self.mu
    
    def std(self):
        return self.sigma

    def rms(self):
        avg=self.avg()
        std=self.std()
        return np.sqrt(std*std + avg*avg)
    
class File1d(Dist1d):
    
    def __init__(self,var,verbose=0,**kwargs):
        
        self.xstr=var
        
        if("file" in kwargs):
            self.distfile = kwargs["file"]
        else:
            raise ValueError("File 1D distribution required parameter 'file' not found.")
        
        if("units" in kwargs):
            units = kwargs["units"]
        else:
            units = "dimensionless"
        
        vprint(var+"-distribution file: '"+self.distfile+"'",verbose>0,0,True)
        f = open(self.distfile,'r')
        headers = f.readline().split()
        f.close()

        if(len(headers)!=2):
            raise ValueError("file1D distribution file must have two columns")
            
        #if(headers[0]!=self.xstr):
        #    raise ValueError("Input distribution file variable must be = "+var)
        #if(headers[1]!="P"+self.xstr):
        #    raise ValueError("Input distribution file pdf name must be = P"+var)    
            
        data = np.loadtxt(self.distfile,skiprows=1)

        xs = data[:,0]*unit_registry(units)
        Px =data[:,1]*unit_registry.parse_expression("1/"+units)
        
        super().__init__(xs,Px,self.xstr)
        
class TemporalLaserPulseStacking(Dist1d):

    xstr="t" 
    ts = []
    Pt = []

    def __init__(self,lengths=None, angles=None, dv=None, wc=None, pulse_FWHM=None,verbose=0,**params):

        self.verbose=verbose

        vprint("crystal temporal laser shaping",self.verbose>0,0,True)
        
        if(lengths is None):
            lengths=[]
            for key in params:
                if("crystal_length_" in key):
                    lengths.append(params[key])

        if(angles is None):
            angles=[]
            for key in params:
                if("crystal_angle_" in key):
                    angles.append(params[key])
                    
        if(dv is None and "dv" not in params):
            dv=1.05319*unit_registry("ps/mm")
        elif(dv is None):
            dv=params["dv"]
            
        if(wc is None and "wc" not in params):  
            wc=3622.40686*unit_registry("THz")
        elif(wc is None):
            wc=params["wc"]
        
        if(pulse_FWHM is None and "pulse_FWHM" not in params):
            pulse_FWHM=1.8*unit_registry("ps")
        elif(pulse_FWHM is None):
            pulse_FWHM = params["pulse_FWHM"]
        
        self.dV = dv;
        self.w0 = wc;
        self.laser_pulse_FWHM = pulse_FWHM;

        self.crystals = []
        self.total_crystal_length=0     
     
        self.set_crystals(lengths,angles);
        self.propagate_pulses();

        self.ts = self.get_t_pts(10000)
        self.set_pdf()
        self.set_cdf()

    def set_crystals(self,lengths,angles):
    
        assert_with_message(len(lengths)==len(angles), "Number of crystal lengths must be the same as the number of angles.")

        self.lengths=lengths
        self.angles=angles
        self.angle_offsets=np.zeros(len(angles))

        for ii in range(len(lengths)):
            assert_with_message(lengths[ii]>0,"Crystal length must be > 0.")              
            if(ii % 2 ==0):
                angle_offset= -45*unit_registry("deg")
            else:
                angle_offset=   0*unit_registry("deg")
                 
            vprint("crystal "+str(ii+1)+ " length = {:0.3f~P}".format(self.lengths[ii]),self.verbose>0,2,False)
            vprint(", angle = {:0.3f~P}".format(self.angles[ii]),self.verbose>0,0,True)

            self.crystals.append({"length":lengths[ii],"angle":angles[ii],"angle_offset":angle_offset})    

    def propagate_pulses(self):

        self.total_crystal_length = 0;
        self.pulses=[]

        initial_pulse={"intensity":1,"polarization_angle":0*unit_registry("rad"),"relative_delay":0}
        self.pulses.append(initial_pulse)

        for ii in range(len(self.crystals)): 
            vprint("applying crystal: "+str(ii+1),self.verbose>1,3,True);
            self.apply_crystal(self.crystals[ii]);

        #[t_min t_max] is the default range over which to sample rho(u)
        self.t_max = 0.5*self.total_crystal_length*self.dV + 5.0*self.laser_pulse_FWHM;  
        self.t_min = -self.t_max;

        vprint("Pulses propagated: min t = {:0.3f~P}".format(self.t_min) + ", max t = {:0.3f~P}".format(self.t_max),self.verbose>0,2,True) 

    def apply_crystal(self,next_crystal):
        #FUNCTION TO GENERATE TWO PULSES WHEN PASSING THROUGH CRYSTAL:

        #add to total crystal length
        self.total_crystal_length += next_crystal["length"];
        
        theta_fast = next_crystal["angle"]-next_crystal["angle_offset"];  
        theta_slow = theta_fast + 0.5*pi;

        new_pulses = []

        for initial_pulse in self.pulses:

            #the sign convention is chosen so that (-) time represents the head of the electron bunch,
            #and (+) time represents the tail

            #create new pulses:
            pulse_fast={}
            pulse_slow={}

            pulse_fast["intensity"] = initial_pulse["intensity"]*np.cos(initial_pulse["polarization_angle"] - theta_fast);
            pulse_fast["polarization_angle"] = theta_fast;
            pulse_fast["relative_delay"] = initial_pulse["relative_delay"] - self.dV*next_crystal["length"]*0.5;

            pulse_slow["intensity"] = initial_pulse["intensity"]*np.cos(initial_pulse["polarization_angle"] - theta_slow);
            pulse_slow["polarization_angle"] = theta_slow;
            pulse_slow["relative_delay"] = initial_pulse["relative_delay"] + self.dV*next_crystal["length"]*0.5;

            new_pulses.append(pulse_fast);
            new_pulses.append(pulse_slow);

        self.pulses=new_pulses

    def evaluate_sech_fields(self, axis_angle, pulse, t, field):

        #Evaluates the real and imaginary parts of one component of the E-field:
        normalization = pulse["intensity"]*np.cos(pulse["polarization_angle"] - axis_angle);
        w = 2*np.arccosh(np.sqrt(2))/self.laser_pulse_FWHM;

        field[0] = field[0] + normalization*np.cos(self.w0*(t-pulse["relative_delay"])) / np.cosh(w*(t-pulse["relative_delay"]))
        field[1] = field[1] + normalization*np.sin(self.w0*(t-pulse["relative_delay"])) / np.cosh(w*(t-pulse["relative_delay"]))

    def get_t_pts(self,n):
        return linspace(self.t_min,self.t_max.to(self.t_min),n)
  
    def get_x_pts(self,n):
        return self.get_t_pts(n)

    def set_pdf(self):

        ex=np.zeros((2,len(self.ts)))*unit_registry("")
        ey=np.zeros((2,len(self.ts)))*unit_registry("")

        for pulse in self.pulses: 
            self.evaluate_sech_fields(0.5*pi,pulse,self.ts,ex);
            self.evaluate_sech_fields(0.0,   pulse,self.ts,ey);

        self.Pt = ( (ex[0,:]**2 + ex[1,:]**2) + (ey[0,:]**2 + ey[1,:]**2) ).magnitude * unit_registry("THz")
        self.Pt = self.Pt/trapz(self.Pt,self.ts)

    def set_cdf(self):
        self.Ct = cumtrapz(self.Pt,self.ts,initial=0)

    def pdf(self,t):
        return interp(t,self.ts,self.Pt)

    def cdf(self,t):
        return interp(t,self.ts,self.Ct)

    def cdfinv(self,rns):
        return interp(rns*unit_registry(""),self.Ct,self.ts)

    def avg(self):
        return trapz(self.ts*self.Pt,self.ts)

    def std(self):
        return np.sqrt(trapz(self.ts*self.ts*self.Pt,self.ts))

    def get_params_list(self,var):
        return (["crystal_length_$N","crystal_angle_$N"],["laser_pulse_FWHM","avg_"+var,"std_"+var])

class Tukey(Dist1d):

    def __init__(self,var,verbose=0,**kwargs):
        
        self.xstr = var
        
        sigmastr = "sigma_"+var
        if(sigmastr in kwargs.keys()):
            self.sigma = kwargs[sigmastr]
        else:
            self.sigma=1.0
         
        if('ratio' in kwargs.keys()):
            self.r = kwargs['ratio']
        else:
            raise ValueError("Tukey 1D dist required parameter 'ratio' not found in input parameters.")

        if('length' in kwargs.keys()):
            self.L = kwargs['length']
        else:
            raise ValueError("Tukey 1D dist required parameter 'length' not found in input parameters.")

        vprint("Tukey",verbose>0,0,True)
        vprint("legnth = {:0.3f~P}".format(self.L)+", ratio = {:0.3f~P}".format(self.r),verbose>0,2,True)
            
    def get_x_pts(self,n):
        return 1.1*linspace(-self.L/2.0,self.L/2.0,n)

    def pdf(self,x):        
        res = np.zeros(x.shape)

        if(self.r==0):
           flat_region = np.logical_and(x <= self.L/2.0, x >= -self.L/2.0)
           res[flat_region]=1/self.L
       
        else:
            
            Lflat = self.L*(1-self.r)
            Lcos = self.r*self.L/2.0
            pcos_region = np.logical_and(x >= +Lflat/2.0, x<=+self.L/2.0)
            mcos_region = np.logical_and(x <= -Lflat/2.0, x>=-self.L/2.0)
            flat_region = np.logical_and(x < Lflat/2.0, x > -Lflat/2.0)
            res[pcos_region]=0.5*(1+np.cos( (pi/Lcos)*(x[pcos_region]-Lflat/2.0) ))/self.L
            res[mcos_region]=0.5*(1+np.cos( (pi/Lcos)*(x[mcos_region]+Lflat/2.0) ))/self.L
            res[flat_region]=1.0/self.L

            res[x<-self.L]=0 

        res = res*unit_registry('1/'+str(self.L.units))
        
        return res/trapz(res,x)

    def cdf(self,x):
        xpts = self.get_x_pts(10000)
        pdfs = self.pdf(xpts)
        cdfs = cumtrapz(self.pdf(xpts),xpts,initial=0)
        cdfs = cdfs/cdfs[-1]
        cdfs = interp(x,xpts,cdfs)
        cdfs = cdfs/cdfs[-1]
        return cdfs

    def cdfinv(self,p):
        xpts = self.get_x_pts(10000)
        cdfs = self.cdf(xpts)
        return interp(p,cdfs,xpts)
    
    def avg(self):
        xpts = self.get_x_pts(10000)
        return trapz(self.pdf(xpts)*xpts,xpts)

    def std(self):
        xpts = self.get_x_pts(10000)
        avgx=self.avg()
        return np.sqrt(trapz(self.pdf(xpts)*(xpts-avgx)*(xpts-avgx),xpts))

    def rms(self):
        avg=self.avg()
        std=self.std()
        return np.sqrt(std*std + avg*avg)

    
class DistRad():

    rgen = RandGen()

    rs = []    # r pts [0,inf]
    Pr = []    # Probability Distribution Function Pr(r)
    Cr = []    # Cumulative Disrtirbution Functoin Cr(r) = int(r Pr dr), has length(Pr) + 1

    def __init__(self,rs,Pr):

        self.rs = rs
        self.Pr = Pr

        norm = radint(self.Pr,self.rs)
        if(norm<=0):
            raise ValueError("Normalization of PDF was <= 0")
       
        self.Pr = self.Pr/norm
        self.Cr,self.rb = radcumint(self.Pr,self.rs)
        
    def get_r_pts(self,n):
        return linspace(self.rs[0],self.rs[-1],n)

    def pdf(self,r):
        return interp(r,self.rs,self.rs*self.Pr)

    def cdf(self,r):
        return interp(r**2,self.rs**2,self.Cr)

    def cdfinv(self,rns):

        rns=np.squeeze(rns)
        indsL = np.searchsorted(self.Cr,rns)-1
        indsH = indsL+1

        c1 = self.Cr[indsL]
        c2 = self.Cr[indsH]

        r1 = self.rb[indsL]
        r2 = self.rb[indsH]

        same_rs = ( (r1.magnitude)==(r2.magnitude))
        diff_rs = np.logical_not(same_rs)        

        r = np.zeros(rns.shape)*unit_registry(str(self.rs.units))  

        r[same_rs]=r1[same_rs]
        r[diff_rs] = np.sqrt( ( r2[diff_rs]*r2[diff_rs]*(rns[diff_rs]-c1[diff_rs]) + r1[diff_rs]*r1[diff_rs]*(c2[diff_rs]-rns[diff_rs]))/(c2[diff_rs]-c1[diff_rs]) )

        return r
    
    def sample(self,N,sequence=None,params=None):
        return self.cdfinv(self.rgen.rand( (N,1),sequence,params)*unit_registry("dimensionless"))

    def plot_pdf(self,n=1000):
        r=self.get_r_pts(n)
        plt.figure()
        plt.plot(r,self.pdf(r))
        plt.xlabel("r")
        plt.ylabel("PDF(r)")

    def plot_cdf(self,n=1000):
        r=self.get_r_pts(n)
        plt.figure()
        plt.plot(r,self.cdf(r))
        plt.xlabel("r")
        plt.ylabel("CDF(r)")

    def avg(self):
        return np.sum( ((self.rb[1:]**3 - self.rb[:-1]**3)/3.0)*self.Pr ) 

    def rms(self):
        return np.sqrt( np.sum( ((self.rb[1:]**4 - self.rb[:-1]**4)/4.0)*self.Pr ) )

    def std(self):
        avg=self.avg()
        rms=self.rms()
        return np.sqrt(rms*rms-avg*avg)

    def test_sampling(self):
     
        rs=self.sample(100000,sequence="hammersley")    
        r = self.get_r_pts(1000)
        pdf = self.pdf(r)

        rho,edges = histogram(rs,bins=100)
        rhist = (edges[1:] + edges[:-1]) / 2
        rho = rho/np.trapz(rho,rhist)

        avgr = rs.mean()
        avgr_str = "{:0.3f~P}".format(avgr)
        stdr = rs.std()
        stdr_str = "{:0.3f~P}".format(stdr)
        
        davgr = self.avg()
        dstdr = self.std()

        davgr_str = "{:0.3f~P}".format(davgr)
        dstdr_str = "{:0.3f~P}".format(dstdr)  

        plt.figure()
        plt.plot(r, pdf/r, rhist, rho/rhist, 'or')
        plt.xlabel("r")
        plt.ylabel("PDF(r)/r")
        plt.title("Sample stats: <r> = "+avgr_str+", $\sigma_r$ = "+stdr_str+"\nDist. stats: <r> = "+davgr_str+", $\sigma_r$ = "+dstdr_str)
        plt.legend(["PDF/r","Hist. of Sampling/r"])
  
        plt.show()

        
class UniformRad(DistRad):
 
    rL=0
    rR=0

    def __init__(self,verbose=0,**kwargs):
            
        maxstr = "max_r"
        if(maxstr in kwargs.keys()):
            self.rR = kwargs[maxstr]
        else:
            raise ValueError("Radial uniform dist required parameter "+maxstr+" not found in input parameters.")
        
        minstr = "min_r"
        if(minstr in kwargs.keys()):
            self.rL = kwargs[minstr]
        else:
            self.rL=0*unit_registry(str(self.rR.units))
        
        if(self.rL>=self.rR):
            raise ValueError("Radial uniform dist must have rL < rR")
        if(self.rR<0):
            raise ValueError("Radial uniform dist must have rR >= 0")
        
        vprint("radial uniform",verbose>0,0,True)
        vprint(minstr+" = {:0.3f~P}".format(self.rL)+", "+maxstr+" = {:0.3f~P}".format(self.rR),verbose>0,2,True)

    def get_r_pts(self,n):
        f = 0.2
        dr = f*np.abs(self.avg())
        return np.linspace(self.rL-dr,self.rR+dr,n)

    def avg(self):
        return (2.0/3.0)*(self.rR**3 - self.rL**3)/(self.rR**2-self.rL**2)

    def rms(self):
        return np.sqrt( (self.rR**2 + self.rL**2)/2.0 )

    def pdf(self,r):
        nonzero = (r >= self.rL) & (r <= self.rR)
        res = np.zeros(len(r))
        res[nonzero]=r[nonzero]*2.0/(self.rR**2-self.rL**2)
        return res

    def cdfinv(self,rns):
        return np.sqrt( self.rL**2 + (self.rR**2 - self.rL**2)*rns) 

class NormRad(DistRad):

    sigma = 0 

    def __init__(self,verbose=0,**kwargs):
        
        if("sigma_xy" in kwargs):
            self.sigma = kwargs["sigma_xy"]
        else:
            raise ValueError("Radial Gaussian required parameter sigma_xy not found.")
         
        vprint("radial Gaussian",verbose>0,0,True)
        vprint("sigma_xy = {:0.3f~P}".format(self.sigma),verbose>0,2,True)

    def pdf(self,r):
        return (1/self.sigma/self.sigma)*np.exp(-r*r/2.0/self.sigma/self.sigma )*r 

    def cdf(self,r):
        return 1 - np.exp(-r*r/2.0/self.sigma/self.sigma)

    def get_r_pts(self,n=1000):
        return np.linspace(0,+5*self.sigma.magnitude,n)*unit_registry(str(self.sigma.units))
        
    def avg(self):
        return math.sqrt(math.pi/2)*self.sigma

    def rms(self):
        return np.sqrt(2)*self.sigma

    def cdfinv(self,rns):
        return self.sigma*np.sqrt(-2*np.log(1-rns))

class NormRadTrunc(DistRad):

    f = 0
    R = 0
    sigma_inf = 0
    
    def __init__(self,radius=None,fraction=None,**params):
    
        if(radius is None and "pinhole_size" not in params):
            raise ValueError("Radial truncated Gaussian requires either a radius or pinhole size as input parameter.")
        elif(radius is None):
            radius = params["pinhole_size"]/2.0
          
        if(fraction is None and "fraction" not in params):
            raise ValueError("Radial truncated Gaussian input parameter 'fraction' not found.")
        elif(radius is None):
            fraction = params["fraction"]  
        
        if(radius<=0):
            raise ValueError("For truncated gaussian, radius has to be > 0")
        if(fraction > 1 or fraction < 0):
            raise ValueErorr("For truncated gaussian, fraction must satisfy: f > 0 and f < 1")

        self.f = fraction
        self.R = radius
        self.sigma_inf = radius/np.sqrt(2)/np.sqrt(np.log(1/fraction))

    def pdf(self,r):
        res = np.zeros(len(r))
        nonzero = r<=self.R 
        res[nonzero]=1/self.sigma_inf**2/(1-self.f)*np.exp(-r[nonzero]*r[nonzero]/2/self.sigma_inf**2)*r[nonzero]
        return res

    def cdf(self,r):
        return (1-np.exp(-r*r/2/self.sigma_inf**2))/(1-self.f)

    def cdfinv(self,rns):
        return np.sqrt( 2*self.sigma_inf**2 * ( np.log(1/(rns*(self.f-1)+1)) )) 

    def get_r_pts(self,n=1000):
        return np.linspace(0,1.2*self.R.magnitude,n)*unit_registry(str(self.R.units))

    def avg(self):
        return (self.sigma_inf*math.sqrt(math.pi/2)*erf(self.R/np.sqrt(2)/self.sigma_inf)-self.R*self.f)/(1-self.f)

    def rms(self):
        return np.sqrt( 2*self.sigma_inf**2 - self.R**2 * self.f/(1-self.f) )

class RadFile(DistRad):

    def __init__(self,**params):

        if("file" not in params):
            raise ValueError("Radial distribution file required input parameter 'distribution file' not found.")
        if("units" not in params):
            raise ValueError("Radial distribution file required input parameter 'units' not found.")    
        
        distfile = params["file"]
        units = params["units"]
        
        self.distfile = distfile
        f = open(distfile,'r')
        headers = f.readline().split()
        f.close()

        if(len(headers)!=2):
            raise ValueError("radial distribution file must have two columns")
        data = np.loadtxt(distfile,skiprows=1)

        rs = data[:,0]*unit_registry(units)
        Pr = data[:,1]*unit_registry.parse_expression("1/"+units+"/"+units)
      
        if(np.count_nonzero(rs < 0 )):
            raise ValueError("Radial distribution r-values must be >= 0.")
       
        super().__init__(rs,Pr)

class TukeyRad(DistRad):

    def __init__(self,verbose=0,**kwargs):
         
        if('ratio' in kwargs.keys()):
            self.r = kwargs['ratio']
        else:
            raise ValueError("TukeyRad dist required parameter 'ratio' not found in input parameters.")

        if('length' in kwargs.keys()):
            self.L = kwargs['length']
        else:
            raise ValueError("TukeyRad dist required parameter 'length' not found in input parameters.")

        vprint("TukeyRad",verbose>0,0,True)
        vprint("legnth = {:0.3f~P}".format(self.L)+", ratio = {:0.3f~P}".format(self.r),verbose>0,2,True)

    def get_r_pts(self,n=1000):
        return np.linspace(0,1.2*self.L.magnitude,n)*unit_registry(str(self.L.units))

    def pdf(self,r):        
        res = np.zeros(r.shape)

        if(self.r==0):
           flat_region = np.logical_and(r <= self.L, x >= 0.0)
           res[flat_region]=1.0
       
        else:
            
            Lflat = self.L*(1-self.r)
            Lcos = self.r*self.L
            cos_region = np.logical_and(r >= +Lflat, r <=+self.L)
            flat_region = np.logical_and(r < Lflat, r >= 0)
            res[cos_region]=0.5*(1+np.cos( (pi/Lcos)*(r[cos_region]-Lflat) ))
            res[flat_region]=1.0/self.L
        
        res = res*r
        res = res*unit_registry('1/'+str(self.L.units))
        
        return res/radint(res,r)

    def pdfr(self,r):

        res = np.zeros(r.shape)

        if(self.r==0):
           flat_region = np.logical_and(r <= self.L, x >= 0.0)
           res[flat_region]=1.0
       
        else:
            
            Lflat = self.L*(1-self.r)
            Lcos = self.r*self.L
            cos_region = np.logical_and(r >= +Lflat, r <=+self.L)
            flat_region = np.logical_and(r < Lflat, r >= 0)
            res[cos_region]=0.5*(1+np.cos( (pi/Lcos)*(r[cos_region]-Lflat) ))
            res[flat_region]=1.0
        
        #res = res*r
        res = res*unit_registry('1/'+str(self.L.units))
        
        return res/radint(res,r)
   

    def cdf(self,r):
        rpts = self.get_r_pts(10000)
        pdfs = self.pdfr(rpts)
        
        cdfs,rbins = radcumint(pdfs,rpts,initial=0)
        cdfs = cdfs/cdfs[-1]
        cdfs = interp(r,rbins,cdfs)
        cdfs = cdfs/cdfs[-1]
        return cdfs

    def cdfinv(self,p):
        rpts = self.get_r_pts(10000)
        cdfs = self.cdf(rpts)
        return interp(p,cdfs,rpts)

    def avg(self):
        rpts = self.get_r_pts(10000)
        pdfs = self.pdfr(rpts)
        return radint(pdfs*rpts,rpts)

    def rms(self):
        rpts = self.get_r_pts(10000)
        pdfs = self.pdfr(rpts)
        return np.sqrt(radint(pdfs*rpts*rpts,rpts))

class Dist2d():

    xstr = ""
    ystr = ""
    xs = []    # x pts
    ys = []
    Pxy = []    # Probability Distribution Function P(x,y)
    
    rgen = RandGen()

    def __init__(self,xs,ys,Pxy,xstr="x",ystr="y"):

        self.xs=xs
        self.ys=ys
        self.Pxy = Pxy
        self.xstr=xstr
        self.ystr=ystr
    
        self.xb = np.zeros(len(self.xs.magnitude)+1)*unit_registry(str(self.xs.units))
        self.xb[1:-1] = (self.xs[1:]+self.xs[:-1])/2.0

        dxL = self.xb[+1]-self.xs[+0]
        dxR = self.xs[-1]-self.xb[-2]

        self.xb[+0] = self.xs[+0]-dxL
        self.xb[-1] = self.xs[-1]+dxR
        
        self.yb = np.zeros(len(ys)+1)*unit_registry(str(self.ys.units))
        self.yb[1:-1] = (self.ys[1:]+self.ys[:-1])/2.0

        dyL = self.yb[+1]-self.ys[+0]
        dyR = self.ys[-1]-self.yb[-2]

        self.yb[+0] = self.ys[+0]-dyL
        self.yb[-1] = self.ys[-1]+dyR

        # Integrate out y to get rho(x) = int(rho(x,y)dy)
        self.dx = self.xb[1:]-self.xb[:-1] 
        self.dy = self.yb[1:]-self.yb[:-1] 

        self.Px = np.matmul(np.transpose(Pxy.magnitude),self.dy.magnitude)*unit_registry("1/"+str(self.ys.units))
        self.Px = self.Px/np.sum(self.Px*self.dx)
        
        self.Cx = np.zeros(len(self.xb))*unit_registry("dimensionless")
        self.Cx[1:] = np.cumsum(self.Px*self.dx)

        # Get cumulative distributions along y as a function of x:
        self.Cys=np.zeros((len(self.yb),len(self.xs)))

        norms = np.sum(np.multiply(self.Pxy.magnitude,np.transpose(mlib.repmat(self.dy.magnitude,len(self.xs),1))), axis=0)

        self.Cys[1:,:] = np.cumsum(np.multiply(self.Pxy.magnitude,np.transpose(mlib.repmat(self.dy.magnitude,len(self.xs),1))),axis=0)/norms
        self.Cys=self.Cys*unit_registry("dimensionless")

    def pdf(self,x,y):
        x=0
   
    def plot_pdf(self):
        plt.figure()
        extent = [(self.xs.min()).magnitude,(self.xs.max()).magnitude,(self.ys.min()).magnitude,(self.ys.max()).magnitude]
        plt.imshow(self.Pxy,extent=extent)
        plt.xlabel(self.xstr+" ("+str(self.xs.units)+")")
        plt.ylabel(self.ystr+" ("+str(self.ys.units)+")")
      
    def pdfx(self,x):
        return interp(x,self.xs,self.Px)

    def plot_pdfx(self):
        plt.figure()
        plt.plot(self.xs,self.Px)

    def cdfx(self,x):
        return interp(x,self.xb,self.Cx)

    def plot_cdfx(self):
        plt.figure()
        plt.plot(self.xb,self.Cx)    

    def cdfxinv(self,ps):
        return interp(ps,self.Cx,self.xb)

    def plot_cdfys(self):
        plt.figure()
        for ii in range(len(self.xs)):
            plt.plot(self.yb,self.Cys[:,ii])    

    def sample(self,N,sequence=None,params=None):
        rns = self.rgen.rand((N,2),sequence,params)*unit_registry("dimensionless")
        x,y = self.cdfinv(rns)       
        return (x,y)

    def cdfinv(self,rns):

        x = self.cdfxinv(rns[0,:])
        indx = np.searchsorted(self.xb,x)-1
        
        y = np.zeros(x.shape)*unit_registry(str(x.units))
        for ii in range(self.Cys.shape[1]):
            in_column = (ii==indx)
            if(np.count_nonzero(in_column)>0):
                y[in_column] = interp(rns[1,in_column],self.Cys[:,ii],self.yb)

        return (x,y)

    def test_sampling(self):
        x,y = self.sample(100000,sequence="hammersley")
        #plt.plot(x,y,'*')
        #plt.figure()
        #xhist,xedges=np.histogram(x,bins=100)
        #xhistx = (xedges[1:]+xedges[:-1])/2.0
        #plt.plot(xhistx,xhist)
 
        plt.figure()
        plt.plot(x,y,'*')

class File2d(Dist2d):

    def __init__(self, var1, var2, filename=None,**params):

        
        
        if(filename is None and "file" not in params):
            raise ValueError("File 2D distribution requires an input file.")
        elif(filename is None):
            filename = params['file']
        
        xs,ys,Pxy,xstr,ystr = read_2d_file(filename)
        super().__init__(xs,ys,Pxy,xstr=xstr,ystr=ystr)

#def test():

    #tukey = Tukey1D('x')#,'sigma_x'=1*unit_registry('mm'),'tukey_window_ratio'=0*unit_registry('dimensionless'))
    #tukwy.test_sampling()

    #udist = uniform(-2*unit_registry("m"),1*unit_registry("m"))
    #udist.test_sampling()

    #ndist = norm(-2*unit_registry("mm"),3*unit_registry("mm"))
    #ndist.test_sampling()

    #urad = uniformrad(1*unit_registry("mm"),2*unit_registry("mm"))
    #urad.test_sampling()

    #nrad = normrad(3*unit_registry("mm"))
    #nrad.test_sampling()

    #tnrad = normrad_trunc(1*unit_registry("mm"),0.5*unit_registry("dimensionless"))
    #tnrad.test_sampling()

    #lengths = 1.887*np.array([8,4,2,1])*unit_registry("mm")
    #angles = [0.6,1.8,-0.9,-0.5]*unit_registry("deg")

    #cdist = temporal_laser_pulse_stacking(lengths,angles,verbose=1)
    #cdist.test_sampling()

    #fdist = file1d("cutgauss.1d.txt",xstr="x")
    #fdist.test_sampling()
   
    #rfile = radfile("cutgauss.rad.txt",units="mm")
    #rfile.test_sampling()

    #f2d = file2d('checker.test.dist.txt')
    #f2d = file2d('laser.prof.example.txt')
    #f2d.plot_cdfys()
    #f2d.plot_pdfx()
    #f2d.plot_cdfx()
    #f2d.test_sampling()
    #f2d.plot_pdf()
    #x,y=f2d.sample(100000,sequence="hammersley")
    #plt.plot(x,y,'*')

    #plt.show()

    
# ---------------------------------------------------------------------------- 
#   This allows the main function to be at the beginning of the file
# ---------------------------------------------------------------------------- 
#if __name__ == '__main__':
#    test()



