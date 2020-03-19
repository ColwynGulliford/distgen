"""
Defines the random number generator class as well all distribution function objects.
"""

from .hammersley import create_hammersley_samples
from .physical_constants import *
from .tools import *

import numpy as np
import numpy.matlib as mlib
#import scipy
import math

from matplotlib import pyplot as plt

def random_generator(shape,sequence=None,params=None):

    if(sequence is None):
        return np.random.random(shape)

    elif(sequence=="hammersley"):
        
        dim = shape[0]
        N = shape[1] 

        if(params is None):
            return np.squeeze(create_hammersley_samples(N, dim=dim, burnin=-1, primes=()))
        else:
            return np.squeeze(create_hammersley_samples(N, dim=dim, burnin=params["burnin"], primes=params["primes"]))
    else:
        raise ValueError("Sequence: "+str(sequence)+" is not supported")

def get_dist(var,params,verbose=0):
    """
    Translates user input strings and evaluated corrector corresponding distribution function.
    Inputs: var [str] name of variable (x,y,px,...,etc) for distribution,
            dtype [str] user string or shorthand for distribution function
            params [dict] required user parameters for distribution function
            verbose [bool] flag for more or less output to screen
    """
    assert 'type' in params, 'No distribution type for '+var+' specified.'
    dtype = params['type']

    if(dtype=="uniform" or dtype=="u"):
        dist = Uniform(var,verbose=verbose,**params)
    elif(dtype=="gaussian" or dtype=="g"):
        dist = Norm(var,verbose=verbose,**params)
    elif(dtype=="file1d"):
        dist = File1d(var,verbose=verbose,**params)
    elif(dtype=='tukey'):
        dist = Tukey(var,verbose=verbose,**params)
    elif(dtype=='super_gaussian' or dtype=='sg'):
        dist = SuperGaussian(var,verbose=verbose,**params)
    elif((dtype=="radial_uniform" or dtype=="ru") and var=="r"):
        dist = UniformRad(verbose=verbose,**params)
    elif((dtype=="radial_gaussian" or dtype=="rg") and var=="r"):
        dist = NormRad(verbose=verbose,**params)
    elif((dtype=="radial_super_gaussian" or dtype=="rsg") and var=="r"):
        dist = SuperGaussianRad(verbose=verbose,**params)
    elif(dtype=="radfile" and var=="r"):
        dist = RadFile(verbose=verbose,**params)
    elif(dtype=="radial_tukey"):
        dist = TukeyRad(verbose=verbose,**params)
    elif(dtype=="file2d"):
        dist = File2d("x","y",verbose=verbose,**params)
    elif(dtype=="crystals"):
        dist = TemporalLaserPulseStacking(verbose=verbose,**params)
    else:
        raise ValueError("Distribution type '"+dtype+"' is not supported.")
            
    return dist


class Dist():

    def __init__(self):
        pass

    def check_inputs(self,params):

        # Make sure user isn't passing the wrong parameters:
        allowed_params = self.optional_params + self.required_params + ['verbose','type']
        for param in params:
            assert param in allowed_params, 'Incorrect param given to '+self.__class__.__name__+ '.__init__(**kwargs): '+param+'\nAllowed params: '+str(allowed_params)

        # Make sure all required parameters are specified
        for req in self.required_params:
            assert req in params, 'Required input parameter '+req+' to '+self.__class__.__name__+'.__init__(**kwargs) was not found.'
    
class Dist1d(Dist):

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
    
    #rgen = RandGen()
    
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
        return self.cdfinv( random_generator((1,N),sequence,params)*unit_registry("dimensionless") )

    def plot_pdf(self,n=1000):
        """
        Plots the associated pdf function sampled with n points
        """
        x=self.get_x_pts(n)
        p=self.pdf(x)
        plt.figure()
        plt.plot(x,self.pdf(x))
        plt.xlabel(self.xstr+" ({:~P}".format(x.units)+")")
        plt.ylabel("PDF("+self.xstr+") ({:~P}".format(p.units)+")")

    def plot_cdf(self,n=1000):
        """ 
        Plots the associtated cdf function sampled with n points
        """
        x=self.get_x_pts(n)
        plt.figure()
        plt.plot(x,self.cdf(x))
        plt.xlabel(self.xstr+" ({:~P}".format(x.units)+")")
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

        """
        Useful for verifying the distribution object works correctly when sampling.

        Plots the pdf, cdf, and histogram of 10000 samples from the PDF.
        """
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
        plt.legend(["PDF","Normalized Sampling"])    

class Uniform(Dist1d):

    """
    Implements a the uniform 1d distribution over a range a <= x <= b.


    """

    def __init__(self,var,verbose=0,**kwargs):
        
        """
        Sets the required parameters for the 1d uniform dist:
        var [str] =  the name of the distribution variable
        verbose [int] controls the level of string output to the terminal
        **kwargs [dict] provides all other input parameters.  
        The class requires physical parameters with keys "min_{var}" and "max_{var}" 
        in order to set the range for the distribution.

        """

        self.xstr = var
        minstr = "min_"+var
        maxstr = "max_"+var
        
        self.required_params = [minstr, maxstr]
        self.optional_params = []

        self.check_inputs(kwargs)
       
        self.xL = kwargs[minstr]           
        self.xR = kwargs[maxstr]
        vprint("uniform",verbose>0,0,True)
        vprint(minstr+" = {:0.3f~P}".format(self.xL)+", "+maxstr+" = {:0.3f~P}".format(self.xR),verbose>0,2,True)
  
    def get_x_pts(self,n):
        """
        Returns n equally spaced x values that sample just over the relevant range of [a,b] (DOES NOT SAMPLE DISTRIBUTION)
        Inputs: n [int]
        """
        f = 0.2
        dx = f*np.abs(self.avg())
        return np.linspace(self.xL-dx,self.xR+dx,n)

    def pdf(self,x):
        """
        Returns the PDF at the values in x [array w/units].  PDF has units of 1/[x]
        """
        nonzero = (x >= self.xL) & (x <= self.xR)
        res = np.zeros(len(x))*unit_registry('1/'+str(self.xL.units))
        res[nonzero]=1/(self.xR-self.xL)
        return res

    def cdf(self,x):
        """
        Returns the CDF at the values of x [array w/units].  CDF is dimensionless
        """
        nonzero = (x >= self.xL) & (x <= self.xR)
        res = np.zeros(len(x))
        res[nonzero]=(x[nonzero]-self.xL)/(self.xR-self.xL)*unit_registry('dimensionless')
       
        return res

    def cdfinv(self,rns):
        """
        Returns the inverse of the CDF function for probabilies rns [array], providing a sampling of the PDF.
        """
        return (self.xR-self.xL)*rns + self.xL

    def avg(self):
        """
        Returns the first moment of the PDF: <x> = (a + b)/2
        """
        return 0.5*(self.xR+self.xL)

    def std(self):
        """
        Returns the square root of the variance of the PDF: <x> = (b-a)/sqrt(12)
        """
        return (self.xR-self.xL)/np.sqrt(12) 
  
    def rms(self):
        """
        Returns the rms of the distribution computed from the avg and std.
        """
        avg=self.avg()
        std=self.std()
        return np.sqrt(std*std + avg*avg)
    
class Norm(Dist1d):

    def __init__(self,var,verbose=0,**kwargs):
        

        self.type='Norm'
        self.xstr = var

        sigmastr = "sigma_"+var
        self.required_params=[sigmastr]

        sigma_cutoff_str   = "n_sigma_cutoff"
        sigma_cutoff_left  = "n_sigma_cutoff_left"
        sigma_cutoff_right = "n_sigma_cutoff_right"
        avgstr = "avg_"+var
        self.optional_params=[sigma_cutoff_str,sigma_cutoff_left,sigma_cutoff_right,avgstr]

        self.check_inputs(kwargs)

        self.sigma = kwargs[sigmastr]
            
        if(avgstr in kwargs.keys()):
            self.mu = kwargs[avgstr]
        else:
            self.mu = 0*unit_registry(str(self.sigma.units))

        left_cut_set = False
        right_cut_set = False

        assert not (sigma_cutoff_str in kwargs.keys() and (sigma_cutoff_left in kwargs.keys() or sigma_cutoff_right in kwargs.keys()) )

        if(sigma_cutoff_str in kwargs.keys()):

            self.a = -kwargs[sigma_cutoff_str]*self.sigma
            self.b = +kwargs[sigma_cutoff_str]*self.sigma

            left_cut_set = True
            right_cut_set = True

        if(sigma_cutoff_left in kwargs.keys()):

            self.a = kwargs[sigma_cutoff_left]*self.sigma
            left_cut_set = True

        if(sigma_cutoff_right in kwargs.keys()):

            self.b = kwargs[sigma_cutoff_right]*self.sigma
            right_cut_set = True 

        if(not left_cut_set):
            self.a = -float('Inf')*unit_registry(str(self.sigma.units))

        if(not right_cut_set):
            self.b = +float('Inf')*unit_registry(str(self.sigma.units))

        assert self.a < self.b, 'Right side cut off a = {:0.3f~P}'.format(self.a) + ' must be < left side cut off b = {:0.3f~P}'.format(self.b)

        self.A = (self.a - self.mu)/self.sigma
        self.B = (self.b - self.mu)/self.sigma
        
        self.pA = self.canonical_pdf(self.A)
        self.pB = self.canonical_pdf(self.B)

        self.PA = self.canonical_cdf(self.A)
        self.PB = self.canonical_cdf(self.B)

        self.Z = self.PB - self.PA

        vprint("Gaussian",verbose>0,0,True)
        vprint("avg_"+var+" = {:0.3f~P}".format(self.mu)+", sigma_"+var+" = {:0.3f~P}".format(self.sigma),verbose>0,2,True)
        vprint("Left n_sigma_cutoff = {:0.3f~P}".format(self.b/self.sigma)+', Right n_sigma_cutoff = {:0.3f~P}'.format(self.a/self.sigma),verbose>0 and self.b.magnitude<float('Inf'),2,True)
            
    def get_x_pts(self,n):
        return self.mu + linspace(-5*self.sigma,+5*self.sigma,1000)

    def canonical_pdf(self,csi):
        return (1/np.sqrt(2*math.pi))*np.exp( -csi**2/2.0 ) 

    def pdf(self,x):        
        csi = (x-self.mu)/self.sigma
        res = self.canonical_pdf(csi)/self.Z/self.sigma
        x_out_of_range = (x<self.a) | (x>self.b)
        res[x_out_of_range] = 0*unit_registry('1/'+str(self.sigma.units))
        return res

    def canonical_cdf(self,csi):
        return 0.5*(1+erf(csi/math.sqrt(2) ) )

    def cdf(self,x):
        csi = (x-self.mu)/self.sigma
        res = (self.canonical_cdf(csi) - self.PA)/self.Z
        x_out_of_range = (x<self.a) | (x>self.b)
        res[x_out_of_range] = 0*unit_registry('dimensionless')
        return res

    def canonical_cdfinv(self,rns):
        return math.sqrt(2)*erfinv((2*rns-1))

    def cdfinv(self,rns):
        scaled_rns = rns*self.Z + self.PA
        return self.mu + self.sigma*self.canonical_cdfinv(scaled_rns)

    def avg(self):
        return self.mu + self.sigma*(self.pA - self.pB)/self.Z 
    
    def std(self):

        if(self.A.magnitude == -float('Inf')):
            ApA = 0*unit_registry('dimensionless')
        else:
            ApA = self.A*self.pA

        if(self.B.magnitude == +float('Inf')):
            BpB = 0*unit_registry('dimensionless')
        else:
            BpB = self.B*self.pB
       
        return self.sigma*np.sqrt( 1 + (ApA - BpB)/self.Z - ((self.pA - self.pB)/self.Z)**2 )

    def rms(self):
        avg=self.avg()
        std=self.std()
        return np.sqrt(std*std + avg*avg)

class SuperGaussian(Dist1d):

    def __init__(self,var,verbose=0,**kwargs):

        self.type='Norm'
        self.xstr = var

        lambda_str = 'lambda'
        power_str = 'p'
        alpha_str = 'alpha'
        avg_str = 'avg_'+var

        self.required_params=[lambda_str]
        self.optional_params=[avg_str,power_str,alpha_str]
        self.check_inputs(kwargs)

        assert not (alpha_str in kwargs and power_str in kwargs), 'SuperGaussian power parameter must be set using "p" or "alpha", not both.' 
        assert (alpha_str in kwargs or power_str in kwargs), 'SuperGaussian power parameter must be set using "p" or "alpha". Neither provided.' 

        self.Lambda = kwargs[lambda_str]
        if(power_str in kwargs):
            self.p = kwargs[power_str]
        elif(alpha_str):
            alpha = kwargs[alpha_str]
            assert alpha >= 0 and alpha <= 1, 'SugerGaussian parameter must satisfy 0 <= alpha <= 1, not = '+str(alpha)
            if(alpha.magnitude==0): 
                self.p = float('Inf')*unit_registry('dimensionless')
            else:
                self.p = 1/alpha 

        if(avg_str in kwargs):
            self.mu = kwargs[avg_str]
        else:
            self.mu = 0*unit_registry(str(self.Lambda.units))
 
    def pdf(self,x=None):  

        if(x is None):
            x=self.get_x_pts(10000)
      
        xi = (x-self.mu)/self.Lambda
        nu1 = 0.5*xi**2
        rho = np.exp(-np.float_power(nu1.magnitude,self.p.magnitude))*unit_registry('1/'+str(self.Lambda.units))

        return rho/trapz(rho,x)
        
    def get_x_pts(self,n):
        return self.mu + linspace(-5*self.Lambda, +5*self.Lambda,1000)

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

    
class File1d(Dist1d):
    
    def __init__(self,var,verbose=0,**kwargs):
        

        self.required_params = ['file','units']
        self.optional_params = []
        self.check_inputs(kwargs)

        self.xstr=var
        
        self.distfile = kwargs["file"]
        self.units = kwargs["units"]
        
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

        xs = data[:,0]*unit_registry(self.units)
        Px = data[:,1]*unit_registry.parse_expression("1/"+self.units)
        
        assert np.count_nonzero(xs.magnitude) > 0, 'Supplied 1d distribution coordinate vector '+var+' is zero everywhere.'
        assert np.count_nonzero(Px.magnitude) > 0, 'Supplied 1d distribution P'+var+' is zero everywhere.'

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

        for param in params:
            assert 'crystal_angle_' in param or 'crystal_length' in param or param=='type', 'Unknown keyword parameter sent to '+self.__class__.__name__+': '+param
                    
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
         
        self.required_params = ['ratio','length']
        self.optional_params = []
        self.check_inputs(kwargs)
            
        self.r = kwargs['ratio']
        self.L = kwargs['length']

        vprint("Tukey",verbose>0,0,True)
        vprint("legnth = {:0.3f~P}".format(self.L)+", ratio = {:0.3f~P}".format(self.r),verbose>0,2,True)
            
    def get_x_pts(self,n):
        return 1.1*linspace(-self.L/2.0,self.L/2.0,n)

    def pdf(self,x):        
        res = np.zeros(x.shape)*unit_registry('1/'+str(self.L.units))

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

            res[x<-self.L]=0*unit_registry('1/'+str(self.L.units))
        
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

    
class DistRad(Dist):

    #rs = []    # r pts [0,inf]
    #Pr = []    # Probability Distribution Function Pr(r)
    #Cr = []    # Cumulative Disrtirbution Functoin Cr(r) = int(r Pr dr), has length(Pr) + 1

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

    def rho(self,r):
        return interp(r,self.rs,self.Pr)

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
        return self.cdfinv(random_generator( (1,N),sequence,params)*unit_registry("dimensionless"))

    def plot_pdf(self,n=1000):

        r=self.get_r_pts(n)
        p = self.rho(r)
        P = self.pdf(r)
        fig, (a1,a2) = plt.subplots(1,2)
        #print(a1,a2)
        #print(r)
        a1.plot(r,p)
        a1.set(xlabel='r ({:~P})'.format(r.units))
        a1.set(ylabel='2$\pi\\rho$(r) ({:~P})'.format(p.units))
        
        a2.plot(r,P)
        a2.set(xlabel='r ({:~P})'.format(r.units))
        a2.set(ylabel='PDF(r) ({:~P})'.format(P.units))

        plt.tight_layout()

    def plot_cdf(self,n=1000):
        r=self.get_r_pts(n)
        plt.figure()
        plt.plot(r,self.cdf(r))
        plt.xlabel('r ({:~P})'.format(r.units))
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
        p = self.rho(r)
        P = self.pdf(r)
        
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
        plt.plot(r, P/r, rhist, rho/rhist, 'or')
        plt.xlabel("r ({:~P})".format(r.units))
        plt.ylabel("2$\pi\\rho$(r) ({:~P})".format(p.units))
        plt.title("Sample stats: <r> = "+avgr_str+", $\sigma_r$ = "+stdr_str+"\nDist. stats: <r> = "+davgr_str+", $\sigma_r$ = "+dstdr_str)
        plt.legend(["2$\pi\\rho$(r)","Normalized Sampling"])
  
        plt.show()

class UniformRad(DistRad):

    """
    Implements a uniform (constant) distribution between 0 <= min_r < r <= max_r.  

    Typical use example in YAML format:

    r_dist: 
    type: uniform
    params: 
        min_r: 
            value: 1
            units: mm
        max_t:
            value: 2
            units: ps
    """

    def __init__(self,verbose=0,**kwargs):
            
        maxstr = "max_r"
        minstr = "min_r"

        self.required_params=[maxstr]
        self.optional_params=[minstr]
        self.check_inputs(kwargs)

        self.rR = kwargs[maxstr]

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
        res = res*unit_registry('1/'+str(r.units))
        return res

    def rho(self,r):
        nonzero = (r >= self.rL) & (r <= self.rR)
        res = np.zeros(len(r))
        res[nonzero]=2/(self.rR**2-self.rL**2)
        res = res*unit_registry('1/'+str(r.units)+'/'+str(r.units))
        return res

    def cdf(self,r):
        nonzero = (r >= self.rL) & (r <= self.rR)
        res = np.zeros(len(r))
        res[nonzero]=(r[nonzero]*r[nonzero] - self.rL**2)/(self.rR**2-self.rL**2)
        res = res*unit_registry('dimensionless')
        return res

    def cdfinv(self,rns):
        return np.sqrt( self.rL**2 + (self.rR**2 - self.rL**2)*rns) 

#class NormRad(DistRad):

#    sigma = 0 

#    def __init__(self,verbose=0,**kwargs):
        
#        if("sigma_xy" in kwargs):
#            self.sigma = kwargs["sigma_xy"]
#        else:
#            raise ValueError("Radial Gaussian required parameter sigma_xy not found.")
#         
#        vprint("radial Gaussian",verbose>0,0,True)
#        vprint("sigma_xy = {:0.3f~P}".format(self.sigma),verbose>0,2,True)

#    def pdf(self,r):
#        return (1/self.sigma/self.sigma)*np.exp(-r*r/2.0/self.sigma/self.sigma )*r 

#    def rho(self,r):

#        return (1/self.sigma/self.sigma)*np.exp(-r*r/2.0/self.sigma/self.sigma ) 
#
#   def cdf(self,r):
#        return 1 - np.exp(-r*r/2.0/self.sigma/self.sigma)

#    def get_r_pts(self,n=1000):
#        return np.linspace(0,+5*self.sigma.magnitude,n)*unit_registry(str(self.sigma.units))
        
#    def avg(self):
#        return math.sqrt(math.pi/2)*self.sigma

#    def rms(self):
#        return np.sqrt(2)*self.sigma

#    def cdfinv(self,rns):
#        return self.sigma*np.sqrt(-2*np.log(1-rns))

#class NormRadTrunc(DistRad):

#    f = 0
#    R = 0
#    sigma_inf = 0
    
#   def __init__(self,radius=None,fraction=None,**params):
#    
#        if(radius is None and "pinhole_size" not in params):
#            raise ValueError("Radial truncated Gaussian requires either a radius or pinhole size as input parameter.")
#        elif(radius is None):
#            radius = params["pinhole_size"]/2.0
          
#        if(fraction is None and "fraction" not in params):
#            raise ValueError("Radial truncated Gaussian input parameter 'fraction' not found.")
#        elif(radius is None):
#            fraction = params["fraction"]  
        
#        if(radius<=0):
#            raise ValueError("For truncated gaussian, radius has to be > 0")
#        if(fraction > 1 or fraction < 0):
#            raise ValueErorr("For truncated gaussian, fraction must satisfy: f > 0 and f < 1")

#        self.f = fraction
#        self.R = radius
#        self.sigma_inf = radius/np.sqrt(2)/np.sqrt(np.log(1/fraction))

#    def pdf(self,r):

#        print(r[0],self.R)
#        res = np.zeros(len(r))
#        nonzero = r<=self.R 
#        res[nonzero]=1/self.sigma_inf**2/(1-self.f)*np.exp(-r[nonzero]*r[nonzero]/2/self.sigma_inf**2)*r[nonzero]
#        res = res*unit_registry('1/'+str(r.units))
#        return res

#    def rho(self,r):
#        res = np.zeros(len(r))
 #       nonzero = r<=self.R 
#        res[nonzero]=1/self.sigma_inf**2/(1-self.f)*np.exp(-r[nonzero]*r[nonzero]/2/self.sigma_inf**2)
#        res = res*unit_registry('1/'+str(r.units)+'/'+str(r.units))
#        return res

#    def cdf(self,r):
#        res = np.zeros(len(r))
#        nonzero = r<=self.R 
#        res[nonzero]=(1-np.exp(-r[nonzero]*r[nonzero]/2/self.sigma_inf**2))/(1-self.f)
#        return res

#    def cdfinv(self,rns):
#        return np.sqrt( 2*self.sigma_inf**2 * ( np.log(1/(rns*(self.f-1)+1)) )) 

#    def get_r_pts(self,n=1000):
#        return np.linspace(0,1.2*self.R.magnitude,n)*unit_registry(str(self.R.units))

#    def avg(self):
#        return (self.sigma_inf*math.sqrt(math.pi/2)*erf(self.R/np.sqrt(2)/self.sigma_inf)-self.R*self.f)/(1-self.f)

#    def rms(self):
#        return np.sqrt( 2*self.sigma_inf**2 - self.R**2 * self.f/(1-self.f) )


class NormRad(DistRad):
    
    def __init__(self, **params):

        self.required_params=[]
        self.optional_params=['sigma_xy','truncation_fraction',
				'truncation_radius_left','truncation_radius_right',
				'n_sigma_cutoff_left','n_sigma_cutoff_left','n_sigma_cutoff',
				'truncation_radius','truncation_radius_left','truncation_radius_right']

        self.check_inputs(params)

        assert (not ('sigma_xy' in params and 'truncation_fraction' in params)), 'User must specify either a sigma_xy or truncation fraction, not both'
        assert (not ('sigma_xy' not in params and 'truncation_fraction' not in params)), 'User must specify sigma_xy or a truncation fraction for radial normal distribution'

        self.rR = None
        self.rL = None

        if('sigma_xy' in params):
            self.sigma = params['sigma_xy']

            if('truncation_radius_left' in params and 'truncation_radius_right' in params):

                self.rL = params['truncation_radius_left']
                self.rR = params['truncation_radius_right']

            elif('truncation_radius' in params):

                self.rL = 0*unit_registry('mm')
                self.rR = params['truncation_radius']

            elif('n_sigma_cutoff_left' in params and 'n_sigma_cutoff_right' in params):

                self.rL = params['n_sigma_cutoff_left']*self.sigma
                self.rR = params['n_sigma_cutoff_right']*self.sigma

            elif('n_sigma_cutoff' in params):

                self.rL = 0*unit_registry('mm')
                self.rR = params['n_sigma_cutoff']*self.sigma

            else:
                self.rL = 0*unit_registry('mm')
                self.rR = float('Inf')*unit_registry('mm')

        elif('truncation_fraction' in params):
            f = params['truncation_fraction']

            if('truncation_radius' in params):

                R = params['truncation_radius']
                
                self.sigma = R*np.sqrt( 1.0/2.0/np.log(1/f) )
                self.rL=0*unit_registry('mm')
                self.rR = R  

            elif('truncation_radius_left' in params and 'truncation_radius_right' in params):
                
                self.rL = params['truncation_radius_right']
                self.rR = params['truncation_radius_right']
                self.sigma = self.rR*np.sqrt( 1.0/2.0/np.log(1/f) )

        #print(self.sigma, self.rL, self.rR)

        assert self.rR.magnitude >= 0, "Radial Gaussian right cut radius must be >= 0"
        assert self.rL < self.rR, "Radial Gaussian left cut radius must be < right cut radius"

        self.pR = self.canonical_rho(self.rR/self.sigma)
        self.pL = self.canonical_rho(self.rL/self.sigma)
        self.dp = self.pL-self.pR

        #print('boot')

    def canonical_rho(self,xi):
        return (1/2.0/math.pi)*np.exp(-xi**2/2)

    def rho(self,r):

        xi = (r/self.sigma)
        res = np.zeros(len(r))*unit_registry('1/'+str(r.units)+'/'+str(r.units))
        nonzero =  (r>=self.rL) & (r<=self.rR)
        res[nonzero]= (1/2.0/math.pi)*self.canonical_rho(xi[nonzero])/self.dp/(self.sigma**2)
        return res*unit_registry('1/'+str(r.units)+'/'+str(r.units))

    def pdf(self,r):
   
        xi = (r/self.sigma)
        res = np.zeros(len(r))
        nonzero =  (r>=self.rL) & (r<=self.rR)
        res[nonzero] = r[nonzero]*self.canonical_rho(xi[nonzero])/self.dp/self.sigma**2
        return res*unit_registry('1/'+str(r.units))

    def cdf(self,r):

        res = np.zeros(len(r))
        nonzero =  (r>=self.rL) & (r<=self.rR)
        xi = (r/self.sigma)
        res[nonzero]=(self.pL - self.canonical_rho(xi[nonzero]))/self.dp
        return res

    def cdfinv(self,rns):
        return np.sqrt( 2*self.sigma**2 * np.log(1/2/math.pi/( self.pL - self.dp*rns )) ) 

    def get_r_pts(self,n=1000):
        if(self.rR.magnitude==float('Inf')):
            endr = 5*self.sigma
        else:
            endr = 1.2*self.rR
        return linspace(0.88*self.rL,endr,n)

    def avg(self):

        xiL = self.rL/self.sigma
        xiR = self.rR/self.sigma
   
        erfL = erf(xiL/np.sqrt(2))
        erfR = erf(xiR/np.sqrt(2))

        if(self.rR.magnitude==float('Inf')):
            xiRpR = 0*unit_registry('')
        else:
            xiRpR = xiR*self.pR

        return self.sigma*( (xiL*self.pL - xiRpR) + (1.0/2.0/np.sqrt(2*math.pi))*(erfR-erfL) )/self.dp 

    def rms(self):

        if(self.rR.magnitude==float('Inf')):
            pRrR2 = 0*unit_registry('mm^2')
        else:
            pRrR2 = self.pR*self.rR**2

        pRrL2 = self.pR*self.rL**2
        return np.sqrt( 2*self.sigma**2 - self.rL**2 + (pRrL2 - pRrR2)/self.dp )

class RadFile(DistRad):

    def __init__(self,**params):

        self.required_params=['file','units']
        self.optional_params=[]
        self.check_inputs(params)
        
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

        self.required_params=['ratio','length']
        self.optional_params=[]
        self.check_inputs(kwargs)
         
        self.r = kwargs['ratio']
        self.L = kwargs['length']

        vprint("TukeyRad",verbose>0,0,True)
        vprint("legnth = {:0.3f~P}".format(self.L)+", ratio = {:0.3f~P}".format(self.r),verbose>0,2,True)

    def get_r_pts(self,n=1000):
        return np.linspace(0,1.2*self.L.magnitude,n)*unit_registry(str(self.L.units))

    def pdf(self,r):        
        rho = self.rho(r)
        return r*rho

    def rho(self,r):

        ustr = '1/'+str(self.L.units)+"/"+str(self.L.units)

        res = np.zeros(r.shape)*unit_registry(ustr)

        if(self.r==0):
           flat_region = np.logical_and(r <= self.L, x >= 0.0)
           res[flat_region]=1.0*unit_registry(ustr)
       
        else:
            
            Lflat = self.L*(1-self.r)
            Lcos = self.r*self.L
            cos_region = np.logical_and(r >= +Lflat, r <=+self.L)
            flat_region = np.logical_and(r < Lflat, r >= 0)
            res[cos_region]=0.5*(1+np.cos( (pi/Lcos)*(r[cos_region]-Lflat) ))*unit_registry(ustr)
            res[flat_region]=1.0*unit_registry(ustr)
        
        res = res
        res = res/radint(res,r)
        return res
   

    def cdf(self,r):
        rpts = self.get_r_pts(10000)
        pdfs = self.rho(rpts)
        
        cdfs,rbins = radcumint(pdfs,rpts,initial=0)
        cdfs = cdfs/cdfs[-1]
        cdfs = interp(r,rbins,cdfs)
        cdfs = cdfs/cdfs[-1]
        cdfs*unit_registry('dimensionless')

        return cdfs

    def cdfinv(self,p):
        rpts = self.get_r_pts(10000)
        cdfs = self.cdf(rpts)
        return interp(p,cdfs,rpts)

    def avg(self):
        rpts = self.get_r_pts(10000)
        pdfs = self.rho(rpts)
        return radint(pdfs*rpts,rpts)

    def rms(self):
        rpts = self.get_r_pts(10000)
        pdfs = self.rho(rpts)
        return np.sqrt(radint(pdfs*rpts*rpts,rpts))


class SuperGaussianRad(DistRad):

    def __init__(self,verbose=0,**kwargs):

        self.required_params=['lambda']
        self.optional_params=['p','alpha']
        self.check_inputs(kwargs)

        assert not ('alpha' in kwargs and 'p' in kwargs), 'Radial Super Gaussian power parameter must be set using "p" or "alpha", not both.' 
        assert ('alpha' in kwargs or 'p' in kwargs), 'Radial Super Gaussian power parameter must be set using "p" or "alpha". Neither provided.' 

        self.Lambda = kwargs['lambda']

        if('p' in kwargs):
            self.p = kwargs['p']
        elif('alpha' in kwargs):
            alpha = kwargs['alpha']
            assert alpha >= 0 and alpha <= 1, 'SugerGaussian parameter must satisfy 0 <= alpha <= 1, not = '+str(alpha)
            if(alpha.magnitude==0): 
                self.p = float('Inf')*unit_registry('dimensionless')
            else:
                self.p = 1/alpha 

        vprint("SuperGaussianRad",verbose>0,0,True)
        vprint("lambda = {:0.3f~P}".format(self.Lambda)+", power = {:0.3f~P}".format(self.p),verbose>0,2,True)

    def get_r_pts(self,n=1000):
        return np.linspace(0,5*self.Lambda.magnitude,n)*unit_registry(str(self.Lambda.units))

    def pdf(self,r):        
        rho = self.rho(r)
        return r*rho

    def rho(self,r):

        ustr = '1/'+str(self.Lambda.units)+"/"+str(self.Lambda.units)

        csi = r/self.Lambda
        nur = 0.5*csi**2

        rho = np.exp(-np.float_power(nur.magnitude,self.p.magnitude))*unit_registry(ustr)
        rho = rho/radint(rho,r)/2/pi
        return rho
   

    def cdf(self,r):
        rpts = self.get_r_pts(10000)
        pdfs = self.rho(rpts)
        
        cdfs,rbins = radcumint(pdfs,rpts,initial=0)
        cdfs = cdfs/cdfs[-1]
        cdfs = interp(r,rbins,cdfs)
        cdfs = cdfs/cdfs[-1]
        cdfs*unit_registry('dimensionless')

        return cdfs

    def cdfinv(self,p):
        rpts = self.get_r_pts(10000)
        cdfs = self.cdf(rpts)
        return interp(p,cdfs,rpts)

    def avg(self):
        rpts = self.get_r_pts(10000)
        pdfs = self.rho(rpts)
        return radint(pdfs*rpts,rpts)

    def rms(self):
        rpts = self.get_r_pts(10000)
        pdfs = self.rho(rpts)
        return np.sqrt(radint(pdfs*rpts*rpts,rpts))

class Dist2d(Dist):

    #xstr = ""
    #ystr = ""
    #xs = []    # x pts
    #ys = []
    #Pxy = []    # Probability Distribution Function P(x,y)
    
    #rgen = RandGen()

    def __init__(self,xs,ys,Pxy,xstr="x",ystr="y"):

        self.xs=xs
        self.ys=ys
        self.Pxy = Pxy
        self.xstr=xstr
        self.ystr=ystr

        assert np.count_nonzero(Pxy.magnitude) > 0, 'Supplied 2d distribution is zero everywhere.'
    
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
        pass
   
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

    def __init__(self, var1, var2, **params):

        self.required_params=['file']
        self.optional_params=[]

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



