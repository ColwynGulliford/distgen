"""
Defines the random number generator class and distribution function objects.
"""

from .hammersley import create_hammersley_samples
from .physical_constants import unit_registry
from .physical_constants import pi

from .tools import vprint

from .tools import interp
from .tools import linspace
from .tools import centers

from .tools import trapz
from .tools import cumtrapz
from .tools import radint
from .tools import radcumint

from .tools import histogram
from .tools import radial_histogram

from .tools import  concatenate

from .tools import erf
from .tools import erfinv
from .tools import gamma

from .tools import get_vars
from .tools import read_2d_file
from .tools import read_image_file

from pint import Quantity

import numpy as np
import numpy.matlib as mlib
import os

from matplotlib import pyplot as plt

def random_generator(shape,sequence=None,params=None):
    """ Returns a set of 'random' (either numpy.random.random or from a Hammersley sequence) numbers """
    if(sequence is None or sequence=='pseudo'):
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
        dist = Uniform(var,verbose=verbose, **params)
    elif(dtype=="gaussian" or dtype=="g"):
        dist = Norm(var,verbose=verbose, **params)
    elif(dtype=="file1d"):
        dist = File1d(var,verbose=verbose, **params)
    elif(dtype=='tukey'):
        dist = Tukey(var,verbose=verbose, **params)
    elif(dtype=='super_gaussian' or dtype=='sg'):
        dist = SuperGaussian(var,verbose=verbose, **params)
    elif(dtype=="superposition" or dtype=='sup'):
        dist = Superposition(var, verbose=verbose, **params)
    elif(dtype =='product' or dtype=='pro'):
        dist = Product(var, verbose=verbose, **params)
    elif(dtype=='deformable'):
        dist = Deformable(var, verbose=verbose, **params)
    elif((dtype=="radial_uniform" or dtype=="ru") and var=="r"):
        dist = UniformRad(verbose=verbose, **params)
    elif((dtype=="radial_gaussian" or dtype=="rg") and var=="r"):
        dist = NormRad(verbose=verbose ,**params)
    elif((dtype=="radial_super_gaussian" or dtype=="rsg") and var=="r"):
        dist = SuperGaussianRad(verbose=verbose, **params)
    elif(dtype=="radfile" and var=="r"):
        dist = RadFile(verbose=verbose, **params)
    elif(dtype=="radial_tukey"):
        dist = TukeyRad(verbose=verbose, **params)
    elif(dtype=='raddeformable' or dtype=='dr'):
        dist = DeformableRad(verbose=verbose, **params)
    elif(dtype=="file2d"):
        dist = File2d("x","y",verbose=verbose, **params)
    elif(dtype=="crystals"):
        dist = TemporalLaserPulseStacking(verbose=verbose, **params)
    elif(dtype=='uniform_theta' or dtype=='ut'):
        dist =  UniformTheta(verbose=verbose, **params)
    elif(dtype=='image2d'):
        dist = Image2d(var, verbose=verbose, **params)
    else:
        raise ValueError(f'Distribution type "{dtype}" is not supported.')
            
    return dist    


class Dist():
    """
    Defines a base class for all distributions, and includes functionality for strict input checking
    """
    def __init__(self):
        self._n_indent=2

    def check_inputs(self, params):
        """
        Checks the input dictionary to derived class.  Derived class supplies lists of required and optional params.
        """

        # Make sure user isn't passing the wrong parameters:
        allowed_params = self.optional_params + self.required_params + ['verbose','type','indent']
        #print(allowed_params)
        for param in params:
            assert param in allowed_params, f'Incorrect param given to {self.__class__.__name__}.__init__(**kwargs): {param}\nAllowed params: {allowed_params}'

        # Make sure all required parameters are specified
        for req in self.required_params:
            assert req in params, f'Required input parameter {req} to {self.__class__.__name__}.__init__(**kwargs) was not found.'

        if('indent' in params):
            self._n_indent = params['indent']


class Dist1d(Dist):

    """
    Defines the base class for 1 dimensional distribution functions.  
    Assumes user will pass in [x,f(x)] as the pdf. 
    Numerically intergates to find the cdf and to sample the distribution.  
    Methods should be overloaded for specific derived classes, particularly if
    the distribution allows analytic treatment.
    """
    
    def __init__(self, xs=None, Px=None, xstr="x"):

        super().__init__()

        self.xs = xs
        self.Px = Px
        self.xstr = xstr
          
        if(Px is not None):
            norm = np.trapz(self.Px, self.xs)
            if(norm<=0):
                raise ValueError('Normalization of PDF was <= 0')

            self.Px = self.Px/norm
            self.Cx = cumtrapz(self.Px, self.xs)
    
    def get_x_pts(self, n):
        """
        Returns a vector of x pts suitable for sampling the PDF Px(x)
        """
        return linspace(self.xs[0], self.xs[-1],n)

    def pdf(self, x):
        """"
        Evaluates the pdf at the user supplied points in x
        """
        return interp(x, self.xs, self.Px)
 
    def cdf(self, x):
        """"
        Evaluates the cdf at the user supplied points in x
        """  
        return interp(x, self.xs, self.Cx)

    def cdfinv(self, rns):
        """
        Evaluates the inverse of the cdf at probabilities rns
        """
        return interp(rns, self.Cx, self.xs)

    def sample(self, N, sequence=None, params=None):
        """
        Generate coordinates by sampling the underlying pdf
        """
        return self.cdfinv( random_generator((1,N),sequence,params)*unit_registry("dimensionless") )

    def plot_pdf(self, n=1000):
        """
        Plots the associated pdf function sampled with n points
        """
        x=self.get_x_pts(n)
        p=self.pdf(x)
        plt.figure()
        plt.plot(x, p)
        plt.xlabel(f'{self.xstr} ({x.units:~P})')
        plt.ylabel(f'PDF({self.xstr}) ({p.units:~P})')

    def plot_cdf(self, n=1000):
        """ 
        Plots the associtated cdf function sampled with n points
        """
        x=self.get_x_pts(n)
        P=self.cdf(x)
        plt.figure()
        plt.plot(x,P)
        plt.xlabel(f'{self.xstr} ({x.units:~P})')
        plt.ylabel(f'CDF({self.xstr}) ({P.units})')
        
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

        rho, edges = histogram(xs, nbins=100)
        xc = centers(edges)
        rho = rho/np.trapz(rho,xc)

        savgx = xs.mean()
        sstdx = xs.std()

        davgx = self.avg()
        dstdx = self.std()      

        plt.figure()
        plt.plot(x, pdf, xc, rho, 'or')
        plt.xlabel(f'{self.xstr} ({x.units:~P})')   
        plt.ylabel(f'PDF ({pdf.units:~P})')  

        stat_line = f'Sample stats: <{self.xstr}> = {savgx:G~P}, '+'$\sigma_{'+str(self.xstr)+'}$'+f' = {sstdx:G~P}'
        dist_line =  f'Dist. stats: <{self.xstr}> = {davgx:G~P}, '+'$\sigma_{'+str(self.xstr)+'}$'+f' = {dstdx:G~P}'  
 
        plt.title(stat_line+'\n'+dist_line)
        plt.legend(["PDF","Sampling"])    


class Superposition(Dist1d):

    """Dist object that allows user to superimpose multiple 1d distributions together to form a new PDF for sampling"""

    def __init__(self, var, verbose, **kwargs):

        self.xstr = var
        assert 'dists' in kwargs, 'SuperPositionDist1d must be supplied the key word argument "dists"'

        dist_defs = kwargs['dists']

        dists={}

        min_var=0
        max_var=0

        if('weights' not in kwargs):
            weights={name:1 for name in dist_defs}
        else:
            weights = kwargs['weights']

        vprint('superpostion',verbose>0,0,True)

        for ii, name in enumerate(dist_defs.keys()):

            if(name not in weights):
                weights[name]=1

            dist_defs[name]['indent']=3

            vprint(f'{ii+1}. distribution name: {name}, type: ', verbose>0, 2, False)
            dists[name] = get_dist(var, dist_defs[name],  verbose=verbose)
            
            xi = dists[name].get_x_pts(10)

            if(xi[0] <min_var): min_var = xi[0]
            if(xi[-1]>max_var): max_var = xi[-1]

        xs = linspace(min_var, max_var, 10000)

        for ii, name in enumerate(dists.keys()):

            pi = dists[name].pdf(xs)

            assert weights[name]>=0, 'Weights for superpostiion dist must be >= 0.'

            if(ii==0):
                ps = weights[name]*pi/np.max(pi.magnitude)
            else:
                ps = ps + weights[name]*pi/np.max(pi.magnitude)

        super().__init__(xs, ps, var)

        
        #vprint(f'min_{var} = {self.xL:G~P}, max_{var} = {self.xR:G~P}', verbose>0, 2, True)

class Product(Dist1d):
    
    """Dist object that allows user to multiply multiple 1d distributions together to form a new PDF for sampling"""

    def __init__(self, var, verbose, **kwargs):

        self.xstr = var
        assert 'dists' in kwargs, 'ProductDist 1d must be supplied the key word argument "dists"'
        dist_defs = kwargs['dists']

        dists={}

        min_var=0
        max_var=0

        for ii, name in enumerate(dist_defs.keys()):

            vprint(f'\ndistribution name: {name}', verbose>0, 0, True)
            dists[name] = get_dist(var, dist_defs[name], verbose=verbose)
            
            xi = dists[name].get_x_pts(10)

            if(xi[0] <min_var): min_var = xi[0]
            if(xi[-1]>max_var): max_var = xi[-1]

        xs = linspace(min_var, max_var, 10000)

        for ii, name in enumerate(dists.keys()):

            pi = dists[name].pdf(xs)

            if(ii==0):
                ps = pi/np.max(pi.magnitude)
            else:
                ps = ps*pi/np.max(pi.magnitude)

        super().__init__(xs, ps, var)


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
        self.required_params = []
        self.optional_params = [f'max_{var}', f'min_{var}', f'avg_{var}', f'sigma_{var}']

        self.check_inputs(kwargs)

        use_min_max = f'max_{var}' in kwargs and f'min_{var}' in kwargs
        use_avg_sigma = f'avg_{var}' in kwargs and f'sigma_{var}' in kwargs

        assert use_min_max ^ use_avg_sigma, f'User must specify either min_{var} and max_{var}] or [avg_{var} and sigma_{var}]'

        if(f'min_{var}' in kwargs):
            self.xL = kwargs[f'min_{var}']           
            self.xR = kwargs[f'max_{var}']
        else:
            length = np.sqrt(12)*kwargs[f'sigma_{var}']
            avgv = kwargs[f'avg_{var}']
            self.xL = avgv-length/2
            self.xR = avgv+length/2



        #assert (f'max_{var}' in kwargs and f'min_{var}' in kwargs) or (f'avg_{var}' in kwargs and f'sigma_{var}' in kwargs), f'User must specify either min_{var} and max_{var}] or [avg_{var} and sigma_{var}], not both.'
        #self.xL = kwargs[minstr]           
        #self.xR = kwargs[maxstr]
        vprint('uniform',verbose>0,0,True)
        vprint(f'min_{var} = {self.xL:G~P}, max_{var} = {self.xR:G~P}', verbose>0, 2, True)
  
    def get_x_pts(self, n, f=0.2):
        """
        Returns n equally spaced x values that sample just over the relevant range of [a,b] (DOES NOT SAMPLE DISTRIBUTION)
        Inputs: n [int]
        """
        dx = f*np.abs(self.avg())
        return np.linspace(self.xL-dx, self.xR+dx, n)

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
        res = np.zeros(len(x))*unit_registry('dimensionless')
        res[nonzero]=(x[nonzero]-self.xL)/(self.xR-self.xL)
       
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


class Linear(Dist1d):

    
    """ Defines the PDF and CDF for a linear function in 1d """

    def __init__(self, var, verbose=0, **kwargs):

        self._n_indent = 2

        self.type='Linear'
        self.xstr = var

        xa_str = f'min_{var}'
        xb_str = f'max_{var}'

        self.required_params = ['slope_fraction', xa_str, xb_str]
        self.optional_params = []

        self.check_inputs(kwargs)

        self.a = kwargs[xa_str]
        self.b = kwargs[xb_str]
        self.r = kwargs['slope_fraction']
        self.f = 1-np.abs(self.r)

        assert self.a < self.b, f'Error: {xa_str} must be < {xb_str}.'
        assert self.r>=-1 and self.r <= 1, 'Error: slope fraction must be: -1 <= r < 1.'

        self.dx = self.b-self.a

        if(self.r >= 0):
            # Do the maths
            self.pb = 2/(1+self.f)/self.dx
            self.pa = self.f*self.pb

        else:
            # Relabel the other way
            self.pa = 2/(1+self.f)/self.dx
            self.pb = self.f*self.pa

        self.dp = self.pb-self.pa
        self.m = self.dp/self.dx

        vprint('Linear',verbose>0,0,True)
        #vprint(f'avg_{var} = {self.mu:G~P}, sigma_{var} = {self.sigma:0.3f~P}',verbose>0,self._n_indent,True)
        #if(self.sigma>0):
        #    vprint(f'Left n_sigma_cutoff = {self.b/self.sigma:G~P}, Right n_sigma_cutoff = {self.a/self.sigma:G~P}',verbose>0 and self.b.magnitude<float('Inf'),2,True)
        #else:
        #    vprint(f'Left n_sigma_cutoff = {self.b:G~P}, Right n_sigma_cutoff = {self.a:G~P}',verbose>0 and self.b.magnitude<float('Inf'),2,True)

    def get_x_pts(self, n, f=0.2):
        """
        Returns n equally spaced x values that sample just over the relevant range of [a,b] (DOES NOT SAMPLE DISTRIBUTION)
        Inputs: n [int]
        """
        dx = f*np.abs(self.avg())
        return np.linspace(self.a-dx, self.b+dx, n)

    def pdf(self, x):
        """
        Returns the PDF at the values in x [array w/units].  PDF has units of 1/[x]
        """
        nonzero = (x >= self.a) & (x <= self.b)
        res = np.zeros(len(x))*unit_registry('1/'+str(self.a.units))
        res[nonzero]=self.m*(x[nonzero]-self.a) + self.pa
        return res

    def cdf(self,x):
        """
        Returns the CDF at the values of x [array w/units].  CDF is dimensionless
        """
        nonzero = (x >= self.a) & (x <= self.b)
        res = np.zeros(len(x))*unit_registry('dimensionless')
        delta = x[nonzero]-self.a
        res[nonzero]=0.5*self.m*delta**2 + self.pa*delta
       
        return res

    def cdfinv(self, p):
        return self.a + (np.sqrt( self.pa**2 + 2*self.m*p ) - self.pa)/self.m


    def avg(self):
        """
        Returns the first moment of the PDF: 
        """
        d2 = self.b**2-self.a**2
        d3 = self.b**3-self.a**3

        return self.pa*d2/2.0 + self.m*( d3/3.0 - self.a*d2/2.0)

    def std(self):
        """
        Returns the square root of the variance of the PDF: 
        """
        return np.sqrt( self.rms()**2 - self.avg()**2 )
  
    def rms(self):
        """
        Returns the rms of the distribution computed from the avg and std.
        """

        d3 = self.b**3-self.a**3
        d4 = self.b**4-self.a**4

        ta = self.pa*d3/3.0
        tm = self.m*(d4/4.0 -self.a*d3/3.0)

        return np.sqrt(ta+tm)

    
class Norm(Dist1d):

    """ Defines the PDF and CDF for a normal distribution with truncation on either side """

    def __init__(self, var, verbose=0, **kwargs):

        self._n_indent = 2

        self.type='Norm'
        self.xstr = var

        sigmastr = f'sigma_{var}'
        self.required_params=[sigmastr]

        sigma_cutoff_str   = "n_sigma_cutoff"
        sigma_cutoff_left  = "n_sigma_cutoff_left"
        sigma_cutoff_right = "n_sigma_cutoff_right"
        avgstr = f'avg_{var}'
        self.optional_params=[sigma_cutoff_str,sigma_cutoff_left,sigma_cutoff_right,avgstr]

        self.check_inputs(kwargs)

        self.sigma = kwargs[sigmastr]

        assert self.sigma.magnitude>=0, 'Error: sigma for Norm(1d) must be >= 0'
            
        if(avgstr in kwargs.keys()):
            self.mu = kwargs[avgstr]
        else:
            self.mu = 0*unit_registry(str(self.sigma.units))

        left_cut_set = False
        right_cut_set = False

        assert not (sigma_cutoff_str in kwargs.keys() and (sigma_cutoff_left in kwargs.keys() or sigma_cutoff_right in kwargs.keys()) )

        if(sigma_cutoff_str in kwargs.keys()):

            self.a = -kwargs[sigma_cutoff_str]*self.sigma + self.mu
            self.b = +kwargs[sigma_cutoff_str]*self.sigma + self.mu

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

        if(self.sigma.magnitude>0):

            assert self.a < self.b, 'Right side cut off a = {a:G~P} must be < left side cut off b = {b:G~P}'

            self.A = (self.a - self.mu)/self.sigma
            self.B = (self.b - self.mu)/self.sigma
        
            self.pA = self.canonical_pdf(self.A)
            self.pB = self.canonical_pdf(self.B)

            self.PA = self.canonical_cdf(self.A)
            self.PB = self.canonical_cdf(self.B)

            self.Z = self.PB - self.PA

        else:

            self.A = 0*unit_registry('dimensionless')
            self.B = 0*unit_registry('dimensionless')
        
            self.pA = 0
            self.pB = 0

            self.PA = 0
            self.PB = 0

            self.Z = 1.0

        vprint('Gaussian',verbose>0,0,True)
        vprint(f'avg_{var} = {self.mu:G~P}, sigma_{var} = {self.sigma:0.3f~P}',verbose>0,self._n_indent,True)

        if(self.sigma>0):
            vprint(f'Left n_sigma_cutoff = {self.b/self.sigma:G~P}, Right n_sigma_cutoff = {self.a/self.sigma:G~P}',verbose>0 and self.b.magnitude<float('Inf'),2,True)
        else:
            vprint(f'Left n_sigma_cutoff = {self.b:G~P}, Right n_sigma_cutoff = {self.a:G~P}',verbose>0 and self.b.magnitude<float('Inf'),2,True)

    def get_x_pts(self, n=1000, f=0.1):

        """ Returns xpts from [a,b] or +/- 5 sigma, depending on the defintion of PDF """

        if(-float('Inf') < self.a.magnitude):
            lhs=self.a*(1-f*np.sign(self.a.magnitude))
        else:
            lhs=-5*self.sigma

        if(self.b.magnitude < float('Inf')):
            rhs = self.b*(1+f*np.sign(self.b.magnitude))
        else:
            rhs=+5*self.sigma

        return self.mu + linspace(lhs, rhs, n)

    def canonical_pdf(self,csi):
        """ Definies the canonical normal distribution """
        return (1/np.sqrt(2*pi))*np.exp( -csi**2/2.0 ) 

    def pdf(self,x):     

        """ Define the PDF for non-canonical normal dist including truncations on either side"""   
        csi = (x-self.mu)/self.sigma
        res = self.canonical_pdf(csi)/self.Z/self.sigma
        x_out_of_range = (x<self.a) | (x>self.b)
        res[x_out_of_range] = 0*unit_registry('1/'+str(self.sigma.units))
        return res

    def canonical_cdf(self, csi):
        """ Defines the canonical cdf function """
        return 0.5*(1+erf(csi/np.sqrt(2) ) )

    def cdf(self,x):
        """ Define the CDF for non-canonical normal dist including truncations on either side"""
        csi = (x-self.mu)/self.sigma
        res = (self.canonical_cdf(csi) - self.PA)/self.Z
        x_out_of_range = (x<self.a) | (x>self.b)
        res[x_out_of_range] = 0*unit_registry('dimensionless')
        return res

    def canonical_cdfinv(self,rns):
        """ Define the inverse of the CDF for canonical normal dist including truncations on either side"""
        return np.sqrt(2)*erfinv((2*rns-1))

    def cdfinv(self, rns):
        """ Define the inverse of the CDF for non-canonical normal dist including truncations on either side"""
        scaled_rns = rns*self.Z + self.PA
        return self.mu + self.sigma*self.canonical_cdfinv(scaled_rns)

    def avg(self):
        """ Computes the <x> value of the distribution: <x> = int( x rho(x) dx) """
        if(self.sigma.magnitude>0):
            return self.mu + self.sigma*(self.pA - self.pB)/self.Z 
        else:
            return self.mu
    
    def std(self):
        """ Computes the sigma of the distribution: sigma_x = sqrt(int( (x-<x>)^2 rho(x) dx)) """
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
        """ Computes the rms of the distribution: sigma_x = sqrt(int( x^2 rho(x) dx)) """
        avg=self.avg()
        std=self.std()
        return np.sqrt(std*std + avg*avg)

class SuperGaussian(Dist1d):

    """ Distribution  object that samples a 1d Super Gaussian PDF"""

    def __init__(self,var,verbose=0,**kwargs):

        self.type='SuperGaussian'
        self.xstr = var

        lambda_str = 'lambda'
        sigma_str = f'sigma_{var}'
        power_str = 'p'
        alpha_str = 'alpha'
        avg_str = f'avg_{var}'

        self.required_params=[]
        self.optional_params=[avg_str, power_str, alpha_str, 'lambda', sigma_str, 'n_sigma_cutoff']
        self.check_inputs(kwargs)

        assert not (alpha_str in kwargs and power_str in kwargs), 'SuperGaussian power parameter must be set using "p" or "alpha", not both.' 
        assert (alpha_str in kwargs or power_str in kwargs), 'SuperGaussian power parameter must be set using "p" or "alpha". Neither provided.' 

        assert not (sigma_str in kwargs and lambda_str in kwargs), 'SuperGaussian length scale must either be set using "lambda" or "{sigma_str}", not both.' 
        assert (alpha_str in kwargs or power_str in kwargs), 'SuperGaussian length scale must be set using "lambda" or "{sigma_str}", Neither provided.' 

        if(power_str in kwargs):
            self.p = kwargs[power_str]
        else:
            alpha = kwargs[alpha_str]
            assert alpha >= 0 and alpha <= 1, 'SugerGaussian parameter must satisfy 0 <= alpha <= 1.'
            if(alpha.magnitude==0): 
                self.p = float('Inf')*unit_registry('dimensionless')
            else:
                self.p = 1/alpha 

        assert self.p > 0, 'SuperGaussian power "p" must be > 0, not p = {self.p}'
 
        if('lambda' in kwargs):
            self.Lambda = kwargs[lambda_str]
        else: 
            self.Lambda = self.get_lambda(kwargs[sigma_str])

        if(avg_str in kwargs):
            self.mu = kwargs[avg_str]
        else:
            self.mu = 0*unit_registry(str(self.Lambda.units))

        if('n_sigma_cutoff' in kwargs):
            self.n_sigma_cutoff=kwargs['n_sigma_cutoff']
        else:
            self.n_sigma_cutoff=3

        vprint('Super Gaussian', verbose>0, 0, True)
        vprint(f'sigma_{var} = {self.std():G~P}, power = {self.p:G~P}', verbose, 2, True)
        vprint(f'n_sigma_cutoff = {self.n_sigma_cutoff}', int(verbose>=1 and self.n_sigma_cutoff!=3), 2, True)
 
    def pdf(self,x=None):  

        """ Defines the PDF for super Gaussian function """
        if(x is None):
            x=self.get_x_pts(10000)
      
        xi = (x-self.mu)/self.Lambda
        nu1 = 0.5*(xi**2)

        N = 1./2/np.sqrt(2)/self.Lambda/gamma(1+1.0/2.0/self.p)

        rho = N*np.exp(-np.float_power(nu1.magnitude,self.p.magnitude))
        
        return rho
        
    def get_x_pts(self, n=None):
        """
        Returns n equally spaced x values from +/- n_sigma_cutoff*sigma
        """
        if(n is None):
            n=10000
        return self.mu + linspace(-self.n_sigma_cutoff*self.std(), +self.n_sigma_cutoff*self.std(),n)

    def cdf(self,x):
        """ Defines the CDF for the super Gaussian function """
        xpts = self.get_x_pts(10000)
        pdfs = self.pdf(xpts)
        cdfs = cumtrapz(self.pdf(xpts), xpts)

        cdfs = cdfs/cdfs[-1]

        cdfs = interp(x, xpts, cdfs)
        cdfs = cdfs/cdfs[-1]
        #print(x[0],cdfs[0])
        return cdfs

    def cdfinv(self,p):
        """ Definess the inverse of the CDF for the super Gaussian function """
        xpts = self.get_x_pts(10000)
        cdfs = self.cdf(xpts)
        return interp(p,cdfs,xpts)
    
    def avg(self):
        """ Returns the average value of x for super Gaussian """
        return self.mu

    def std(self):
        """ Returns the standard deviation of the super Gausssian dist """
        G1 = gamma(1 + 3.0/2.0/self.p)
        G2 = gamma(1+1/2/self.p)
        return self.Lambda * np.sqrt(2*G1/3/G2)

    def get_lambda(self,sigma):
        """ Returns the length scale of the super Gausssian dist """
        G1 = gamma(1 + 3.0/2.0/self.p)
        G2 = gamma(1+1/2/self.p)
        return np.sqrt(3*G2/2.0/G1)*sigma

    def rms(self):
        """ Returns the rms of the super Gausssian dist """
        avg=self.avg()
        std=self.std()
        return np.sqrt(std*std + avg*avg)

    
class File1d(Dist1d):

    """Defines an object for loading a 1d PDF from a file and using for particle sampling"""
    
    def __init__(self,var,verbose=0,**kwargs):
        

        self.required_params = ['file','units']
        self.optional_params = []
        self.check_inputs(kwargs)

        self.xstr=var
        
        self.distfile = kwargs["file"]
        self.units = kwargs["units"]
        
        vprint(f'{var}-distribution file: "{self.distfile}"',verbose>0,0,True)
        with open(self.distfile,'r') as f:
            headers = f.readline().split()

        if(len(headers)!=2):
            raise ValueError("file1D distribution file must have two columns")
            
        #if(headers[0]!=self.xstr):
        #    raise ValueError("Input distribution file variable must be = "+var)
        #if(headers[1]!="P"+self.xstr):
        #    raise ValueError("Input distribution file pdf name must be = P"+var)    
            
        data = np.loadtxt(self.distfile,skiprows=1)

        xs = data[:,0]*unit_registry(self.units)
        Px = data[:,1]*unit_registry.parse_expression("1/"+self.units)
        
        assert np.count_nonzero(xs.magnitude) > 0, f'Supplied 1d distribution coordinate vector {var} is zero everywhere.'
        assert np.count_nonzero(Px.magnitude) > 0, f'Supplied 1d distribution P{var} is zero everywhere.'

        super().__init__(xs,Px,self.xstr)
        
class TemporalLaserPulseStacking(Dist1d):

    """ CU style model of birefringent crystal pulse stacking """

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

    def set_crystals(self, lengths, angles):

        """ Sets the crytal parameters for propagating sech pulses """
    
        assert len(lengths)==len(angles), 'Number of crystal lengths must be the same as the number of angles.'

        self.lengths=lengths
        self.angles=angles
        self.angle_offsets=np.zeros(len(angles))

        for ii in range(len(lengths)):
            assert lengths[ii]>0,"Crystal length must be > 0."           
            if(ii % 2 ==0):
                angle_offset= -45*unit_registry("deg")
            else:
                angle_offset=   0*unit_registry("deg")
                 
            vprint(f'crystal {ii+1} length = {self.lengths[ii]:G~P}',self.verbose>0,2,False)
            vprint(f', angle = {self.angles[ii]:G~P}',self.verbose>0,0,True)

            self.crystals.append({"length":lengths[ii],"angle":angles[ii],"angle_offset":angle_offset})    

    def propagate_pulses(self):

        """ Propagates the sech pulses through each crystal, resulting in two new pulses """

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

        vprint(f'Pulses propagated: min t = {self.t_min:G~P}, max t = {self.t_max:G~P}',self.verbose>0,2,True) 

    def apply_crystal(self,next_crystal):
        """ Generates two new pulses from an incoming pulse in a given crystal """

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

        """ Evaluates the electric field of the sech pulses """

        #Evaluates the real and imaginary parts of one component of the E-field:
        normalization = pulse["intensity"]*np.cos(pulse["polarization_angle"] - axis_angle);
        w = 2*np.arccosh(np.sqrt(2))/self.laser_pulse_FWHM;

        field[0] = field[0] + normalization*np.cos(self.w0*(t-pulse["relative_delay"])) / np.cosh(w*(t-pulse["relative_delay"]))
        field[1] = field[1] + normalization*np.sin(self.w0*(t-pulse["relative_delay"])) / np.cosh(w*(t-pulse["relative_delay"]))

    def get_t_pts(self, n):
        return linspace(self.t_min,self.t_max.to(self.t_min),n)
  
    def get_x_pts(self, n):
        return self.get_t_pts(n)

    def set_pdf(self):

        """ Evaluates the sech fields and computes the square of the 
        fields for intenstity in order to set the distribution """

        ex=np.zeros((2,len(self.ts)))*unit_registry("")
        ey=np.zeros((2,len(self.ts)))*unit_registry("")

        for pulse in self.pulses: 
            self.evaluate_sech_fields(0.5*pi,pulse,self.ts,ex);
            self.evaluate_sech_fields(0.0,   pulse,self.ts,ey);

        self.Pt = ( (ex[0,:]**2 + ex[1,:]**2) + (ey[0,:]**2 + ey[1,:]**2) ).magnitude * unit_registry("THz")
        self.Pt = self.Pt/trapz(self.Pt,self.ts)

    def set_cdf(self):
        """ Computes the CDF of the distribution """
        self.Ct = cumtrapz(self.Pt, self.ts)

    def pdf(self, t):
        """ Returns the PDF at the values in t """
        return interp(t, self.ts, self.Pt)

    def cdf(self, t):
        """ Returns the CDF at the values of t """
        return interp(t, self.ts, self.Ct)

    def cdfinv(self, rns):
        """ Computes the inverse of the CDF at probabilities rns """
        return interp(rns*unit_registry(''),self.Ct,self.ts)

    def avg(self):
        """ Computes the expectation value of t of the distribution """
        return trapz(self.ts*self.Pt,self.ts)

    def std(self):
        """ Computes the sigma of the PDF """
        return np.sqrt(trapz(self.ts*self.ts*self.Pt,self.ts))

    #def get_params_list(self,var):
     #   """ Returns the crystal parameter list"""
    #    return (["crystal_length_$N","crystal_angle_$N"],["laser_pulse_FWHM","avg_"+var,"std_"+var])

class Tukey(Dist1d):

    """ Defines a 1d Tukey distribution """

    def __init__(self, var, verbose=0, **kwargs):
        
        self.xstr = var
         
        self.required_params = ['ratio','length']
        self.optional_params = []
        self.check_inputs(kwargs)
            
        self.r = kwargs['ratio']
        self.L = kwargs['length']

        vprint('Tukey',verbose>0,0,True)
        vprint(f'length = {self.L:G~P}, ratio = {self.r:G~P}',verbose>0,2,True)
            
    def get_x_pts(self,n):
        return 1.1*linspace(-self.L/2.0,self.L/2.0,n)

    def pdf(self, x):

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

    def cdf(self, x):
        xpts = self.get_x_pts(10000)
        pdfs = self.pdf(xpts)
        cdfs = cumtrapz(self.pdf(xpts), xpts)
        cdfs = cdfs/cdfs[-1]
        cdfs = interp(x,xpts,cdfs)
        cdfs = cdfs/cdfs[-1]
        return cdfs

    def cdfinv(self, p):
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


class Deformable(Dist1d):

    def __init__(self, var, verbose=0, **kwargs):

        self.xstr = var

        sigstr = f'sigma_{var}'
        avgstr = f'avg_{var}'
         
        self.required_params = ['slope_fraction', 'alpha', sigstr, avgstr]
        self.optional_params = ['n_sigma_cutoff']

        self.check_inputs(kwargs)

        self.sigma = kwargs[sigstr]
        self.mean = kwargs[avgstr]

        if('n_sigma_cutoff' in kwargs):
            n_sigma_cutoff=kwars['n_sigma_cutoff']
        else:
            n_sigma_cutoff=3

        sg_params = {'alpha':kwargs['alpha'],  
                    sigstr:self.sigma, 
                    'n_sigma_cutoff':n_sigma_cutoff}

        self.dist={}
        self.dist['super_gaussian'] = SuperGaussian(var, verbose=verbose, **sg_params)

        # SG
        xs = self.dist['super_gaussian'].get_x_pts(10000)
        Px = self.dist['super_gaussian'].pdf(xs)

        # Linear

        lin_params={'slope_fraction':kwargs['slope_fraction'], f'min_{var}':xs[0], f'max_{var}':xs[-1]}
        self.dist['linear'] = Linear(var, verbose=verbose, **lin_params)

        Px = Px*self.dist['linear'].pdf(xs)


        norm = np.trapz(Px, xs)
        assert norm > 0, 'Error: derformable distribution can not be normalized.'
        Px = Px/norm

        avgx = np.trapz(xs*Px, xs)
        stdx = np.sqrt(np.trapz( Px*(xs-avgx)**2, xs))

        #print(avgx, stdx)

        xs = self.mean + (self.sigma/stdx)*(xs-avgx)

        super().__init__(xs=xs, Px=Px, xstr=var)

    def std(self):
        return self.sigma

    def avg(self):
        return self.mean

    def rms(self):
        return np.sqrt(self.std()**2 + self.avg()**2)




class DistTheta(Dist):    

    def __init__(self):
        pass

    def get_theta_pts(self, n):
        return linspace(0*unit_registry('rad'), 2*pi, n)


    def plot_pdf(self, n=1000):

        theta =self.get_theta_pts(n)
        
        p=self.pdf(theta)

        plt.figure()
        plt.plot(theta,p)
        plt.xlabel(f'$theta$ ({str(theta.unit)}))')
        plt.ylabel(f'PDF(${self.theta_str}$) ({str(p.unit)})')


class UniformTheta(DistTheta):
    """
    Defines a uniformly distributed theta over t0 <= min_theta < max_theta <= 2 pi
    """
    def __init__(self, verbose=0, **kwargs):

        self.required_params=['min_theta', 'max_theta']
        self.optional_params=[]
        self.check_inputs(kwargs)
   
        min_theta = kwargs['min_theta']
        max_theta = kwargs['max_theta']

        assert min_theta >= 0.0,  'Min theta value must be >= 0 rad'
        assert max_theta <= 2*pi, 'Max theta value must be <= 2 pi rad'

        self.a = min_theta
        self.b = max_theta
 
        self.range = max_theta-min_theta

        self.Ca = np.cos(self.a)
        self.Sa = np.sin(self.a)

        self.Cb = np.cos(self.b)
        self.Sb = np.sin(self.b)

        vprint('uniform theta', verbose>0, 0, True)
        vprint(f'min_theta = {self.a:G~P}, max_theta = {self.b:G~P}', verbose>0, 2, True)
        
    def avgCos(self):
        return (np.sin(b)-np.sin(a))/self.range

    def avgSin(self):
        return (np.cos(a)-np.cos(b))/self.range

    def avgCos2(self):
        return 0.5*(1 + (self.Cb*self.Sb - self.Ca*self.Sa)/self.range)

    def avgSin2(self):
        return 0.5*(1 - (self.Cb*self.Sb - self.Ca*self.Sa)/self.range)

    def mod2pi(self, thetas):
        return tnp.mod(thetas, 2*pi)

    def pdf(self, thetas):
        return np.full( (len(thetas),), 1/self.range) 

    def cdf(self, thetas):
        return self.mod2pi(thetas)/self.range;
        
    def cdfinv(self, rns):
        return rns*self.range


class DistRad(Dist):

    def __init__(self, rs, Pr):

        self.rs = rs
        self.Pr = Pr
        #self.rb =  centers(rs)

        norm = radint(self.Pr, self.rs)
        if(norm<=0):
            raise ValueError('Normalization of PDF was <= 0')
       
        self.Pr = self.Pr/norm
        self.Cr, self.rb = radcumint(self.Pr, self.rs)
        
    def get_r_pts(self, n):
        return linspace(self.rs[0], self.rs[-1], n)

    def rho(self, r):
        return interp(r, self.rs, self.Pr)

    def pdf(self, r):
        return interp(r, self.rs, self.rs*self.Pr)

    def cdf(self, r):
        return interp(r**2, self.rb**2, self.Cr)

    def cdfinv(self, rns):

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

        a1.plot(r,p)
        a1.set(xlabel=f'r ({r.units:~P})')
        a1.set(ylabel=f'$\\rho_r$(r) ({p.units:~P})')
        
        a2.plot(r,P)
        a2.set(xlabel=f'r ({r.units:~P})')
        a2.set(ylabel=f'PDF(r) ({P.units:~P})')

        plt.tight_layout()

    def plot_cdf(self, n=1000, ax=None):
 
        if(ax is None):
            plt.figure()
            ax = plt.gca() 

        r=self.get_r_pts(n)

        ax.plot(r,self.cdf(r))
        ax.set_xlabel(f'$r$ ({r.units:~P})')
        ax.set_ylabel('CDF$(r)$')

    def avg(self):
        return np.sum( ((self.rb[1:]**3 - self.rb[:-1]**3)/3.0)*self.Pr ) 

    def rms(self):
        return np.sqrt( np.sum( ((self.rb[1:]**4 - self.rb[:-1]**4)/4.0)*self.Pr ) )

    def std(self):
        avg=self.avg()
        rms=self.rms()
        return np.sqrt(rms*rms-avg*avg)

    def test_sampling(self,ax=None):
     
        if(ax is None):
            plt.figure()
            ax = plt.gca()

        rs=self.sample(100000,sequence="hammersley")    
        r = self.get_r_pts(1000)
        p = self.rho(r)
        P = self.pdf(r)

        r_hist, r_edges = radial_histogram(rs, nbins=500)   
        r_bins = centers(r_edges)  
        r_hist = r_hist/radint(r_hist, r_bins)

        avgr = rs.mean()
        stdr = rs.std()

        davgr = self.avg()
        dstdr = self.std()

        ax.plot(r, p, r_bins, r_hist, 'or')
        ax.set_xlabel(f'r ({r.units:~P})')
        ax.set_ylabel(f'$\\rho_r(r)$ ({r_hist.units:~P})')
        ax.set_title(f'Sample stats: <r> = {avgr:G~P}, $\sigma_r$ = {stdr:G~P}\nDist. stats: <r> = {davgr:G~P}, $\sigma_r$ = {dstdr:G~P}')
        ax.legend(['$\\rho_r$(r)','Sampling'])

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

    def __init__(self, verbose=0, **kwargs):
            
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
        vprint(f'{minstr} = {self.rL:G~P}, {maxstr} = {self.rR:G~P}',verbose>0,2,True)

    def get_r_pts(self, n, f=0.2):
        dr = f*np.abs(self.avg())
        return np.linspace(self.rL-dr,self.rR+dr,n)

    def avg(self):
        return (2.0/3.0)*(self.rR**3 - self.rL**3)/(self.rR**2-self.rL**2)

    def rms(self):
        return np.sqrt( (self.rR**2 + self.rL**2)/2.0 )

    def pdf(self, r):
        nonzero = (r >= self.rL) & (r <= self.rR)
        res = np.zeros(len(r))*unit_registry('1/'+str(r.units))
        res[nonzero]=r[nonzero]*2.0/(self.rR**2-self.rL**2)
        #res = res*unit_registry('1/'+str(r.units))
        return res

    def rho(self, r):
        nonzero = (r >= self.rL) & (r <= self.rR)
        res = np.zeros(len(r))*unit_registry('1/'+str(r.units)+'/'+str(r.units))
        res[nonzero]=2/(self.rR**2-self.rL**2)
        #res = res*unit_registry('1/'+str(r.units)+'/'+str(r.units))
        return res

    def cdf(self, r):
        nonzero = (r >= self.rL) & (r <= self.rR)
        res = np.zeros(len(r))*unit_registry('dimensionless')
        res[nonzero]=(r[nonzero]*r[nonzero] - self.rL**2)/(self.rR**2-self.rL**2)
        #res = res*unit_registry('dimensionless')
        return res

    def cdfinv(self, rns):
        return np.sqrt( self.rL**2 + (self.rR**2 - self.rL**2)*rns) 


class LinearRad(DistRad):

    def __init__(self, verbose=0, **kwargs):

        self._n_indent = 2

        self.type='LinearRad'
        #self.xstr = var

        ra_str = f'min_r'
        rb_str = f'max_r'

        self.required_params = ['slope_fraction', ra_str, rb_str]
        self.optional_params = []

        self.check_inputs(kwargs)

        self.a = kwargs[ra_str]
        self.b = kwargs[rb_str]
        self.ratio = kwargs['slope_fraction']
        self.f = 1-np.abs(self.ratio)

        assert self.a < self.b, f'Error: {ra_str} must be < {rb_str}.'
        assert self.a >= 0, f'Error: {ra_str} must be >= 0.'
        assert self.ratio>=-1 and self.ratio <= 1, 'Error: slope fraction must be: -1 <= r < 1.'

        self.dr = self.b-self.a

        if(self.ratio >= 0):
            # Do the maths
            self.pb = 2/(1+self.f)/self.dr
            self.pa = self.f*self.pb

        else:
            # Relabel the other way
            self.pa = 2/(1+self.f)/self.dr
            self.pb = self.f*self.pa

        self.dp = self.pb-self.pa
        self.m = self.dp/self.dr

        vprint('LinearRad',verbose>0,0,True)

    def get_r_pts(self, n, f=0.2):
        return np.linspace(self.a*(1-f), self.b*(1+f), n) 

    def norm(self):

        dr2 = (self.b**2-self.a**2)/2.0
        dr3 = (self.b**3-self.a**3)/3.0

        return 1.0/(self.m*dr3 - self.m*self.a*dr2 + self.pa*dr2)

    def rho(self, r):
        nonzero = (r >= self.a) & (r <= self.b)
        res = np.zeros(len(r))*unit_registry('1/'+str(r.units)+'/'+str(r.units))
        res[nonzero] = self.norm()*(  self.m*(r[nonzero]-self.a) + self.pa)
        return res

    def pdf(self, r):
        return r*self.rho(r)

    def cdf(self, r):

        nonzero = (r >= self.a) & (r <= self.b)
        dr2 = (r[nonzero]**2-self.a**2)/2.0
        dr3 = (r[nonzero]**3-self.a**3)/3.0
        
        res = np.zeros(len(r))*unit_registry('')
        res[nonzero] = self.norm()*(  self.m*dr3 - self.m*self.a*dr2 + self.pa*dr2 )
        return res

    def cdfinv(self, p):
        rpts = self.get_r_pts(10000, f=0)
        cdfs = self.cdf(rpts)
        return interp(p, cdfs, rpts)

    def avg(self):
        rpts = self.get_r_pts(10000, f=0)
        pdfs = self.rho(rpts)
        return radint(pdfs*rpts,rpts)

    def rms(self):
        rpts = self.get_r_pts(10000, f=0)
        pdfs = self.rho(rpts)
        return np.sqrt(radint(pdfs*rpts*rpts,rpts))



class NormRad(DistRad):
    
    def __init__(self, verbose=False, **params):

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

        vprint('radial Gaussian', verbose, 0, True)
        #vprint('underlying sigma_xy

    def canonical_rho(self,xi):
        return (1.0/2.0/pi)*np.exp(-xi**2/2)

    def rho(self, r):

        xi = (r/self.sigma)
        res = np.zeros(len(r))*unit_registry('1/'+str(r.units)+'/'+str(r.units))
        nonzero =  (r>=self.rL) & (r<=self.rR)
        res[nonzero]= self.canonical_rho(xi[nonzero])/self.dp/(self.sigma**2)
        return res

    def pdf(self, r):
   
        xi = (r/self.sigma)
        res = np.zeros(len(r))*unit_registry('1/'+str(r.units))
        nonzero =  (r>=self.rL) & (r<=self.rR)
        res[nonzero] = r[nonzero]*self.canonical_rho(xi[nonzero])/self.dp/self.sigma**2
        return res

    def cdf(self, r):

        res = np.zeros(len(r))*unit_registry('dimensionless')
        nonzero =  (r>=self.rL) & (r<=self.rR)
        xi = (r/self.sigma)
        res[nonzero]=(self.pL - self.canonical_rho(xi[nonzero]))/self.dp
        return res

    def cdfinv(self,rns):
        return np.sqrt( 2*self.sigma**2 * np.log(1/2/pi/( self.pL - self.dp*rns )) ) 

    def get_r_pts(self, n=1000):
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

        return self.sigma*( (xiL*self.pL - xiRpR) + (1.0/2.0/np.sqrt(2*pi))*(erfR-erfL) )/self.dp 

    def rms(self):

        if(self.rR.magnitude==float('Inf')):
            pRrR2 = 0*unit_registry('mm^2')
        else:
            pRrR2 = self.pR*self.rR**2

        pRrL2 = self.pR*self.rL**2
        return np.sqrt( 2*self.sigma**2 + self.rL**2 + (pRrL2 - pRrR2)/self.dp )

class RadFile(DistRad):

    def __init__(self, verbose=0, **params):

        self.required_params=['file','units']
        self.optional_params=[]
        self.check_inputs(params)
        
        distfile = params["file"]
        units = params["units"]
        
        self.distfile = distfile
        
        with open(distfile,'r') as f:
            headers = f.readline().split()

        if(len(headers)!=2):
            raise ValueError("radial distribution file must have two columns")
        data = np.loadtxt(distfile,skiprows=1)

        rs = data[:,0]*unit_registry(units)
        Pr = data[:,1]*unit_registry.parse_expression("1/"+units+"/"+units)
      
        if(np.count_nonzero(rs < 0 )):
            raise ValueError("Radial distribution r-values must be >= 0.")
       
        super().__init__(rs, Pr)

        vprint('radial file', verbose>0, 0, True)
        vprint(f'r-dist file: "{distfile}"', verbose>0, 2, True)

class TukeyRad(DistRad):

    def __init__(self, verbose=0, **kwargs):

        self.required_params=['ratio','length']
        self.optional_params=[]
        self.check_inputs(kwargs)
         
        self.r = kwargs['ratio']
        self.L = kwargs['length']

        vprint("TukeyRad",verbose>0,0,True)
        vprint("legnth = {:0.3f~P}".format(self.L)+", ratio = {:0.3f~P}".format(self.r),verbose>0,2,True)

    def get_r_pts(self, n=1000, f=0.2):
        return np.linspace(0, (1+f)*self.L.magnitude, n)*unit_registry(str(self.L.units))

    def pdf(self, r):        
        return r*self.rho(r)

    def rho(self, r):

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
        res = res/radint(res, r)
        return res
   

    def cdf(self, r):
        rpts = self.get_r_pts(10000)
        pdfs = self.rho(rpts)
        
        cdfs,rbins = radcumint(pdfs, rpts)
        cdfs = cdfs/cdfs[-1]
        cdfs = interp(r,rbins,cdfs)
        cdfs = cdfs/cdfs[-1]
        cdfs*unit_registry('dimensionless')

        return cdfs

    def cdfinv(self, p):
        rpts = self.get_r_pts(10000)
        cdfs = self.cdf(rpts)
        return interp(p, cdfs, rpts)

    def avg(self):
        rpts = self.get_r_pts(10000)
        pdfs = self.rho(rpts)
        return radint(pdfs*rpts, rpts)

    def rms(self):
        rpts = self.get_r_pts(10000)
        pdfs = self.rho(rpts)
        return np.sqrt(radint(pdfs*rpts*rpts, rpts))


class SuperGaussianRad(DistRad):

    def __init__(self, verbose=0, **kwargs):

        self.required_params=[]
        self.optional_params=['p', 'alpha', 'lambda', 'sigma_xy']
        self.check_inputs(kwargs)

        assert not ('alpha' in kwargs and 'p' in kwargs), 'Radial Super Gaussian power parameter must be set using "p" or "alpha", not both.' 
        assert ('alpha' in kwargs or 'p' in kwargs), 'Radial Super Gaussian power parameter must be set using "p" or "alpha". Neither provided.' 

        assert not ('lambda' in kwargs and 'sigma_xy' in kwargs), 'Radial Super Gaussian power parameter must be set using "sigma_xy" or "lambda", not both.' 
        assert ('lambda' in kwargs or 'sigma_xy' in kwargs), 'Radial Super Gaussian power parameter must be set using "sigma_xy" or "lambda". Neither provided.' 

        if('p' in kwargs):
            self.p = kwargs['p']
        elif('alpha' in kwargs):
            alpha = kwargs['alpha']
            assert alpha >= 0 and alpha <= 1, f'SugerGaussian parameter must satisfy 0 <= alpha <= 1, not = {alpha}'
            if(alpha.magnitude==0): 
                self.p = float('Inf')*unit_registry('dimensionless')
            else:
                self.p = 1/alpha 

        if('lambda' in kwargs):
            self.Lambda = kwargs['lambda']
        else:
            self.Lambda = self.get_lambda(kwargs['sigma_xy'])
  
        assert self.p > 0, 'Radial Super Gaussian power p must be > 0.'
 
        vprint('SuperGaussianRad',verbose>0,0,True)
        vprint(f'lambda = {self.Lambda:G~P}, power = {self.p:G~P}',verbose>0,2,True)

    def get_r_pts(self, n=1000):
        
        if(self.p < float('Inf')):
            f = self.p.magnitude
        else:
            f=1

        return np.linspace(0, 5*self.Lambda.magnitude, n)*unit_registry(str(self.Lambda.units))

    def pdf(self, r):        
        rho = self.rho(r)
        return r*rho

    def rho(self, r):

        csi = r/self.Lambda
        nur = 0.5*(csi**2)
        N = (1.0/gamma(1+1.0/self.p)/self.Lambda**2)
        rho = N*np.exp(-np.float_power(nur.magnitude,self.p.magnitude))
        return rho
   

    def cdf(self, r):
        rpts = self.get_r_pts(10000)
        pdfs = self.rho(rpts)
        
        cdfs,rbins = radcumint(pdfs,rpts)
        cdfs = cdfs/cdfs[-1]
        cdfs = interp(r,rbins,cdfs)
        cdfs = cdfs/cdfs[-1]
        cdfs*unit_registry('dimensionless')

        return cdfs

    def cdfinv(self, p):
        rpts = self.get_r_pts(10000)
        cdfs = self.cdf(rpts)
        return interp(p,cdfs,rpts)

    def avg(self):
        return (2.0*np.sqrt(2.0)/3.0)*(gamma(1+3.0/2.0/self.p)/gamma(1+1.0/self.p))*self.Lambda

    def rms(self):
        return np.sqrt( gamma(1+2.0/self.p)/gamma(1+1.0/self.p) )*self.Lambda

    def get_lambda(self, sigma_xy):
        rrms = sigma_xy/np.sqrt(0.5)
        return np.sqrt(gamma(1+1.0/self.p)/gamma(1+2.0/self.p))*rrms

class DeformableRad(DistRad):

    def __init__(self, verbose=0, **kwargs):

        

        sigstr = f'sigma_xy'
        #avgstr = f'avg_{var}'
         
        self.required_params = ['slope_fraction', 'alpha', sigstr]
        self.optional_params = []#['n_sigma_cutoff']

        self.check_inputs(kwargs)

        self.sigma = kwargs[sigstr]
        #self.mean = kwargs[avgstr]

        if('n_sigma_cutoff' in kwargs):
            n_sigma_cutoff=kwars['n_sigma_cutoff']
        else:
            n_sigma_cutoff=3

        sg_params = {'alpha':kwargs['alpha'], sigstr:self.sigma}
                    #'n_sigma_cutoff':n_sigma_cutoff}

        self.dist={}
        self.dist['super_gaussian'] = SuperGaussianRad(verbose=verbose, **sg_params)

        # SG
        rs = self.dist['super_gaussian'].get_r_pts(10000)
        Pr = self.dist['super_gaussian'].rho(rs)

        # Linear
        lin_params={'slope_fraction':kwargs['slope_fraction'], f'min_r':rs[0], f'max_r':rs[-1]}
        self.dist['linear'] = LinearRad(verbose=verbose, **lin_params)

        Pr = Pr*self.dist['linear'].rho(rs)

        norm = radint(Pr, rs)
        assert norm > 0, 'Error: derformable distribution can not be normalized.'
        Pr = Pr/norm

        #avgx = np.trapz(xs*Px, xs)
        stdx = np.sqrt( radint( Pr*rs**2, rs) )/np.sqrt(2)
        rs = (self.sigma/stdx)*rs

        super().__init__(rs=rs, Pr=Pr)

    def rms(self):
        return np.sqrt(2)*self.sigma

    #def avg(self):
    #    return self.mean

    #def rms(self):
    #    return np.sqrt(self.sigma()**2 + self.avg()**2)



class Dist2d(Dist):


    def __init__(self, xs=None, ys=None, Pxy=None, xstr='x', ystr='y', x_unit='', y_unit='', verbose=False):

        if(not isinstance(xs, Quantity)):
            xs = xs*unit_registry(x_unit)

        if(not isinstance(ys, Quantity)):
            ys = ys*unit_registry(y_unit)

        if(not isinstance(Pxy, Quantity)):
            Pxy = Pxy*unit_registry(f'1/{x_unit}/{y_unit}')

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
        norms[norms==0] = 1

        self.Cys[1:,:] = np.cumsum(np.multiply(self.Pxy.magnitude,np.transpose(mlib.repmat(self.dy.magnitude,len(self.xs),1))),axis=0)/norms
        self.Cys=self.Cys*unit_registry("dimensionless")

    def pdf(self, x, sy):
        pass
   
    def plot_pdf(self):
        plt.figure()
        extent = [(self.xs.min()).magnitude,(self.xs.max()).magnitude,(self.ys.min()).magnitude,(self.ys.max()).magnitude]
        plt.imshow(self.Pxy,extent=extent)
        plt.xlabel(self.xstr+" ("+str(self.xs.units)+")")
        plt.ylabel(self.ystr+" ("+str(self.ys.units)+")")
      
    def pdfx(self, x):
        return interp(x,self.xs,self.Px)

    def plot_pdfx(self):
        plt.figure()
        plt.plot(self.xs,self.Px)

    def cdfx(self, x):
        return interp(x,self.xb,self.Cx)

    def plot_cdfx(self):
        plt.figure()
        plt.plot(self.xb,self.Cx)    

    def cdfxinv(self, ps):
        return interp(ps,self.Cx,self.xb)

    def plot_cdfys(self):
        plt.figure()
        for ii in range(len(self.xs)):
            plt.plot(self.yb,self.Cys[:,ii])    

    def sample(self, N, sequence=None, params=None):
        rns = self.rgen.rand((N,2),sequence,params)*unit_registry("dimensionless")
        x,y = self.cdfinv(rns[0,:], rns[1,:])       
        return (x,y)

    def cdfinv(self, rnxs, rnys):

        x = self.cdfxinv(rnxs)
        indx = np.searchsorted(self.xb,x)-1
        
        y = np.zeros(x.shape)*unit_registry(str(x.units))
        for ii in range(self.Cys.shape[1]):
            in_column = (ii==indx)
            if(np.count_nonzero(in_column)>0):
                y[in_column] = interp(rnys[in_column],self.Cys[:,ii],self.yb)

        return (x, y)

    def test_sampling(self):
        x,y = self.sample(100000,sequence="hammersley") 
        plt.figure()
        plt.plot(x, y, '*')


class Image2d(Dist2d):

    def __init__(self, variables, verbose, **params):
        
        vstrs = get_vars(variables)

        assert len(vstrs)==2, f'Wrong number of variables given to Image2d: {len(vstrs)}'
        
        v1 = vstrs[0]
        v2 = vstrs[1]

        self.required_params=['P']
        self.optional_params=[f'min_{v1}',  f'max_{v1}', f'min_{v2}',  f'max_{v2}', v1, v2]
    
        self.check_inputs(params)

        Pxy = params['P']
        assert np.min(np.min(Pxy)) >= 0, 'Error in Image2d: the 2d probability function must be >= 0'   

        if(v1 not in params):

            xmin = params[f'min_{v1}']
            xmax = params[f'max_{v1}'].to(xmin.units)

            assert xmin<xmax, f'Error in Image2d: min {v1} must < max {v1}.'

            xs = linspace(xmin, xmax, Pxy.shape[1])

        else:
            xs = params[v1]

        if(v2 not in params):

            ymin = params[f'min_{v2}'].to(xmin.units)
            ymax = params[f'max_{v2}'].to(xmin.units)

            assert ymin<ymax, f'Error in Image2d: min {v2} must < max {v2}.'

            ys = linspace(ymin, ymax, Pxy.shape[0])

        else:
            ys = params[v2]

        Pxy = Pxy*unit_registry(f'1/{xs.units}/{ys.units}')

        super().__init__(xs, ys, Pxy, xstr=v1, ystr=v2)
        
class File2d(Dist2d):

    def __init__(self, var1, var2, verbose, **params):

        self.required_params=['file']
        self.optional_params=[f'min_{var1}',  f'max_{var1}', f'min_{var2}',  f'max_{var2}', var1, var2, 'threshold']

        self.check_inputs(params)

        filename = params['file']

        ext = (os.path.splitext(filename)[1]).lower()

        if(ext in ['.png', '.jpg', '.jpeg']):

            Pxy  = read_image_file(filename)

            xstr = var1
            ystr = var2

            min_var1_str = f'min_{var1}'
            max_var1_str = f'max_{var1}'

            assert min_var1_str in params, f'Error in File2d: user must specify {min_var1_str}.'
            assert max_var1_str in params, f'Error in File2d: user must specify {max_var1_str}.'

            min_var1 = params[min_var1_str]
            max_var1 = params[max_var1_str]

            xs = linspace(min_var1, max_var1, Pxy.shape[1])

            min_var2_str = f'min_{var2}'
            max_var2_str = f'max_{var2}'

            assert min_var2_str in params, f'Error in File2d: user must specify {min_var2_str}.'
            assert max_var2_str in params, f'Error in File2d: user must specify {max_var2_str}.'

            min_var2 = params[min_var2_str]
            max_var2 = params[max_var2_str]

            ys = linspace(min_var2, max_var2, Pxy.shape[0])

            Pxy = np.flipud(Pxy)
            Pxy = Pxy*unit_registry(f'1/{str(xs.units)}/{str(ys.units)}')

            super().__init__(xs, ys, Pxy, xstr=xstr, ystr=ystr)

        elif(ext=='.txt'):
        
            xs, ys, Pxy, xstr, ystr = read_2d_file(filename)

        else:
            raise ValueError(f'Error: unknown file extension: "{ext}" for filename = {filename}')
    
        if('threshold' in params):
            threshold=params['threshold']
        else:
            threshold=0
        
        assert threshold>=0 and threshold<1, 'Error: image threshold must be >=0 and < 1.'



        under_threshold = Pxy.magnitude < threshold*Pxy.magnitude.max()
        Pxy.magnitude[under_threshold]=0

        print(Pxy.magnitude.max())

        super().__init__(xs, ys, Pxy, xstr=xstr, ystr=ystr)

        vprint('2D File PDF', verbose>0, 0, True)
        vprint(f'2D pdf file: {params["file"]}', verbose>0, 2, True)
        vprint(f'min_{var1} = {min(xs):G~P}, max_{var1} = {max(xs):G~P}', verbose>0, 2, True)
        vprint(f'min_{var2} = {min(ys):G~P}, max_{var2} = {max(ys):G~P}', verbose>0, 2, True)

    
# ---------------------------------------------------------------------------- 
#   This allows the main function to be at the beginning of the file
# ---------------------------------------------------------------------------- 
if __name__ == '__main__':
    
    read_2d_file('')