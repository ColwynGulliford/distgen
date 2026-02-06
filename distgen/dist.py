"""
Defines the random number generator class and distribution function objects.
"""

from .fermi_dirac_3step_barrier_photocathode_model import (
    fermi_dirac_3step_barrier_pdf_bounds_spherical,
    fermi_dirac_3step_barrier_pdf_spherical,
)

from .laser_speckle import generate_speckle_pattern_with_filter

from .hammersley import create_hammersley_samples

from .parsing import convert_input_quantities

from .physical_constants import unit_registry
from .physical_constants import PHYSICAL_CONSTANTS

from .tools import vprint
from .tools import isscalar

from .tools import interp
from .tools import interp2d
from .tools import meshgrid
from .tools import linspace
from .tools import centers
from .tools import spline1d

# Numerical integration
from .tools import trapz
from .tools import cumtrapz
from .tools import radint
from .tools import radcumint

# Histogramming
from .tools import histogram
from .tools import radial_histogram

# Array manipluation
from .tools import flipud

# Special functions
from .tools import erf
from .tools import erfinv
from .tools import gamma

# Imagine handling
from .tools import get_vars
from .tools import SUPPORTED_IMAGE_EXTENSIONS
from .tools import read_2d_file
from .tools import read_image_file

from pint import Quantity

import numpy as np

# import numpy.matlib as mlib
import os

from matplotlib import pyplot as plt

from scipy import integrate, optimize

import warnings


def random_generator(shape, sequence, **kwargs):
    """Returns a set of 'random' (either numpy.random.random or from a Hammersley sequence) numbers"""

    if sequence == "pseudo":

        if "seed" in kwargs:
            rng = np.random.default_rng(kwargs["seed"])    #np.random.seed(kwargs["seed"])
        else:
            rng = np.random.default_rng()

        return rng.random(shape)

    elif sequence == "hammersley":
        dim = shape[0]
        N = shape[1]

        if "burnin" not in kwargs:
            kwargs["burnin"] = -1

        if "primes" not in kwargs:
            kwargs["primes"] = ()

        return np.squeeze(create_hammersley_samples(N, dim=dim, **kwargs))
    else:
        raise ValueError("Sequence: " + str(sequence) + " is not supported")


def get_dist(var, params, verbose=0):
    """
    Translates user input strings and evaluated corrector corresponding distribution function.
    Inputs: var [str] name of variable (x,y,px,...,etc) for distribution,
            dtype [str] user string or shorthand for distribution function
            params [dict] required user parameters for distribution function
            verbose [bool] flag for more or less output to screen
    """
    assert "type" in params, "No distribution type for " + var + " specified."
    dtype = params["type"]

    if dtype == "dist1d":
        dist = Dist1d(xstr=var, **params)
    elif dtype == "uniform" or dtype == "u":
        dist = Uniform(var, verbose=verbose, **params)
    elif dtype == "gaussian" or dtype == "g":
        dist = Norm(var, verbose=verbose, **params)
    elif dtype == "file1d":
        dist = File1d(var, verbose=verbose, **params)
    elif dtype == "tukey":
        dist = Tukey(var, verbose=verbose, **params)
    elif dtype == "maxell_boltzmann" or dtype == "mb":
        dist = MaxwellBoltzmannDist(var, verbose=verbose, **params)
    elif dtype == "fermi_dirac_3step_barrier_photocathode" or dtype == "fd3sb":
        dist = FermiDirac3StepBarrierMomentumDist(verbose=verbose, **params)
    elif dtype == "maxell_boltzmann_kinetic_energy" or dtype == "mbe":
        dist = MaxwellBoltzmannEnergyDist(var, verbose=verbose, **params)
    elif dtype == "super_gaussian" or dtype == "sg":
        dist = SuperGaussian(var, verbose=verbose, **params)
    elif dtype == "superposition" or dtype == "sup":
        dist = Superposition(var, verbose=verbose, **params)
    elif (dtype == "product" or dtype == "pro") and len(var) == 1:
        dist = Product(var, verbose=verbose, **params)
    elif dtype == "interp":
        dist = Interpolation1d(var, verbose=verbose, **params)
    elif (dtype == "product" or dtype == "pro") and len(var) == 2:
        dist = Product2d(var, verbose=verbose, **params)
    elif dtype == "deformable":
        dist = Deformable(var, verbose=verbose, **params)
    elif dtype == 'radial':
        dist = DistRad(params['r'], params['Pr'], verbose=verbose)
    elif (dtype == "radial_uniform" or dtype == "ru") and var == "r":
        dist = UniformRad(verbose=verbose, **params)
    elif (dtype == "radial_gaussian" or dtype == "rg") and var == "r":
        dist = NormRad(verbose=verbose, **params)
    elif (dtype == "radial_super_gaussian" or dtype == "rsg") and var == "r":
        dist = SuperGaussianRad(verbose=verbose, **params)
    elif dtype == "radfile" and var == "r":
        dist = RadFile(verbose=verbose, **params)
    elif dtype == "radial_tukey":
        dist = TukeyRad(verbose=verbose, **params)
    elif dtype == "raddeformable" or dtype == "dr":
        dist = DeformableRad(verbose=verbose, **params)
    elif (dtype == "radial_interpolation" or dtype == "ri") and var == "r":
        dist = InterpolationRad(verbose=verbose, **params)
    elif dtype == "file2d":
        dist = File2d("x", "y", verbose=verbose, **params)
    elif dtype == "crystals":
        dist = TemporalLaserPulseStacking(verbose=verbose, **params)
    elif dtype == "sech2":
        dist = Sech2(verbose=verbose, **params)
    elif dtype == "uniform_theta" or dtype == "ut":
        dist = UniformTheta(verbose=verbose, **params)
    elif dtype == "uniform_phi" or dtype == "up":
        dist = UniformPhi(verbose=verbose, **params)
    elif dtype == "image2d":
        dist = Image2d(var, verbose=verbose, **params)
    elif dtype == "uniform_laser_speckle" and var == "xy":
        dist = UniformLaserSpeckle(verbose=verbose, **params)
    else:
        raise ValueError(f'Distribution type "{dtype}" is not supported.')

    return dist


class Dist:
    """
    Defines a base class for all distributions, and includes functionality for strict input checking
    """

    def __init__(self):
        self._n_indent = 2
        self.optional_params = []

    def check_inputs(self, params):
        """
        Checks the input dictionary to derived class.  Derived class supplies lists of required and optional params.
        """

        params = convert_input_quantities(params)

        # Make sure user isn't passing the wrong parameters:
        allowed_params = (
            self.optional_params + self.required_params + ["verbose", "type", "indent"]
        )
        # print(allowed_params)
        for param in params:
            assert (
                param in allowed_params
            ), f"Incorrect param given to {self.__class__.__name__}.__init__(**kwargs): {param}\nAllowed params: {allowed_params}"

        # Make sure all required parameters are specified
        for req in self.required_params:
            assert (
                req in params
            ), f"Required input parameter {req} to {self.__class__.__name__}.__init__(**kwargs) was not found."

        if "indent" in params:
            self._n_indent = params["indent"]

    def parse_params(params):
        return convert_input_quantities(params)

    def print_dist_methods(self, verbose):
            pass


class Dist1d(Dist):
    """
    Defines the base class for 1 dimensional distribution functions.
    Assumes user will pass in [x,f(x)] as the pdf.
    
    Numerically intergates to find the cdf and to sample the distribution.
    
    Methods should be overloaded for specific derived classes, particularly if
    the distribution allows analytic treatment.
    """

    def __init__(self, xs=None, Px=None, xstr="x", **params):
        super().__init__()

        if Px is None:  # User may supply NumPy arrays for dist
            pstr = f"P{xstr}"

            self.required_params = [xstr, pstr, "units"]
            self.optional_params = []

            self.check_inputs(params)

            xs = params[xstr] * unit_registry(params["units"])
            Px = params[pstr] * unit_registry(f'1/{params["units"]}')

        self.xs = xs
        self.Px = Px
        self.xstr = xstr

        norm = np.trapezoid(self.Px, self.xs)
        if norm <= 0:
            raise ValueError("Normalization of PDF was <= 0")

        self.Px = self.Px / norm
        self.Cx = cumtrapz(self.Px, self.xs)

        self._domain_lower_bound = self.xs[0]
        self._domain_upper_bound = self.xs[-1]

        self._pdf_evaluation_method = 'interp'
        self._pdf_integration_method = 'trapezoid'
        self._cdf_evaluation_method = 'interp'
        self._cdf_inverse_method = 'interp'

    def print_dist_methods(self, verbose):
        vprint(f"PDF evaluation method: {self._pdf_evaluation_method}, domain = [{self._domain_lower_bound:G~P}, {self._domain_lower_bound:G~P}]", verbose > 1, 2, True)
        vprint(f"PDF integration method: {self._pdf_integration_method}, domain = [{self._domain_lower_bound:G~P}, {self._domain_lower_bound:G~P}]", verbose > 1, 2, True)
        vprint(f"CDF evaluation method: {self._cdf_evaluation_method}", verbose > 1, 2, True)
        vprint(f"Inverse CDF method: {self._cdf_inverse_method}", verbose > 1, 2, True)

    def get_x_pts(self, n):
        """
        Returns a vector of x pts suitable for sampling the PDF Px(x)
        """
        return np.linspace(self._domain_lower_bound, self._domain_upper_bound, n)

    def pdf(self, x):
        """ "
        Evaluates the pdf at the user supplied points in x
        """
        return interp(x, self.xs, self.Px)

    def cdf(self, x):
        """ "
        Evaluates the cdf at the user supplied points in x
        """
        return interp(x, self.xs, self.Cx)

    def cdfinv(self, rns):
        """
        Evaluates the inverse of the cdf at probabilities rns
        """
        return interp(rns, self.Cx, self.xs)

    def sample(self, N, sequence, **kwargs):
        """
        Generate coordinates by sampling the underlying pdf
        """
        return self.cdfinv(
            random_generator((1, N), sequence, **kwargs)
            * unit_registry("dimensionless")
        )

    def plot_pdf(self, n=1000):
        """
        Plots the associated pdf function sampled with n points
        """
        
        x = self.get_x_pts(n)
        p = self.pdf(x)  
        plt.figure()
        plt.plot(x, p)
        plt.xlabel(f"{self.xstr} ({x.units:~P})")
        plt.ylabel(f"PDF({self.xstr}) ({p.units:~P})")

    def plot_cdf(self, n=1000):
        """
        Plots the associtated cdf function sampled with n points
        """
        x = self.get_x_pts(n)
        P = self.cdf(x)
        plt.figure()
        plt.plot(x, P)
        plt.xlabel(f"{self.xstr} ({x.units:~P})")
        plt.ylabel(f"CDF({self.xstr}) ({P.units})")

    def avg(self):
        """
        Defines the 1st moment of the pdf, defaults to using trapz integration
        """
        return trapz(self.xs * self.Px, self.xs)

    def rms(self):
        """
        Defines the rms of the pdf, defaults to using trapz integration
        """
        return np.sqrt(trapz(self.xs * self.xs * self.Px, self.xs))

    def std(self):
        """
        Defines the sqrt of the variance of the pdf, defaults to using trapz integration
        """
        avg = self.avg()
        rms = self.rms()
        return np.sqrt(rms * rms - avg * avg)

    def test_sampling(self):
        """
        Useful for verifying the distribution object works correctly when sampling.

        Plots the pdf, cdf, and histogram of 10000 samples from the PDF.
        """
        xs = self.sample(100000, sequence="hammersley")
        x = self.get_x_pts(1000)
        pdf = self.pdf(x)

        rho, edges = histogram(xs, nbins=100)
        xc = centers(edges)
        rho = rho / np.trapezoid(rho, xc)

        savgx = xs.mean()
        sstdx = xs.std()

        davgx = self.avg()
        dstdx = self.std()

        assert np.isclose(savgx, davgx, rtol=1e-3, atol=1e-3)
        assert np.isclose(sstdx, dstdx, rtol=1e-3, atol=1e-3)

        plt.figure()
        plt.plot(x, pdf, xc, rho, "or")
        plt.xlabel(f"{self.xstr} ({x.units:~P})")
        plt.ylabel(f"PDF ({pdf.units:~P})")

        stat_line = (
            f"Sample stats: <{self.xstr}> = {savgx:G~P}, "
            + r"$\sigma_{"
            + str(self.xstr)
            + "}$"
            + f" = {sstdx:G~P}"
        )
        dist_line = (
            f"Dist. stats: <{self.xstr}> = {davgx:G~P}, "
            + r"$\sigma_{"
            + str(self.xstr)
            + "}$"
            + f" = {dstdx:G~P}"
        )

        plt.title(stat_line + "\n" + dist_line)
        plt.legend(["PDF", "Sampling"])


class Superposition(Dist1d):
    """Dist object that allows user to superimpose multiple 1d distributions together to form a new PDF for sampling"""

    def __init__(self, var, verbose, **kwargs):
        self.xstr = var
        assert (
            "dists" in kwargs
        ), 'SuperPositionDist1d must be supplied the key word argument "dists"'

        dist_defs = kwargs["dists"]

        dists = {}

        min_var = 0
        max_var = 0

        if "weights" not in kwargs:
            weights = {name: 1 for name in dist_defs}
        else:
            weights = kwargs["weights"]

        vprint("superpostion", verbose > 0, 0, True)

        for ii, name in enumerate(dist_defs.keys()):
            if name not in weights:
                weights[name] = 1

            dist_defs[name]["indent"] = 3

            vprint(f"{ii+1}. distribution name: {name}, type: ", verbose > 0, 2, False)
            dists[name] = get_dist(var, dist_defs[name], verbose=verbose)

            xi = dists[name].get_x_pts(10)

            if xi[0] < min_var:
                min_var = xi[0]
            if xi[-1] > max_var:
                max_var = xi[-1]

        xs = linspace(min_var, max_var, 10000)

        for ii, name in enumerate(dists.keys()):
            pii = dists[name].pdf(xs)

            assert weights[name] >= 0, "Weights for superpostiion dist must be >= 0."

            if ii == 0:
                ps = weights[name] * pii / np.max(pii.magnitude)
            else:
                ps = ps + weights[name] * pii / np.max(pii.magnitude)

        super().__init__(xs, ps, var)

        #print(f'min_{var} = {self.xL:G~P}, max_{var} = {self.xR:G~P}', verbose>0, 2, True)


class Product(Dist1d):
    """Dist object that allows user to multiply multiple 1d distributions together to form a new PDF for sampling"""

    def __init__(self, var, verbose, **kwargs):
        self.xstr = var
        assert (
            "dists" in kwargs
        ), 'ProductDist 1d must be supplied the key word argument "dists"'
        dist_defs = kwargs["dists"]

        dists = {}

        min_var = 0
        max_var = 0

        for ii, name in enumerate(dist_defs.keys()):
            vprint(f"\ndistribution name: {name}", verbose > 0, 0, True)
            dists[name] = get_dist(var, dist_defs[name], verbose=verbose)

            xi = dists[name].get_x_pts(10)

            if xi[0] < min_var:
                min_var = xi[0]
            if xi[-1] > max_var:
                max_var = xi[-1]

        self.min_var = min_var
        self.max_var = max_var

        if 'integration_method' in kwargs:
            self.integration_method = kwargs['integration_method']
        else:
            self.integration_method = 'trapezoid'

        if self.integration_method == 'trapezoid':
   
            xs = linspace(min_var, max_var, 10000)

            for ii, name in enumerate(dists.keys()):
                pii = dists[name].pdf(xs)
              
                if ii == 0:
                    ps = pii / np.max(pii)
                else:
                    ps = ps * pii / np.max(pii)

            ps = ps.to('dimensionless') * unit_registry(f'1/{xs.units}')
 
            super().__init__(xs, ps, var)

        else:
            raise ValueError('Unsupported integration method.')

        

    def pdf(self, x):

        if self.integration_method == 'trapezoid':
            return super().pdf(x)
        
        else: 

            for ii, name in enumerate(dists.keys()):
                pii = dists[name].pdf(x)
    
                if ii == 0:
                    ps = pii
                else:
                    ps = ps * pii

            



class Uniform(Dist1d):
    """
    Implements a the uniform 1d distribution over a range a <= x <= b.

    """

    def __init__(self, var, verbose=0, **kwargs):
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
        self.optional_params = [
            f"max_{var}",
            f"min_{var}",
            f"avg_{var}",
            f"sigma_{var}",
        ]

        self.check_inputs(kwargs)

        use_min_max = f"max_{var}" in kwargs and f"min_{var}" in kwargs
        use_avg_sigma = f"avg_{var}" in kwargs and f"sigma_{var}" in kwargs

        assert (
            use_min_max ^ use_avg_sigma
        ), f"User must specify either min_{var} and max_{var}] or [avg_{var} and sigma_{var}]"

        if f"min_{var}" in kwargs:
            self.xL = kwargs[f"min_{var}"]
            self.xR = kwargs[f"max_{var}"]
        else:
            length = np.sqrt(12) * kwargs[f"sigma_{var}"]
            avgv = kwargs[f"avg_{var}"]
            self.xL = avgv - length / 2
            self.xR = avgv + length / 2

        # assert (f'max_{var}' in kwargs and f'min_{var}' in kwargs) or (f'avg_{var}' in kwargs and f'sigma_{var}' in kwargs), f'User must specify either min_{var} and max_{var}] or [avg_{var} and sigma_{var}], not both.'
        # self.xL = kwargs[minstr]
        # self.xR = kwargs[maxstr]
        vprint("uniform", verbose > 0, 0, True)
        vprint(
            f"min_{var} = {self.xL:G~P}, max_{var} = {self.xR:G~P}, avg_{var} = {self.avg():G~P}, sigma_{var}: {self.std(): G~P}",
            verbose > 0,
            2,
            True,
        )

        self._pdf_evaluation_method = 'analytic'
        self._pdf_integration_method = 'analytic'
        self._cdf_evaluation_method = 'analytic'
        self._cdf_inverse_method = 'analytic'
        self._domain_lower_bound = -np.inf * self.xL.units
        self._domain_upper_bound = +np.inf * self.xR.units

        self.print_dist_methods(verbose)

    def get_x_pts(self, n, f=0.2):
        """
        Returns n equally spaced x values that sample just over the relevant range of [a,b] (DOES NOT SAMPLE DISTRIBUTION)
        Inputs: n [int]
        """
        dx = f * np.abs(self.avg())
        return np.linspace(self.xL - dx, self.xR + dx, n)

    def pdf(self, x):
        """
        Returns the PDF at coordinate value(s) x
        """

        x_geq_xL_and_leq_xR_int = 1 * ((x >= self.xL) & (x <= self.xR))  
        return x_geq_xL_and_leq_xR_int / (self.xR - self.xL)


    def cdf(self, x):
        """
        Returns the CDF at the values of x [array w/units].  CDF is dimensionless
        """
        
        x_geq_xL_and_leq_xR_int = 1 * ((x >= self.xL) & (x <= self.xR))          
        return x_geq_xL_and_leq_xR_int * (x-self.xL) / (self.xR - self.xL)


    def cdfinv(self, rns):
        """
        Returns the inverse of the CDF function for probabilies rns [array], providing a sampling of the PDF.
        """
        return (self.xR - self.xL) * rns + self.xL

    def avg(self):
        """
        Returns the first moment of the PDF: <x> = (a + b)/2
        """
        return 0.5 * (self.xR + self.xL)

    def std(self):
        """
        Returns the square root of the variance of the PDF: <x> = (b-a)/sqrt(12)
        """
        return (self.xR - self.xL) / np.sqrt(12)

    def rms(self):
        """
        Returns the rms of the distribution computed from the avg and std.
        """
        avg = self.avg()
        std = self.std()
        return np.sqrt(std * std + avg * avg)


class Linear(Dist1d):
    """Defines the PDF and CDF for a linear function in 1d"""

    def __init__(self, var, verbose=0, **kwargs):
        self._n_indent = 2

        self.type = "Linear"
        self.xstr = var

        xa_str = f"min_{var}"
        xb_str = f"max_{var}"

        self.required_params = ["slope_fraction", xa_str, xb_str]
        self.optional_params = []

        self.check_inputs(kwargs)

        self.a = kwargs[xa_str]
        self.b = kwargs[xb_str]
        self.r = kwargs["slope_fraction"]
        self.f = 1 - np.abs(self.r)

        assert self.a < self.b, f"Error: {xa_str} must be < {xb_str}."
        assert (
            self.r >= -1 and self.r <= 1
        ), "Error: slope fraction must be: -1 <= r < 1."

        self.dx = self.b - self.a

        if self.r >= 0:
            # Do the maths
            self.pb = 2 / (1 + self.f) / self.dx
            self.pa = self.f * self.pb

        else:
            # Relabel the other way
            self.pa = 2 / (1 + self.f) / self.dx
            self.pb = self.f * self.pa

        self.dp = self.pb - self.pa
        self.m = self.dp / self.dx

        vprint("Linear", verbose > 0, 0, True)
        # vprint(f'avg_{var} = {self.mu:G~P}, sigma_{var} = {self.sigma:0.3f~P}',verbose>0,self._n_indent,True)
        # if(self.sigma>0):
        #    vprint(f'Left n_sigma_cutoff = {self.b/self.sigma:G~P}, Right n_sigma_cutoff = {self.a/self.sigma:G~P}',verbose>0 and self.b.magnitude<float('Inf'),2,True)
        # else:
        #    vprint(f'Left n_sigma_cutoff = {self.b:G~P}, Right n_sigma_cutoff = {self.a:G~P}',verbose>0 and self.b.magnitude<float('Inf'),2,True)

        self._pdf_evaluation_method = 'analytic'
        self._pdf_integration_method = 'analytic'
        self._cdf_evaluation_method = 'analytic'
        self._cdf_inverse_method = 'analytic'
        self._domain_lower_bound = -np.inf * self.a.units
        self._domain_upper_bound = +np.inf * self.b.units

        self.print_dist_methods(verbose)

    def get_x_pts(self, n, f=0.2):
        """
        Returns n equally spaced x values that sample just over the relevant range of [a,b] (DOES NOT SAMPLE DISTRIBUTION)
        Inputs: n [int]
        """
        dx = f * np.abs(self.avg())
        return np.linspace(self.a - dx, self.b + dx, n)

    def pdf(self, x):
        """
        Returns the PDF at the values in x [array w/units].  PDF has units of 1/[x]
        """
        nonzero = 1 * ((x >= self.a) & (x <= self.b))   
        return nonzero * (self.m * (x - self.a) + self.pa)

    def cdf(self, x):
        """
        Returns the CDF at the values of x [array w/units].  CDF is dimensionless
        """
        nonzero = 1 * ((x >= self.a) & (x <= self.b))
        delta = x - self.a
        return nonzero * (0.5 * self.m * delta**2 + self.pa * delta)

    def cdfinv(self, p):
        return self.a + (np.sqrt(self.pa**2 + 2 * self.m * p) - self.pa) / self.m

    def avg(self):
        """
        Returns the first moment of the PDF:
        """
        d2 = self.b**2 - self.a**2
        d3 = self.b**3 - self.a**3

        return self.pa * d2 / 2.0 + self.m * (d3 / 3.0 - self.a * d2 / 2.0)

    def std(self):
        """
        Returns the square root of the variance of the PDF:
        """
        return np.sqrt(self.rms() ** 2 - self.avg() ** 2)

    def rms(self):
        """
        Returns the rms of the distribution computed from the avg and std.
        """

        d3 = self.b**3 - self.a**3
        d4 = self.b**4 - self.a**4

        ta = self.pa * d3 / 3.0
        tm = self.m * (d4 / 4.0 - self.a * d3 / 3.0)

        return np.sqrt(ta + tm)


class Norm(Dist1d):
    """Defines the PDF and CDF for a normal distribution with truncation on either side"""

    def __init__(self, var, verbose=0, **kwargs):
        self._n_indent = 2

        self.type = "Norm"
        self.xstr = var

        sigmastr = f"sigma_{var}"
        self.required_params = [sigmastr]

        sigma_cutoff_str = "n_sigma_cutoff"
        sigma_cutoff_left = "n_sigma_cutoff_left"
        sigma_cutoff_right = "n_sigma_cutoff_right"
        avgstr = f"avg_{var}"
        self.optional_params = [
            sigma_cutoff_str,
            sigma_cutoff_left,
            sigma_cutoff_right,
            avgstr,
        ]

        self.check_inputs(kwargs)

        self.sigma = kwargs[sigmastr]

        assert self.sigma.magnitude >= 0, "Error: sigma for Norm(1d) must be >= 0"

        if avgstr in kwargs.keys():
            self.mu = kwargs[avgstr]
        else:
            self.mu = 0 * unit_registry(str(self.sigma.units))

        left_cut_set = False
        right_cut_set = False

        assert not (
            sigma_cutoff_str in kwargs.keys()
            and (
                sigma_cutoff_left in kwargs.keys()
                or sigma_cutoff_right in kwargs.keys()
            )
        )

        if sigma_cutoff_str in kwargs.keys():
            self.a = -kwargs[sigma_cutoff_str] * self.sigma + self.mu
            self.b = +kwargs[sigma_cutoff_str] * self.sigma + self.mu

            left_cut_set = True
            right_cut_set = True

        if sigma_cutoff_left in kwargs.keys():
            self.a = kwargs[sigma_cutoff_left] * self.sigma
            left_cut_set = True

        if sigma_cutoff_right in kwargs.keys():
            self.b = kwargs[sigma_cutoff_right] * self.sigma
            right_cut_set = True

        if not left_cut_set:
            self.a = -float("Inf") * self.sigma.units

        if not right_cut_set:
            self.b = +float("Inf") * self.sigma.units

        if self.sigma.magnitude > 0:
            assert (
                self.a < self.b
            ), f"Right side cut off a = {a:G~P} must be < left side cut off b = {b:G~P}"

            self.A = (self.a - self.mu) / self.sigma
            self.B = (self.b - self.mu) / self.sigma

            self.pA = self.canonical_pdf(self.A)
            self.pB = self.canonical_pdf(self.B)

            self.PA = self.canonical_cdf(self.A)
            self.PB = self.canonical_cdf(self.B)

            self.Z = self.PB - self.PA

        else:
            self.A = 0 * unit_registry("dimensionless")
            self.B = 0 * unit_registry("dimensionless")

            self.pA = 0
            self.pB = 0

            self.PA = 0
            self.PB = 0

            self.Z = 1.0

        vprint("Gaussian", verbose > 0, 0, True)
        vprint(
            f"avg_{var} = {self.mu:G~P}, sigma_{var} = {self.sigma:0.3f~P}",
            verbose > 0,
            self._n_indent,
            True,
        )

        if self.sigma > 0:
            vprint(
                f"Left n_sigma_cutoff = {self.b/self.sigma:G~P}, Right n_sigma_cutoff = {self.a/self.sigma:G~P}",
                verbose > 0 and self.b.magnitude < float("Inf"),
                2,
                True,
            )
        else:
            vprint(
                f"Left n_sigma_cutoff = {self.b:G~P}, Right n_sigma_cutoff = {self.a:G~P}",
                verbose > 0 and self.b.magnitude < float("Inf"),
                2,
                True,
            )

        self._pdf_evaluation_method = 'analytic'
        self._pdf_integration_method = 'analytic'
        self._cdf_evaluation_method = 'analytic'
        self._cdf_inverse_method = 'analytic'
        self._domain_lower_bound = -np.inf * self.a.units
        self._domain_upper_bound = +np.inf * self.b.units

        self.print_dist_methods(verbose)

    def get_x_pts(self, n=1000, f=0.1):
        """Returns xpts from [a,b] or +/- 5 sigma, depending on the defintion of PDF"""

        if -float("Inf") < self.a.magnitude:
            lhs = self.a * (1 - f * np.sign(self.a.magnitude))
        else:
            lhs = -5 * self.sigma

        if self.b.magnitude < float("Inf"):
            rhs = self.b * (1 + f * np.sign(self.b.magnitude))
        else:
            rhs = +5 * self.sigma

        return self.mu + linspace(lhs, rhs, n)

    def canonical_pdf(self, csi):
        """Definies the canonical normal distribution"""
        return (1 / np.sqrt(2 * PHYSICAL_CONSTANTS.pi)) * np.exp(-(csi**2) / 2.0)

    def pdf(self, x=None):
        """Define the PDF for non-canonical normal dist including truncations on either side"""

        if x is None:
            x = self.get_x_pts()

        csi = (x - self.mu) / self.sigma
        x_geq_a_and_leq_b_as_int = 1 * (x >= self.a) & (x <= self.b)

        return x_geq_a_and_leq_b_as_int * self.canonical_pdf(csi) / self.Z / self.sigma

    def canonical_cdf(self, csi):
        """Defines the canonical cdf function"""
        return 0.5 * (1 + erf(csi / np.sqrt(2)))

    def cdf(self, x):
        """Define the CDF for non-canonical normal dist including truncations on either side"""
        csi = (x - self.mu) / self.sigma
        x_geq_a_and_leq_b_as_int = 1 * (x >= self.a) & (x <= self.b)
        return  x_geq_a_and_leq_b_as_int * (self.canonical_cdf(csi) - self.PA) / self.Z
        
        #x_out_of_range = (x < self.a) | (x > self.b)
        #res[x_out_of_range] = 0 * unit_registry("dimensionless")
        #return res

    def canonical_cdfinv(self, rns):
        """Define the inverse of the CDF for canonical normal dist including truncations on either side"""
        return np.sqrt(2) * erfinv((2 * rns - 1))

    def cdfinv(self, rns):
        """Define the inverse of the CDF for non-canonical normal dist including truncations on either side"""
        scaled_rns = rns * self.Z + self.PA
        return self.mu + self.sigma * self.canonical_cdfinv(scaled_rns)

    def avg(self):
        """Computes the <x> value of the distribution: <x> = int( x rho(x) dx)"""
        if self.sigma.magnitude > 0:
            return self.mu + self.sigma * (self.pA - self.pB) / self.Z
        else:
            return self.mu

    def std(self):
        """Computes the sigma of the distribution: sigma_x = sqrt(int( (x-<x>)^2 rho(x) dx))"""
        if self.A.magnitude == -float("Inf"):
            ApA = 0 * unit_registry("dimensionless")
        else:
            ApA = self.A * self.pA

        if self.B.magnitude == +float("Inf"):
            BpB = 0 * unit_registry("dimensionless")
        else:
            BpB = self.B * self.pB

        return self.sigma * np.sqrt(
            1 + (ApA - BpB) / self.Z - ((self.pA - self.pB) / self.Z) ** 2
        )

    def rms(self):
        """Computes the rms of the distribution: sigma_x = sqrt(int( x^2 rho(x) dx))"""
        avg = self.avg()
        std = self.std()
        return np.sqrt(std * std + avg * avg)


class SuperGaussian(Dist1d):
    """Distribution  object that samples a 1d Super Gaussian PDF"""

    def __init__(self, var, verbose=0, **kwargs):
        self.type = "SuperGaussian"
        self.xstr = var

        lambda_str = "lambda"
        sigma_str = f"sigma_{var}"
        power_str = "p"
        alpha_str = "alpha"
        avg_str = f"avg_{var}"

        self.required_params = []
        self.optional_params = [
            avg_str,
            power_str,
            alpha_str,
            "lambda",
            sigma_str,
            "n_sigma_cutoff",
        ]
        self.check_inputs(kwargs)

        assert not (
            alpha_str in kwargs and power_str in kwargs
        ), 'SuperGaussian power parameter must be set using "p" or "alpha", not both.'
        assert (
            alpha_str in kwargs or power_str in kwargs
        ), 'SuperGaussian power parameter must be set using "p" or "alpha". Neither provided.'

        assert not (
            sigma_str in kwargs and lambda_str in kwargs
        ), 'SuperGaussian length scale must either be set using "lambda" or "{sigma_str}", not both.'
        assert (
            alpha_str in kwargs or power_str in kwargs
        ), 'SuperGaussian length scale must be set using "lambda" or "{sigma_str}", Neither provided.'

        if power_str in kwargs:
            self.p = kwargs[power_str]

            if isinstance(self.p, float) or isinstance(self.p, int):
                self.p = float(self.p) * unit_registry("dimensionless")

        else:
            alpha = kwargs[alpha_str]
            assert (
                alpha >= 0 and alpha <= 1
            ), "SugerGaussian parameter must satisfy 0 <= alpha <= 1."
            if alpha.magnitude == 0:
                self.p = float("Inf") * unit_registry("dimensionless")
            else:
                self.p = 1 / alpha

        assert self.p > 0, 'SuperGaussian power "p" must be > 0, not p = {self.p}'

        if "lambda" in kwargs:
            self.Lambda = kwargs[lambda_str]
        else:
            self.Lambda = self.get_lambda(kwargs[sigma_str])

        if avg_str in kwargs:
            self.mu = kwargs[avg_str]
        else:
            self.mu = 0 * unit_registry(str(self.Lambda.units))

        if "n_sigma_cutoff" in kwargs:
            self.n_sigma_cutoff = kwargs["n_sigma_cutoff"]
        else:
            self.n_sigma_cutoff = 3

        vprint("Super Gaussian", verbose > 0, 0, True)
        vprint(
            f"sigma_{var} = {self.std():G~P}, power = {self.p:G~P}", verbose, 2, True
        )
        vprint(
            f"n_sigma_cutoff = {self.n_sigma_cutoff}",
            int(verbose >= 1 and self.n_sigma_cutoff != 3),
            2,
            True,
        )

        self._pdf_evaluation_method = 'analytic'
        self._pdf_integration_method = 'trapezoid'
        self._cdf_evaluation_method = 'interpolation'
        self._cdf_inverse_method = 'interpolation'
        self._domain_lower_bound = -np.inf * self.Lambda.units
        self._domain_upper_bound = +np.inf * self.Lambda.units

        self.print_dist_methods(verbose) 

    def pdf(self, x=None):
        """Defines the PDF for super Gaussian function"""
        if x is None:
            x = self.get_x_pts(10000)

        xi = (x - self.mu) / self.Lambda
        nu1 = 0.5 * (xi**2)

        N = 1.0 / 2 / np.sqrt(2) / self.Lambda / gamma(1 + 1.0 / 2.0 / self.p)

        rho = N * np.exp(-np.float_power(nu1.magnitude, self.p.magnitude))

        return rho

    def get_x_pts(self, n=None):
        """
        Returns n equally spaced x values from +/- n_sigma_cutoff*sigma
        """
        if n is None:
            n = 10000
        return self.mu + linspace(
            -self.n_sigma_cutoff * self.std(), +self.n_sigma_cutoff * self.std(), n
        )

    def cdf(self, x):
        """Defines the CDF for the super Gaussian function"""
        xpts = self.get_x_pts(10000)
        # pdfs = self.pdf(xpts)
        cdfs = cumtrapz(self.pdf(xpts), xpts)

        cdfs = cdfs / cdfs[-1]

        cdfs = interp(x, xpts, cdfs)
        cdfs = cdfs / cdfs[-1]
        # print(x[0],cdfs[0])
        return cdfs

    def cdfinv(self, p):
        """Definess the inverse of the CDF for the super Gaussian function"""
        xpts = self.get_x_pts(10000)
        cdfs = self.cdf(xpts)
        return interp(p, cdfs, xpts)

    def avg(self):
        """Returns the average value of x for super Gaussian"""
        return self.mu

    def std(self):
        """Returns the standard deviation of the super Gausssian dist"""
        G1 = gamma(1 + 3.0 / 2.0 / self.p)
        G2 = gamma(1 + 1 / 2 / self.p)
        return self.Lambda * np.sqrt(2 * G1 / 3 / G2)

    def get_lambda(self, sigma):
        """Returns the length scale of the super Gausssian dist"""
        G1 = gamma(1 + 3.0 / 2.0 / self.p)
        G2 = gamma(1 + 1 / 2 / self.p)
        return np.sqrt(3 * G2 / 2.0 / G1) * sigma

    def rms(self):
        """Returns the rms of the super Gausssian dist"""
        avg = self.avg()
        std = self.std()
        return np.sqrt(std * std + avg * avg)


class File1d(Dist1d):
    """Defines an object for loading a 1d PDF from a file and using for particle sampling"""

    def __init__(self, var, verbose=0, **kwargs):
        self.required_params = ["file", "units"]
        self.optional_params = []
        self.check_inputs(kwargs)

        self.xstr = var

        self.distfile = kwargs["file"]
        self.units = kwargs["units"]

        vprint(f'{var}-distribution file: "{self.distfile}"', verbose > 0, 0, True)
        with open(self.distfile, "r") as f:
            headers = f.readline().split()

        if len(headers) != 2:
            raise ValueError("file1D distribution file must have two columns")

        # if(headers[0]!=self.xstr):
        #    raise ValueError("Input distribution file variable must be = "+var)
        # if(headers[1]!="P"+self.xstr):
        #    raise ValueError("Input distribution file pdf name must be = P"+var)

        data = np.loadtxt(self.distfile, skiprows=1)

        xs = data[:, 0] * unit_registry(self.units)
        Px = data[:, 1] * unit_registry.parse_expression("1/" + self.units)

        assert (
            np.count_nonzero(xs.magnitude) > 0
        ), f"Supplied 1d distribution coordinate vector {var} is zero everywhere."
        assert (
            np.count_nonzero(Px.magnitude) > 0
        ), f"Supplied 1d distribution P{var} is zero everywhere."

        super().__init__(xs, Px, self.xstr)


class TemporalLaserPulseStacking(Dist1d):
    """CU style model of birefringent crystal pulse stacking"""

    xstr = "t"
    ts = []
    Pt = []

    def __init__(
        self,
        lengths=None,
        angles=None,
        dv=None,
        wc=None,
        pulse_FWHM=None,
        verbose=0,
        **params,
    ):
        self.verbose = verbose

        vprint("crystal temporal laser shaping", self.verbose > 0, 0, True)

        if lengths is None:
            lengths = []
            for key in params:
                if "crystal_length_" in key:
                    lengths.append(params[key])

        if angles is None:
            angles = []
            for key in params:
                if "crystal_angle_" in key:
                    angles.append(params[key])

        for param in params:
            assert (
                "crystal_angle_" in param
                or "crystal_length" in param
                or param == "type"
            ), (
                "Unknown keyword parameter sent to "
                + self.__class__.__name__
                + ": "
                + param
            )

        if dv is None and "dv" not in params:
            dv = 1.05319 * unit_registry("ps/mm")
        elif dv is None:
            dv = params["dv"]

        if wc is None and "wc" not in params:
            wc = 3622.40686 * unit_registry("THz")
        elif wc is None:
            wc = params["wc"]

        if pulse_FWHM is None and "pulse_FWHM" not in params:
            pulse_FWHM = 1.8 * unit_registry("ps")
        elif pulse_FWHM is None:
            pulse_FWHM = params["pulse_FWHM"]

        self.dV = dv
        self.w0 = wc
        self.laser_pulse_FWHM = pulse_FWHM

        self.crystals = []
        self.total_crystal_length = 0

        self.set_crystals(lengths, angles)
        self.propagate_pulses()

        self.ts = self.get_t_pts(10000)
        self.set_pdf()
        self.set_cdf()

    def set_crystals(self, lengths, angles):
        """Sets the crytal parameters for propagating sech pulses"""

        assert len(lengths) == len(
            angles
        ), "Number of crystal lengths must be the same as the number of angles."

        self.lengths = lengths
        self.angles = angles
        self.angle_offsets = np.zeros(len(angles))

        for ii in range(len(lengths)):
            assert lengths[ii] > 0, "Crystal length must be > 0."
            if ii % 2 == 0:
                angle_offset = -45 * unit_registry("deg")
            else:
                angle_offset = 0 * unit_registry("deg")

            vprint(
                f"crystal {ii+1} length = {self.lengths[ii]:G~P}",
                self.verbose > 0,
                2,
                False,
            )
            vprint(f", angle = {self.angles[ii]:G~P}", self.verbose > 0, 0, True)

            self.crystals.append(
                {
                    "length": lengths[ii],
                    "angle": angles[ii],
                    "angle_offset": angle_offset,
                }
            )

    def propagate_pulses(self):
        """Propagates the sech pulses through each crystal, resulting in two new pulses"""

        self.total_crystal_length = 0
        self.pulses = []

        initial_pulse = {
            "intensity": 1,
            "polarization_angle": 0 * unit_registry("rad"),
            "relative_delay": 0,
        }
        self.pulses.append(initial_pulse)

        for ii in range(len(self.crystals)):
            vprint("applying crystal: " + str(ii + 1), self.verbose > 1, 3, True)
            self.apply_crystal(self.crystals[ii])

        # [t_min t_max] is the default range over which to sample rho(u)
        self.t_max = (
            0.5 * self.total_crystal_length * self.dV + 5.0 * self.laser_pulse_FWHM
        )
        self.t_min = -self.t_max

        vprint(
            f"Pulses propagated: min t = {self.t_min:G~P}, max t = {self.t_max:G~P}",
            self.verbose > 0,
            2,
            True,
        )

    def apply_crystal(self, next_crystal):
        """Generates two new pulses from an incoming pulse in a given crystal"""

        # add to total crystal length
        self.total_crystal_length += next_crystal["length"]

        theta_fast = next_crystal["angle"] - next_crystal["angle_offset"]
        theta_slow = theta_fast + 0.5 * PHYSICAL_CONSTANTS.pi

        new_pulses = []

        for initial_pulse in self.pulses:
            # the sign convention is chosen so that (-) time represents the head of the electron bunch,
            # and (+) time represents the tail

            # create new pulses:
            pulse_fast = {}
            pulse_slow = {}

            pulse_fast["intensity"] = initial_pulse["intensity"] * np.cos(
                initial_pulse["polarization_angle"] - theta_fast
            )
            pulse_fast["polarization_angle"] = theta_fast
            pulse_fast["relative_delay"] = (
                initial_pulse["relative_delay"] - self.dV * next_crystal["length"] * 0.5
            )

            pulse_slow["intensity"] = initial_pulse["intensity"] * np.cos(
                initial_pulse["polarization_angle"] - theta_slow
            )
            pulse_slow["polarization_angle"] = theta_slow
            pulse_slow["relative_delay"] = (
                initial_pulse["relative_delay"] + self.dV * next_crystal["length"] * 0.5
            )

            new_pulses.append(pulse_fast)
            new_pulses.append(pulse_slow)

        self.pulses = new_pulses

    def evaluate_sech_fields(self, axis_angle, pulse, t, field):
        """Evaluates the electric field of the sech pulses"""

        # Evaluates the real and imaginary parts of one component of the E-field:
        normalization = pulse["intensity"] * np.cos(
            pulse["polarization_angle"] - axis_angle
        )
        w = 2 * np.arccosh(np.sqrt(2)) / self.laser_pulse_FWHM

        field[0] = field[0] + normalization * np.cos(
            self.w0 * (t - pulse["relative_delay"])
        ) / np.cosh(w * (t - pulse["relative_delay"]))
        field[1] = field[1] + normalization * np.sin(
            self.w0 * (t - pulse["relative_delay"])
        ) / np.cosh(w * (t - pulse["relative_delay"]))

    def get_t_pts(self, n):
        return linspace(self.t_min, self.t_max.to(self.t_min), n)

    def get_x_pts(self, n):
        return self.get_t_pts(n)

    def set_pdf(self):
        """Evaluates the sech fields and computes the square of the
        fields for intenstity in order to set the distribution"""

        ex = np.zeros((2, len(self.ts))) * unit_registry("")
        ey = np.zeros((2, len(self.ts))) * unit_registry("")

        for pulse in self.pulses:
            self.evaluate_sech_fields(0.5 * PHYSICAL_CONSTANTS.pi, pulse, self.ts, ex)
            self.evaluate_sech_fields(0.0, pulse, self.ts, ey)

        self.Pt = (
            (ex[0, :] ** 2 + ex[1, :] ** 2) + (ey[0, :] ** 2 + ey[1, :] ** 2)
        ).magnitude * unit_registry("THz")
        self.Pt = self.Pt / trapz(self.Pt, self.ts)

    def set_cdf(self):
        """Computes the CDF of the distribution"""
        self.Ct = cumtrapz(self.Pt, self.ts)

    def pdf(self, t):
        """Returns the PDF at the values in t"""
        return interp(t, self.ts, self.Pt)

    def cdf(self, t):
        """Returns the CDF at the values of t"""
        return interp(t, self.ts, self.Ct)

    def cdfinv(self, rns):
        """Computes the inverse of the CDF at probabilities rns"""
        return interp(rns * unit_registry(""), self.Ct, self.ts)

    def avg(self):
        """Computes the expectation value of t of the distribution"""
        return trapz(self.ts * self.Pt, self.ts)

    def std(self):
        """Computes the sigma of the PDF"""
        return np.sqrt(trapz(self.ts * self.ts * self.Pt, self.ts))

    # def get_params_list(self,var):
    #   """ Returns the crystal parameter list"""
    #    return (["crystal_length_$N","crystal_angle_$N"],["laser_pulse_FWHM","avg_"+var,"std_"+var])


class Sech2(Dist1d):
    """
    Defines a (sech(t))^2 function, which is useful for modeling the temporal shape of a pulsed laser
    """

    def __init__(self, verbose=0, **kwargs):
        self.required_params = []
        self.optional_params = ["avg_t", "n_tau_cutoff", "tau", "sigma_t"]
        self.check_inputs(kwargs)

        assert not (
            "tau" in kwargs and "sigma_t" in kwargs
        ), "User must specify either tau or sigma_t."

        self.xstr = "t"

        if "tau" in kwargs:
            self.tau = kwargs["tau"]
            reset_tau = False
        elif "sigma_t" in kwargs:
            self.tau = kwargs[
                "sigma_t"
            ]  # Incorrect, but will be used as a starting guess
            self.sigma = kwargs["sigma_t"]
            reset_tau = True

        if "avg_t" in kwargs:
            self.mu = kwargs["avg_t"]
        else:
            self.mu = 0 * unit_registry(str(self.tau.units))

        if "n_tau_cutoff" in kwargs:
            self.n_tau_cutoff = kwargs["n_tau_cutoff"]
        else:
            self.n_tau_cutoff = 3

        self.set_dist_params(self.tau)

        if reset_tau:
            self.tau = self.reset_tau_from_sigma()

        else:
            self.sigma = self.get_sigma_from_tau()

        vprint("Sech2", verbose > 0, 0, True)
        vprint(
            f"sigma_t = {self.sigma:G~P}, tau = {self.tau:G~P}", verbose > 0, 2, True
        )

    def get_x_pts(self, n):
        return linspace(self.min_t, self.max_t, 10_000)

    def pdf(self, t):
        eta = (t - self.mu) / self.tau
        return self.norm / np.cosh(eta) ** 2

    def cdf(self, t):
        eta = (t - self.mu) / self.tau
        return (self.norm * self.tau) * (np.tanh(eta) - np.tanh(self.eta_min))

    def cdfinv(self, p):
        alpha = p / (self.norm * self.tau)
        return self.mu + self.tau * np.arctanh(alpha + np.tanh(self.eta_min))

    def avg(self):
        return self.mu

    def std(self):
        return self.sigma

    def tau(self):
        return self.tau

    @property
    def t(self):
        return self.get_x_pts(10_000)

    def get_sigma_from_tau(self):
        rmsx = integrate.quad(
            lambda x: x**2 / np.cosh(x) ** 2, self.eta_min, self.eta_max
        )[0]

        return self.tau * np.sqrt(self.NT * rmsx)

    def reset_tau_from_sigma(self):
        def sigma_squared_error(T):
            self.set_dist_params(T[0] * unit_registry(str(self.sigma.units)))
            return (
                self.get_sigma_from_tau().magnitude - self.sigma.magnitude
            ) ** 2 / self.sigma.magnitude**2

        res = optimize.minimize(sigma_squared_error, self.sigma)

        return res["x"][0] * unit_registry(str(self.sigma.units))

    def set_dist_params(self, T):
        self.tau = T

        self.min_t = self.mu - self.n_tau_cutoff * self.tau
        self.max_t = self.mu + self.n_tau_cutoff * self.tau

        self.eta_min = (self.min_t - self.mu) / self.tau
        self.eta_max = (self.max_t - self.mu) / self.tau

        self.NT = 1 / (np.tanh(self.eta_max) - np.tanh(self.eta_min))
        self.norm = self.NT / self.tau


class Tukey(Dist1d):
    """
    Defines a 1d Tukey Window distribution.  This is a flat distribution with
    cosine like tails on each end.
    """

    def __init__(self, var, verbose=0, **kwargs):
        self.xstr = var

        self.required_params = ["ratio", "length"]
        self.optional_params = []
        self.check_inputs(kwargs)

        self.r = kwargs["ratio"]
        self.L = kwargs["length"]

        vprint("Tukey", verbose > 0, 0, True)
        vprint(f"length = {self.L:G~P}, ratio = {self.r:G~P}", verbose > 0, 2, True)

    def get_x_pts(self, n):
        return 1.1 * linspace(-self.L / 2.0, self.L / 2.0, n)

    def pdf(self, x):
        res = np.zeros(x.shape) * unit_registry("1/" + str(self.L.units))

        if self.r == 0:
            flat_region = np.logical_and(x <= self.L / 2.0, x >= -self.L / 2.0)
            res[flat_region] = 1 / self.L

        else:
            Lflat = self.L * (1 - self.r)
            Lcos = self.r * self.L / 2.0
            pcos_region = np.logical_and(x >= +Lflat / 2.0, x <= +self.L / 2.0)
            mcos_region = np.logical_and(x <= -Lflat / 2.0, x >= -self.L / 2.0)
            flat_region = np.logical_and(x < Lflat / 2.0, x > -Lflat / 2.0)
            res[pcos_region] = (
                0.5
                * (
                    1
                    + np.cos(
                        (PHYSICAL_CONSTANTS.pi / Lcos) * (x[pcos_region] - Lflat / 2.0)
                    )
                )
                / self.L
            )
            res[mcos_region] = (
                0.5
                * (
                    1
                    + np.cos(
                        (PHYSICAL_CONSTANTS.pi / Lcos) * (x[mcos_region] + Lflat / 2.0)
                    )
                )
                / self.L
            )
            res[flat_region] = 1.0 / self.L

            res[x < -self.L] = 0 * unit_registry("1/" + str(self.L.units))

        return res / trapz(res, x)

    def cdf(self, x):
        xpts = self.get_x_pts(10000)
        # pdfs = self.pdf(xpts)
        cdfs = cumtrapz(self.pdf(xpts), xpts)
        cdfs = cdfs / cdfs[-1]
        cdfs = interp(x, xpts, cdfs)
        cdfs = cdfs / cdfs[-1]
        return cdfs

    def cdfinv(self, p):
        xpts = self.get_x_pts(10000)
        cdfs = self.cdf(xpts)
        return interp(p, cdfs, xpts)

    def avg(self):
        xpts = self.get_x_pts(10000)
        return trapz(self.pdf(xpts) * xpts, xpts)

    def std(self):
        xpts = self.get_x_pts(10000)
        avgx = self.avg()
        return np.sqrt(trapz(self.pdf(xpts) * (xpts - avgx) * (xpts - avgx), xpts))

    def rms(self):
        avg = self.avg()
        std = self.std()
        return np.sqrt(std * std + avg * avg)


class Deformable(Dist1d):
    def __init__(self, var, verbose=0, **kwargs):
        self.xstr = var

        sigstr = f"sigma_{var}"
        avgstr = f"avg_{var}"

        self.required_params = ["slope_fraction", "alpha", sigstr, avgstr]
        self.optional_params = ["n_sigma_cutoff"]

        self.check_inputs(kwargs)

        self.sigma = kwargs[sigstr]
        self.mean = kwargs[avgstr]

        if "n_sigma_cutoff" in kwargs:
            n_sigma_cutoff = kwargs["n_sigma_cutoff"]
        else:
            n_sigma_cutoff = 3

        sg_params = {
            "alpha": kwargs["alpha"],
            sigstr: self.sigma,
            "n_sigma_cutoff": n_sigma_cutoff,
        }

        self.dist = {}
        self.dist["super_gaussian"] = SuperGaussian(var, verbose=verbose, **sg_params)

        # SG
        xs = self.dist["super_gaussian"].get_x_pts(10000)
        Px = self.dist["super_gaussian"].pdf(xs)

        # Linear

        lin_params = {
            "slope_fraction": kwargs["slope_fraction"],
            f"min_{var}": xs[0],
            f"max_{var}": xs[-1],
        }
        self.dist["linear"] = Linear(var, verbose=verbose, **lin_params)

        Px = Px * self.dist["linear"].pdf(xs)

        norm = np.trapezoid(Px, xs)
        assert norm > 0, "Error: derformable distribution can not be normalized."
        Px = Px / norm

        avgx = np.trapezoid(xs * Px, xs)
        stdx = np.sqrt(np.trapezoid(Px * (xs - avgx) ** 2, xs))

        # print(avgx, stdx)

        xs = self.mean + (self.sigma / stdx) * (xs - avgx)

        super().__init__(xs=xs, Px=Px, xstr=var)

    def std(self):
        return self.sigma

    def avg(self):
        return self.mean

    def rms(self):
        return np.sqrt(self.std() ** 2 + self.avg() ** 2)


class MaxwellBoltzmannDist(Dist1d):
    def __init__(self, var, verbose, **kwargs):
        self.xstr = var

        self.required_params = [f"scale_{var}"]
        self.optional_params = []

        self.check_inputs(kwargs)

        self.a = kwargs[f"scale_{var}"]
        self.f = np.sqrt(2 / np.pi)

        self.xs = self.get_x_pts(10000)
        self.Cx = self.cdf(self.xs)

        vprint("Maxwell-Boltzmann", verbose > 0, 0, True)
        vprint(f"{var} scale = {self.a:G~P}", verbose > 0, 2, True)

    def get_x_pts(self, n):
        return linspace(0 * unit_registry(str(self.a.units)), 5 * self.a, n)

    def pdf(self, x):
        xhat = x / self.a
        return (1 / self.a) * self.f * xhat**2 * np.exp(-(xhat**2) / 2.0)

    def cdf(self, x):
        xhat = x / self.a
        return erf(xhat / np.sqrt(2)) - self.f * xhat * np.exp(-(xhat**2) / 2.0)

    def avg(self):
        return 2 * self.a * self.f

    def std(self):
        return self.a * np.sqrt((3 * np.pi - 8) / PHYSICAL_CONSTANTS.pi)

    def rms(self):
        return np.sqrt(self.std() ** 2 + self.avg() ** 2)


class MaxwellBoltzmannEnergyDist(Dist1d):
    def __init__(self, var, verbose, **kwargs):
        self.xstr = var

        self.required_params = ["kT"]
        self.optional_params = []
        self.check_inputs(kwargs)

        self.a = kwargs["kT"]
        self.f = np.sqrt(2 / np.pi)

        xs = self.get_x_pts(10000)
        self.xs = xs

        super().__init__(xs=xs, Px=self.pdf(xs), xstr=var)

        vprint("Maxwell-Boltzmann Energy", verbose > 0, 0, True)
        vprint(f"kT = {self.a:G~P}", verbose > 0, 2, True)

    def get_x_pts(self, n):
        return linspace(0 * unit_registry(str(self.a.units)), 10 * self.a, n)

    def pdf(self, x):
        xhat = x / self.a
        return (1 / self.a) * self.f * np.sqrt(xhat) * np.exp(-xhat)

    # def cdf(self, x):
    #    xhat = x/self.a
    #    return erf(xhat/np.sqrt(2)) - self.f * xhat * np.exp(-xhat**2/2.0)

    # def avg(self):
    #    return 2*self.a*self.f

    # def std(self):
    #    return self.a*np.sqrt( (3*np.pi-8)/np.pi )

    # def rms(self):
    #    return np.sqrt(self.std()**2 + self.avg()**2)


class Interpolation1d(Dist1d):
    def __init__(self, var, verbose=0, **kwargs):
        self.xstr = var

        sigstr = f"sigma_{var}"
        avgstr = f"avg_{var}"

        self.required_params = [f"P{var}", sigstr, avgstr, "method"]
        self.optional_params = [f"{var}", "n_pts"]

        self.check_inputs(kwargs)

        self.sigma = kwargs[sigstr]
        self.mean = kwargs[avgstr]

        pts = kwargs[f"P{var}"]
        self.method = kwargs["method"]

        if isinstance(pts, list):
            pts = np.array(pts)

        elif isinstance(pts, dict):
            pts = np.array([v for k, v in pts.items()])

        if "n_pts" in kwargs:
            n_pts = kwargs["n_pts"]
        else:
            n_pts = 1000

        if f"{var}s" in kwargs:
            pass
        else:
            xs = np.linspace(-1, 1, len(pts))

        units = self.mean.units

        xs = xs * unit_registry(str(units))
        Px = pts * unit_registry.parse_expression(f"1/{units}")

        # Save the original curve
        self.x0s = xs

        # Do interpolation
        xs, Px = self.interpolate1d(xs, Px, method=kwargs["method"], n_pts=n_pts)

        Px = Px / trapz(Px, xs)
        avgx = trapz(xs * Px, xs)
        stdx = np.sqrt(trapz((xs - avgx) ** 2 * Px, xs))
        scale = self.sigma / stdx

        xs = scale * xs
        Px = Px / scale

        avgx = trapz(xs * Px, xs)
        stdx = np.sqrt(trapz((xs - avgx) ** 2 * Px, xs))

        xs = xs - avgx + self.mean

        # Make sure interoplation doesn't yield negative values
        Px[Px.magnitude < 0] = 0 * unit_registry.parse_expression(f"1/{units}")

        super().__init__(xs=xs, Px=Px, xstr=var)

    def interpolate1d(self, x, y, method="spline", n_pts=1000, s=0.0, k=3):
        xs = linspace(x[0], x[-1], n_pts)

        if method == "spline":
            Px = spline1d(xs, x, y, s=s, k=k)

        return xs, Px

    def avg(self):
        return self.mean

    def std(self):
        return self.sigma


class DistTheta(Dist):
    def __init__(self):
        pass

    def get_theta_pts(self, n):
        return linspace(0 * unit_registry("rad"), 2 * PHYSICAL_CONSTANTS.pi, n)

    def plot_pdf(self, n=1000):
        theta = self.get_theta_pts(n)

        p = self.pdf(theta)

        plt.figure()
        plt.plot(theta, p)
        plt.xlabel(f"$\theta$ ({str(theta.unit)}))")
        plt.ylabel(f"PDF(${self.theta_str}$) ({str(p.unit)})")


class UniformTheta(DistTheta):
    """
    Defines a uniformly distributed theta over t0 <= min_theta < max_theta <= 2 pi
    """

    def __init__(self, verbose=0, **kwargs):
        self.required_params = ["min_theta", "max_theta"]
        self.optional_params = []
        self.check_inputs(kwargs)

        min_theta = kwargs["min_theta"]
        max_theta = kwargs["max_theta"]

        assert min_theta >= 0.0, "Min theta value must be >= 0 rad"
        assert (
            max_theta <= 2 * PHYSICAL_CONSTANTS.pi
        ), "Max theta value must be <= 2 pi rad"

        self.a = min_theta
        self.b = max_theta

        self.range = max_theta - min_theta

        self.Ca = np.cos(self.a)
        self.Sa = np.sin(self.a)

        self.Cb = np.cos(self.b)
        self.Sb = np.sin(self.b)

        vprint("uniform theta", verbose > 0, 0, True)
        vprint(
            f"min_theta = {self.a:G~P}, max_theta = {self.b:G~P}", verbose > 0, 2, True
        )

    def avgCos(self):
        return (np.sin(self.b) - np.sin(self.a)) / self.range

    def avgSin(self):
        return (np.cos(self.a) - np.cos(self.b)) / self.range

    def avgCos2(self):
        return 0.5 * (1 + (self.Cb * self.Sb - self.Ca * self.Sa) / self.range)

    def avgSin2(self):
        return 0.5 * (1 - (self.Cb * self.Sb - self.Ca * self.Sa) / self.range)

    def mod2pi(self, thetas):
        return np.mod(thetas, 2 * PHYSICAL_CONSTANTS.pi)

    def pdf(self, thetas):
        return np.full((len(thetas),), 1 / self.range)

    def cdf(self, thetas):
        return self.mod2pi(thetas) / self.range

    def cdfinv(self, rns):
        return rns * self.range


class DistPhi(Dist):
    def __init__(self):
        pass

    def get_phi_pts(self, n):
        return linspace(0 * unit_registry("rad"), PHYSICAL_CONSTANTS.pi, n)

    def plot_pdf(self, n=1000):
        phi = self.get_phi_pts(n)

        p = self.pdf(phi)

        plt.figure()
        plt.plot(phi, p)
        plt.xlabel(f"$theta$ ({str(phi.unit)}))")
        plt.ylabel(f"PDF(${self.phi_str}$) ({str(p.unit)})")


class UniformPhi(DistPhi):
    """
    Defines a uniformly distributed theta over t0 <= min_phi < max_phi <= pi
    """

    def __init__(self, verbose=0, **kwargs):
        self.required_params = ["min_phi", "max_phi"]
        self.optional_params = []
        self.check_inputs(kwargs)

        min_phi = kwargs["min_phi"]
        max_phi = kwargs["max_phi"]

        assert min_phi >= 0.0, "Min phi value must be >= 0 rad"
        assert max_phi <= PHYSICAL_CONSTANTS.pi, "Max phi value must be <= pi rad"

        self.a = min_phi
        self.b = max_phi

        self.range = max_phi - min_phi

        self.Ca = np.cos(self.a)
        self.Sa = np.sin(self.a)

        self.Cb = np.cos(self.b)
        self.Sb = np.sin(self.b)

        vprint("uniform phi", verbose > 0, 0, True)
        vprint(f"min_phi = {self.a:G~P}, max_phi = {self.b:G~P}", verbose > 0, 2, True)

    def avgCos(self):
        return (self.Sb - self.Sa) / (self.Ca - self.Cb)

    def avgSin(self):
        return 0.5 * (self.range - 0.5 * (np.sin(2 * self.b) - np.sin(2 * self.a)))

    def avgCos2(self):
        return (1 / 3.0) * (self.Ca**3 - self.Cb**3) / (self.Ca - self.Cb)

    def avgSin2(self):
        return 1 - self.avgCos2()

    def mod2pi(self, phis):
        return np.mod(phis, 2 * PHYSICAL_CONSTANTS.pi)

    def pdf(self, phis):
        return np.sin(phis) / (self.Ca - self.Cb)

    def cdf(self, phis):
        return (self.Ca - np.cos(phis)) / (self.Ca - self.Cb)

    def cdfinv(self, rns):
        return np.arccos(self.Ca - rns * (self.Ca - self.Cb))


class DistRad(Dist):
    def __init__(self, rs, Pr, verbose=0,):
        
        self.rs = rs
        self.Pr = Pr
        # self.rb =  centers(rs)

        norm = radint(self.Pr, self.rs)
        if norm <= 0:
            raise ValueError("Normalization of PDF was <= 0")

        self.Pr = self.Pr / norm
        self.Cr, self.rb = radcumint(self.Pr, self.rs)

        self.var_type = "radial"

        vprint('radial dist', verbose > 0, 0, True)

    def get_r_pts(self, n):
        return linspace(self.rs[0], self.rs[-1], n)

    def rho(self, r):
        return interp(r, self.rs, self.Pr)

    def rho_xy(self, x, y):
        X, Y = meshgrid(x, y)
        return interp(np.sqrt(X**2 + Y**2), self.rs, self.Pr)

    def pdf(self, r):
        return interp(r, self.rs, self.rs * self.Pr)

    def pdf_xy(self, x, y):
        X, Y = meshgrid(x, y)
        return interp(np.sqrt(X**2 + Y**2), self.rs, self.rs * self.Pr)

    def cdf(self, r):
        return interp(r**2, self.rb**2, self.Cr)

    def cdfinv(self, rns):
        rns = np.squeeze(rns)
        indsL = np.searchsorted(self.Cr, rns) - 1
        indsH = indsL + 1

        c1 = self.Cr[indsL]
        c2 = self.Cr[indsH]

        r1 = self.rb[indsL]
        r2 = self.rb[indsH]

        same_rs = (r1.magnitude) == (r2.magnitude)
        diff_rs = np.logical_not(same_rs)

        r = np.zeros(rns.shape) * unit_registry(str(self.rs.units))

        r[same_rs] = r1[same_rs]
        r[diff_rs] = np.sqrt(
            (
                r2[diff_rs] * r2[diff_rs] * (rns[diff_rs] - c1[diff_rs])
                + r1[diff_rs] * r1[diff_rs] * (c2[diff_rs] - rns[diff_rs])
            )
            / (c2[diff_rs] - c1[diff_rs])
        )

        return r

    def sample(self, N, sequence="hammersley", **kwargs):
        return self.cdfinv(
            random_generator((1, N), sequence, *kwargs) * unit_registry("dimensionless")
        )

    def plot_pdf(self, n=1000):
        r = self.get_r_pts(n)
        p = self.rho(r)
        P = self.pdf(r)
        fig, (a1, a2) = plt.subplots(1, 2)

        a1.plot(r, p)
        a1.set(xlabel=f"r ({r.units:~P})")
        a1.set(ylabel=f"$\\rho_r$(r) ({p.units:~P})")

        a2.plot(r, P)
        a2.set(xlabel=f"r ({r.units:~P})")
        a2.set(ylabel=f"PDF(r) ({P.units:~P})")

        plt.tight_layout()

    def plot_cdf(self, n=1000, ax=None):
        if ax is None:
            plt.figure()
            ax = plt.gca()

        r = self.get_r_pts(n)

        ax.plot(r, self.cdf(r))
        ax.set_xlabel(f"$r$ ({r.units:~P})")
        ax.set_ylabel("CDF$(r)$")

    def avg(self):
        return np.sum(((self.rb[1:] ** 3 - self.rb[:-1] ** 3) / 3.0) * self.Pr)

    def rms(self):
        return np.sqrt(np.sum(((self.rb[1:] ** 4 - self.rb[:-1] ** 4) / 4.0) * self.Pr))

    def std(self):
        avg = self.avg()
        rms = self.rms()
        return np.sqrt(rms * rms - avg * avg)

    def test_sampling(self, ax=None):
        if ax is None:
            plt.figure()
            ax = plt.gca()

        rs = self.sample(100000, sequence="hammersley")
        r = self.get_r_pts(1000)
        p = self.rho(r)
        # P = self.pdf(r)

        r_hist, r_edges = radial_histogram(rs, nbins=500)
        r_bins = centers(r_edges)
        r_hist = r_hist / radint(r_hist, r_bins)

        avgr = rs.mean()
        stdr = rs.std()

        davgr = self.avg()
        dstdr = self.std()

        assert np.isclose(avgr, davgr, rtol=1e-3, atol=1e-3)
        assert np.isclose(stdr, dstdr, rtol=1e-3, atol=1e-3)

        ax.plot(r, p, r_bins, r_hist, "or")
        ax.set_xlabel(f"r ({r.units:~P})")
        ax.set_ylabel(f"$\\rho_r(r)$ ({r_hist.units:~P})")
        ax.set_title(
            rf"Sample stats: <r> = {avgr:G~P}, $\sigma_r$ = {stdr:G~P}\nDist. stats: <r> = {davgr:G~P}, $\sigma_r$ = {dstdr:G~P}"
        )
        ax.legend(["$\\rho_r$(r)", "Sampling"])

    def get_x_pts(self, m):
        r = self.get_r_pts(2)
        return linspace(-r[-1], r[-1], m)

    def get_y_pts(self, n):
        r = self.get_r_pts(2)
        return linspace(-r[-1], r[-1], n)

    def get_xy_pts(self, m, n=None):
        if n is None:
            n = m

        return (self.get_x_pts(m), self.get_y_pts(n))

    def min_dx(self):
        return np.diff(self.rs).min()

    def min_dy(self):
        return np.diff(self.rs).min()


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

        self.required_params = [maxstr]
        self.optional_params = [minstr]
        self.check_inputs(kwargs)

        self.rR = kwargs[maxstr]

        if minstr in kwargs.keys():
            self.rL = kwargs[minstr]
        else:
            self.rL = 0 * unit_registry(str(self.rR.units))

        if self.rL >= self.rR:
            raise ValueError("Radial uniform dist must have rL < rR")
        if self.rR < 0:
            raise ValueError("Radial uniform dist must have rR >= 0")

        vprint("radial uniform", verbose > 0, 0, True)
        vprint(
            f"{minstr} = {self.rL:G~P}, {maxstr} = {self.rR:G~P}", verbose > 0, 2, True
        )

    def get_r_pts(self, n, f=0.2):
        dr = f * np.abs(self.avg())
        return np.linspace(self.rL - dr, self.rR + dr, n)

    def avg(self):
        return (2.0 / 3.0) * (self.rR**3 - self.rL**3) / (self.rR**2 - self.rL**2)

    def rms(self):
        return np.sqrt((self.rR**2 + self.rL**2) / 2.0)

    def pdf(self, r):
        nonzero = (r >= self.rL) & (r <= self.rR)
        res = np.zeros(len(r)) * unit_registry("1/" + str(r.units))
        res[nonzero] = r[nonzero] * 2.0 / (self.rR**2 - self.rL**2)
        # res = res*unit_registry('1/'+str(r.units))
        return res

    def pdf_xy(self, x, y):
        X, Y = meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)

        nonzero = (R >= self.rL) & (R <= self.rR)
        res = np.zeros(R.shape) * unit_registry("1/" + str(x.units))
        res[nonzero] = R[nonzero] * 2.0 / (self.rR**2 - self.rL**2)

    def rho(self, r):
        nonzero = (r >= self.rL) & (r <= self.rR)
        res = np.zeros(len(r)) * unit_registry("1/" + str(r.units) + "/" + str(r.units))
        res[nonzero] = 2 / (self.rR**2 - self.rL**2)
        # res = res*unit_registry('1/'+str(r.units)+'/'+str(r.units))
        return res

    def rho_xy(self, x, y):
        X, Y = meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)
        nonzero = (R >= self.rL) & (R <= self.rR)
        res = np.zeros(R.shape) * unit_registry(f"1/{x.units}/{y.units}")
        res[nonzero] = 2.0 / (self.rR**2 - self.rL**2)

        return res

    def cdf(self, r):
        nonzero = (r >= self.rL) & (r <= self.rR)
        res = np.zeros(len(r)) * unit_registry("dimensionless")
        res[nonzero] = (r[nonzero] * r[nonzero] - self.rL**2) / (
            self.rR**2 - self.rL**2
        )
        # res = res*unit_registry('dimensionless')
        return res

    def cdfinv(self, rns):
        return np.sqrt(self.rL**2 + (self.rR**2 - self.rL**2) * rns)

    @property
    def min_dx(self):
        return None

    @property
    def min_dy(self):
        return None


class LinearRad(DistRad):
    def __init__(self, verbose=0, **kwargs):
        self._n_indent = 2

        self.type = "LinearRad"
        # self.xstr = var

        ra_str, rb_str = "min_r", "max_r"

        self.required_params = ["slope_fraction", ra_str, rb_str]
        self.optional_params = []

        self.check_inputs(kwargs)

        self.a = kwargs[ra_str]
        self.b = kwargs[rb_str]
        self.ratio = kwargs["slope_fraction"]
        self.f = 1 - np.abs(self.ratio)

        assert self.a < self.b, f"Error: {ra_str} must be < {rb_str}."
        assert self.a >= 0, f"Error: {ra_str} must be >= 0."
        assert (
            self.ratio >= -1 and self.ratio <= 1
        ), "Error: slope fraction must be: -1 <= r < 1."

        self.dr = self.b - self.a

        if self.ratio >= 0:
            # Do the maths
            self.pb = 2 / (1 + self.f) / self.dr
            self.pa = self.f * self.pb

        else:
            # Relabel the other way
            self.pa = 2 / (1 + self.f) / self.dr
            self.pb = self.f * self.pa

        self.dp = self.pb - self.pa
        self.m = self.dp / self.dr

        vprint("LinearRad", verbose > 0, 0, True)

    def get_r_pts(self, n, f=0.2):
        return np.linspace(self.a * (1 - f), self.b * (1 + f), n)

    def norm(self):
        dr2 = (self.b**2 - self.a**2) / 2.0
        dr3 = (self.b**3 - self.a**3) / 3.0

        return 1.0 / (self.m * dr3 - self.m * self.a * dr2 + self.pa * dr2)

    def rho(self, r):
        nonzero = (r >= self.a) & (r <= self.b)
        res = np.zeros(len(r)) * unit_registry("1/" + str(r.units) + "/" + str(r.units))
        res[nonzero] = self.norm() * (self.m * (r[nonzero] - self.a) + self.pa)
        return res

    def rho_xy(self, x, y):
        X, Y = meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)

        nonzero = (R >= self.a) & (R <= self.b)
        res = np.zeros(R.shape) * unit_registry(
            "1/" + str(x.units) + "/" + str(y.units)
        )
        res[nonzero] = self.norm() * (self.m * (R[nonzero] - self.a) + self.pa)
        return res

    def pdf(self, r):
        return r * self.rho(r)

    def cdf(self, r):
        nonzero = (r >= self.a) & (r <= self.b)
        dr2 = (r[nonzero] ** 2 - self.a**2) / 2.0
        dr3 = (r[nonzero] ** 3 - self.a**3) / 3.0

        res = np.zeros(len(r)) * unit_registry("")
        res[nonzero] = self.norm() * (
            self.m * dr3 - self.m * self.a * dr2 + self.pa * dr2
        )
        return res

    def cdfinv(self, p):
        rpts = self.get_r_pts(10000, f=0)
        cdfs = self.cdf(rpts)
        return interp(p, cdfs, rpts)

    def avg(self):
        rpts = self.get_r_pts(10000, f=0)
        pdfs = self.rho(rpts)
        return radint(pdfs * rpts, rpts)

    def rms(self):
        rpts = self.get_r_pts(10000, f=0)
        pdfs = self.rho(rpts)
        return np.sqrt(radint(pdfs * rpts * rpts, rpts))


class NormRad(DistRad):
    def __init__(self, verbose=False, **params):
        self.required_params = []
        self.optional_params = [
            "sigma_xy",
            "truncation_fraction",
            "truncation_radius_left",
            "truncation_radius_right",
            "n_sigma_cutoff_left",
            "n_sigma_cutoff_left",
            "n_sigma_cutoff",
            "truncation_radius",
            "truncation_radius_left",
            "truncation_radius_right",
        ]

        self.check_inputs(params)
      
        assert not (
            "sigma_xy" in params and "truncation_fraction" in params
        ), "User must specify either a sigma_xy or truncation fraction, not both"
        assert not (
            "sigma_xy" not in params and "truncation_fraction" not in params
        ), "User must specify sigma_xy or a truncation fraction for radial normal distribution"

        self.rR = None
        self.rL = None

        if "sigma_xy" in params:
            self.sigma = params["sigma_xy"]

            if (
                "truncation_radius_left" in params
                and "truncation_radius_right" in params
            ):
                self.rL = params["truncation_radius_left"]
                self.rR = params["truncation_radius_right"]

            elif "truncation_radius_left" in params:
                self.rL = params["truncation_radius_left"]
                self.rR = np.inf * self.rL.units

            elif "truncation_radius_right" in params:
                self.rR = params["truncation_radius_right"]
                self.rL =  0.0 * self.rR

            elif "truncation_radius" in params:
                self.rL = 0 * unit_registry("mm")
                self.rR = params["truncation_radius"]

            elif "n_sigma_cutoff_left" in params and "n_sigma_cutoff_right" in params:
                self.rL = params["n_sigma_cutoff_left"] * self.sigma
                self.rR = params["n_sigma_cutoff_right"] * self.sigma

            elif "n_sigma_cutoff" in params:
                self.rL = 0 * unit_registry("mm")
                self.rR = params["n_sigma_cutoff"] * self.sigma

            else:
                self.rL = 0 * unit_registry("mm")
                self.rR = float("Inf") * unit_registry("mm")

        elif "truncation_fraction" in params:
            f = params["truncation_fraction"]

            if "truncation_radius" in params:
                R = params["truncation_radius"]

                self.sigma = R * np.sqrt(1.0 / 2.0 / np.log(1 / f))
                self.rL = 0 * unit_registry("mm")
                self.rR = R

            elif (
                "truncation_radius_left" in params
                and "truncation_radius_right" in params
            ):
                self.rL = params["truncation_radius_right"]
                self.rR = params["truncation_radius_left"]
                self.sigma = self.rR * np.sqrt(1.0 / 2.0 / np.log(1 / f))

        assert self.rR.magnitude >= 0, "Radial Gaussian right cut radius must be >= 0"
        assert (
            self.rL < self.rR
        ), "Radial Gaussian left cut radius must be < right cut radius"

        self.pR = self.canonical_rho(self.rR / self.sigma)
        self.pL = self.canonical_rho(self.rL / self.sigma)
        self.dp = self.pL - self.pR

        vprint("radial Gaussian", verbose, 0, True)
        vprint(f'Non-truncated sigma_xy: {self.sigma:G~P}', verbose, 2, True)

    def canonical_rho(self, xi):
        return (1.0 / 2.0 / PHYSICAL_CONSTANTS.pi) * np.exp(-(xi**2) / 2)

    def rho(self, r):
        #print(r)
        xi = r / self.sigma

        r_geq_rL_and_leq_rR_as_int = 1 * (r >= self.rL) & (r <= self.rR)
        return r_geq_rL_and_leq_rR_as_int * self.canonical_rho(xi) / self.dp / (self.sigma**2)
        
        
        #res = np.zeros(len(r)) * unit_registry("1/" + str(r.units) + "/" + str(r.units))
        #nonzero = (r >= self.rL) & (r <= self.rR)
        #res[nonzero] = self.canonical_rho(xi[nonzero]) / self.dp / (self.sigma**2)
        #return res

    def rho_xy(self, x, y):
        X, Y = meshgrid(x, y)

        R = np.sqrt(X**2 + Y**2)

        xi = R / self.sigma
        res = np.zeros(R.shape) * unit_registry(
            "1/" + str(x.units) + "/" + str(y.units)
        )

        r_geq_rL_and_leq_rR_as_int = 1 * (R >= self.rL) & (R <= self.rR)
        return r_geq_rL_and_leq_rR_as_int * self.canonical_rho(xi) / self.dp / (self.sigma**2)

        #nonzero = (R >= self.rL) & (R <= self.rR)
        #res[nonzero] = self.canonical_rho(xi[nonzero]) / self.dp / (self.sigma**2)

        #return res

    def pdf(self, r):
        xi = r / self.sigma
        r_geq_rL_and_leq_rR_as_int = 1 * (r >= self.rL) & (r <= self.rR)
        return r_geq_rL_and_leq_rR_as_int * self.canonical_rho(xi) / self.dp / self.sigma**2
        #res = np.zeros(len(r)) * unit_registry("1/" + str(r.units))
        #nonzero = (r >= self.rL) & (r <= self.rR)
        #res[nonzero] = (
        #    r[nonzero] * self.canonical_rho(xi[nonzero]) / self.dp / self.sigma**2
        #)
        #return res

    def cdf(self, r):
        r_geq_rL_and_leq_rR_as_int = 1 * (r >= self.rL) & (r <= self.rR)
        #res = np.zeros(len(r)) * unit_registry("dimensionless")
        #nonzero = (r >= self.rL) & (r <= self.rR)
        xi = r / self.sigma
        return r_geq_rL_and_leq_rR_as_int * (self.pL - self.canonical_rho(xi)) / self.dp
        #res[nonzero] = (self.pL - self.canonical_rho(xi[nonzero])) / self.dp
        #return res

    def cdfinv(self, rns):
        return np.sqrt(
            2
            * self.sigma**2
            * np.log(1 / 2 / PHYSICAL_CONSTANTS.pi / (self.pL - self.dp * rns))
        )

    def get_r_pts(self, n=1000):
        if self.rR.magnitude == float("Inf"):
            endr = 5 * self.sigma
        else:
            endr = 1.2 * self.rR
        return linspace(0.88 * self.rL, endr, n)

    def avg(self):
        xiL = self.rL / self.sigma
        xiR = self.rR / self.sigma

        erfL = erf(xiL / np.sqrt(2))
        erfR = erf(xiR / np.sqrt(2))

        if self.rR.magnitude == float("Inf"):
            xiRpR = 0 * unit_registry("")
        else:
            xiRpR = xiR * self.pR

        return (
            self.sigma
            * (
                (xiL * self.pL - xiRpR)
                + (1.0 / 2.0 / np.sqrt(2 * PHYSICAL_CONSTANTS.pi)) * (erfR - erfL)
            )
            / self.dp
        )

    def rms(self):
        if self.rR.magnitude == float("Inf"):
            pRrR2 = 0 * unit_registry("mm^2")
        else:
            pRrR2 = self.pR * self.rR**2

        pRrL2 = self.pR * self.rL**2

        return np.sqrt(2 * self.sigma**2 + self.rL**2 + (pRrL2 - pRrR2) / self.dp)

    @property
    def min_dx(self):
        return None

    @property
    def min_dy(self):
        return None


class RadFile(DistRad):
    def __init__(self, verbose=0, **params):
        self.required_params = ["file", "units"]
        self.optional_params = []
        self.check_inputs(params)

        distfile = params["file"]
        units = params["units"]

        self.distfile = distfile

        with open(distfile, "r") as f:
            headers = f.readline().split()

        if len(headers) != 2:
            raise ValueError("radial distribution file must have two columns")

        data = np.loadtxt(distfile, skiprows=1)

        rs = data[:, 0] * unit_registry(units)
        Pr = data[:, 1] * unit_registry.parse_expression("1/" + units + "/" + units)

        if np.count_nonzero(rs < 0):
            raise ValueError("Radial distribution r-values must be >= 0.")

        super().__init__(rs, Pr)

        vprint("radial file", verbose > 0, 0, True)
        vprint(f'r-dist file: "{distfile}"', verbose > 0, 2, True)


class TukeyRad(DistRad):
    def __init__(self, verbose=0, **kwargs):
        self.required_params = ["ratio", "length"]
        self.optional_params = []
        self.check_inputs(kwargs)

        self.r = kwargs["ratio"]
        self.L = kwargs["length"]

        vprint("TukeyRad", verbose > 0, 0, True)
        vprint(
            "legnth = {:0.3f~P}".format(self.L) + ", ratio = {:0.3f~P}".format(self.r),
            verbose > 0,
            2,
            True,
        )

    def get_r_pts(self, n=1000, f=0.2):
        return np.linspace(0, (1 + f) * self.L.magnitude, n) * unit_registry(
            str(self.L.units)
        )

    def pdf(self, r):
        return r * self.rho(r)

    def rho(self, r):
        ustr = "1/" + str(self.L.units) + "/" + str(self.L.units)

        res = np.zeros(r.shape) * unit_registry(ustr)

        if self.r == 0:
            flat_region = np.logical_and(r <= self.L)
            res[flat_region] = 1.0 * unit_registry(ustr)

        else:
            Lflat = self.L * (1 - self.r)
            Lcos = self.r * self.L
            cos_region = np.logical_and(r >= +Lflat, r <= +self.L)
            flat_region = np.logical_and(r < Lflat, r >= 0)
            res[cos_region] = (
                0.5
                * (1 + np.cos((PHYSICAL_CONSTANTS.pi / Lcos) * (r[cos_region] - Lflat)))
                * unit_registry(ustr)
            )
            res[flat_region] = 1.0 * unit_registry(ustr)

        res = res
        res = res / radint(res, r)
        return res

    def cdf(self, r):
        rpts = self.get_r_pts(10000)
        pdfs = self.rho(rpts)

        cdfs, rbins = radcumint(pdfs, rpts)
        cdfs = cdfs / cdfs[-1]
        cdfs = interp(r, rbins, cdfs)
        cdfs = cdfs / cdfs[-1]
        cdfs * unit_registry("dimensionless")

        return cdfs

    def cdfinv(self, p):
        rpts = self.get_r_pts(10000)
        cdfs = self.cdf(rpts)
        return interp(p, cdfs, rpts)

    def avg(self):
        rpts = self.get_r_pts(10000)
        pdfs = self.rho(rpts)
        return radint(pdfs * rpts, rpts)

    def rms(self):
        rpts = self.get_r_pts(10000)
        pdfs = self.rho(rpts)
        return np.sqrt(radint(pdfs * rpts * rpts, rpts))


class SuperGaussianRad(DistRad):
    def __init__(self, verbose=0, **kwargs):
        self.required_params = []
        self.optional_params = ["p", "alpha", "lambda", "sigma_xy"]
        self.check_inputs(kwargs)

        assert not (
            "alpha" in kwargs and "p" in kwargs
        ), 'Radial Super Gaussian power parameter must be set using "p" or "alpha", not both.'
        assert (
            "alpha" in kwargs or "p" in kwargs
        ), 'Radial Super Gaussian power parameter must be set using "p" or "alpha". Neither provided.'

        assert not (
            "lambda" in kwargs and "sigma_xy" in kwargs
        ), 'Radial Super Gaussian power parameter must be set using "sigma_xy" or "lambda", not both.'
        assert (
            "lambda" in kwargs or "sigma_xy" in kwargs
        ), 'Radial Super Gaussian power parameter must be set using "sigma_xy" or "lambda". Neither provided.'

        if "p" in kwargs:
            self.p = kwargs["p"]
        elif "alpha" in kwargs:
            alpha = kwargs["alpha"]
            assert (
                alpha >= 0 and alpha <= 1
            ), f"SugerGaussian parameter must satisfy 0 <= alpha <= 1, not = {alpha}"
            if alpha.magnitude == 0:
                self.p = float("Inf") * unit_registry("dimensionless")
            else:
                self.p = 1 / alpha

        if "lambda" in kwargs:
            self.Lambda = kwargs["lambda"]
        else:
            self.Lambda = self.get_lambda(kwargs["sigma_xy"])

        assert self.p > 0, "Radial Super Gaussian power p must be > 0."

        vprint("SuperGaussianRad", verbose > 0, 0, True)
        vprint(
            f"lambda = {self.Lambda:G~P}, power = {self.p:G~P}", verbose > 0, 2, True
        )

    def get_r_pts(self, n=1000):
        # if(self.p < float('Inf')):
        #    f = self.p.magnitude
        # else:
        #    f=1

        return np.linspace(0, 5 * self.Lambda.magnitude, n) * unit_registry(
            str(self.Lambda.units)
        )

    def pdf(self, r):
        rho = self.rho(r)
        return r * rho

    def rho(self, r):
        csi = r / self.Lambda
        nur = 0.5 * (csi**2)
        N = 1.0 / gamma(1 + 1.0 / self.p) / self.Lambda**2
        rho = N * np.exp(-np.float_power(nur.magnitude, self.p.magnitude))
        return rho

    def cdf(self, r):
        rpts = self.get_r_pts(10000)
        pdfs = self.rho(rpts)

        cdfs, rbins = radcumint(pdfs, rpts)
        cdfs = cdfs / cdfs[-1]
        cdfs = interp(r, rbins, cdfs)
        cdfs = cdfs / cdfs[-1]
        cdfs * unit_registry("dimensionless")

        return cdfs

    def cdfinv(self, p):
        rpts = self.get_r_pts(10000)
        cdfs = self.cdf(rpts)
        return interp(p, cdfs, rpts)

    def avg(self):
        return (
            (2.0 * np.sqrt(2.0) / 3.0)
            * (gamma(1 + 3.0 / 2.0 / self.p) / gamma(1 + 1.0 / self.p))
            * self.Lambda
        )

    def rms(self):
        return np.sqrt(gamma(1 + 2.0 / self.p) / gamma(1 + 1.0 / self.p)) * self.Lambda

    def get_lambda(self, sigma_xy):
        rrms = sigma_xy / np.sqrt(0.5)
        return np.sqrt(gamma(1 + 1.0 / self.p) / gamma(1 + 2.0 / self.p)) * rrms


class DeformableRad(DistRad):
    def __init__(self, verbose=0, **kwargs):
        sigstr = "sigma_xy"
        # avgstr = f'avg_{var}'

        self.required_params = ["slope_fraction", "alpha", sigstr]
        self.optional_params = []  # ['n_sigma_cutoff']

        self.check_inputs(kwargs)

        self.sigma = kwargs[sigstr]
        # self.mean = kwargs[avgstr]

        # if('n_sigma_cutoff' in kwargs):
        #    n_sigma_cutoff=kwargs['n_sigma_cutoff']
        # else:
        #    n_sigma_cutoff=3

        sg_params = {"alpha": kwargs["alpha"], sigstr: self.sigma}
        #'n_sigma_cutoff':n_sigma_cutoff}

        self.dist = {}
        self.dist["super_gaussian"] = SuperGaussianRad(verbose=verbose, **sg_params)

        # SG
        rs = self.dist["super_gaussian"].get_r_pts(10000)
        Pr = self.dist["super_gaussian"].rho(rs)

        # Linear
        lin_params = {
            "slope_fraction": kwargs["slope_fraction"],
            "min_r": rs[0],
            "max_r": rs[-1],
        }
        self.dist["linear"] = LinearRad(verbose=verbose, **lin_params)

        Pr = Pr * self.dist["linear"].rho(rs)

        norm = radint(Pr, rs)
        assert norm > 0, "Error: derformable distribution can not be normalized."
        Pr = Pr / norm

        # avgx = np.trapz(xs*Px, xs)
        stdx = np.sqrt(radint(Pr * rs**2, rs)) / np.sqrt(2)
        rs = (self.sigma / stdx) * rs

        super().__init__(rs=rs, Pr=Pr)

    def rms(self):
        return np.sqrt(2) * self.sigma

    # def avg(self):
    #    return self.mean

    # def rms(self):
    #    return np.sqrt(self.sigma()**2 + self.avg()**2)


class InterpolationRad(DistRad):
    def __init__(self, verbose=0, **kwargs):
        # sigstr = 'sigma_xy'
        # avgstr = f'avg_xy'

        self.required_params = ["Pr", "method"]
        self.optional_params = ["r", "n_pts"]

        self.check_inputs(kwargs)

        # self.sigma_xy = kwargs[sigstr]
        # self.mean = kwargs[avgstr]

        pts = kwargs["Pr"]
        self.method = kwargs["method"]

        if isinstance(pts, list):
            pts = np.array(pts)

        elif isinstance(pts, dict):
            pts = np.array([v for k, v in pts.items()])

        if "n_pts" in kwargs:
            n_pts = kwargs["n_pts"]
        else:
            n_pts = 1000

        if "rs" in kwargs:
            pass
        else:
            rs = np.linspace(0, 1, len(pts))

        rs = rs * unit_registry("mm")
        Pr = pts * unit_registry.parse_expression("1/mm/mm")

        # Save the original curve
        self.r0s = rs

        # Do interpolation
        rs, Pr = self.interpolate1d(rs, Pr, method=kwargs["method"], n_pts=n_pts)

        # Make sure interoplation doesn't yield negative values
        Pr[Pr.magnitude < 0] = 0 * unit_registry.parse_expression("1/mm/mm")

        vprint("radial interpolation", verbose > 0, 0, True)
        # vprint(f'lambda = {self.Lambda:G~P}, power = {self.p:G~P}',verbose>0,2,True)

        super().__init__(rs=rs, Pr=Pr)

    def interpolate1d(self, r, P, method="spline", n_pts=1000, s=0.0, k=3):
        rs = linspace(r[0], r[-1], n_pts)

        if method == "spline":
            Pr = spline1d(rs, r, P, s=s, k=k)

        return rs, Pr


def is_radial_dist(dtype):
    abrevs = ["rg", "ru", "rsg", "dr", "ri"]

    if dtype in abrevs:
        return True

    full_names = [
        "radial_uniform",
        "radial_gaussian",
        "radial_super_gaussian",
        "radfile",
        "radial_tukey",
        "raddeformable",
        "radial_interpolation",
    ]

    if dtype in full_names:
        return True

    return False


class Dist2d(Dist):
    def __init__(
        self,
        xs=None,
        ys=None,
        Pxy=None,
        xstr="x",
        ystr="y",
        x_unit="",
        y_unit="",
        verbose=False,
    ):
        if not isinstance(xs, Quantity):
            xs = xs * unit_registry(x_unit)

        if not isinstance(ys, Quantity):
            ys = ys * unit_registry(y_unit)

        if not isinstance(Pxy, Quantity):
            Pxy = Pxy * unit_registry(f"1/{x_unit}/{y_unit}")

        if len(xs) != Pxy.shape[1]:
            raise ValueError("Length of input vector x must = Pxy.shape[1]")
        if len(ys) != Pxy.shape[0]:
            raise ValueError("Length of input vector y must = Pxy.shape[0]")

        # Pxy = flipud(Pxy)

        if len(xs) < 100:
            warnings.warn(
                "Specificed grid was sparse (< 100) in x direction. This may cause blurring near sharp edges in the distribition due to interpolation."
            )
        if len(ys) < 100:
            warnings.warn(
                "Specificed grid was sparse (< 100) in y direction. This may cause blurring near sharp edges in the distribition due to interpolation."
            )

        self.xs = xs
        self.ys = ys
        self.Pxy = Pxy
        self.xstr = xstr
        self.ystr = ystr

        self.var_type = "2d"

        assert (
            np.count_nonzero(Pxy.magnitude) > 0
        ), "Supplied 2d distribution is zero everywhere."

        self.xb = np.zeros(len(self.xs.magnitude) + 1) * unit_registry(
            str(self.xs.units)
        )
        self.xb[1:-1] = (self.xs[1:] + self.xs[:-1]) / 2.0

        dxL = self.xb[+1] - self.xs[+0]
        dxR = self.xs[-1] - self.xb[-2]

        self.xb[+0] = self.xs[+0] - dxL
        self.xb[-1] = self.xs[-1] + dxR

        self.yb = np.zeros(len(ys) + 1) * unit_registry(str(self.ys.units))
        self.yb[1:-1] = (self.ys[1:] + self.ys[:-1]) / 2.0

        dyL = self.yb[+1] - self.ys[+0]
        dyR = self.ys[-1] - self.yb[-2]

        self.yb[+0] = self.ys[+0] - dyL
        self.yb[-1] = self.ys[-1] + dyR

        # Integrate out y to get rho(x) = int(rho(x,y)dy)
        self.dx = self.xb[1:] - self.xb[:-1]
        self.dy = self.yb[1:] - self.yb[:-1]

        self.Px = np.matmul(
            np.transpose(Pxy.magnitude), self.dy.magnitude
        ) * unit_registry("1/" + str(self.ys.units))
        self.Px = self.Px / np.sum(self.Px * self.dx)

        self.Cx = np.zeros(len(self.xb)) * unit_registry("dimensionless")
        self.Cx[1:] = np.cumsum(self.Px * self.dx)

        # Get cumulative distributions along y as a function of x:
        self.Cys = np.zeros((len(self.yb), len(self.xs)))

        # norms = np.sum(np.multiply(self.Pxy.magnitude, np.transpose(mlib.repmat(self.dy.magnitude,len(self.xs),1))), axis=0)
        norms = np.sum(
            np.multiply(
                self.Pxy.magnitude,
                np.transpose(np.tile(self.dy.magnitude, (len(self.xs), 1))),
            ),
            axis=0,
        )
        norms[norms == 0] = 1

        # self.Cys[1:,:] = np.cumsum(np.multiply(self.Pxy.magnitude, np.transpose(mlib.repmat(self.dy.magnitude,len(self.xs),1))),axis=0)/norms
        self.Cys[1:, :] = (
            np.cumsum(
                np.multiply(
                    self.Pxy.magnitude,
                    np.transpose(np.tile(self.dy.magnitude, (len(self.xs), 1))),
                ),
                axis=0,
            )
            / norms
        )
        self.Cys = self.Cys * unit_registry("dimensionless")

    def pdf(self, x, y):
        return interp2d(x, y, self.xs, self.ys, self.Pxy)

    def plot_pdf(self):
        plt.figure()
        extent = [
            (self.xs.min()).magnitude,
            (self.xs.max()).magnitude,
            (self.ys.min()).magnitude,
            (self.ys.max()).magnitude,
        ]
        plt.imshow(self.Pxy, extent=extent)
        plt.xlabel(self.xstr + " (" + str(self.xs.units) + ")")
        plt.ylabel(self.ystr + " (" + str(self.ys.units) + ")")

    def pdfx(self, x):
        return interp(x, self.xs, self.Px)

    def plot_pdfx(self):
        plt.figure()
        plt.plot(self.xs, self.Px)

    def cdfx(self, x):
        return interp(x, self.xb, self.Cx)

    def plot_cdfx(self):
        plt.figure()
        plt.plot(self.xb, self.Cx)

    def cdfxinv(self, ps):
        return interp(ps, self.Cx, self.xb)

    def plot_cdfys(self):
        plt.figure()
        for ii in range(len(self.xs)):
            plt.plot(self.yb, self.Cys[:, ii])

    def sample(self, N, sequence=None, params=None):
        rns = self.rgen.rand((N, 2), sequence, params) * unit_registry("dimensionless")
        x, y = self.cdfinv(rns[0, :], rns[1, :])
        return (x, y)

    def cdfinv(self, rnxs, rnys):
        x = self.cdfxinv(rnxs)
        indx = np.searchsorted(self.xb, x) - 1

        y = np.zeros(x.shape) * unit_registry(str(self.ys.units))
        for ii in range(self.Cys.shape[1]):
            in_column = ii == indx
            if np.count_nonzero(in_column) > 0:
                y[in_column] = interp(rnys[in_column], self.Cys[:, ii], self.yb)

        return (x, y)

    @property
    def min_dx(self):
        return np.diff(self.xs).min()

    @property
    def min_dy(self):
        return np.diff(self.ys).min()

    def test_sampling(self):
        x, y = self.sample(100000, sequence="hammersley")
        plt.figure()
        plt.plot(x, y, "*")

    def get_x_pts(self, m):
        return linspace(self.xs[0], self.xs[-1], m)

    def get_y_pts(self, n):
        return linspace(self.ys[0], self.ys[-1], n)

    def get_xy_pts(self, m, n=None):
        if n is None:
            n = m

        return (self.get_x_pts(m), self.get_y_pts(n))


class SuperPosition2d(Dist2d):
    def __init__(self, variables, verbose, **kwargs):
        vstrs = get_vars(variables)

        # self, variables, verbose, **kwargs

        assert (
            len(vstrs) == 2
        ), f"Wrong number of variables given to Image2d: {len(vstrs)}"
        assert (
            "dists" in kwargs
        ), 'ProductDist 2d must be supplied the key word argument "dists"'

        vprint("Superposition 2d", verbose, 1, new_line=True)

        dist_defs = kwargs["dists"]

        dists = {}

        if "weights" in kwargs:
            weights = kwargs["weights"]
        else:
            weights = np.linspace(len(dist_defs.keys()))

        weights = weights / np.sum(weights)

        for ii, name in enumerate(dist_defs.keys()):
            vprint(f"distribution name: {name}", verbose > 0, 2, True)

            # handle 2d or radial here
            if is_radial_dist(dist_defs[name]["type"]):
                dists[name] = get_dist("r", dist_defs[name], verbose=verbose)
            else:
                dists[name] = get_dist(variables, dist_defs[name], verbose=verbose)

            xi, yi = dists[name].get_xy_pts(2)

            if ii == 0:
                min_x, max_x = xi[0], xi[-1]
                min_y, max_y = yi[0], yi[-1]

                if dists[name].min_dx is not None:
                    min_dx = dists[name].min_dx
                else:
                    min_dx = np.inf * xi.units

                if dists[name].min_dy is not None:
                    min_dy = dists[name].min_dy
                else:
                    min_dy = np.inf * yi.units

            else:
                if xi[0] > min_x:
                    min_x = xi[0]
                if xi[-1] < max_x:
                    max_x = xi[-1]

                if yi[0] > min_y:
                    min_y = yi[0]
                if yi[-1] < max_y:
                    max_y = yi[-1]

                if dists[name].min_dx is not None and dists[name].min_dx < min_dx:
                    min_dx = dists[name].min_dx

                if dists[name].min_dy is not None and dists[name].min_dy < min_dy:
                    min_dy = dists[name].min_dy

        nx, ny = (
            int(((max_x - min_x) / min_dx).magnitude),
            int(((max_y - min_y) / min_dy).magnitude),
        )

        xs = linspace(min_x, max_x, nx)
        ys = linspace(min_y, max_y, ny)

        for ii, name in enumerate(dists.keys()):
            if is_radial_dist(dist_defs[name]["type"]):
                pii = dists[name].rho_xy(xs, ys)
            else:
                pii = dists[name].pdf(xs, ys)

            pii = pii / np.sum(np.sum(pii))

            if ii == 0:
                ps = pii / np.max(pii.magnitude)
            else:
                ps = weights[ii] * pii / np.max(pii.magnitude) + ps

        ps = ps.magnitude * unit_registry(f"1/{xs.units}/{ys.units}")

        # Update the base class with the superposition distribution
        super().__init__(xs, ys, ps, xstr=vstrs[0], ystr=vstrs[-1])


class Product2d(Dist2d):
    """Dist object that allows user to multiply multiple 2d distributions together to form a new PDF for sampling"""

    def __init__(self, variables, verbose, **kwargs):
        vstrs = get_vars(variables)
        assert (
            len(vstrs) == 2
        ), f"Wrong number of variables given to Image2d: {len(vstrs)}"
        assert (
            "dists" in kwargs
        ), 'ProductDist 2d must be supplied the key word argument "dists"'

        vprint("Product 2d", verbose, 1, new_line=True)

        dist_defs = kwargs["dists"]

        dists = {}

        for ii, name in enumerate(dist_defs.keys()):
            vprint(f"distribution name: {name}", verbose > 0, 2, True)

            # handle 2d or radial here
            if is_radial_dist(dist_defs[name]["type"]):
                dists[name] = get_dist("r", dist_defs[name], verbose=verbose)
            else:
                dists[name] = get_dist(variables, dist_defs[name], verbose=verbose)

            xi, yi = dists[name].get_xy_pts(2)

            if ii == 0:
                min_x, max_x = xi[0], xi[-1]
                min_y, max_y = yi[0], yi[-1]

                if dists[name].min_dx is not None:
                    min_dx = dists[name].min_dx
                else:
                    min_dx = np.inf * xi.units

                if dists[name].min_dy is not None:
                    min_dy = dists[name].min_dy
                else:
                    min_dy = np.inf * yi.units

            else:
                if xi[0] > min_x:
                    min_x = xi[0]
                if xi[-1] < max_x:
                    max_x = xi[-1]

                if yi[0] > min_y:
                    min_y = yi[0]
                if yi[-1] < max_y:
                    max_y = yi[-1]

                if dists[name].min_dx is not None and dists[name].min_dx < min_dx:
                    min_dx = dists[name].min_dx

                if dists[name].min_dy is not None and dists[name].min_dy < min_dy:
                    min_dy = dists[name].min_dy

        nx, ny = (
            int(((max_x - min_x) / min_dx).magnitude),
            int(((max_y - min_y) / min_dy).magnitude),
        )

        xs = linspace(min_x, max_x, nx)
        ys = linspace(min_y, max_y, ny)

        for ii, name in enumerate(dists.keys()):
            if is_radial_dist(dist_defs[name]["type"]):
                pii = dists[name].rho_xy(xs, ys)
            else:
                pii = dists[name].pdf(xs, ys)

            if ii == 0:
                ps = pii / np.max(pii.magnitude)
            else:
                ps = ps * pii / np.max(pii.magnitude)

        ps = ps.magnitude * unit_registry(f"1/{xs.units}/{ys.units}")

        # Update the base class with the product distribution
        super().__init__(xs, ys, ps, xstr=vstrs[0], ystr=vstrs[-1])


class Image2d(Dist2d):
    def __init__(self, variables, verbose, **params):
        vstrs = get_vars(variables)

        assert (
            len(vstrs) == 2
        ), f"Wrong number of variables given to Image2d: {len(vstrs)}"

        v1, v2 = vstrs[0], vstrs[1]

        self.required_params = ["P"]
        self.optional_params = [
            f"min_{v1}",
            f"max_{v1}",
            f"min_{v2}",
            f"max_{v2}",
            v1,
            v2,
        ]

        self.check_inputs(params)

        Pxy = flipud(params["P"])
        assert (
            np.min(np.min(Pxy)) >= 0
        ), "Error in Image2d: the 2d probability function must be >= 0"

        if v1 not in params:
            xmin = params[f"min_{v1}"]
            xmax = params[f"max_{v1}"].to(xmin.units)

            assert xmin < xmax, f"Error in Image2d: min {v1} must < max {v1}."

            xs = linspace(xmin, xmax, Pxy.shape[1])

        else:
            xs = params[v1]

        if v2 not in params:
            ymin = params[f"min_{v2}"].to(xmin.units)
            ymax = params[f"max_{v2}"].to(xmin.units)

            assert ymin < ymax, f"Error in Image2d: min {v2} must < max {v2}."

            ys = linspace(ymin, ymax, Pxy.shape[0])

        else:
            ys = params[v2]

        Pxy = Pxy * unit_registry(f"1/{xs.units}/{ys.units}")

        super().__init__(xs, ys, Pxy, xstr=v1, ystr=v2)


class File2d(Dist2d):
    def __init__(self, var1, var2, verbose, **params):
        self.required_params = ["file"]
        self.optional_params = [
            f"min_{var1}",
            f"max_{var1}",
            f"min_{var2}",
            f"max_{var2}",
            var1,
            var2,
            "threshold",
            "invert",
        ]

        self.check_inputs(params)

        filename = params["file"]

        ext = (os.path.splitext(filename)[1]).lower()

        if ext in SUPPORTED_IMAGE_EXTENSIONS:
            Pxy = read_image_file(filename)

            xstr = var1
            ystr = var2

            min_var1_str = f"min_{var1}"
            max_var1_str = f"max_{var1}"

            assert (
                min_var1_str in params
            ), f"Error in File2d: user must specify {min_var1_str}."
            assert (
                max_var1_str in params
            ), f"Error in File2d: user must specify {max_var1_str}."

            min_var1 = params[min_var1_str]
            max_var1 = params[max_var1_str]

            xs = linspace(min_var1, max_var1, Pxy.shape[1])

            min_var2_str = f"min_{var2}"
            max_var2_str = f"max_{var2}"

            assert (
                min_var2_str in params
            ), f"Error in File2d: user must specify {min_var2_str}."
            assert (
                max_var2_str in params
            ), f"Error in File2d: user must specify {max_var2_str}."

            min_var2 = params[min_var2_str]
            max_var2 = params[max_var2_str]

            ys = linspace(min_var2, max_var2, Pxy.shape[0])

            Pxy = np.flipud(Pxy)
            Pxy = Pxy * unit_registry(f"1/{str(xs.units)}/{str(ys.units)}")

            super().__init__(xs, ys, Pxy, xstr=xstr, ystr=ystr)

        elif ext == ".txt":
            xs, ys, Pxy, xstr, ystr = read_2d_file(filename)

        else:
            raise ValueError(
                f'Error: unknown file extension: "{ext}" for filename = {filename}'
            )

        if "invert" in params and params["invert"]:
            Pxy = Pxy.max() - Pxy

        if "threshold" in params:
            threshold = params["threshold"]
        else:
            threshold = 0

        assert (
            threshold >= 0 and threshold < 1
        ), "Error: image threshold must be >=0 and < 1."

        under_threshold = Pxy.magnitude < threshold * Pxy.magnitude.max()
        Pxy.magnitude[under_threshold] = 0

        super().__init__(xs, ys, Pxy, xstr=xstr, ystr=ystr)

        vprint("2D File PDF", verbose > 0, 0, True)
        vprint(f'2D pdf file: {params["file"]}', verbose > 0, 2, True)
        vprint(
            f"min_{var1} = {min(xs):G~P}, max_{var1} = {max(xs):G~P}",
            verbose > 0,
            2,
            True,
        )
        vprint(
            f"min_{var2} = {min(ys):G~P}, max_{var2} = {max(ys):G~P}",
            verbose > 0,
            2,
            True,
        )


class FermiDirac3StepBarrierMomentumDist(Dist2d):
    def __init__(self, verbose=0, **params):
        """
        Class for sampling the 2d distribution in |p| and spherical polar angle (angle between pvec and z axis in momentum space)

        based on the Dowell-Schmerge cathode model.
        """

        self.required_params = [
            "photon_wavelength",
            "work_function",
            "temperature",
            "fermi_energy",
        ]
        self.optional_params = ['n_tails']

        self.check_inputs(params)

        h = PHYSICAL_CONSTANTS["Planck constant in eV/Hz"]
        c = PHYSICAL_CONSTANTS["speed of light in vacuum"]
        kb = PHYSICAL_CONSTANTS["Boltzmann constant in eV/K"]

        self.photon_wavelength = params["photon_wavelength"]
        self.photon_energy = (h * c / params["photon_wavelength"]).to("eV")
        self.cathode_temperature = params["temperature"]
        self.kT = (kb * self.cathode_temperature).to("eV")
        self.Wf = params["work_function"]
        self.fermi_energy = params["fermi_energy"]
        
        if 'n_tails' in params:
            self.n_tails = params['n_tails']
        else:
            self.n_tails = 4

        vprint("Fermi-Dirac 3 Step Barrier Photocathode Model", verbose > 0, 0, True)
        vprint(
            f'laser wavelength = {self.photon_wavelength.to("nm"):G~P}, photon energy = {self.photon_energy.to("eV"):G~P}',
            verbose > 0,
            2,
            True,
        )
        vprint(
            f'cathode temperature = {self.cathode_temperature.to("K"):G~P}, cathode work function = {self.Wf.to("eV"):G~P}, Fermi energy = {self.fermi_energy.to("eV"):G~P}',
            verbose > 0,
            2,
            True,
        )

        self.p_bounds, self.polar_angle_bounds = (
            fermi_dirac_3step_barrier_pdf_bounds_spherical(
                self.photon_energy, self.Wf, self.cathode_temperature, self.fermi_energy, n_tails=self.n_tails
            )
        )

        p, a = self.p_pts(1000), self.polar_angle_pts(2000)
        P, A = self.p_by_polar_angle_meshgrid(len(p), len(a))
        rho = self.rho_p_polar_angle(P, A)

        Ppa = rho * P.magnitude**2 * np.sin(A) * unit_registry(f"1/{p.units}/{a.units}")

        super().__init__(p, a, Ppa, xstr="p", ystr="polar_angle")

    def p_pts(self, n):
        return linspace(self.p_bounds[0], self.p_bounds[1], n).to("eV/c")

    def polar_angle_pts(self, n):
        return linspace(self.polar_angle_bounds[0], self.polar_angle_bounds[1], n).to(
            "rad"
        )

    def p_by_polar_angle_meshgrid(self, n, m):
        return np.meshgrid(self.p_pts(n), self.polar_angle_pts(m))

    def rho_p_polar_angle(self, p, polar_angle):
        return fermi_dirac_3step_barrier_pdf_spherical(
            p,
            polar_angle,
            self.photon_energy,
            self.Wf,
            self.cathode_temperature,
            self.fermi_energy
        )


class UniformLaserSpeckle(Dist2d):
    def __init__(self, verbose=0, **params):
        self.required_params = ["min_x", "max_x", "min_y", "max_y", "sigma"]
        self.optional_params = [
            "image_size_x",
            "image_size_y",
            "random_seed",
            "pixel_threshold",
        ]

        self.check_inputs(params)

        self.min_x, self.max_x = params["min_x"], params["max_x"]
        self.min_y, self.max_y = params["min_y"], params["max_y"]

        self.scale = params["sigma"]

        if "image_size_x" in params:
            self.image_size_x = params["image_size_x"]
        else:
            self.image_size_x = 512

        if "image_size_y" in params:
            self.image_size_y = params["image_size_y"]
        else:
            self.image_size_y = 512

        if "random_seed" in params:
            random_seed = params["random_seed"]
        else:
            random_seed = None

        if "pixel_threshold" in params:
            self.threshold = params["pixel_threshold"]
        else:
            self.threshold = 0

        # pixels_per_x_scale = self.image_size_x / ( self.max_x - self.min_x )
        # pixels_per_y_scale = self.image_size_y / ( self.max_y - self.min_y )

        # speckle_size_in_pixels_x = pixels_per_x_scale * self.scale
        # speckle_size_in_pixels_y = pixels_per_y_scale * self.scale

        # self.speckle_size_in_pixels_avg = 0.5*( speckle_size_in_pixels_x + speckle_size_in_pixels_y )

        # print(self.speckle_size_in_pixels_avg, self.speckle_size_in_pixels_avg.magnitude)

        image_size = (self.image_size_x, self.image_size_y)

        # speckle_pattern = generate_laser_speckle_fft(image_size, self.speckle_size_in_pixels_avg.magnitude, random_seed=random_seed)
        speckle_pattern = generate_speckle_pattern_with_filter(
            image_size, self.scale, random_seed=random_seed
        )

        # print(speckle_pattern.min(), speckle_pattern.max())

        speckle_pattern[speckle_pattern < self.threshold] = 0

        xs = linspace(self.min_x, self.max_x, self.image_size_x)
        ys = linspace(self.min_y, self.max_y, self.image_size_y)

        Pxy = speckle_pattern * unit_registry(f"1/{xs.units}/{ys.units}")

        super().__init__(xs, ys, Pxy, xstr="x", ystr="y")


# ----------------------------------------------------------------------------
#   This allows the main function to be at the beginning of the file
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    pass
