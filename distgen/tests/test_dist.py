#!/usr/bin/env python

from distgen.physical_constants import unit_registry
import numpy as np
from matplotlib import pyplot as plt

from distgen.tests.conftest import EXAMPLES_DATA_PATH


def test_rng_1():
    # Random Number Generation
    # ---

    # To sample various distributions requires generating random numbers and
    # supplying them to the $CDF^{-1}$ functions for each corresponding
    # distribution.  Currently, this is handled using
    #
    # `distgen.dist.random_generator(shape, sequence, **params)`.
    #
    # Here `shape = (n_dimension, n_particle)` determines the shape of the
    # random numbers returned.  The keyword 'sequence' can be used to set the
    # sequence to Hammerlsey for quasi-random numbers.
    #
    # The difference is shown below:

    from distgen.dist import random_generator

    shape = (2, 100)

    p1 = random_generator(shape, "hammersley")
    p2 = random_generator(shape, "pseudo")

    fig, ax = plt.subplots(1, 2, constrained_layout=True)

    ax[0].plot(p1[0, :], p1[1, :], ".")
    ax[0].set(xlabel="rx", ylabel="ry", title="hammersley")
    ax[1].plot(p2[0, :], p2[1, :], "*")
    ax[1].set(xlabel="rx", ylabel="ry", title="random.rand")


def test_rng_seed():
    from distgen.dist import random_generator
    # When using pseudo random numbers via NumPy, it is possible to set the generator seed:

    shape = (2, 100)
    fig, ax = plt.subplots(1, 1, constrained_layout=True)

    p2 = random_generator(shape, "pseudo", seed=0)
    p3 = random_generator(shape, "pseudo", seed=0)

    ax.plot(p2[0, :], p2[1, :], "*")
    ax.plot(p3[0, :], p3[1, :], ".")
    ax.set(xlabel="rx", ylabel="ry", title="random.rand")


def test_generator():
    from distgen import Generator

    gen = Generator(str(EXAMPLES_DATA_PATH / "jpeg.image.in.yaml"), verbose=0)
    gen.input

    gen.run().plot("x", "y")

    inputs = gen.input.copy()
    inputs["random"] = {"type": "pseudo", "seed": 0}

    gen = Generator(inputs, verbose=0)
    gen.run().plot("x", "y")


def test_1d_distributions():
    # Distgen supports several one dimensional distribution types.
    #
    # # Uniform 1D
    #
    # The uniform distirbuition is defined by a probability distribution function:
    #
    # $\rho(x) = \frac{1}{b-a}$ for $a\leq x\leq b$ and zero elsewhere.
    #
    # The corresponding CDF is
    #
    # $P(x) = \frac{x-a}{b-a}$ for $a\leq x\leq b$ and zero elsewhere.
    #
    # The first and second moments of this distribution are:
    #
    # $\langle x \rangle = \frac{1}{2}(a+b)$ and $\sigma_x = \frac{b-a}{\sqrt{12}}$

    from distgen.dist import Uniform

    var = "x"
    verbose = 1
    params = {"min_x": 2 * unit_registry("mm"), "max_x": 4 * unit_registry("mm")}
    uniform = Uniform(var, verbose=verbose, **params)
    uniform.plot_pdf()
    uniform.plot_cdf()
    uniform.test_sampling()

def test_linear_distributin():

    from distgen.dist import Linear
    
    var = "x"
    verbose = 1
    params = {"min_x": 2 * unit_registry("mm"), "max_x": 4 * unit_registry("mm"), "slope_fraction": 0.25}
    linear = Linear(var, verbose=verbose, **params)
    linear.plot_pdf()
    linear.plot_cdf()
    linear.test_sampling()

def test_normal_distribution():
    # # Normal Distribution (including truncation)
    #
    # The general form of a normal distribution PDF with truncation is given by
    #
    # $\rho(x) =
    # \frac{1}{\sigma}\frac{\phi\left(\frac{x-\mu}{\sigma}\right)}{\Phi\left(\frac{b-\mu}{\sigma}\right)-\Phi\left(\frac{a-\mu}{\sigma}\right)}$.
    #
    # In this expression $\phi(\xi) =
    # \frac{1}{\sqrt{2\pi}}e^{-\frac{1}{2}\xi^2}$ is the canonical normal
    # distribution, $\Phi(\xi) = \frac{1}{2}\left[1 +
    # \text{erf}\left(\frac{\xi}{\sqrt{2}}\right) \right]$ is the canonical
    # normal CDF, and $a=-N_{\text{cutoff}}\cdot\sigma$ and
    # $b=-N_{\text{cutoff}}\cdot\sigma$ are the left and right truncation
    # points.  The CDF if given by
    #
    # $P(x) = \frac{\Phi\left(\frac{x-\mu}{\sigma}\right) -
    # \Phi\left(\frac{a-\mu}{\sigma}\right)}{\Phi\left(\frac{b-\mu}{\sigma}\right)-\Phi\left(\frac{a-\mu}{\sigma}\right)}$.
    #
    # Defining $\alpha = \frac{a-\mu}{\sigma}$ and $\beta =
    # \frac{b-\mu}{\sigma}$, the first and second moments of the distribution
    # are:
    #
    # $\langle x\rangle = \mu + \frac{\phi\left(\alpha\right) -
    # \phi\left(\beta\right)}{\Phi\left(\beta\right)-\Phi\left(\alpha\right)}\sigma$
    # and $\sigma_x = \sigma \left\{1 + \frac{\alpha\phi\left(\alpha\right) -
    # \beta\phi(\beta) }{\Phi(\beta) - \Phi(\alpha)} -
    # \left(\frac{\phi\left(\alpha\right) - \phi(\beta)}{\Phi(\beta) -
    # \Phi(\alpha)}\right)^{2} \right\}^{1/2} $.
    #
    # When using this distribution, if the $N_{\text{cutoff}}$ is not set then
    # the distribution reduces to an infinite range normal distribution, as
    # first shown below:

    from distgen.dist import Norm

    var = "x"
    verbose = 1
    params = {"sigma_x": 2 * unit_registry("mm"), "avg_x": -1 * unit_registry("mm")}
    norm = Norm(var, verbose=verbose, **params)
    norm.plot_pdf()
    norm.plot_cdf()
    norm.test_sampling()


def test_normal_distribution1():
    # Below the $N_{\text{cutoff}}$ parameter is set to cut the distribution
    # symmetrically:

    from distgen.dist import Norm

    var = "x"
    verbose = 1
    params = {
        "sigma_x": 2 * unit_registry("mm"),
        "avg_x": 0 * unit_registry("mm"),
        "n_sigma_cutoff": 2,
    }
    norm = Norm(var, verbose=verbose, **params)
    norm.plot_pdf()
    norm.plot_cdf()
    norm.test_sampling()


def test_normal_distribution_2():
    # The distribution can be truncated asymmetrically using the
    # $N_{\text{cutoff},R}$ and $N_{\text{cutoff},L}$ parameters, as shown
    # below.  Note in this case, it is only required that $N_{\text{cutoff},L}
    # < N_{\text{cutoff},R}$, allowing for completley arbtitray location of the
    # truncation points.  This requires a minus sign for the cut off parameters
    # for truncation values less than zero.

    from distgen.dist import Norm

    params = {
        "sigma_x": 2 * unit_registry("mm"),
        "avg_x": 0 * unit_registry("mm"),
        "n_sigma_cutoff_left": -1.5,
        "n_sigma_cutoff_right": 1,
    }

    norm = Norm("x", verbose=1, **params)
    norm.plot_pdf()
    norm.plot_cdf()
    norm.test_sampling()


def test_super_gaussian():
    # # Super Gaussian
    #
    # In additional to the regular Gaussian function, it is also possible to
    # sample a super-Gaussian distribution defined by
    #
    # $\rho(x; \lambda, p) =
    # \frac{1}{2\sqrt{2}\Gamma\left(1+\frac{1}{2p}\right)\lambda }
    # \exp\left[-\left(\frac{(x-\mu)^2 }{2\lambda^2}\right)^{p}\right]$
    #
    # Here $\sigma_1$ is the length scale and $p$ is the power of the
    # super-Gaussian. Note when $p=1$ reduces to a Normal distirbution, in
    # which case $\sigma_x=\lambda$.  As $p\rightarrow\infty$ the distribution
    # reduces to a flat-top (uniform). The full range of powers is given by
    # $p\in\left(0,\infty\right]$.
    #
    # The first and second moments of the distribution are given by:
    #
    # $\langle x\rangle = \mu$, and $\sigma_x =
    # \left(\frac{2\Gamma\left(1+\frac{3}{2p}\right)}{3\Gamma\left(1+\frac{1}{2p}\right)}\right)^{1/2}\lambda$.
    #
    #
    # Often, it is convenient to scan the distribution from the uniform limit
    # to the Gaussian limit.  To do some, the input $p$ can be parameterized by
    # $\alpha\in[0,1]$ where $p = 1/\alpha$.  Here $\alpha=0$ corresponds to a
    # flat-top (uniform) and $\alpha=1$ corresponds to a Gaussian.  Examples of
    # both types of usage are shown below.

    from distgen.dist import SuperGaussian

    ps = [0.5, 1, 5, float("Inf")]
    alphas = [0, 0.25, 0.5, 1]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, sharex="col", figsize=(12, 4), constrained_layout=True
    )

    plegs = ["p = " + str(p) for p in ps]
    alegs = ["$\\alpha$ = " + str(a) for a in alphas]

    for ii, p in enumerate(ps):
        pparams = {
            "lambda": 2 * unit_registry("mm"),
            "p": p * unit_registry("dimensionless"),
        }

        supG = SuperGaussian("x", verbose=0, **pparams)
        x = supG.get_x_pts(1000)
        rho = supG.pdf(x)
        ax1.plot(x, rho)
        a = alphas[ii]
        aparams = {
            "lambda": 2 * unit_registry("mm"),
            "alpha": a * unit_registry("dimensionless"),
        }

        x = np.linspace(-3 * aparams["lambda"], 3 * aparams["lambda"], 100)
        supG = SuperGaussian("x", verbose=0, **aparams)
        rho = supG.pdf(x)
        ax2.plot(x, rho)

    ax1.set_xlabel("x (mm)")
    ax2.set_xlabel("x (mm)")
    ax1.set_ylabel("pdf (1/mm)")
    ax2.set_ylabel("pdf (1/mm)")
    ax1.legend(plegs)
    ax2.legend(alegs)
    # To set the length scale of the distribution, the user must either supply 'sigma_[var]' or 'lambda'. See usage below:

    params = {
        "sigma_x": 2 * unit_registry("mm"),
        #'alpha': 0.75*unit_registry('dimensionless'),
        "alpha": 0.003 * unit_registry("dimensionless"),
        "avg_x": 0.25 * unit_registry("mm"),
    }

    supG = SuperGaussian("x", verbose=1, **params)
    supG.plot_pdf()
    supG.plot_cdf()
    supG.test_sampling()


def test_1d_pdf_from_file():
    # # 1D PDF from a file

    # Distgen supports importing a 1D PDF saved in column form in.  The input
    # form of the file should have space separated headers such as $x$ and
    # $Px$, with corresponding column data below it.  The PDF is normalized
    # numerically using the numpy.trapz numerical integration routine. The CDF
    # is computed using the scipy.cumtrapz cumulative numerical intgration
    # routine.
    #
    # The following example shows a gaussian PDF with cuts added to it.

    from distgen.dist import File1d

    var = "t"
    verbose = 1
    params = {"file": str(EXAMPLES_DATA_PATH / "cutgauss.1d.txt"), "units": "ps"}
    file1d = File1d(var, verbose=verbose, **params)
    file1d.plot_pdf()
    file1d.plot_cdf()
    file1d.test_sampling()


def test_pulsed_laser_soliton():
    # # $sech^2$ - Pulsed Laser Soliton

    from distgen.dist import Sech2

    verbose = 1
    # params={'tau':1.0*unit_registry('ps'), 'avg_t':1.0*unit_registry('ps')}
    params = {"sigma_t": 1.0 * unit_registry("ps"), "avg_t": 1.0 * unit_registry("ps")}
    laser_pulse = Sech2(verbose=verbose, **params)

    laser_pulse.plot_pdf()
    laser_pulse.plot_cdf()
    laser_pulse.test_sampling()


def test_laser_pulse_stacking():
    # # Laser pulse stacking

    from distgen.dist import TemporalLaserPulseStacking

    verbose = 1
    params = {
        "crystal_length_1": 15.096 * unit_registry("mm"),
        "crystal_length_2": 7.548 * unit_registry("mm"),
        "crystal_length_3": 3.774 * unit_registry("mm"),
        "crystal_length_4": 1.887 * unit_registry("mm"),
        "crystal_angle_1": 0.6 * unit_registry("deg"),
        "crystal_angle_2": 1.8 * unit_registry("deg"),
        "crystal_angle_3": -0.9 * unit_registry("deg"),
        "crystal_angle_4": -0.5 * unit_registry("deg"),
    }

    laser_pulse = TemporalLaserPulseStacking(verbose=verbose, **params)
    laser_pulse.plot_pdf()
    laser_pulse.plot_cdf()
    laser_pulse.test_sampling()


def test_tukey_1d():
    # # Tukey 1D

    from distgen.dist import Tukey

    var = "y"
    params = {
        "length": 2 * unit_registry("mm"),
        "ratio": 0.75 * unit_registry("dimensionless"),
    }
    tukey = Tukey(var, verbose=1, **params)
    tukey.plot_pdf()
    tukey.plot_cdf()
    tukey.test_sampling()


def test_superposition_1d():
    # # Superposition 1D

    from distgen.dist import Superposition

    params = {
        "dists": {
            "d1": {
                "avg_z": -1 * unit_registry("mm"),
                "sigma_z": 1 * unit_registry("mm"),
                "type": "gaussian",
            },
            "d2": {
                "avg_z": +1 * unit_registry("mm"),
                "sigma_z": 1 * unit_registry("mm"),
                "type": "gaussian",
            },
        }
    }

    sup = Superposition("z", 1, **params)
    sup.plot_pdf()
    sup.plot_cdf()
    sup.test_sampling()


def test_maxwell_boltzmann_distribution():
    # # Maxwell-Boltzmann Distribution

    from distgen.dist import MaxwellBoltzmannDist

    params = {"scale_p": 10 * unit_registry("eV/c")}

    mb = MaxwellBoltzmannDist("p", verbose=0, **params)

    mb.plot_pdf()
    mb.plot_cdf()
    mb.test_sampling()


def test_radial_distributions():
    # # Radial Distributions
    # ---

    from distgen.dist import UniformRad

    params = {"min_r": 1 * unit_registry("mm"), "max_r": 2 * unit_registry("mm")}
    urad = UniformRad(verbose=1, **params)
    urad.plot_pdf()
    urad.plot_cdf()
    urad.test_sampling()


def test_radial_normal_distribution_truncation():
    # # Radial Normal Distribution (with truncation)
    #
    # The radial normal distribution including truncation(s) has a probability
    # function given by
    #
    # $\rho_r(r) =
    # \frac{1}{\sigma^2}\frac{\phi(r/\sigma)}{\phi\left(\frac{r_L}{\sigma}\right)-\phi\left(\frac{r_R}{\sigma}\right)}
    # $ for $0 \leq r_L \leq r \leq r_R$ and zero everywhere else.
    #
    # In this expresion $\phi(\xi) = \frac{1}{2\pi}\exp\left(-\xi^2/2\right)$
    # is the canonical raidial normal distirbution (no truncation), and the
    # scale parameter $\sigma$ follows from the product of two normal
    # distributions in $x$ and $y$ when $\sigma=\sigma_x=\sigma_y$.  The
    # corresponding CDF is given by
    #
    # $P(r)=
    # \frac{\phi\left(\frac{r_L}{\sigma}\right)-\phi\left(\frac{r}{\sigma}\right)}{\phi\left(\frac{r_L}{\sigma}\right)-\phi\left(\frac{r_R}{\sigma}\right)}
    # $ for $0 \leq r_L \leq r$.
    #
    # The corresponding first and second moments are:
    #
    # $\langle r\rangle =
    # \frac{\frac{r_L}{\sigma}\phi\left(\frac{r_L}{\sigma}\right)
    # -\frac{r_R}{\sigma}\phi\left(\frac{r_R}{\sigma}\right)
    # +\frac{1}{2\sqrt{2\pi}}\left(
    # \text{erf}\left(\frac{r_R}{\sigma\sqrt{2}}\right) -
    # \text{erf}\left(\frac{r_L}{\sigma\sqrt{2}}\right) \right) }
    # {\phi\left(\frac{r_L}{\sigma}\right)-\phi\left(\frac{r_R}{\sigma}\right)}$,
    #
    # $r_{rms} = \sqrt{ 2\sigma^2 + r_L^2 -
    # \frac{(r_R^2-r_L^2)\phi(r_R/\sigma)}{\phi\left(\frac{r_L}{\sigma}\right)-\phi\left(\frac{r_R}{\sigma}\right)}
    # }$.
    #
    # Note that in the limits $r_L\rightarrow 0$ and $r_R -> \infty$ the above
    # expressions reduce to the underlying radial normal distribution:
    #
    # $\rho_r(r)\rightarrow
    # \frac{\phi\left(\frac{r}{\sigma}\right)}{\sigma^2}$, $P(r)\rightarrow 1 -
    # \phi\left(\frac{r}{\sigma}\right)$, $\langle r\rangle\rightarrow
    # \sqrt{\frac{\pi}{2}}\sigma$, and $r_{rms}\rightarrow \sqrt{2}\sigma$.
    # This limiting case is shown first below.
    #

    from distgen.dist import NormRad

    params = {"sigma_xy": 1 * unit_registry("mm")}
    nrad = NormRad(verbose=1, **params)
    nrad.plot_pdf()
    nrad.plot_cdf()
    nrad.test_sampling()


def test_radial_normal_distribution_truncation1():
    # For laser scientists it can be convenient to to work with a pinhole radius and a fraction of the laser intensity to clip a transverse normal laser mode at.  In this case the user can supply a truncation radius ($=r_R$) and a truncation fraction $f = \exp\left(-\frac{r_R^2}{2\sigma}\right)$ from which distgen determines the underlying $\sigma$.  The example below demonstrates this usage:

    from distgen.dist import NormRad

    params = {
        "truncation_radius": 1 * unit_registry("mm"),
        "truncation_fraction": 0.5 * unit_registry("dimensionless"),
    }
    nrad = NormRad(verbose=1, **params)
    nrad.plot_pdf()
    nrad.plot_cdf()
    nrad.test_sampling()


def test_radial_normal_distribution_truncation2():
    from distgen.dist import NormRad

    params = {"sigma_xy": 2 * unit_registry("mm"), "n_sigma_cutoff": 1}
    nrad = NormRad(verbose=1, **params)
    nrad.plot_pdf()
    nrad.plot_cdf()
    nrad.test_sampling()


def test_radial_tukey():
    # # Radial Tukey

    from distgen.dist import TukeyRad

    params = {
        "length": 1 * unit_registry("mm"),
        "ratio": 0.75 * unit_registry("dimensionless"),
    }
    rtukey = TukeyRad(verbose=1, **params)
    rtukey.plot_pdf()
    rtukey.plot_cdf()
    rtukey.test_sampling()


def test_radial_super_gaussian():
    # # Radial Super Gaussian This implements a radial version of the Super
    # Gaussian function discussed above.  Here the radial function takes the
    # form:
    #
    # $2\pi\rho(r;\lambda,p) =
    # \frac{1}{\Gamma\left(1+\frac{1}{p}\right)\lambda^2}
    # \exp\left[-\left(\frac{r^2}{2\lambda^2}\right)^p\right]$. The
    # corrsponding CDF is: ?
    #
    # The first and (rms) second moment of the distribution are given by:
    #
    # $\langle r\rangle =
    # \frac{2\sqrt{2}}{3}\frac{\Gamma\left(1+\frac{3}{2p}\right)}{\Gamma\left(1+\frac{1}{p}\right)}\lambda$,
    #
    # $r_{\text{rms}} =
    # \sqrt{\frac{\Gamma\left(1+\frac{2}{p}\right)}{\Gamma\left(1+\frac{1}{p}\right)}}\lambda$.
    #

    from distgen.dist import SuperGaussianRad

    params = {
        "sigma_xy": 1 * unit_registry("mm"),
        "alpha": 0.50 * unit_registry("dimensionless"),
    }
    supG = SuperGaussianRad(verbose=1, **params)
    supG.plot_pdf()
    supG.plot_cdf()
    supG.test_sampling()


def test_radial_file_distribution():
    # Radial File Distribution

    from distgen.dist import RadFile

    params = {"file": str(EXAMPLES_DATA_PATH / "cutgauss.rad.txt"), "units": "mm"}

    rfd = RadFile(verbose=1, **params)
    rfd.plot_pdf()
    rfd.plot_cdf()
    rfd.test_sampling()


# Angular Distributions (TODO)
# ---
# Angular distributions define one dimensional probability functions for the cylindrical variable $\theta$.
