from matplotlib import pyplot as plt
import numpy as np
#f#rom .tools import *

from .physical_constants import unit_registry, pi
from .tools import histogram
from .tools import radial_histogram
from .tools import trapz
from .tools import interp
from .tools import radint
from .tools import linspace
from .tools import mean
from .tools import centers
from .tools import zeros

LABELS = {'x':'x', 'y':'y', 'z':'z', 'px':'p_x', 'py':'p_y', 'pz':'p_z', 't':'t', 'r':'r', 'pr':'p_r', 'ptheta':'p_{\\theta}','thetax':'\\theta_x'}

def get_scale(beam, scale):

    """
    Maps a scale factor name to the underlying beam quantity
    """
    if(scale=='charge'):
        return beam.q

    elif(scale=='number'):
        return beam['n_particle']

    else:
        raise ValueError('Could not get scale factor for plot.')
   

def hist_to_pdf(hist, edges, scale=1.0, is_radial=False):

    """
    Function that prepares 1d or radial histogram data for plotting as a pdf or radial distribution
    inputs: hist: histogram values [array], 
            edges: histogram bin edges [array]
            sytle: flag for plotting style [string] in {'hist', 'smooth'}
    """
    xc = centers(edges)

    if(is_radial):  # normalize the distribution
        norm = radint(hist,xc)
    else:
        norm = trapz(hist, xc)      

    hist = scale*hist/norm

    if(is_radial):  # integrate the distribution to get back scale 
        final_scale = radint(hist,xc)
    else:
        final_scale = trapz(hist, xc)

    rho = zeros((hist.size*2,), hist.units)
    x = zeros((hist.size*2,), edges.units)
        
    rho[0::2] = hist
    rho[1::2] = hist
    x[0::2] = edges[:-1]
    x[1::2] = edges[1:]

    if(not is_radial):
        rho = np.insert(rho, 0, 0*hist.units)
        x = np.insert(x, 0, edges[0])

    rho = np.append(rho, 0*hist.units)
    x = np.append(x, edges[-1])

    return (rho, x, final_scale)


def plot_dist1d(beam, var, units, scale='charge', dist_units=None, ax=None, **params):

    """
    Plot a 1d distrbution by histogramming beam particle coordinates 'var'
    """
    if('nbins' in params):
        nbins = params['nbins']
    else:
        nbins = int(np.sqrt(len(beam[var])))

    scale_factor = get_scale(beam, scale)

    if('title_on' in params):
        title_on = params['title_on']
    else:
        title_on = False
 
    hist_x, x_edges = histogram(beam[var].to(units), weights=beam['w'], nbins=nbins)
    (rho_x, x, total_scale) = hist_to_pdf(hist_x, x_edges, scale=scale_factor)
    
    if(dist_units):
        rho_x.ito(dist_units)

    avgt = beam.avg(var, units)
    stdt = beam.std(var, units)
    
    label = LABELS[var]

    if(ax is None):
        ax = plt.gca()

    ax.plot(x, rho_x)
    ax.set_xlabel(f'{var} ({units})')
    ax.set_ylabel(f'{scale} density ({rho_x.units:~P})')

    if(title_on):
        ax.set_title('$<'+label+'>$ = '+f'{avgt:G~P}, '+'$\sigma_{'+label+'}$ = '+f'{stdt:G~P}, total {scale} = {total_scale:G~P}')


def plot_current_profile(beam, t_units, current_units, ax=None, **params):
    """
    Plots the 1D histogram of the time coordinate
    """
    return plot_dist1d(beam, 't', t_units, scale='charge', dist_units=current_units, ax=ax, **params)


def plot_radial_dist(beam, r_units, scale='charge', dist_units=None, ax=None, **params):
    """
    Plots the 1D histogram of the radial coordinate r
    """
    if(ax is None):
        ax = plt.gca()

    if('nbins' in params):
        nbins = params['nbins']
    else:
        nbins = 50
    
    scale_factor = get_scale(beam, scale)

    r_hist, r_edges = radial_histogram(beam['r'].to(r_units), weights=beam['w'], nbins=nbins)
    (rho, r, scale_factor) = hist_to_pdf(r_hist, r_edges, scale=scale_factor, is_radial=True)

    ax.plot(r, rho)
    ax.set_xlabel(f'${LABELS["r"]}$ ({r.units:G~P})')
    ax.set_ylabel(f'{scale} density ({rho.units:G~P})')

    ax.set_xlim([0, ax.get_xlim()[1]])
    ax.set_ylim([0, ax.get_ylim()[1]])
    
    if('title_on' in params and params['title_on']):
        avgr = mean(beam['r'], beam['w'])
        rmsr = np.sqrt( mean(beam['r']*beam['r'], beam['w']) )
        ax.set_title(f'$<{LABELS["r"]}>$ = {avgr:G~P}, '+'$r_{rms}$'+f' = {rmsr:G~P}, total {scale} = {scale_factor:G~P}')

    return ax


def plot_dist2d(beam, var1, units1, var2, units2, style='scatter_hist2d', ax=None, Nfig=None,  **params):
    
    """
    Plot a 2d distribution by histogramming particle coordinates var1 and var2
    """
    if(style=="scatter"):
        fig,ax =plt.plot(beam[var1].to(units1).magnitude,beam[var2].to(units2).magnitude,'*')

    if(style=="scatter_hist2d"):
        if("nbins" in params):
            nbins=params["nbins"]
        else:
            nbins=int(np.sqrt(len(beam[var1]))/3)

        scatter_hist2d(beam[var1].to(units1).magnitude,beam[var2].to(units2).magnitude, bins=[nbins,nbins], s=5, cmap=plt.get_cmap('jet'),ax=ax)
        
    if(ax is None):
        ax = plt.gca()

    if("axis" in params and params["axis"]=="equal"):
        ax.set_aspect('equal', adjustable='box')
    
    avgx = beam.avg(var1).to(units1)
    avgy = beam.avg(var2).to(units2)

    stdx = beam[var1].std().to(units1)
    stdy = beam[var2].std().to(units2)

    ax.set_xlabel(f'${LABELS[var1]}$ ({stdx.units:~P})')
    ax.set_ylabel(f'${LABELS[var2]}$ ({stdy.units:~P})')

    if(stdx==0):
        plt.xlim([-1,1])
    if(stdy==0):
        plt.ylim([-1,1])
  
    if('title_on' in params and params['title_on']):
        line1 = f'$<{LABELS[var1]}>$ = {avgx:G~P}, '+'$\sigma_{'+LABELS[var1]+'}$ = '+f'{stdx:G~P}'
        line2 = f'$<{LABELS[var2]}>$ = {avgy:G~P}, '+'$\sigma_{'+LABELS[var2]+'}$ = '+f'{stdy:G~P}'
        ax.set_title(line1+'\n'+line2)
    return ax


def map_hist(x, y, h, bins):
    xi = np.digitize(x, bins[0]) - 1
    yi = np.digitize(y, bins[1]) - 1
    inds = np.ravel_multi_index((xi, yi),
                                (len(bins[0]) - 1, len(bins[1]) - 1),
                                mode='clip')
    vals = h.flatten()[inds]
    bads = ((x < bins[0][0]) | (x > bins[0][-1]) |
            (y < bins[1][0]) | (y > bins[1][-1]))
    vals[bads] = np.NaN
    return vals


def scatter_hist2d(x, y,
                   s=20, marker=u'o',
                   mode='mountain',
                   bins=10, range=None,
                   normed=False, weights=None,  # np.histogram2d args
                   edgecolors='none',
                   ax=None, dens_func=None,
                   **kwargs):
    """
    Make a scattered-histogram plot.
    Parameters
    ----------
    x, y : array_like, shape (n, )
        Input data
    s : scalar or array_like, shape (n, ), optional, default: 20
        size in points^2.
    marker : `~matplotlib.markers.MarkerStyle`, optional, default: 'o'
        See `~matplotlib.markers` for more information on the different
        styles of markers scatter supports. `marker` can be either
        an instance of the class or the text shorthand for a particular
        marker.
    mode: [None | 'mountain' (default) | 'valley']
       Possible values are:
       - None : The points are plotted as one scatter object, in the
         order in-which they are specified at input.
       - 'mountain' : The points are sorted/plotted in the order of
         the number of points in their 'bin'. This means that points
         in the highest density will be plotted on-top of others. This
         cleans-up the edges a bit, the points near the edges will
         overlap.
       - 'valley' : The reverse order of 'mountain'. The low density
         bins are plotted on top of the high-density ones.
    bins : int or array_like or [int, int] or [array, array], optional
        The bin specification:
          * If int, the number of bins for the two dimensions (nx=ny=bins).
          * If array_like, the bin edges for the two dimensions
            (x_edges=y_edges=bins).
          * If [int, int], the number of bins in each dimension
            (nx, ny = bins).
          * If [array, array], the bin edges in each dimension
            (x_edges, y_edges = bins).
          * A combination [int, array] or [array, int], where int
            is the number of bins and array is the bin edges.
    range : array_like, shape(2,2), optional
        The leftmost and rightmost edges of the bins along each dimension
        (if not specified explicitly in the `bins` parameters):
        ``[[xmin, xmax], [ymin, ymax]]``. All values outside of this range
        will be considered outliers and not tallied in the histogram.
    normed : bool, optional
        If False, returns the number of samples in each bin. If True,
        returns the bin density ``bin_count / sample_count / bin_area``.
    weights : array_like, shape(N,), optional
        An array of values ``w_i`` weighing each sample ``(x_i, y_i)``.
        Weights are normalized to 1 if `normed` is True. If `normed` is
        False, the values of the returned histogram are equal to the sum of
        the weights belonging to the samples falling into each bin.
    edgecolors : color or sequence of color, optional, default: 'none'
        If None, defaults to (patch.edgecolor).
        If 'face', the edge color will always be the same as
        the face color.  If it is 'none', the patch boundary will not
        be drawn.  For non-filled markers, the `edgecolors` kwarg
        is ignored; color is determined by `c`.
    ax : an axes instance to plot into.
    dens_func : function or callable (default: None)
        A function that modifies (inputs and returns) the dens
        values (e.g., np.log10). The default is to not modify the
        values, which will modify their coloring.
    kwargs : these are all passed on to scatter.
    Returns
    -------
    paths : `~matplotlib.collections.PathCollection`
        The scatter instance.
    """
    if ax is None:
        ax = plt.gca()

    h, xe, ye = np.histogram2d(x, y, bins=bins,
                               range=range, normed=normed,
                               weights=weights)
    # bins = (xe, ye)
    dens = map_hist(x, y, h, bins=(xe, ye))
    if dens_func is not None:
        dens = dens_func(dens)
    iorder = slice(None)  # No ordering by default
    if mode == 'mountain':
        iorder = np.argsort(dens)
    elif mode == 'valley':
        iorder = np.argsort(dens)[::-1]
    x = x[iorder]
    y = y[iorder]
    dens = dens[iorder]
    return ax.scatter(x, y,
                      s=s, c=dens,
                      edgecolors=edgecolors,
                      marker=marker,
                      **kwargs)


if __name__ == '__main__':

    randgen = np.random.RandomState(84309242)
    npoint = 10000
    x = randgen.randn(npoint)
    y = 2 * randgen.randn(npoint) + x

    lims = [-10, 10]
    bins = np.linspace(lims[0], lims[1], 50)

    fig, axs = plt.subplots(3, 1, figsize=[4, 8],
                            gridspec_kw=dict(hspace=0.5))

    ax = axs[0]
    ax.plot(x, y, '.', color='b', )
    ax.set_title("Traditional Scatterplot")

    ax = axs[1]
    ax.hist2d(x, y, bins=[bins, bins])
    ax.set_title("Traditional 2-D Histogram")

    ax = axs[2]
    scatter_hist2d(x, y, bins=[bins, bins], ax=ax, s=5)
    ax.set_title("Scatter histogram combined!")

    for ax in axs:
        ax.set_xlim(lims)
        ax.set_ylim(lims)
    fig.savefig('ScatterHist_Example.png', dpi=200)

