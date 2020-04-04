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

LABELS = {'x':'x', 'y':'y', 'z':'z', 'px':'p_x', 'py':'p_y', 'pz':'p_z', 't':'t', 'r':'r', 'pr':'p_r', 'ptheta':'p_{\\theta}','thetax':'\\theta_x'}

def plot_beam(beam,units={"x":"mm","y":"mm","z":"mm","px":"eV/c","py":"eV/c","pz":"eV/c","t":"ps","q":"pC"}):

    # X-Y
    plt.figure(1)
    nbins = np.sqrt(len(beam["x"]))/3
    scat = scatter_hist2d(beam["x"].to(units["x"]).magnitude, beam["y"].to(units["y"]).magnitude, bins=[nbins,nbins], s=5, cmap=plt.get_cmap('jet'))

    stdx = beam["x"].std().to(units["x"])
    stdy = beam["y"].std().to(units["y"])

    stdx_str = "{:0.3f~P}".format(stdx)
    stdy_str = "{:0.3f~P}".format(stdy)

    if(stdx==0):
        plt.xlim([-1,1])
    if(stdy==0):
        plt.ylim([-1,1])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel("$x$ ("+units["x"]+")");
    plt.ylabel("$y$ ("+units["x"]+")");
    plt.title('Laser Spot: $\sigma_x$ = '+stdx_str+", $\sigma_y$ = "+stdy_str)

    # X-PX
    plt.figure(2)
    scat = scatter_hist2d(beam["x"].to(units["x"]).magnitude, beam["px"].to(units["px"]).magnitude, bins=[nbins,nbins], s=5, cmap=plt.get_cmap('jet'))

    stdx = beam["x"].std().to(units["x"])
    stdpx = beam["px"].std().to(units["px"])

    stdx_str = "{:0.3f~P}".format(stdx)
    stdy_str = "{:0.3f~P}".format(stdpx)

    if(stdx==0):
        plt.xlim([-1,1])
    if(stdy==0):
        plt.ylim([-1,1])
    #plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel("$x$ ("+units["x"]+")");
    plt.ylabel("$p_x$ ("+units["px"]+")");
    plt.title('$\sigma_x$ = '+stdx_str+", $\sigma_{p_x}$ = "+stdy_str)

    plot_2d(beam, 10, "x", units["x"], "y", units["y"], ptype="kde") 

    # Y-PY
    plt.figure(3)
    scat = scatter_hist2d(beam["y"].to(units["y"]).magnitude, beam["py"].to(units["py"]).magnitude, bins=[nbins,nbins], s=5, cmap=plt.get_cmap('jet'))

    stdx = beam["y"].std().to(units["y"])
    stdpx = beam["py"].std().to(units["py"])

    stdx_str = "{:0.3f~P}".format(stdx)
    stdy_str = "{:0.3f~P}".format(stdpx)

    if(stdx==0):
        plt.xlim([-1,1])
    if(stdy==0):
        plt.ylim([-1,1])
    #plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel("$y$ ("+units["y"]+")");
    plt.ylabel("$p_y$ ("+units["py"]+")");
    plt.title('$\sigma_y$ = '+stdx_str+", $\sigma_{p_y}$ = "+stdy_str)

    # T
    thist,tedges = np.histogram(beam["t"].to(units["t"]),bins=100)
    qb = beam.q.to(units["q"])
    ts = (tedges[1:] + tedges[:-1]) / 2
    I0 = qb/(1*unit_registry(units["t"]))
    I0 = I0.to_compact()
    rhot = I0*thist/np.trapz(thist,ts)

    stdt = beam["t"].std().to(units["t"])
    stdt_str = "{:0.3f~P}".format(stdt)
    plt.figure(4)
    plt.xlabel("t ("+units["t"]+")")
    plt.ylabel("t ("+units["t"]+")")
    plt.plot(ts,rhot)
    plt.ylabel("I(t) ("+str(rhot.units)+")")
    plt.title('Laser Temporal Profile: $\sigma_t$ = '+stdt_str)
    plt.show()


def plot_hist_1d_data(x_edges, y, ax=None):

    y_plt = np.empty((y.size*2,), dtype=y.dtype)
    x_edges_plt = np.empty((y.size*2,), dtype=y.dtype)
    y_plt[0::2] = y
    y_plt[1::2] = y
    x_edges_plt[0::2] = x_edges[:-1]
    x_edges_plt[1::2] = x_edges[1:]

    if(ax is None):
       ax = plt.gca()
    
    ax.plot(x_edges_plt,y_plt)

    return ax


def plot_1d(beam, var, units, ax=None, **params):
    
    if('nbins' in params):
        bins = params['nbins']
    else:
        bins = 10
  
    v = beam[var].to(units)

    thist, tedges = np.histogram( v.magnitude, bins=bins)
    ts = centers(tedges)
    rhot = thist/np.trapz(thist,ts)
    
    tst = np.zeros(len(ts)+2)
    tst[1:-1]=ts
    tst[0]=ts[0]; tst[-1]=ts[-1]
    
    rhott = np.zeros(len(ts)+2)
    rhott[1:-1]=rhot
    rhott[0]=0; rhott[-1]=0
    
    avgt = np.mean( v )
    stdt = np.std(  v )
    
    avgt_str = f'{avgt:~P}'
    stdt_str = f'{stdt:~P}'

    if(ax is None):
        ax = plt.gca()

    p = 0*unit_registry(f'1/{avgt.units:~P}')

    ax.set_title(f'$<{var}>$ = {avgt:G~P}, '+'$\sigma_{'+var+'}$ = '+f'{stdt:G~P}, q_b = {beam.q:G~P}')
    ax.set_xlabel(f'{var} ({units})')
    ax.set_ylabel(f'Charge density (1/{avgt.units:~P})')
    ax.plot(tst,rhott)
    
def plot_radial_1d(beam, r_units, ax=None, **params):
        
    if(ax is None):
        ax = plt.gca()

    if('nbins' in params):
        nbins = params['nbins']
    else:
        nbins = 50
 
    if('q_units' in params):
        q  = beam.q.to(params['q_units'])
    else:
        q = beam.q
        
    w = q*beam['w']

    r = beam['r'].to(r_units)

    r_hist, r_edges = radial_histogram(r.magnitude, weights=w.magnitude, nbins=nbins)
    r_bins =  centers(r_edges)*r.units
    r_hist = r_hist*unit_registry(f'1/{r_units}^2')
    r_hist = q*r_hist/radint(r_hist, r_bins)

    qb = radint(r_hist, r_bins)

    if('style' in params):
        style = params['style']
    else:
        style = 'dist'


    if(style=='hist'):
        ax = plot_hist_1d_data(r_edges, r_hist, ax)

    elif(style=='dist'):

        rs = np.zeros((len(r_bins)+2,))*r_bins.units
        Pr = np.zeros((len(r_bins)+2,))*r_hist.units

        rs[1:-1] = r_bins   
        rs[0] = r_edges[0]*r_bins.units  
        rs[-1] = r_edges[-1]*r_bins.units

        Pr[1:-1] = r_hist; 
        Pr[0] = interp(0*r_bins.units, r_bins, r_hist); 
        Pr[-1] = 0*Pr.units

        ax.plot(rs,Pr)

    ax.set_xlabel(f'${LABELS["r"]}$ ({r.units:G~P})')
    ax.set_ylabel(f'Charge Density ({r_hist.units:G~P})')

    ax.set_xlim([0, ax.get_xlim()[1]])
    ax.set_ylim([0, ax.get_ylim()[1]])
    
    if('title_on' in params and params['title_on']):
        avgr = mean(r, beam['w'])
        rmsr = np.sqrt( mean(r*r, beam['w']) )
        ax.set_title(f'$<{LABELS["r"]}>$ = {avgr:G~P}, '+'$r_{rms}$'+f' = {rmsr:G~P}, $q_b$ = {qb:G~P}')

    return ax
    

def plot_2d(beam, Nfig, var1, units1, var2, units2, ptype, ax=None, **params):

    labels = LABELS

    #plt.figure(Nfig)
    
    if(ptype=="scatter"):
        fig,ax =plt.plot(beam[var1].to(units1).magnitude,beam[var2].to(units2).magnitude,'*')
    if(ptype=="scatter_hist2d"):
        if("nbins" in params):
            nbins=params["nbins"]
        else:
            nbins=10
        scatter_hist2d(beam[var1].to(units1).magnitude,beam[var2].to(units2).magnitude, bins=[nbins,nbins], s=5, cmap=plt.get_cmap('jet'),ax=ax)
        
    if(ax is None):
        ax = plt.gca()

    if("axis" in params and params["axis"]=="equal"):
        ax.set_aspect('equal', adjustable='box')
    
    avgx = beam[var1].mean().to(units1)
    avgy = beam[var2].mean().to(units2)
    avgx_str = "{:0.3f~P}".format(avgx)
    avgy_str = "{:0.3f~P}".format(avgy)

    stdx = beam[var1].std().to(units1)
    stdy = beam[var2].std().to(units2)

    stdx_str = f'{stdx:G~P}'
    stdy_str = f'{stdy:G~P}'

    ax.set_xlabel(f'${labels[var1]}$ ({stdx_str.split()[1]})')
    ax.set_ylabel(f'${labels[var2]}$ ({stdy_str.split()[1]})')

    if(stdx==0):
        plt.xlim([-1,1])
    if(stdy==0):
        plt.ylim([-1,1])

    if('title_on' in params and params['title_on']):
        ax.set_title(f'$<{labels[var1]}>$ = {avgx:G~P}, $<{labels[var2]}>$ = {avgy:G~P}\n$\sigma_{labels[var1]}$ = '+stdx_str+', $\sigma_{'+labels[var2]+'}$ = '+stdy_str)
    #plt.title('$\sigma_{'+var1+'}$ = '+stdx_str+', $\sigma_{'+var2+'}$ = '+stdy_str)
    
    return ax
        
def plot_current_profile(beam, Nfig, units={'t':'ps', 'q':'pC', 'I':'A'}, nbins=100, ax=None, title_on=False):
    
    if(ax is None):
        plt.figure(Nfig)
        ax = plt.gca()

    thist, tedges = histogram(beam["t"].to(units["t"]),bins=nbins)
    qb = beam.q.to(units["q"])
    ts = (tedges[1:] + tedges[:-1]) / 2
    I0 = qb/(1*unit_registry(units["t"]))
    I0 = I0.to(units['I'])
    rhot = I0*thist/np.trapz(thist,ts)

    tst = np.zeros(len(ts)+2)
    tst[1:-1]=ts
    tst[0]=ts[0]; tst[-1]=ts[-1]
    
    rhott = np.zeros(len(ts)+2)
    rhott[1:-1]=rhot.magnitude
    rhott[0]=0; rhott[-1]=0
    
    avgt = beam.avg('t').to(units['t'])
    stdt = beam.std('t').to(units['t'])
    
    q = trapz(rhot,ts*unit_registry(units['t'])).to(units['q'])

    ax.set_xlabel(f't ({avgt.units:G~P})')
    ax.set_ylabel(f'I(t) ({units["I"]})')
    ax.plot(tst, rhott)
    ax.set_ylim([0, ax.get_ylim()[1]])

    if(title_on):
        ax.set_title(f'<t> = {avgt:G~P}, $\sigma_t$ = {stdt:G~P}, $q_b$ = {q:G~P}')

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

