from matplotlib import pyplot as plt
import numpy as np
from .tools import *


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

def plot_1d(beam,var,units,**params):
    
    if("nbins" in params):
        bins = params["nbins"]
    else:
        bins = 10

    thist,tedges = np.histogram( (beam[var].to(units)).magnitude,bins=100)
    ts = (tedges[1:] + tedges[:-1]) / 2
    rhot = thist/np.trapz(thist,ts)
    
    tst = np.zeros(len(ts)+2)
    tst[1:-1]=ts
    tst[0]=ts[0]; tst[-1]=ts[-1]
    
    rhott = np.zeros(len(ts)+2)
    rhott[1:-1]=rhot
    rhott[0]=0; rhott[-1]=0
    
    avgt = np.mean( (beam[var].to(units)) )
    stdt = np.std( (beam[var].to(units)) )
    
    avgt_str = "{:0.3f~P}".format(avgt)
    stdt_str = "{:0.3f~P}".format(stdt)
    
    plt.title('$<'+var+'>$ = '+avgt_str+', $\sigma_{'+var+'}$ = '+stdt_str)
    plt.xlabel(var+' ('+units+')')
    plt.ylabel('$pdf('+var+')$')
    plt.plot(tst,rhott)
    
def plot_2d(beam, Nfig, var1, units1, var2, units2, ptype, **params):

    #plt.figure(Nfig)
    
    if(ptype=="scatter"):
        fig,ax =plt.plot(beam[var1].to(units1).magnitude,beam[var2].to(units2).magnitude,'*')
    if(ptype=="scatter_hist2d"):
        if("nbins" in params):
            nbins=params["nbins"]
        else:
            nbins=10
        ax = scatter_hist2d(beam[var1].to(units1).magnitude,beam[var2].to(units2).magnitude, bins=[nbins,nbins], s=5, cmap=plt.get_cmap('jet'))
        
    if("axis" in params and params["axis"]=="equal"):
        plt.gca().set_aspect('equal', adjustable='box')

    plt.xlabel(var1 + " ("+units1+")")
    plt.ylabel(var2 + " ("+units2+")")
    
    stdx = beam[var1].std().to(units1)
    stdy = beam[var2].std().to(units2)
    stdx_str = "{:0.3f~P}".format(stdx)
    stdy_str = "{:0.3f~P}".format(stdy)
    if(stdx==0):
        plt.xlim([-1,1])
    if(stdy==0):
        plt.ylim([-1,1])
    plt.title('$\sigma_{'+var1+'}$ = '+stdx_str+', $\sigma_{'+var2+'}$ = '+stdy_str)
    
    return ax
        
def plot_current_profile(beam,Nfig,units={"t":"ps","q":"pC"},nbins=100):
    
    plt.figure(Nfig)
    thist,tedges = histogram(beam["t"].to(units["t"]),bins=nbins)
    qb = beam.q.to(units["q"])
    ts = (tedges[1:] + tedges[:-1]) / 2
    I0 = qb/(1*unit_registry(units["t"]))
    I0 = I0.to_compact()
    rhot = I0*thist/np.trapz(thist,ts)

    tst = np.zeros(len(ts)+2)
    tst[1:-1]=ts
    tst[0]=ts[0]; tst[-1]=ts[-1]
    
    rhott = np.zeros(len(ts)+2)
    rhott[1:-1]=rhot.magnitude
    rhott[0]=0; rhott[-1]=0
    
    stdt = beam["t"].std().to(units["t"])
    stdt_str = "{:0.3f~P}".format(stdt)
    
    plt.xlabel("t ("+units["t"]+")")
    plt.ylabel("I(t) ("+str(rhot.units)+")")
    plt.plot(tst,rhott)
    
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

