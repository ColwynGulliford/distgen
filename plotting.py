import numpy as np
import matplotlib.pyplot as plt


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
