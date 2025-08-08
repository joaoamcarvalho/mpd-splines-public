import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import os
import scipy.stats
from matplotlib import colors, pyplot as plt
from matplotlib.patches import Ellipse
from scipy import stats as st


def save_fig(fig, name, dir=f"./{os.path.basename(__file__).split('.py')[0]}"):
    fig.tight_layout()
    os.makedirs(dir, exist_ok=True)
    fig.savefig(f"{dir}/{name}.png")
    fig.savefig(f"{dir}/{name}.pdf")


def export_legend(plot_options_d, filename="legend.pdf", plot_dir="", ncol=10, linewidth=7, plot_type="plot"):
    fig2 = plt.figure()
    ax2 = fig2.add_subplot()
    for k, v in plot_options_d.items():
        v["linewidth"] = linewidth
        if plot_type == "plot":
            ax2.plot([], [], label=k, **v)
        elif plot_type == "scatter":
            ax2.scatter([], [], label=k, **v)
        else:
            raise NotImplementedError
    ax2.axis("off")
    legend = ax2.legend(frameon=False, loc="lower center", ncol=ncol)
    # for legobj in legend.legendHandles:
    #    legobj.set_linewidth(linewidth)
    fig1 = legend.figure
    fig1.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig1.dpi_scale_trans.inverted())
    fig1.savefig(os.path.join(plot_dir, filename), dpi="figure", bbox_inches=bbox)
    plt.close(fig1)
    plt.close(fig2)


def set_small_ticks(ax, fontsize=6, set_minor_ticks=False):
    if set_minor_ticks:
        ax.tick_params(which="minor", grid_linestyle="--")
    else:
        ax.get_yaxis().set_tick_params(which="minor", size=0)
        ax.get_yaxis().set_tick_params(which="minor", width=0)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)

    for tick in ax.xaxis.get_minor_ticks():
        tick.label.set_fontsize(fontsize)

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)

    for tick in ax.yaxis.get_minor_ticks():
        tick.label.set_fontsize(fontsize)


def get_borderless_figure(size=(4, 4), **figkwargs):
    fig = plt.figure(**figkwargs)
    fig.set_size_inches(size)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    return fig, ax


def remove_borders(ax, top=False, right=False, bottom=False, left=False):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)


def remove_axes_labels_ticks(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")


def confidence_ellipse(x, y, ax, n_std=3.0, facecolor="none", **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def mean_confidence_interval(data, confidence=0.95, axis=0):
    n = data.shape[axis]
    m, se = np.mean(data, axis=axis), scipy.stats.sem(data, axis=axis)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return m, m - h, m + h


def plot_trajectories_kde(ax, trajs, cmap_color="Reds", **kwargs):
    if trajs is not None:
        # https://stackoverflow.com/a/30146280
        data = np.reshape(trajs, (-1, 2))
        x = data[:, 0]
        y = data[:, 1]
        xmin, xmax = -1, 1
        ymin, ymax = -1, 1

        # Perform the kernel density estimate
        xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        values = np.vstack([x, y])
        kernel = st.gaussian_kde(values, bw_method="scott")
        f = np.reshape(kernel(positions).T, xx.shape)

        # Plot
        ax.imshow(
            np.rot90(f),
            extent=[xmin, xmax, ymin, ymax],
            cmap=cmap_color,
            zorder=5,
            alpha=0.5,
            norm=colors.PowerNorm(gamma=0.75),
        )
