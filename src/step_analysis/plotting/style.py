"""Figure style of the previous DeCIMo figures and plotting helpers.

Style taken from ``Fig_UnifiedClassif_DtFb_PerturbationAngle.py`` and
``Fig_Separation_DtFb_LSS-UML-UCS.py`` of the previous analysis: Arial 20 pt,
italic axis labels, strategy colour palette, ``o`` = NI / ``s`` = SI markers
with black edges, dotted light grid lines, custom legends.
"""

import matplotlib.pyplot as plt
import numpy as np

from .. import config

FONT_NAME = "Arial"
FONT_SIZE = 20
FIG_WIDTH, FIG_HEIGHT = 8.1962, 6.396  # inches, base figure size of the previous work
MARKER_SIZE = 100  # scatter marker area (pt^2)
MARKER_ALPHA = 0.7

PALETTE = {
    "UMS": "#54df1f",
    "UCS": "#FF9000",
    "LSS": "#00E4E0",
    "NS": "#9D55D6",
    "FS": "#D03939",
    "SCS": "#C1C1C1",
}
BOX_PALETTE = {"UMS": "#399814", "UCS": "#FFAB3D", "LSS": "#2AABA8"}
CONDITION_MARKERS = {"NI": "o", "SI": "s"}
STRATEGY_ORDER = ["UMS", "LSS", "UCS"]

GRID_KW = dict(linestyle=":", alpha=0.5, color="k", lw=0.7, zorder=0)


def apply_style():
    """Set the global matplotlib parameters of the previous figures.

    Returns
    -------
    None
        Updates ``matplotlib.rcParams`` in place (font family and size, vector
        friendly fonts in PDF output).
    """
    plt.rcParams.update({
        "font.family": FONT_NAME,
        "font.size": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "pdf.fonttype": 42,  # keep text editable in PDF
        "figure.dpi": 100,
    })


def set_labels(ax, xlabel=None, ylabel=None, title=None, fontsize=FONT_SIZE):
    """Write italic axis labels and title in the previous style.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    xlabel, ylabel, title : str, optional
        Texts; None leaves the current value unchanged.
    fontsize : float, optional

    Returns
    -------
    None
    """
    kw = dict(fontname=FONT_NAME, fontsize=fontsize, fontstyle="italic")
    if xlabel is not None:
        ax.set_xlabel(xlabel, **kw)
    if ylabel is not None:
        ax.set_ylabel(ylabel, **kw)
    if title is not None:
        ax.set_title(title, **kw)


def dotted_grid(ax, xs=None, ys=None):
    """Draw the dotted grid lines of the previous figures.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    xs, ys : array_like, optional
        Positions of the vertical / horizontal lines. Default: current ticks
        inside the axis limits.

    Returns
    -------
    None
        Axis limits are frozen so the lines span the full plot.
    """
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    xs = [x for x in (ax.get_xticks() if xs is None else xs) if xlim[0] <= x <= xlim[1]]
    ys = [y for y in (ax.get_yticks() if ys is None else ys) if ylim[0] <= y <= ylim[1]]
    ax.vlines(xs, *ylim, **GRID_KW)
    ax.hlines(ys, *xlim, **GRID_KW)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def strategy_handles(strategies=STRATEGY_ORDER, size=MARKER_SIZE / 10 + 3):
    """Legend handles for the stepping strategies (coloured dots, white edge).

    Parameters
    ----------
    strategies : list of str, optional
    size : float, optional
        Marker size (pt).

    Returns
    -------
    list of matplotlib.lines.Line2D
    """
    return [
        plt.Line2D([0], [0], lw=0, marker="o", markersize=size, markerfacecolor=PALETTE[s],
                   markeredgecolor="#FFFFFF", markeredgewidth=1)
        for s in strategies
    ]


def condition_handles(size=MARKER_SIZE / 10):
    """Legend handles for the sensory conditions (o = NI, s = SI).

    Parameters
    ----------
    size : float, optional
        Marker size (pt).

    Returns
    -------
    handles : list of matplotlib.lines.Line2D
    labels : list of str
    """
    handles = [
        plt.Line2D([0], [0], lw=0, marker=m, markersize=size, markerfacecolor="#FFFFFF",
                   markeredgecolor="k", markeredgewidth=1)
        for m in CONDITION_MARKERS.values()
    ]
    return handles, [f"{c} - condition" for c in CONDITION_MARKERS]


def scatter_strategies(ax, df, x, y, strategies=STRATEGY_ORDER, **kwargs):
    """Scatter plot coloured by strategy, marker shape by sensory condition.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    df : pandas.DataFrame
        Must contain the columns ``x``, ``y``, ``strategy`` and ``condition``.
    x, y : str
        Column names plotted on the horizontal / vertical axes.
    strategies : list of str, optional
        Strategies to plot (drawing order).
    **kwargs
        Extra arguments for ``Axes.scatter`` (override the defaults).

    Returns
    -------
    None
    """
    style = dict(s=MARKER_SIZE, edgecolors="k", linewidths=1, alpha=MARKER_ALPHA, zorder=3)
    style.update(kwargs)
    for strategy in strategies:
        for condition, marker in CONDITION_MARKERS.items():
            sub = df[(df["strategy"] == strategy) & (df["condition"] == condition)]
            ax.scatter(sub[x], sub[y], c=PALETTE[strategy], marker=marker, **style)


def mean_feet_outlines(df, mirror=True):
    """Mean initial outlines of both feet relative to the stepping toe.

    Parameters
    ----------
    df : pandas.DataFrame
        First-step table rows, with the ``sf_*`` / ``nsf_*`` polygon columns,
        ``stepping_foot`` and ``height``.
    mirror : bool, optional
        If True, x is multiplied by :func:`step_analysis.methods.geometry.lateral_sign` so that +x is
        the lateral side of the stepping leg (as in ``ml_lateral_pct``).

    Returns
    -------
    dict of str -> numpy.ndarray
        ``{"sf": (5, 2), "nsf": (5, 2)}`` closed polygons (TOE, TARI, HEE, TAR,
        TOE) in % body height: mean over trials of each vertex position.
    """
    sign = np.where(df["stepping_foot"] == "R", 1.0, -1.0) if mirror else np.ones(len(df))
    out = {}
    for prefix in ("sf", "nsf"):
        pts = []
        for marker in config.FOOT_MARKERS:
            px = 100 * sign * df[f"{prefix}_{marker}_x"] / df["height"]
            py = 100 * df[f"{prefix}_{marker}_y"] / df["height"]
            pts.append([px.mean(), py.mean()])
        pts.append(pts[0])
        out[prefix] = np.array(pts)
    return out


def symmetric_limits(values, step=5.0, margin=2.0):
    """Axis limits rounded to a tick step, covering all values with a margin.

    Parameters
    ----------
    values : array_like
        Plotted values.
    step : float, optional
        Rounding step (same unit as values).
    margin : float, optional
        Margin added before rounding.

    Returns
    -------
    (float, float)
        Lower and upper limits.
    """
    v = np.asarray(values, dtype=float)
    lo = np.floor((np.nanmin(v) - margin) / step) * step
    hi = np.ceil((np.nanmax(v) + margin) / step) * step
    return lo, hi
