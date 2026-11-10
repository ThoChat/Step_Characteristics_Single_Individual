"""Figures of the analysis and of the test report, in the style of the previous work."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .. import config
from . import style
from .style import FONT_SIZE, PALETTE, STRATEGY_ORDER

DIRECTION_TEXT_KW = dict(fontsize=FONT_SIZE - 6, fontstyle="italic", color="0.3", zorder=6,
                         bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1))


def _direction_labels(ax, left, right, top="Anterior", bottom="Posterior"):
    """Write the anatomical directions at the edges of a decomposition plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    left, right : str
        Meaning of the negative / positive ML direction.
    top, bottom : str, optional
        Meaning of the positive / negative AP direction.

    Returns
    -------
    None
    """
    ax.text(0.015, 0.015, f"← {left}", transform=ax.transAxes, ha="left", va="bottom", **DIRECTION_TEXT_KW)
    ax.text(0.985, 0.015, f"{right} →", transform=ax.transAxes, ha="right", va="bottom", **DIRECTION_TEXT_KW)
    ax.text(0.5, 0.985, f"↑ {top}", transform=ax.transAxes, ha="center", va="top", **DIRECTION_TEXT_KW)
    ax.text(0.5, 0.015, f"↓ {bottom}", transform=ax.transAxes, ha="center", va="bottom", **DIRECTION_TEXT_KW)


def plot_decomposition(ax, df, frame="lateral", strategies=STRATEGY_ORDER, xlim=None, ylim=None,
                       tick_step=10, show_feet=True, show_medians=True):
    """Scatter of the first-step end positions decomposed on the ML / AP axes.

    Each point is the stepping toe position at t_end relative to its position
    at t_begin (origin of the step vector), in % of body height.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    df : pandas.DataFrame
        Selected rows of the first-step table (``ml_pct``, ``ml_lateral_pct``,
        ``ap_pct``, ``strategy``, ``condition``; polygon columns for the feet).
    frame : {"lateral", "anatomical"}, optional
        ``"anatomical"``: x = ``ml_pct`` (+ participant's right).
        ``"lateral"``: x = ``ml_lateral_pct`` (+ lateral side of the stepping
        leg, - towards the other foot).
    strategies : list of str, optional
        Strategies plotted (and their drawing order).
    xlim, ylim : (float, float), optional
        Axis limits (% body height). Default: fitted to the data.
    tick_step : float, optional
        Tick and grid spacing (% body height).
    show_feet : bool, optional
        In the lateral frame, draw the mean initial outline of the stepping
        foot (dashed) and of the non-stepping foot (grey).
    show_medians : bool, optional
        Draw the median end position of each strategy (large diamond) with a
        direct label.

    Returns
    -------
    None
    """
    x = "ml_lateral_pct" if frame == "lateral" else "ml_pct"
    y = "ap_pct"
    sub = df[df["strategy"].isin(strategies)]
    xlim = xlim or style.symmetric_limits(sub[x], step=tick_step)
    ylim = ylim or style.symmetric_limits(sub[y], step=tick_step)

    if show_feet and frame == "lateral":
        feet = style.mean_feet_outlines(sub)
        ax.fill(*feet["nsf"].T, facecolor="0.88", edgecolor="0.5", lw=1.5, zorder=1)
        ax.plot(*feet["sf"].T, color="0.35", lw=1.5, ls="--", zorder=1)

    ax.axhline(0, color="k", lw=1, alpha=0.6, zorder=1)
    ax.axvline(0, color="k", lw=1, alpha=0.6, zorder=1)
    style.scatter_strategies(ax, sub, x, y, strategies)

    if show_medians:
        for s in strategies:
            med = sub.loc[sub["strategy"] == s, [x, y]].median()
            ax.scatter(med[x], med[y], marker="D", s=260, c=PALETTE[s], edgecolors="k", linewidths=2.5, zorder=7)
            ax.annotate(s, (med[x], med[y]), xytext=(12, 10), textcoords="offset points", fontsize=FONT_SIZE - 4,
                        fontweight="bold", zorder=8,
                        bbox=dict(facecolor="white", edgecolor="0.3", alpha=0.85, boxstyle="round,pad=0.2"))

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(xlim[0], xlim[1] + 1e-9, tick_step))
    ax.set_yticks(np.arange(ylim[0], ylim[1] + 1e-9, tick_step))
    ax.tick_params(labelsize=FONT_SIZE - 2)
    style.dotted_grid(ax)
    left, right = ("Medial", "Lateral") if frame == "lateral" else ("Left", "Right")
    _direction_labels(ax, left, right)
    style.set_labels(ax, xlabel="Mediolateral (% body height)", ylabel="Anteroposterior (% body height)")


def decomposition_legends(ax, df, strategies=STRATEGY_ORDER, loc=(1.03, 0.45), show_feet=True):
    """Legends of the decomposition figure, placed outside the axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes the legends are attached to (usually the right-most panel).
    df : pandas.DataFrame
        Plotted rows, used for the counts per strategy.
    strategies : list of str, optional
    loc : (float, float), optional
        Position (axes coordinates) of the strategy legend's lower-left corner.
    show_feet : bool, optional
        Add the entries of the feet outlines.

    Returns
    -------
    None
    """
    counts = df["strategy"].value_counts()
    leg1 = ax.legend(style.strategy_handles(strategies), [f"{s} (n={counts.get(s, 0)})" for s in strategies],
                     title="Stepping Strategies:", title_fontsize=FONT_SIZE - 6, fontsize=FONT_SIZE - 4,
                     loc=loc, frameon=True)
    ax.add_artist(leg1)
    handles, labels = style.condition_handles()
    handles.append(plt.Line2D([0], [0], lw=0, marker="D", markersize=12, markerfacecolor="white",
                              markeredgecolor="k", markeredgewidth=2.5))
    labels.append("Median")
    if show_feet:
        handles += [plt.Line2D([0], [0], color="0.35", lw=1.5, ls="--"),
                    plt.Rectangle((0, 0), 1, 1, facecolor="0.88", edgecolor="0.5")]
        labels += ["Stepping foot at t_begin (mean)", "Other foot at t_begin (mean)"]
    ax.legend(handles, labels, fontsize=FONT_SIZE - 6, loc=(loc[0], loc[1] - 0.5), frameon=True)


def plot_component_boxes(ax, df, column, ylabel, strategies=STRATEGY_ORDER):
    """Box plot + strip plot of one step component per strategy (previous DtFb figure style).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    df : pandas.DataFrame
        Selected rows of the first-step table.
    column : str
        Column plotted (e.g. ``"ml_lateral_pct"``).
    ylabel : str
        Y axis label.
    strategies : list of str, optional
        Order of the boxes.

    Returns
    -------
    None
    """
    sub = df[df["strategy"].isin(strategies)]
    sns.boxplot(data=sub, x="strategy", y=column, order=strategies, hue="strategy", hue_order=strategies,
                palette=style.BOX_PALETTE, fliersize=0, width=0.5, linewidth=2.5, saturation=1,
                legend=False, ax=ax)
    for condition, marker in style.CONDITION_MARKERS.items():
        sns.stripplot(data=sub[sub["condition"] == condition], x="strategy", y=column, order=strategies,
                      hue="strategy", hue_order=strategies, palette=PALETTE, marker=marker, size=7,
                      jitter=0.22, alpha=style.MARKER_ALPHA, edgecolor="k", linewidth=1, legend=False, ax=ax)
    ax.axhline(0, linestyle=":", alpha=0.6, color="k", lw=1, zorder=0)
    ylim = ax.get_ylim()
    ax.vlines(np.arange(len(strategies) - 1) + 0.5, *ylim, **style.GRID_KW)
    ax.set_ylim(ylim)
    ax.tick_params(labelsize=FONT_SIZE - 2)
    style.set_labels(ax, xlabel="", ylabel=ylabel)


# ---------------------------------------------------------------------------
# Test-report figures
# ---------------------------------------------------------------------------
AGREEMENT_COLORS = ["0.25", "0.65", "0.85"]


def plot_agreement(ax, labels, series, xlabel="Agreement (% of compared trials)", annotations=None):
    """Horizontal grouped bars of agreement percentages.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    labels : list of str
        One label per compared quantity (bar group), drawn top to bottom.
    series : dict of str -> array_like
        ``{series name: agreement in % per quantity}`` (one to three series).
    xlabel : str, optional
    annotations : dict of str -> list of str, optional
        Text written at the end of each bar of a series (e.g. "640/640").
        Default: the percentage with one decimal.

    Returns
    -------
    None
        A legend is drawn when there is more than one series.
    """
    n, k = len(labels), len(series)
    height = 0.8 / k
    y = np.arange(n)[::-1]
    for j, (name, values) in enumerate(series.items()):
        values = np.asarray(values, dtype=float)
        pos = y + 0.4 - height * (j + 0.5)
        ax.barh(pos, values, height=height * 0.9, color=AGREEMENT_COLORS[j], label=name)
        texts = (annotations or {}).get(name) or [f"{v:.1f} %" for v in values]
        for p, v, t in zip(pos, values, texts):
            ax.text(min(v, 100) + 0.5, p, t, va="center", fontsize=FONT_SIZE - 9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=FONT_SIZE - 7)
    ax.set_xlim(0, 118)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.axvline(100, color="k", lw=1, ls=":")
    ax.tick_params(axis="x", labelsize=FONT_SIZE - 6)
    style.set_labels(ax, xlabel=xlabel, fontsize=FONT_SIZE - 4)
    if k > 1:  # outside the axes: the bars span the full width
        ax.legend(fontsize=FONT_SIZE - 8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
def plot_step_detection(ax, trial, diagnostics, steps, stepping_feet, perturbation=None, dt=config.DT):
    """Step detection signals of one foot (manuscript Fig. 2.9, single axis).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    trial : dict
        Trial time series (``ForceSensor`` used if present).
    diagnostics : dict
        Third output of :func:`step_analysis.methods.kinematics.step_detector` with
        ``return_diagnostics=True``.
    steps, stepping_feet : list
        First two outputs of :func:`step_analysis.methods.kinematics.step_detector`.
    perturbation : (int, int), optional
        Push start and end frames, drawn as a shaded band.
    dt : float, optional
        Sampling period (s).

    Returns
    -------
    None
        All signals are normalised by their maximum (% of max value), so they
        share one axis; t_begin / t_end of every detected step of the first
        stepping foot are drawn as vertical lines.
    """
    foot = stepping_feet[0]
    diag = diagnostics[foot]
    t = np.arange(len(diag["dist"])) * dt
    if "ForceSensor" in trial:
        force = trial["ForceSensor"][: len(t)]
        ax.plot(t, 100 * force / np.max(force), color="0.5", ls="--", lw=2, label="Force F(t)")
    if diag["vtrans"] is not None:
        ax.plot(t, 100 * diag["vtrans"] / np.max(diag["vtrans"]), color="k", lw=2, label=f"{foot}TOE transversal speed")
    ax.plot(t, 100 * diag["dist"] / np.max(diag["dist"]), color="#D03939", lw=2, label=f"{foot}TOE distance")
    if perturbation is not None:
        ax.axvspan(perturbation[0] * dt, perturbation[1] * dt, color="0.9", zorder=0, label="Push")
    for (b, e), f in zip(steps, stepping_feet):
        if f != foot:
            continue
        ax.axvline(b * dt, color="#54df1f", lw=2.5)
        ax.axvline(e * dt, color="#9D55D6", lw=2.5)
    ax.plot([], [], color="#54df1f", lw=2.5, label="t_begin")
    ax.plot([], [], color="#9D55D6", lw=2.5, label="t_end")
    ax.set_ylim(-5, 105)
    ax.tick_params(labelsize=FONT_SIZE - 4)
    style.set_labels(ax, xlabel="Time (s)", ylabel="% of max value", fontsize=FONT_SIZE - 2)


def plot_top_view(ax, trial, stepping_foot, begin, end, result=None, title=None):
    """Top view of a first step: initial feet, front boundary, CoM path, step vector.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    trial : dict
        Trial time series.
    stepping_foot : {"R", "L"}
    begin, end : int
        First step t_begin and t_end frames.
    result : classification.UnifiedResult, optional
        If given, the frame that triggered the label is marked on the CoM path.
    title : str, optional

    Returns
    -------
    None
        Lab axes in m; feet polygons at t_begin (stepping foot dashed), feet
        at t_end in light grey, CoM ground projection over the whole trial.
    """
    from ..methods.geometry import foot_polygon

    other = "L" if stepping_foot == "R" else "R"
    for side, ls in ((stepping_foot, "--"), (other, "-")):
        poly = np.vstack([foot_polygon(trial, side, begin), foot_polygon(trial, side, begin)[:1]])
        ax.plot(*poly.T, color="k", ls=ls, lw=1.8, zorder=3)
        poly_end = np.vstack([foot_polygon(trial, side, end), foot_polygon(trial, side, end)[:1]])
        ax.fill(*poly_end.T, color="0.85", zorder=1)
    toes = np.array([trial[stepping_foot + "TOE"][:2, begin], trial[other + "TOE"][:2, begin]])
    ax.plot(*toes.T, color="#D03939", lw=2, ls=":", zorder=3, label="Front boundary (toes)")
    com = trial["COM"][:2]
    ax.plot(*com, color="0.55", lw=1.5, zorder=2, label="CoM path")
    ax.plot(*com[:, begin:], color="#9D55D6", lw=2, zorder=2, label="CoM after t_begin")
    toe = trial[stepping_foot + "TOE"][:2]
    ax.annotate("", xy=toe[:, end], xytext=toe[:, begin], zorder=5,
                arrowprops=dict(arrowstyle="-|>", color="k", lw=2.5, mutation_scale=18))
    if result is not None:
        frame = result.front_crossing_frame if result.front_crossing_frame is not None else result.polygon_hit_frame
        if frame is not None:
            ax.scatter(*com[:, frame], s=120, marker="*", c="gold", edgecolors="k", zorder=6, label="Decisive frame")
    ax.set_aspect("equal")
    ax.tick_params(labelsize=FONT_SIZE - 8)
    style.set_labels(ax, xlabel="x (m)", ylabel="y (m)", fontsize=FONT_SIZE - 4)
    if title is not None:
        style.set_labels(ax, title=title, fontsize=FONT_SIZE - 6)
