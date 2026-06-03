"""
Plot.py  —  SPT-3G map plotting helpers.

Standalone functions (from quan26 example notebook style):
    from Plot import apply_spt_style
    from Plot import show_map_full_field, show_map_thumbnail
    from Plot import show_1d_functions_of_ell, show_alm_triangle

    apply_spt_style()                         # call once at top of notebook

    show_map_full_field(m, vmin, vmax, title="T", unit="Tcmb")
    show_map_thumbnail( m, vmin, vmax, title="T", unit="Tcmb")

MapPlotter class (multi-panel grid helper):
    from Plot import MapPlotter

    plotter = MapPlotter(frames)   # frames: dict  map_id → {T, Q, U, TT, QQ, UU}
    layout  = [("Left90GHz","Right90GHz"), ("Left150GHz","Right150GHz"), ...]
    plotter.plot_grid(map_key="T", layout=layout, rot=[0, -44.75])

Helper:
    from Plot import mask_map
    m = mask_map(arr, weight)    # set unobserved pixels to hp.UNSEEN
"""

import warnings
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import matplotlib

# ── rcParams ──────────────────────────────────────────────────────────────────

SPT_RCPARAMS = {
    "font.size":           14,
    "font.family":         "DeJavu Serif",
    "font.serif":          ["Times New Roman"],
    "mathtext.fontset":    "dejavuserif",
    "legend.fontsize":     14,
    "axes.grid":           True,
    "axes.grid.which":     "both",
    "grid.linestyle":      "dotted",
    "grid.linewidth":      0.35,
    "xtick.direction":     "in",
    "ytick.direction":     "in",
    "axes.labelsize":      14,
    "axes.labelpad":       6,
    "axes.titlesize":      14,
    "axes.titlepad":       10,
    "xtick.labelsize":     14,
    "ytick.labelsize":     14,
}


def apply_spt_style():
    """Apply the SPT-3G standard matplotlib rcParams (call once per notebook)."""
    matplotlib.rcParams.update(SPT_RCPARAMS)


# ── map preparation helper ────────────────────────────────────────────────────

def mask_map(arr, weight=None, threshold=0.0):
    """
    Return a copy of `arr` with unobserved pixels set to hp.UNSEEN.

    Parameters
    ----------
    arr    : array-like, shape (npix,)   — Stokes map (T, Q, or U)
    weight : array-like or None          — corresponding weight map (TT/QQ/UU)
    threshold : float                    — pixels with weight ≤ threshold → UNSEEN

    Returns
    -------
    m : float ndarray, shape (npix,)
    """
    m = np.asarray(arr, float).copy()
    if weight is not None:
        obs = np.asarray(weight, float) > threshold
    else:
        obs = np.isfinite(m) & (m != hp.UNSEEN)
    m[~obs] = hp.UNSEEN
    return m


# ── standalone plot functions (quan26 style) ──────────────────────────────────

def show_map_full_field(
        m, vmin, vmax, title="", unit="",
        cmap="gray", badcolor="white"):
    """
    Full SPT-3G main-field view using healpy.azeqview.

    Shows the entire 4-subfield footprint at once.
    Rotation is fixed to centre the SPT-3G main field.

    Parameters
    ----------
    m         : array-like, shape (npix,) — HEALPix map (use mask_map first)
    vmin/vmax : float  — colour scale limits
    title     : str
    unit      : str    — colourbar label
    cmap      : str    — matplotlib colourmap name
    badcolor  : str    — colour for hp.UNSEEN pixels
    """
    plt.figure(figsize=(13, 7), num=0, facecolor="white")
    hp.azeqview(
        m,
        rot=(0, -59.5, 0),
        xsize=1300, ysize=700, reso=3.5, fig=0,
        half_sky=True, lamb=True,
        cmap=cmap, min=vmin, max=vmax,
        badcolor=badcolor,
        title=title, unit=unit)
    plt.show()
    plt.close()


def show_map_thumbnail(
        m, vmin, vmax, title="", unit="",
        cmap="gray", badcolor="white",
        rot=(32, -51, 0), xsize=800, ysize=800, reso=0.2):
    """
    Zoomed-in thumbnail of a patch using healpy.gnomview.

    Default rotation centres on a representative SPT-3G subfield patch.
    Change `rot` to zoom into a different part of the field.

    Parameters
    ----------
    m         : array-like, shape (npix,) — HEALPix map (use mask_map first)
    vmin/vmax : float  — colour scale limits
    title     : str
    unit      : str    — colourbar label
    rot       : tuple  — (lon, lat, psi) in degrees
    """
    plt.figure(figsize=(8, 8), num=0, facecolor="white")
    hp.gnomview(
        m,
        rot=rot,
        xsize=xsize, ysize=ysize, reso=reso, fig=0,
        cmap=cmap, min=vmin, max=vmax,
        badcolor=badcolor,
        title=title, unit=unit)
    plt.show()
    plt.close()


def show_1d_functions_of_ell(
        ell, functions, labels, xlims, ylims, ylabel,
        yscale="linear", legend_loc="upper right", legend_ncols=1,
        vlines=[]):
    """
    Plot one or more functions of multipole ell on the same axes.

    Parameters
    ----------
    ell       : array-like           — multipole values
    functions : list of array-like   — y-values for each curve
    labels    : list of str          — legend labels (use [""] to suppress legend)
    xlims     : (xmin, xmax)
    ylims     : (ymin, ymax)
    ylabel    : str
    yscale    : "linear" or "log"
    vlines    : list of float        — vertical dotted lines to draw
    """
    plt.figure(figsize=(8, 5))
    for function, label in zip(functions, labels):
        plt.plot(ell, function, label=label, alpha=0.8)
    for vline in vlines:
        plt.axvline(vline, color="black", linestyle="dotted")
    plt.yscale(yscale)
    plt.xlim(left=xlims[0], right=xlims[1])
    plt.ylim(bottom=ylims[0], top=ylims[1])
    if not (len(labels) == 1 and labels[0] == ""):
        plt.legend(loc=legend_loc, ncol=legend_ncols)
    plt.xlabel(r"$\ell$")
    plt.ylabel(ylabel)
    plt.show()
    plt.close()


def show_alm_triangle(
        alm, lmax, real=True,
        vmin=None, vmax=None, cmap="Oranges_r",
        xlims=None, ylims=None,
        title="Triangle"):
    """
    Triangle plot of spherical harmonic coefficients a_lm.

    Displays a 2-D image with ell on the x-axis and m on the y-axis,
    useful for diagnosing mode-coupling or filter effects.

    Parameters
    ----------
    alm  : complex ndarray — as returned by healpy.map2alm
    lmax : int
    real : bool            — plot real part if True, abs if False
    """
    warnings.filterwarnings("ignore")

    triangle = np.empty((lmax + 1, lmax + 1))
    triangle[:, :] = np.nan
    for l in range(lmax + 1):
        for m in range(0, l + 1):
            i = hp.Alm.getidx(lmax, l, m)
            triangle[m, l] = alm[i].real if real else abs(alm[i])

    plt.figure(figsize=(7, 7))
    if vmin is None:
        vmin = np.nanmin(triangle)
    if vmax is None:
        vmax = np.nanmax(triangle)
    plt.imshow(triangle, origin="lower", vmin=vmin, vmax=vmax, cmap=cmap)
    if xlims is None:
        xlims = [0, triangle.shape[1]]
    if ylims is None:
        ylims = [0, triangle.shape[0]]
    plt.xlim(left=xlims[0], right=xlims[1])
    plt.ylim(bottom=ylims[0], top=ylims[1])
    plt.grid(False)
    plt.colorbar(pad=0.03, shrink=0.8)
    plt.grid(True)
    plt.xlabel(r"$\ell$")
    plt.ylabel(r"$m$")
    plt.title(title)
    plt.show()
    plt.close()


# ── MapPlotter class (multi-panel grid helper) ────────────────────────────────

_WEIGHT = {"T": "TT", "Q": "QQ", "U": "UU"}


class MapPlotter:
    """Multi-panel gnomonic grid helper for SPT-3G HEALPix maps."""

    def __init__(self, frames, cmap="coolwarm", percentile=99):
        self.frames     = frames
        self.cmap       = cmap
        self.percentile = percentile

    # ------------------------------------------------------------------

    def _prep(self, map_id, map_key):
        """Return (masked_map, vlim) for frames[map_id][map_key]."""
        d    = self.frames[map_id]
        arr  = np.asarray(d[map_key], float)
        w    = np.asarray(d[_WEIGHT[map_key]], float) if map_key in _WEIGHT else None
        m    = mask_map(arr, w)
        obs  = m != hp.UNSEEN
        vlim = float(np.nanpercentile(np.abs(arr[obs]), self.percentile))
        return m, vlim

    # ------------------------------------------------------------------

    def gnomview(self, map_id, map_key="T", rot=[0, -44.75],
                 xsize=3500, ysize=1200, reso=1.5,
                 unit="", notext=True, sub=None, **kw):
        """Single-panel healpy gnomview for one map in frames."""
        m, vlim = self._prep(map_id, map_key)
        hp.gnomview(m, rot=rot, xsize=xsize, ysize=ysize, reso=reso,
                    cmap=self.cmap, min=-vlim, max=vlim,
                    badcolor="white", bgcolor="white", coord="C",
                    notext=notext, title=map_id, unit=unit, sub=sub, **kw)

    # ------------------------------------------------------------------

    def imshow(self, map_id, map_key="T", rot=[0, -44.75],
               xsize=3500, ysize=1200, reso=1.5,
               label_text=None, unit=r"$T_{\rm CMB}$",
               xlabel="Right Ascension", figsize=(10, 4), ax=None, **kw):
        """Single-panel matplotlib imshow via GnomonicProj."""
        m, vlim = self._prep(map_id, map_key)

        proj = hp.projector.GnomonicProj(rot=rot, xsize=xsize, ysize=ysize,
                                         reso=reso, coord=None)
        img  = proj.projmap(m, vec2pix_func=lambda x, y, z:
                            hp.vec2pix(hp.get_nside(m), x, y, z))
        img[img == hp.UNSEEN] = np.nan

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        im = ax.imshow(img, origin="lower", cmap=self.cmap,
                       vmin=-vlim, vmax=vlim, aspect="auto", **kw)
        if label_text:
            ax.text(0.04, 0.88, label_text, transform=ax.transAxes,
                    fontsize=16, color="black")
        cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.025)
        cbar.set_label(unit)
        ax.set_xlabel(xlabel, fontsize=14)
        return fig, ax

    # ------------------------------------------------------------------

    def plot_grid(self, map_key, layout, rot, suptitle="",
                  xsize=3500, ysize=1200, reso=2.0,
                  figsize=(12, 12), hspace=0.08, wspace=0.05):
        """Grid of gnomview panels.

        layout : list of tuples of map IDs, e.g.
            [("Left90GHz","Right90GHz"), ("Left150GHz","Right150GHz"), ...]
        """
        nrows = len(layout)
        ncols = max(len(row) for row in layout)
        fig   = plt.figure(figsize=figsize, facecolor="white")

        for panel, mid in enumerate((m for row in layout for m in row), start=1):
            self.gnomview(mid, map_key=map_key, rot=rot, xsize=xsize,
                          ysize=ysize, reso=reso, sub=(nrows, ncols, panel))

        if suptitle:
            fig.suptitle(suptitle, fontsize=16, y=0.995)
        fig.subplots_adjust(top=0.94, hspace=hspace, wspace=wspace)
        return fig
