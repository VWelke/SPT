"""
Plot.py  —  SPT-3G map plotting helpers.

Usage:
    from Plot import MapPlotter

    plotter = MapPlotter(frames)   # frames: dict  map_id → {T, Q, U, TT, QQ, UU}

    plotter.gnomview("Left90GHz", map_key="T", rot=[0, -44.75])
    fig, ax = plotter.imshow("Left90GHz", map_key="T", rot=[0, -44.75], label_text="90 GHz")

    layout = [("Left90GHz","Right90GHz"), ("Left150GHz","Right150GHz"), ("Left220GHz","Right220GHz")]
    plotter.plot_grid(map_key="T", layout=layout, rot=[0, -44.75], suptitle="T maps")
"""

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

_WEIGHT = {"T": "TT", "Q": "QQ", "U": "UU"}


class MapPlotter:
    """Gnomonic plotting helper for SPT-3G HEALPix maps."""

    def __init__(self, frames, cmap="coolwarm", percentile=99):
        self.frames     = frames
        self.cmap       = cmap
        self.percentile = percentile

    # ------------------------------------------------------------------

    def _prep(self, map_id, map_key):
        """Return (masked_map, vlim) for frames[map_id][map_key]."""
        d   = self.frames[map_id]
        arr = np.asarray(d[map_key], float)
        w   = np.asarray(d[_WEIGHT[map_key]], float) if map_key in _WEIGHT else None

        m   = arr.copy()
        obs = (w > 0) if w is not None else (m != hp.UNSEEN)
        m[~obs] = hp.UNSEEN
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
        img = proj.projmap(m, vec2pix_func=lambda x, y, z: hp.vec2pix(hp.get_nside(m), x, y, z))
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
        """Grid of gnomview panels from frames.

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
