"""
eb_pipeline.py
--------------
Q/U → E/B diagnostic pipeline for SPT-3G.

WARNING
-------
This pipeline uses ``healpy.map2alm_lsq`` on a partial sky and does NOT
correct for E→B leakage.  It is a quick inspection / sanity-check tool.
For the final BB power spectrum use a proper pseudo-Cl estimator
(e.g. NaMaster) that handles E-B mixing analytically.

Main entry points
-----------------
    from eb_pipeline import maps_to_eb, frame_to_eb

    result = frame_to_eb(frames["Left90GHz"], apod_mask, lmax=512)
    E_map, B_map = result["E_map"], result["B_map"]
"""

import numpy as np
import healpy as hp

from Initial_test.Map_Operation.mask_utils import obs_mask_from_weights


def maps_to_eb(T, Q, U, TT, QQ, UU, apod_mask,
               lmax=512, tol=1e-8, maxiter=20):
    """
    Full Q/U → E/B diagnostic pipeline.

    Steps
    -----
    1. Build observed-pixel boolean mask from weight thresholds.
    2. Intersect with apodisation support (apod > 0).
    3. Apply apodisation window to T, Q, U (in-place on copies).
    4. Solve TQU → TEB alms via ``healpy.map2alm_lsq``
       (iterative least-squares; handles partial sky).
    5. Reconstruct E/B maps via ``healpy.alm2map``.

    Parameters
    ----------
    T, Q, U : array-like, shape (npix,)
        Stokes maps in T_CMB units.
    TT, QQ, UU : array-like, shape (npix,)
        Weight-map diagonals from Wpol.
    apod_mask : array-like, shape (npix,), values in [0, 1]
        Smooth apodisation window (e.g. from ``apodise_mask()``).
    lmax : int
        Maximum multipole for the harmonic transform.
    tol : float
        Convergence tolerance for map2alm_lsq.
    maxiter : int
        Maximum iterations for map2alm_lsq.

    Returns
    -------
    result : dict
        E_map     : float ndarray  — E map (hp.UNSEEN outside obs)
        B_map     : float ndarray  — B map (hp.UNSEEN outside obs)
        almE      : complex ndarray — E spherical harmonic coefficients
        almB      : complex ndarray — B spherical harmonic coefficients
        obs_mask  : bool ndarray   — combined observed-pixel mask used
        rel_res   : float          — relative residual from map2alm_lsq
    """
    T    = np.asarray(T,         float).copy()
    Q    = np.asarray(Q,         float).copy()
    U    = np.asarray(U,         float).copy()
    apod = np.asarray(apod_mask, float)
    nside = hp.get_nside(T)

    # --- observed mask: weight > 0  AND  apod support  AND  finite values ---
    obs  = obs_mask_from_weights(TT, QQ, UU)
    obs &= np.isfinite(T) & np.isfinite(Q) & np.isfinite(U)
    obs &= apod > 0

    if obs.sum() == 0:
        raise ValueError("obs_mask is empty — check weight maps and apod_mask.")

    # --- apply apodisation; zero outside mask for the harmonic transform ---
    T[obs] *= apod[obs];   T[~obs] = 0.0
    Q[obs] *= apod[obs];   Q[~obs] = 0.0
    U[obs] *= apod[obs];   U[~obs] = 0.0

    # --- TQU → TEB alms (least-squares, partial sky) ---
    alms, _res_maps, rel_res = hp.sphtfunc.map2alm_lsq(
        [T, Q, U], lmax=lmax, pol=True, tol=tol, maxiter=maxiter,
    )
    almT, almE, almB = alms

    # --- alms → E/B maps ---
    _, E_map, B_map = hp.alm2map(
        [almT, almE, almB], nside=nside, lmax=lmax, pol=True,
    )

    # mask unobserved pixels so plotter / gnomview can render them correctly
    E_map[~obs] = hp.UNSEEN
    B_map[~obs] = hp.UNSEEN

    return {
        "E_map":    E_map,
        "B_map":    B_map,
        "almE":     almE,
        "almB":     almB,
        "obs_mask": obs,
        "rel_res":  float(rel_res) if np.ndim(rel_res) == 0 else float(rel_res[-1]),
    }


def frame_to_eb(frame, apod_mask, lmax=512, **kwargs):
    """
    Convenience wrapper: run ``maps_to_eb`` on a frame dict.

    Parameters
    ----------
    frame : dict with keys T, Q, U, TT, QQ, UU
        As produced by the frame-loading cell in the notebook.
    apod_mask : array-like
        Apodisation window.
    **kwargs
        Forwarded to ``maps_to_eb`` (lmax, tol, maxiter, …).

    Returns
    -------
    Same dict as ``maps_to_eb``.
    """
    return maps_to_eb(
        frame["T"], frame["Q"], frame["U"],
        frame["TT"], frame["QQ"], frame["UU"],
        apod_mask, lmax=lmax, **kwargs,
    )
