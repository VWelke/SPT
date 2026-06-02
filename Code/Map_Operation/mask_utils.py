"""
mask_utils.py
-------------
Sky-mask utilities for SPT-3G analysis.

Typical workflow
----------------
    from mask_utils import obs_mask_from_weights, apodise_mask, load_mask

    # Build from weight maps (always available):
    obs  = obs_mask_from_weights(TT, QQ, UU)
    apod = apodise_mask(obs, apod_fwhm_deg=1.0)

    # Or load a pre-computed mask from file (cluster):
    apod = load_mask("/path/to/apod_mask.fits", nside_out=2048)
"""

import numpy as np
# healpy is imported lazily inside functions that need it so that
# obs_mask_from_weights (pure numpy) remains importable on systems
# where healpy is not installed (e.g. local Windows development).


def obs_mask_from_weights(TT, QQ=None, UU=None, threshold=0.0):
    """
    Build a boolean observed-pixel mask from weight-map diagonals.

    A pixel is considered observed when all supplied weight components
    exceed *threshold*.

    Parameters
    ----------
    TT : array-like, shape (npix,)
        Temperature weight (required).
    QQ, UU : array-like or None
        Polarisation weight diagonals (recommended for pol analysis).
    threshold : float
        Minimum weight to include a pixel.  Default 0 keeps every
        pixel with any signal.

    Returns
    -------
    mask : bool ndarray, shape (npix,)
    """
    mask = np.asarray(TT, float) > threshold
    if QQ is not None:
        mask &= np.asarray(QQ, float) > threshold
    if UU is not None:
        mask &= np.asarray(UU, float) > threshold
    return mask


def apodise_mask(obs_mask, apod_fwhm_deg=1.0):
    """
    Create a smooth apodisation window from a binary sky mask.

    Gaussian-smooths the binary mask and clips to [0, 1].  The smooth
    transition at the boundary reduces spectral leakage in harmonic
    transforms.

    This is adequate for diagnostics.  For production BB analysis use
    NaMaster's C² apodisation (``nmt.mask_apodization``).

    Parameters
    ----------
    obs_mask : array-like of bool/float, shape (npix,)
        Binary sky mask (1 = observed).
    apod_fwhm_deg : float
        FWHM in degrees of the Gaussian taper.  Typical SPT value: 1–2°.

    Returns
    -------
    apod : float ndarray, shape (npix,), values in [0, 1]
    """
    import healpy as hp
    fwhm_rad = np.radians(apod_fwhm_deg)
    apod = hp.smoothing(np.asarray(obs_mask, float), fwhm=fwhm_rad)
    return np.clip(apod, 0.0, 1.0)


def load_mask(mask_file, nside_out=None):
    """
    Load a pre-computed apodisation mask from a FITS or NumPy file.

    Parameters
    ----------
    mask_file : str
        Path to the mask.  Supported formats:
        - HEALPix FITS map (``*.fits``, ``*.fits.gz``)
        - NumPy binary array (``*.npy``)
    nside_out : int or None
        If given, ud_grade the loaded mask to this nside.

    Returns
    -------
    mask : float ndarray, shape (npix,)
    """
    import healpy as hp

    if mask_file.endswith(".npy"):
        mask = np.load(mask_file).astype(float)
    else:
        mask = hp.read_map(mask_file, dtype=float)

    if nside_out is not None and hp.get_nside(mask) != nside_out:
        mask = hp.ud_grade(mask, nside_out)

    return mask
