"""
test_eb_pipeline.py
-------------------
Unit tests for mask_utils and eb_pipeline.

Tests are split into two groups:
  - "numpy-only" tests  — run locally without healpy (obs_mask logic)
  - "healpy" tests      — require healpy; skipped locally, run on cluster

Run with:
    python -m pytest test_eb_pipeline.py -v
or standalone:
    python test_eb_pipeline.py
"""

import numpy as np
import pytest

# ── optional healpy import ────────────────────────────────────────────────────
try:
    import healpy as hp
    HAS_HEALPY = True
except ImportError:
    hp = None
    HAS_HEALPY = False

needs_healpy = pytest.mark.skipif(
    not HAS_HEALPY,
    reason="healpy not installed (Linux/cluster only)"
)

# ── always importable (lazy healpy in mask_utils) ─────────────────────────────
from Initial_test.Map_Operation.mask_utils import obs_mask_from_weights

# ── conditional imports ───────────────────────────────────────────────────────
if HAS_HEALPY:
    from Initial_test.Map_Operation.mask_utils  import apodise_mask
    from Initial_test.Map_Operation.eb_pipeline import maps_to_eb, frame_to_eb


# ── constants ─────────────────────────────────────────────────────────────────
NSIDE = 32   # ~1.8° pixels; fast for tests
LMAX  = 64
NPIX  = 12 * NSIDE ** 2


# ── healpy-free helpers ───────────────────────────────────────────────────────

def _fake_npix(nside):
    return 12 * nside ** 2


# ── healpy helpers (only defined when available) ──────────────────────────────

def _make_pure_E_tqu(seed=42):
    """Synthetic T/Q/U with pure-E signal (B = 0 exactly)."""
    rng  = np.random.default_rng(seed)
    nalm = hp.Alm.getsize(LMAX)
    almT = np.zeros(nalm, dtype=complex)
    almE = (rng.standard_normal(nalm) + 1j * rng.standard_normal(nalm)) / np.sqrt(2)
    almB = np.zeros(nalm, dtype=complex)
    T, Q, U = hp.alm2map([almT, almE, almB], nside=NSIDE, lmax=LMAX, pol=True)
    return T, Q, U, almE


def _circular_mask(lon_deg=0.0, lat_deg=-45.0, radius_deg=20.0):
    """Boolean cap mask mimicking one SPT subfield."""
    theta0 = np.radians(90.0 - lat_deg)
    vec0   = hp.ang2vec(theta0, np.radians(lon_deg))
    ipix   = hp.query_disc(NSIDE, vec0, np.radians(radius_deg))
    mask   = np.zeros(NPIX, dtype=bool);  mask[ipix] = True
    return mask


def _full_weights():
    return np.ones(NPIX), np.ones(NPIX), np.ones(NPIX)


def _cap_weights(cap):
    w = cap.astype(float)
    return w, w, w


# ══ numpy-only tests (always run) ════════════════════════════════════════════

def test_obs_mask_T_only():
    """Pixels with TT > 0 should be True; rest False."""
    TT       = np.zeros(NPIX)
    TT[:100] = 1.0
    mask     = obs_mask_from_weights(TT)
    assert mask.sum() == 100
    assert mask[:100].all()
    assert not mask[100:].any()


def test_obs_mask_requires_all_components():
    """Mask must be the AND of all weight conditions."""
    TT = np.ones(NPIX)
    QQ = np.ones(NPIX);  QQ[50:]  = 0.0
    UU = np.ones(NPIX);  UU[:80]  = 1.0;  UU[80:] = 0.0
    mask = obs_mask_from_weights(TT, QQ, UU)
    assert mask[:50].all()
    assert not mask[50:].any()


def test_obs_mask_threshold():
    """threshold parameter should exclude low-weight pixels."""
    TT       = np.ones(NPIX)
    TT[:200] = 0.5
    mask     = obs_mask_from_weights(TT, threshold=0.9)
    assert not mask[:200].any()
    assert mask[200:].all()


def test_obs_mask_no_polarisation():
    """Passing only TT (no QQ/UU) should work fine."""
    TT   = np.zeros(NPIX);  TT[10:20] = 2.0
    mask = obs_mask_from_weights(TT)
    assert mask.sum() == 10


# ══ healpy tests (skip locally, run on cluster) ═══════════════════════════════

@needs_healpy
def test_apodise_range():
    """Apodised values must lie in [0, 1]."""
    obs  = _circular_mask()
    apod = apodise_mask(obs, apod_fwhm_deg=3.0)
    assert apod.min() >= -1e-10
    assert apod.max() <=  1.0 + 1e-10


@needs_healpy
def test_apodise_interior():
    """Pixels deep inside the cap should be close to 1."""
    obs   = _circular_mask(radius_deg=25.0)
    apod  = apodise_mask(obs, apod_fwhm_deg=2.0)
    inner = _circular_mask(radius_deg=10.0)
    assert apod[inner].mean() > 0.9


@needs_healpy
def test_apodise_smooth_boundary():
    """There should be a smooth taper (values strictly between 0 and 1)."""
    obs        = _circular_mask()
    apod       = apodise_mask(obs, apod_fwhm_deg=3.0)
    transition = (apod > 0.05) & (apod < 0.95)
    assert transition.sum() > 0


@needs_healpy
def test_output_keys():
    """Result dict must contain all expected keys."""
    T, Q, U, _ = _make_pure_E_tqu()
    TT, QQ, UU = _full_weights()
    result      = maps_to_eb(T, Q, U, TT, QQ, UU, np.ones(NPIX), lmax=LMAX)
    for key in ("E_map", "B_map", "almE", "almB", "obs_mask", "rel_res"):
        assert key in result, f"missing key: {key}"


@needs_healpy
def test_output_shapes():
    """Output arrays must have the correct shapes."""
    T, Q, U, _ = _make_pure_E_tqu()
    TT, QQ, UU = _full_weights()
    result      = maps_to_eb(T, Q, U, TT, QQ, UU, np.ones(NPIX), lmax=LMAX)
    assert result["E_map"].shape    == (NPIX,)
    assert result["B_map"].shape    == (NPIX,)
    assert result["obs_mask"].shape == (NPIX,)
    assert result["almE"].shape     == (hp.Alm.getsize(LMAX),)


@needs_healpy
def test_empty_mask_raises():
    """An all-zero weight map should raise ValueError."""
    T, Q, U, _ = _make_pure_E_tqu()
    TT          = np.zeros(NPIX)    # no observed pixels
    with pytest.raises(ValueError, match="obs_mask is empty"):
        maps_to_eb(T, Q, U, TT, np.ones(NPIX), np.ones(NPIX), np.ones(NPIX), lmax=LMAX)


@needs_healpy
def test_fullsky_B_suppressed():
    """
    Full sky, pure-E input: B map should be negligible vs E map.
    On a full sky map2alm_lsq converges quickly and B ≈ 0.
    """
    T, Q, U, _ = _make_pure_E_tqu()
    TT, QQ, UU = _full_weights()
    result      = maps_to_eb(T, Q, U, TT, QQ, UU, np.ones(NPIX),
                             lmax=LMAX, maxiter=50)

    obs    = result["obs_mask"]
    E_rms  = np.sqrt(np.mean(result["E_map"][obs] ** 2))
    B_rms  = np.sqrt(np.mean(result["B_map"][obs] ** 2))

    assert E_rms > 0, "E_rms is zero — pipeline returned null"
    ratio  = B_rms / E_rms
    assert ratio < 0.05, f"B/E = {ratio:.4f}, expected < 0.05 for pure-E input"


@needs_healpy
def test_fullsky_E_recovered():
    """Full sky: output almE should correlate strongly with the input almE."""
    T, Q, U, almE_in = _make_pure_E_tqu()
    TT, QQ, UU       = _full_weights()
    result            = maps_to_eb(T, Q, U, TT, QQ, UU, np.ones(NPIX),
                                   lmax=LMAX, maxiter=50)
    almE_out = result["almE"]
    corr     = np.corrcoef(almE_in.real, almE_out.real)[0, 1]
    assert corr > 0.99, f"almE correlation = {corr:.4f}, expected > 0.99"


@needs_healpy
def test_masked_finite_inside():
    """Inside the obs mask, E/B maps must be finite."""
    T, Q, U, _  = _make_pure_E_tqu()
    cap          = _circular_mask()
    TT, QQ, UU  = _cap_weights(cap)
    apod         = apodise_mask(cap, apod_fwhm_deg=3.0)
    result       = maps_to_eb(T, Q, U, TT, QQ, UU, apod, lmax=LMAX, maxiter=50)
    obs          = result["obs_mask"]
    assert np.isfinite(result["E_map"][obs]).all()
    assert np.isfinite(result["B_map"][obs]).all()


@needs_healpy
def test_masked_unseen_outside():
    """Outside the obs mask, E/B maps must be hp.UNSEEN."""
    T, Q, U, _  = _make_pure_E_tqu()
    cap          = _circular_mask()
    TT, QQ, UU  = _cap_weights(cap)
    apod         = apodise_mask(cap, apod_fwhm_deg=3.0)
    result       = maps_to_eb(T, Q, U, TT, QQ, UU, apod, lmax=LMAX, maxiter=50)
    obs          = result["obs_mask"]
    assert (result["E_map"][~obs] == hp.UNSEEN).all()
    assert (result["B_map"][~obs] == hp.UNSEEN).all()


@needs_healpy
def test_frame_to_eb_matches():
    """frame_to_eb must give identical results to maps_to_eb."""
    T, Q, U, _  = _make_pure_E_tqu()
    TT, QQ, UU  = _full_weights()
    apod         = np.ones(NPIX)
    frame        = {"T": T, "Q": Q, "U": U, "TT": TT, "QQ": QQ, "UU": UU}
    r1 = maps_to_eb(T.copy(), Q.copy(), U.copy(),
                    TT, QQ, UU, apod.copy(), lmax=LMAX)
    r2 = frame_to_eb(frame, apod, lmax=LMAX)
    np.testing.assert_array_equal(r1["E_map"], r2["E_map"])
    np.testing.assert_array_equal(r1["B_map"], r2["B_map"])


@needs_healpy
def test_convergence_reported():
    """rel_res should be a finite non-negative scalar."""
    T, Q, U, _  = _make_pure_E_tqu()
    TT, QQ, UU  = _full_weights()
    result       = maps_to_eb(T, Q, U, TT, QQ, UU, np.ones(NPIX), lmax=LMAX)
    assert np.isfinite(result["rel_res"])
    assert result["rel_res"] >= 0.0


# ── standalone runner ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    tests   = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    passed  = failed = skipped = 0

    for name, fn in tests:
        mark = getattr(fn, "pytestmark", [])
        if any(getattr(m, "name", "") == "skipif" and m.args[0] for m in mark):
            print(f"  SKIP  {name}  (healpy not available)")
            skipped += 1
            continue
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception as exc:
            print(f"  FAIL  {name}")
            print(f"        {type(exc).__name__}: {exc}")
            failed += 1

    print(f"\n{passed} passed, {skipped} skipped, {failed} failed "
          f"out of {len(tests)} tests")
    sys.exit(failed)
