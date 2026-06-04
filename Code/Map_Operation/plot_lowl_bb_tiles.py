# %% [markdown]
# # Low-ℓ BB — Full-Bundle Coadd Tile Viewer
#
# Loads a full-bundle coadded HEALPix map (FITS), computes the observed field
# extent from the weight map, then tiles the field into overlapping square
# patches and plots each one with `show_map_thumbnail` (reso = 0.2 arcmin/px).
#
# **Cluster-only.** Set all paths in §1 before running.
#
# Workflow
# --------
# 1. Set paths and configuration (§1)
# 2. Load coadd T / Q / U from FITS (§2)
# 3. Load available apodisation + point-source masks (§3)
# 4. Inspect field extent from observed pixels (§4)
# 5. Build tile grid (§5)
# 6. Tile loop — **unmasked** first, to see raw point sources (§6)
# 7. Tile loop — **with mask applied** (§7)

# %% [markdown]
# ## 0. Imports & style

# %%
import os
import sys
import numpy as np
import healpy as hp
import matplotlib
import matplotlib.pyplot as plt
from spt3g import core, maps

# Plot.py lives in the same directory; adjust if running from elsewhere.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Plot import (
    apply_spt_style,
    mask_map,
    show_map_full_field,
    show_map_thumbnail,
)

apply_spt_style()

print("python     :", sys.version)
print("numpy      :", np.__version__)
print("healpy     :", hp.__version__)
print("matplotlib :", matplotlib.__version__)

# %% [markdown]
# ## 1. Paths & configuration
#
# **Edit these on the cluster before running.**
#
# ### Coadd map
# Expected FITS format: full-sky HEALPix with T / Q / U stored in
# fields 0 / 1 / 2 (standard IQU convention).
# If the weight map (TT) is a separate file, set `WEIGHT_FILE` accordingly.
#
# ### Apodisation / point-source masks
# All available masks from `/sptlocal/user/creichardt/bb2020/`:
#
# | Variable        | File                                              |
# |-----------------|---------------------------------------------------|
# | `mask_250_30`   | puremask8192_0p5medwt_250mJy_30arcmin.npz         |
# | `mask_250_60`   | puremask8192_0p5medwt_250mJy_60arcmin.npz         |
# | `mask_250_nd30` | puremask8192_0p5medwt_250mJy_nodisk_30arcmin.npz  |
# | `mask_250_nd60` | puremask8192_0p5medwt_250mJy_nodisk_60arcmin.npz  |
# | `mask_100_30`   | puremask8192_0p5medwt_100mJy_30arcmin.npz         |
# | `mask_apod_30`  | puremask8192_0p5medwt_30arcmin.npz                |
# | `mask_apod_60`  | puremask8192_0p5medwt_60arcmin.npz                |

# %%
# ── Coadd path ────────────────────────────────────────────────────────────────
COADD_FILE  = "/path/to/full_bundle_coadd.fits"   # ← set on cluster
WEIGHT_FILE = None    # path to TT-weight FITS if separate; None → use isfinite

# ── Mask directory ────────────────────────────────────────────────────────────
MASK_DIR = "/sptlocal/user/creichardt/bb2020"

MASK_FILES = {
    "mask_250_30"   : "puremask8192_0p5medwt_250mJy_30arcmin.npz",
    "mask_250_60"   : "puremask8192_0p5medwt_250mJy_60arcmin.npz",
    "mask_250_nd30" : "puremask8192_0p5medwt_250mJy_nodisk_30arcmin.npz",
    "mask_250_nd60" : "puremask8192_0p5medwt_250mJy_nodisk_60arcmin.npz",
    "mask_100_30"   : "puremask8192_0p5medwt_100mJy_30arcmin.npz",
    "mask_apod_30"  : "puremask8192_0p5medwt_30arcmin.npz",
    "mask_apod_60"  : "puremask8192_0p5medwt_60arcmin.npz",
}

# ── Tile & display parameters ─────────────────────────────────────────────────
RESO_ARCMIN  = 0.2     # arcmin per pixel  (reso= argument in show_map_thumbnail)
PATCH_DEG    = 3.0     # square patch width = height in degrees
PATCH_PIX    = int(PATCH_DEG * 60 / RESO_ARCMIN)   # = 900 px for 3° patches

STOKES_KEYS  = ["T", "Q", "U"]
CMAP         = "coolwarm"

# Set to a directory path to save PDFs instead of displaying inline.
# Leave as None to show figures in the notebook.
SAVE_DIR     = None    # e.g. "/path/to/output/tiles/"

print(f"Patch size  : {PATCH_DEG}° × {PATCH_DEG}°  =  {PATCH_PIX} × {PATCH_PIX} px")
print(f"Resolution  : {RESO_ARCMIN} arcmin/px")

# %% [markdown]
# ## 2. Load coadd map (FITS)
#
# `hp.read_map(..., field=(0,1,2))` loads T / Q / U simultaneously.
# If the file only has one Stokes component (e.g., T only), load with
# `field=0` and skip Q / U below.

# %%
T_c, Q_c, U_c = hp.read_map(COADD_FILE, field=(0, 1, 2), partial=False)

# Build the observed-pixel boolean mask
if WEIGHT_FILE is not None:
    TT_c     = hp.read_map(WEIGHT_FILE, field=0, partial=False)
    obs_mask = TT_c > 0
else:
    # Fallback: any pixel that is finite and not UNSEEN
    obs_mask = (
        np.isfinite(T_c) & (T_c != hp.UNSEEN) &
        np.isfinite(Q_c) & (Q_c != hp.UNSEEN) &
        np.isfinite(U_c) & (U_c != hp.UNSEEN)
    )

stokes_maps = {"T": T_c, "Q": Q_c, "U": U_c}
nside_c     = hp.get_nside(T_c)

print(f"nside            : {nside_c}")
print(f"Observed pixels  : {obs_mask.sum():,}  ({obs_mask.sum()/len(T_c)*100:.2f}% of sky)")

# %% [markdown]
# ## 3. Load apodisation / point-source masks
#
# Each `.npz` file contains the mask as `arr_0` (verify with
# `np.load(path).files` if the key differs).
#
# All masks are loaded here so you can switch between them in §6–7
# without reloading from disk.

# %%
masks = {}
for label, fname in MASK_FILES.items():
    path = os.path.join(MASK_DIR, fname)
    if not os.path.exists(path):
        print(f"[skip]  {label} — file not found: {path}")
        continue
    data = np.load(path)
    # Most SPT npz masks store the array under 'arr_0'; check if different.
    key  = data.files[0]
    masks[label] = data[key].astype(float)
    print(f"Loaded  {label:16s}  shape={masks[label].shape}  key='{key}'")

print(f"\nAvailable masks: {list(masks.keys())}")

# %% [markdown]
# ## 4. Inspect field extent
#
# Derive the RA / Dec bounding box from the observed pixels.
# This is used to auto-build the tile grid in §5.

# %%
obs_pix        = np.where(obs_mask)[0]
theta_c, phi_c = hp.pix2ang(nside_c, obs_pix)

ra_obs  = np.degrees(phi_c)
ra_obs  = np.where(ra_obs > 180, ra_obs - 360, ra_obs)   # −180 … +180 deg
dec_obs = 90.0 - np.degrees(theta_c)

ra_min,  ra_max  = ra_obs.min(),  ra_obs.max()
dec_min, dec_max = dec_obs.min(), dec_obs.max()

print(f"RA  range  :  {ra_min:.2f}° → {ra_max:.2f}°   span = {ra_max-ra_min:.2f}°")
print(f"Dec range  : {dec_min:.2f}° → {dec_max:.2f}°   span = {dec_max-dec_min:.2f}°")

# Quick full-field overview (optional sanity check)
m_T_full = mask_map(T_c, obs_mask.astype(float))
rms_T    = float(np.std(m_T_full[m_T_full != hp.UNSEEN]))
show_map_full_field(
    m_T_full, vmin=-3*rms_T, vmax=3*rms_T,
    title="Full-bundle coadd — T  (full field overview)",
    unit="Tcmb", cmap=CMAP,
)

# %% [markdown]
# ## 5. Build tile grid
#
# Tile centres are spaced exactly `PATCH_DEG` apart in both RA and Dec
# (no overlap). The grid starts half a patch-width inside the field boundary
# so every tile is fully inside the observed footprint.
#
# Adjust `PATCH_DEG` in §1 to change the patch size.

# %%
ra_centres  = np.arange(ra_min  + PATCH_DEG/2, ra_max  + PATCH_DEG/2, PATCH_DEG)
dec_centres = np.arange(dec_min + PATCH_DEG/2, dec_max + PATCH_DEG/2, PATCH_DEG)

n_ra, n_dec = len(ra_centres), len(dec_centres)
n_total     = n_ra * n_dec * len(STOKES_KEYS)

print(f"RA  centres  : {n_ra}  ({ra_centres[0]:.1f}° … {ra_centres[-1]:.1f}°)")
print(f"Dec centres  : {n_dec}  ({dec_centres[0]:.1f}° … {dec_centres[-1]:.1f}°)")
print(f"Patches/Stokes : {n_ra * n_dec}")
print(f"Total panels : {n_total}  ({len(STOKES_KEYS)} Stokes × {n_ra*n_dec} patches)")
if n_total > 200:
    print("  ⚠  Many panels — consider setting SAVE_DIR to write PDFs instead.")

# %% [markdown]
# ## 6. Tile loop — unmasked
#
# Plot each patch **without** applying the point-source mask.
# Purpose: visually locate bright point sources in T, Q, U.
#
# After inspecting, decide which mask threshold (100 mJy vs 250 mJy,
# 30 arcmin vs 60 arcmin disk) is appropriate, then run §7.

# %%
def _tile_loop(stokes_maps_in, obs_mask_in, label=""):
    """
    Iterate over all tiles and Stokes components, calling show_map_thumbnail
    for each patch (inline) or saving to SAVE_DIR (if set in §1).
    """
    if SAVE_DIR is not None:
        os.makedirs(SAVE_DIR, exist_ok=True)

    for stokes in STOKES_KEYS:
        arr = stokes_maps_in[stokes].copy()
        arr[~obs_mask_in] = hp.UNSEEN

        rms        = float(np.std(arr[obs_mask_in]))
        vmin, vmax = -3 * rms, 3 * rms

        print(f"  {stokes}  rms = {rms:.4g}  →  colour limits ±{3*rms:.4g}")

        for dec_c in dec_centres:
            for ra_c in ra_centres:
                title = (
                    f"{stokes}{('  ' + label) if label else ''}  |  "
                    f"RA {ra_c:+.1f}°  Dec {dec_c:+.1f}°"
                )
                rot = (ra_c, dec_c, 0)

                if SAVE_DIR is not None:
                    # Save mode: drive gnomview directly so we control savefig
                    fig = plt.figure(figsize=(6, 6), facecolor="white")
                    hp.gnomview(
                        arr,
                        rot=rot,
                        xsize=PATCH_PIX, ysize=PATCH_PIX,
                        reso=RESO_ARCMIN,
                        cmap=CMAP, min=vmin, max=vmax,
                        badcolor="white",
                        title=title, unit="Tcmb",
                        fig=fig.number,
                    )
                    safe_label = label.replace(" ", "_").replace("/", "-")
                    fname = (
                        f"tile_{stokes}{('_' + safe_label) if safe_label else ''}"
                        f"_ra{ra_c:+.1f}_dec{dec_c:+.1f}.pdf"
                    )
                    plt.savefig(
                        os.path.join(SAVE_DIR, fname),
                        bbox_inches="tight", dpi=120,
                    )
                    plt.close(fig)
                else:
                    # Inline mode: use the existing square-thumbnail helper
                    show_map_thumbnail(
                        arr, vmin=vmin, vmax=vmax,
                        title=title, unit="Tcmb",
                        cmap=CMAP,
                        rot=rot,
                        xsize=PATCH_PIX, ysize=PATCH_PIX,
                        reso=RESO_ARCMIN,
                    )

# ──────────────────────────────────────────────────────────────────────────────
print("=== Unmasked tiles ===")
_tile_loop(stokes_maps, obs_mask, label="unmasked")

# %% [markdown]
# ## 7. Tile loop — with point-source mask
#
# Choose a mask loaded in §3 by setting `ACTIVE_MASK` below.
# The mask is multiplied into each Stokes map before plotting.
#
# Suggested starting point: `"mask_250_30"` (250 mJy threshold, 30' disk).
# Then compare with `"mask_100_30"` to check how many extra sources appear.

# %%
ACTIVE_MASK = "mask_250_30"   # ← change to any key from masks dict

assert ACTIVE_MASK in masks, (
    f"'{ACTIVE_MASK}' not loaded. Available: {list(masks.keys())}"
)

apod = masks[ACTIVE_MASK]

# Sanity-check nside compatibility
nside_mask = hp.get_nside(apod)
if nside_mask != nside_c:
    print(f"Warning: mask nside={nside_mask}, map nside={nside_c} — upgranding mask.")
    apod = hp.ud_grade(apod, nside_out=nside_c)

# Combined observed mask: observed pixels AND mask > 0
obs_and_mask = obs_mask & (apod > 0)

# Apply apodisation: multiply map values by mask weight inside observed region
stokes_masked = {}
for stokes in STOKES_KEYS:
    arr = stokes_maps[stokes].copy()
    arr[obs_and_mask]  *= apod[obs_and_mask]   # taper at edges / source holes
    arr[~obs_and_mask]  = hp.UNSEEN
    stokes_masked[stokes] = arr

print(f"Active mask  : {ACTIVE_MASK}  ({MASK_FILES[ACTIVE_MASK]})")
print(f"Pixels after mask  : {obs_and_mask.sum():,}  "
      f"(was {obs_mask.sum():,} unmasked)")

# ──────────────────────────────────────────────────────────────────────────────
print(f"\n=== Masked tiles  [{ACTIVE_MASK}] ===")
_tile_loop(stokes_masked, obs_and_mask, label=ACTIVE_MASK)
