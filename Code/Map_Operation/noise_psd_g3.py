"""
noise_psd_g3.py
Compute BB noise pseudo-Cl from sign-flip g3.gz maps.
Run from terminal: python noise_psd_g3.py
"""

import os
import numpy as np
import healpy as hp
from spt3g import core, maps

# Config — edit these before running
SIGNFLIP_FILES = [
    # "/sptgrid/.../signflip_bundle_000.g3.gz",
    # "/sptgrid/.../signflip_bundle_001.g3.gz",
]

FREQ = "220GHz"   # frame Id to use from each file: "90GHz", "150GHz", "220GHz"

MASK_DIR    = "/sptlocal/user/creichardt/bb2020"
ACTIVE_MASK = "mask_250_nd30"

MASK_FILES = {
    "mask_250_30"   : "puremask8192_0p5medwt_250mJy_30arcmin.npz",
    "mask_250_60"   : "puremask8192_0p5medwt_250mJy_60arcmin.npz",
    "mask_250_nd30" : "puremask8192_0p5medwt_250mJy_nodisk_30arcmin.npz",
    "mask_250_nd60" : "puremask8192_0p5medwt_250mJy_nodisk_60arcmin.npz",
    "mask_100_30"   : "puremask8192_0p5medwt_100mJy_30arcmin.npz",
    "mask_apod_30"  : "puremask8192_0p5medwt_30arcmin.npz",
    "mask_apod_60"  : "puremask8192_0p5medwt_60arcmin.npz",
}

LMAX     = 6144   # 3 * nside for nside=2048
OUT_FILE = "/sptlocal/user/vwelke/lowl_bb_tiles/noise_psd_BB_g3.npz"


def load_qu_from_g3(fpath, freq):
    """Open a g3 file, find the frame matching freq, deweight, return Q and U as numpy arrays."""
    for frame in core.G3File(fpath):
        if frame.type == core.G3FrameType.Map and frame['Id'] == freq:
            maps.RemoveWeights(frame)
            Q = np.array(frame['Q'])
            U = np.array(frame['U'])
            obs = np.array(frame['Wpol'].QQ) > 0
            return Q, U, obs, frame['Q'].nside
    return None, None, None, None


def main():
    if len(SIGNFLIP_FILES) == 0:
        raise RuntimeError("No sign-flip files listed. Add paths to SIGNFLIP_FILES.")

    # Load mask
    mask_path = os.path.join(MASK_DIR, MASK_FILES[ACTIVE_MASK])
    with np.load(mask_path) as d:
        apod = d[d.files[0]].astype(float)
    nside_mask = hp.get_nside(apod)
    print(f"Mask    : {ACTIVE_MASK}  nside={nside_mask}")
    print(f"Freq    : {FREQ}")
    print(f"LMAX    : {LMAX}")
    print(f"N files : {len(SIGNFLIP_FILES)}")
    print()

    cl_BB_sum = np.zeros(LMAX + 1)
    N         = len(SIGNFLIP_FILES)
    n_done    = 0

    for i, fpath in enumerate(SIGNFLIP_FILES):
        Q_sf, U_sf, obs_sf, nside_sf = load_qu_from_g3(fpath, FREQ)

        if Q_sf is None:
            print(f"  [{i+1}/{N}]  WARNING: no '{FREQ}' frame in {os.path.basename(fpath)}, skipping")
            continue

        # Resize mask if nside differs
        apod_sf = hp.ud_grade(apod, nside_out=nside_sf) if nside_sf != nside_mask else apod

        apply_mask = obs_sf & (apod_sf > 0)

        Q_sf[apply_mask]   *= apod_sf[apply_mask]
        Q_sf[~apply_mask]   = 0.0
        U_sf[apply_mask]   *= apod_sf[apply_mask]
        U_sf[~apply_mask]   = 0.0

        T_zeros = np.zeros(len(Q_sf))
        cls      = hp.anafast([T_zeros, Q_sf, U_sf], lmax=LMAX)
        cl_BB_sum += cls[2]
        n_done   += 1

        del Q_sf, U_sf, T_zeros, cls
        print(f"  [{i+1}/{N}]  {os.path.basename(fpath)}")

    if n_done == 0:
        raise RuntimeError("No files were processed successfully.")

    cl_BB_noise = cl_BB_sum / n_done
    ell         = np.arange(LMAX + 1)

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    np.savez(OUT_FILE, ell=ell, N_l=cl_BB_noise, N=n_done, mask=ACTIVE_MASK, freq=FREQ)
    print(f"\nDone. Processed {n_done}/{N} files. Saved → {OUT_FILE}")
    print(f"Load with: data = np.load('{OUT_FILE}'); ell = data['ell']; N_l = data['N_l']")


if __name__ == "__main__":
    main()
