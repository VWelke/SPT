"""
noise_psd_fits_150ghz.py
Compute BB noise pseudo-Cl from sign-flip 150GHz FITS maps.
Run from terminal: python noise_psd_fits_150ghz.py
"""

# cd ~/Initial_test/Code/Map_Operation

# nohup python -u noise_psd_fits_150ghz.py > noise_psd_fits_150ghz.log 2>&1 &

# tail -f noise_psd_fits_150ghz.log

import os
import numpy as np
import healpy as hp
import time

# Config — edit these before running

SIGNFLIP_DIR = "/sptgrid/analysis/spt3g_d1_midell_tqu_healpix/real_data_maps/signflip_noise/"

import glob

N_FILES      = 10
all_sf_files = sorted(glob.glob(os.path.join(SIGNFLIP_DIR, "signflip_noise_permutation*_150ghz.fits")))
rng          = np.random.default_rng()
SIGNFLIP_FILES = list(rng.choice(all_sf_files, size=min(N_FILES, len(all_sf_files)), replace=False))

MASK_DIR    = "/sptlocal/user/creichardt/bb2020"
ACTIVE_MASK = "mask_250_nd30"

MASK_FILES = {
    "mask_250_nd30" : "puremask8192_0p5medwt_250mJy_nodisk_30arcmin.npz",
}

LMAX     = 3000
OUT_FILE = os.path.expanduser("~/Initial_test/outputs/noise_psd_BB_fits_150ghz.npz")


def main():
    t_start = time.time()

    # Load mask
    mask_path = os.path.join(MASK_DIR, MASK_FILES[ACTIVE_MASK])
    with np.load(mask_path) as d:
        apod = d[d.files[0]].astype(np.float32)
    nside_mask = hp.get_nside(apod)
    print(f"Mask    : {ACTIVE_MASK}  nside={nside_mask}")
    print(f"LMAX    : {LMAX}")
    print(f"N files : {len(SIGNFLIP_FILES)}")
    print()

    if not SIGNFLIP_FILES:
        raise RuntimeError(
            f"No sign-flip FITS files matched {SIGNFLIP_DIR!r} for the 150 GHz band. "
            "Expected filenames with a 150ghz suffix."
        )

    cl_BB_sum = np.zeros(LMAX + 1)
    N_done = 0

    for i, fpath in enumerate(SIGNFLIP_FILES):
        t_file_start = time.time()

        try:
            Q_sf = hp.read_map(fpath, field=1, partial=False, dtype=np.float32)
            U_sf = hp.read_map(fpath, field=2, partial=False, dtype=np.float32)
        except FileNotFoundError:
            print(f"  [{i+1}/{len(SIGNFLIP_FILES)}]  MISSING — skipping {os.path.basename(fpath)}")
            continue

        apply_mask         = np.isfinite(Q_sf) & (Q_sf != hp.UNSEEN) & (apod > 0)
        Q_sf[apply_mask]  *= apod[apply_mask]
        Q_sf[~apply_mask]  = 0.0
        U_sf[apply_mask]  *= apod[apply_mask]
        U_sf[~apply_mask]  = 0.0

        T_zeros = np.zeros(len(Q_sf), dtype=np.float32)
        cls     = hp.anafast([T_zeros, Q_sf, U_sf], lmax=LMAX, iter=0)
        cl_BB_sum += cls[2]
        N_done += 1

        del Q_sf, U_sf, T_zeros, cls
        print(f"  [{i+1}/{len(SIGNFLIP_FILES)}]  {os.path.basename(fpath)}  "
              f"({time.time() - t_file_start:.1f}s)")

    cl_BB_noise = cl_BB_sum / N_done
    ell         = np.arange(LMAX + 1)

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    np.savez(OUT_FILE, ell=ell, N_l=cl_BB_noise, N=N_done, mask=ACTIVE_MASK)

    print(f"\nTotal time: {time.time() - t_start:.1f}s")
    print(f"Done. N={N_done} files averaged → {OUT_FILE}")
    print(f"Load with: data = np.load('{OUT_FILE}'); ell = data['ell']; N_l = data['N_l']")


if __name__ == "__main__":
    main()
