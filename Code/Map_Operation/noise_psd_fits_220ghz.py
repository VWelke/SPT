"""
noise_psd_fits.py
Compute BB noise pseudo-Cl from sign-flip FITS maps.
Run from terminal: python noise_psd_fits.py
"""

# cd ~/Initial_test/Code/Map_Operation

# nohup python -u noise_psd_fits_220ghz.py > noise_psd_fits_220ghz.log 2>&1 &

# tail -f noise_psd_fits_220ghz.log

import os
import numpy as np
import healpy as hp
import time

# Config — edit these before running

SIGNFLIP_DIR = "/sptgrid/analysis/spt3g_d1_midell_tqu_healpix/real_data_maps/signflip_noise/"

index = np.random.randint(1, 500, 1) 
#SIGNFLIP_FILES = [
#    os.path.join(SIGNFLIP_DIR, f"signflip_noise_permutation{i}_220ghz.fits") for i in index
#]

SIGNFLIP_FILES = [
    os.path.join(SIGNFLIP_DIR, f"signflip_noise_permutation499_220ghz.fits") for i in index
]

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

LMAX     = 3000
OUT_FILE = "/sptlocal/user/vwelke/lowl_bb_tiles/noise_psd_BB_fits_220ghz.npz"


def main():
    t_start = time.time()

    # Load mask
    mask_path = os.path.join(MASK_DIR, MASK_FILES[ACTIVE_MASK])
    with np.load(mask_path) as d:
        apod = d[d.files[0]].astype(float)
    nside_mask = hp.get_nside(apod)
    print(f"Mask    : {ACTIVE_MASK}  nside={nside_mask}")
    print(f"LMAX    : {LMAX}")
    print(f"N files : {len(SIGNFLIP_FILES)}")
    print()

    cl_BB_sum = np.zeros(LMAX + 1)
    N = len(SIGNFLIP_FILES)

    for i, fpath in enumerate(SIGNFLIP_FILES):
        t_file_start = time.time()
        Q_sf = hp.read_map(fpath, field=1, partial=False)
        U_sf = hp.read_map(fpath, field=2, partial=False)

        apod_sf  =  apod 

        # Build combined mask from observed pixels + apodisation
        obs_sf       = np.isfinite(Q_sf) & (Q_sf != hp.UNSEEN)
        apply_mask   = obs_sf & (apod_sf > 0)

        Q_sf[apply_mask]   *= apod_sf[apply_mask]
        Q_sf[~apply_mask]   = 0.0
        U_sf[apply_mask]   *= apod_sf[apply_mask]
        U_sf[~apply_mask]   = 0.0

        T_zeros = np.zeros(len(Q_sf))
        cls      = hp.anafast([T_zeros, Q_sf, U_sf], lmax=LMAX)
        cl_BB_sum += cls[2]

        del Q_sf, U_sf, T_zeros, cls
        print(f"  [{i+1}/{N}]  {os.path.basename(fpath)}")
        print(f"    Time for file: {time.time() - t_file_start:.2f} seconds")

    cl_BB_noise = cl_BB_sum / N
    ell         = np.arange(LMAX + 1)

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    np.savez(OUT_FILE, ell=ell, N_l=cl_BB_noise, N=N, mask=ACTIVE_MASK)
    t_end = time.time()
    print(f"\nTotal time: {t_end - t_start:.2f} seconds")
    print(f"\nDone. Saved → {OUT_FILE}")
    print(f"Load with: data = np.load('{OUT_FILE}'); ell = data['ell']; N_l = data['N_l']")


if __name__ == "__main__":
    main()
