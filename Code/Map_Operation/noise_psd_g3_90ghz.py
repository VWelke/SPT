"""
noise_psd_g3_90ghz.py
Compute BB noise pseudo-Cl from sign-flip 90GHz G3 maps.
Run from terminal: python noise_psd_g3_90ghz.py
"""

# cd ~/Initial_test/Code/Map_Operation

# nohup python -u noise_psd_g3_90ghz.py > noise_psd_g3_90ghz.log 2>&1 &

# tail -f noise_psd_g3_90ghz.log

import os
import glob
import numpy as np
import healpy as hp
import time
from spt3g import core, maps

# Config

SIGNFLIP_DIR = "/sptgrid/user/javva/baseline_bb_coadd_match/coadd_fullauto/full/"
FREQ         = "90GHz"

# Pick N random sign-flip files from whatever is available in the directory
N_FILES        = 10
all_sf_files   = sorted(glob.glob(os.path.join(SIGNFLIP_DIR, "signflip_*_bundle_000.g3.gz")))  # sorted so input list is deterministic; selection is still random each run
rng            = np.random.default_rng()
SIGNFLIP_FILES = list(rng.choice(all_sf_files, size=min(N_FILES, len(all_sf_files)), replace=False))

MASK_DIR    = "/sptlocal/user/creichardt/bb2020"
ACTIVE_MASK = "mask_100_30"

MASK_FILES = {
    "mask_100_30" : "puremask_0p5medwt_100mJy_30arcmin.npz",
}

LMAX     = 3000
OUT_FILE = os.path.expanduser("~/Initial_test/Code/Map_Operation/outputs/noise_psd_BB_g3_90ghz.npz")


def main():
    t_start = time.time()

    # Load mask
    mask_path = os.path.join(MASK_DIR, MASK_FILES[ACTIVE_MASK])
    with np.load(mask_path) as d:
        apod = d[d.files[0]].astype(np.float32)
    nside_mask = hp.get_nside(apod)
    print(f"Mask      : {ACTIVE_MASK}  nside={nside_mask}")
    print(f"FREQ      : {FREQ}")
    print(f"LMAX      : {LMAX}")
    print(f"N files   : {len(SIGNFLIP_FILES)}")
    print(f"Available : {len(all_sf_files)}")
    print()

    cl_BB_sum = np.zeros(LMAX + 1)
    N_done = 0

    for i, fpath in enumerate(SIGNFLIP_FILES):
        t_file_start = time.time()

        try:
            sf_frame = None
            for frame in core.G3File(fpath):
                if frame.type == core.G3FrameType.Map and frame['Id'] == FREQ:
                    sf_frame = frame
                    break
        except Exception as e:
            print(f"  [{i+1}/{len(SIGNFLIP_FILES)}]  ERROR: {e} — skipping {os.path.basename(fpath)}")
            continue

        if sf_frame is None:
            print(f"  [{i+1}/{len(SIGNFLIP_FILES)}]  WARNING: no '{FREQ}' frame — skipping {os.path.basename(fpath)}")
            continue

        maps.RemoveWeights(sf_frame)
        Q_sf   = np.array(sf_frame['Q']).astype(np.float32)
        U_sf   = np.array(sf_frame['U']).astype(np.float32)
        obs_sf = np.array(sf_frame['Wpol'].TT) > 0
        del sf_frame

        apply_mask         = obs_sf & (apod > 0)
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
    np.savez(OUT_FILE, ell=ell, N_l=cl_BB_noise, N=N_done, mask=ACTIVE_MASK, freq=FREQ)

    print(f"\nTotal time : {time.time() - t_start:.1f}s")
    print(f"Done. N={N_done} files averaged → {OUT_FILE}")
    print(f"Load with  : data = np.load('{OUT_FILE}'); ell = data['ell']; N_l = data['N_l']")


if __name__ == "__main__":
    main()
