"""
run_namaster_midl.py
Run NaMaster (pure-B) on one mid-l (nside=8192) mask against the 90GHz
mid-l coadd map, and pickle the result.

Run from terminal: python run_namaster_midl.py
"""

# cd ~/Initial_test/Code/Map_Operation

# nohup python -u run_namaster_midl.py > run_namaster_midl.log 2>&1 &

# tail -f run_namaster_midl.log

import os
os.environ["OMP_NUM_THREADS"] = "10"

import datetime
import pickle

import healpy as hp
import numpy as np
import pymaster as nmt

# Config — edit these before running

MASK_DIR = "/sptlocal/user/creichardt/bb2020"

MASK_FILES = {
    "mask_250_30":   "puremask8192_0p5medwt_250mJy_30arcmin.npz",
    "mask_250_60":   "puremask8192_0p5medwt_250mJy_60arcmin.npz",
    "mask_250_nd30": "puremask8192_0p5medwt_250mJy_nodisk_30arcmin.npz",
    "mask_250_nd60": "puremask8192_0p5medwt_250mJy_nodisk_60arcmin.npz",
    "mask_100_30":   "puremask8192_0p5medwt_100mJy_30arcmin.npz",
    "mask_apod_30":  "puremask8192_0p5medwt_30arcmin.npz",
    "mask_apod_60":  "puremask8192_0p5medwt_60arcmin.npz",
}
ACTIVE_MASK = "mask_250_nd30"

COADD_FILE = "/sptgrid/analysis/spt3g_d1_midell_tqu_healpix/real_data_maps/full/full_095ghz.fits"

OUT_DIR = os.path.expanduser("~/Initial_test/outputs")


def compute_master(f_a, f_b, wsp):
    cl_coupled = nmt.compute_coupled_cell(f_a, f_b)
    cl_decoupled = wsp.decouple_cell(cl_coupled)
    return cl_decoupled


def run_namaster(dmap, mask):
    """Pure-B pseudo-Cl via NaMaster. Only works for apodized masks."""
    print('1.', datetime.datetime.now())
    b = nmt.NmtBin.from_nside_linear(hp.npix2nside(mask.shape[0]), 10)
    print('2.', datetime.datetime.now())

    f2yp0 = nmt.NmtField(mask, [dmap[1], dmap[2]], purify_e=False, purify_b=True)
    print('3.', datetime.datetime.now())

    w_yp = nmt.NmtWorkspace()
    w_yp.compute_coupling_matrix(f2yp0, f2yp0, b)
    bpwf = w_yp.get_bandpower_windows()
    print('4.', datetime.datetime.now())

    clyp = compute_master(f2yp0, f2yp0, w_yp)
    print('5. finished computing namaster', datetime.datetime.now())

    return b.get_effective_ells(), clyp, bpwf


def main():
    mask_name = MASK_FILES[ACTIVE_MASK]
    mask_path = os.path.join(MASK_DIR, mask_name)
    mask = np.load(mask_path)['mask']
    mask[np.isnan(mask)] = 0
    print(f"Mask    : {ACTIVE_MASK}  ({mask_name})  nside={hp.npix2nside(mask.shape[0])}")
    print(f"Map     : {COADD_FILE}")
    print()

    T = hp.read_map(COADD_FILE, field=0, partial=False)
    Q = hp.read_map(COADD_FILE, field=1, partial=False)
    U = hp.read_map(COADD_FILE, field=2, partial=False)
    dmap = [T, Q, -1 * U]

    output = run_namaster(dmap, mask)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_file = os.path.join(OUT_DIR, f"mask_test_{mask_name}_fits.pkl")
    with open(out_file, 'wb') as f:
        pickle.dump(output, f)

    print(f"Done. Saved to {out_file}")


if __name__ == "__main__":
    main()
