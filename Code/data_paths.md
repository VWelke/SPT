# Data Paths

Quick reference for map and mask locations on the cluster.

---

## Mid-ℓ maps (FITS, HEALPix nside=8192)

```
/sptgrid/analysis/spt3g_d1_midell_tqu_healpix/real_data_maps/full/
    full_220ghz.fits
    full_150ghz.fits
    full_095ghz.fits
```

Columns: PIXEL, TEMPERATURE, Q_POLARISATION, U_POLARISATION + 6 normalised covariance fields.  
`WEIGHTED = F` — T, Q, U are already unweighted Stokes values.

Simulated maps:
```
/sptgrid/analysis/spt3g_d1_midell_tqu_healpix/simulated_maps/
```
(contents TBD — see `bb_sim_inspect.ipynb`)

---

## Low-ℓ maps (g3.gz, HEALPix nside=2048)

```
/sptgrid/user/javva/baseline_bb_coadd_match/coadd_fullauto/full/
    no_signflip_bundle_000.g3.gz
```

Three frames per file — one per frequency band:
- Frame `Id = "90GHz"`
- Frame `Id = "150GHz"`
- Frame `Id = "220GHz"`

Keys per frame: `T`, `Q`, `U`, `Wpol`, `Id`.  
`weighted = True` — use `maps.RemoveWeights(frame)` before extracting arrays.

---

## Masks

```
/sptlocal/user/creichardt/bb2020/
```

| Key | File | Description |
|---|---|---|
| `mask_250_30` | `puremask8192_0p5medwt_250mJy_30arcmin.npz` | 250 mJy sources, 30 arcmin apod, with disk |
| `mask_250_60` | `puremask8192_0p5medwt_250mJy_60arcmin.npz` | 250 mJy sources, 60 arcmin apod, with disk |
| `mask_250_nd30` | `puremask8192_0p5medwt_250mJy_nodisk_30arcmin.npz` | 250 mJy, no disk, 30 arcmin |
| `mask_250_nd60` | `puremask8192_0p5medwt_250mJy_nodisk_60arcmin.npz` | 250 mJy, no disk, 60 arcmin |
| `mask_100_30` | `puremask8192_0p5medwt_100mJy_30arcmin.npz` | 100 mJy sources, 30 arcmin apod |
| `mask_apod_30` | `puremask8192_0p5medwt_30arcmin.npz` | Apodisation only, no source masking, 30 arcmin |
| `mask_apod_60` | `puremask8192_0p5medwt_60arcmin.npz` | Apodisation only, no source masking, 60 arcmin |

All masks: HEALPix nside=8192, single array key `'mask'`, values 0–1.  
When used with nside=2048 maps, downgrade with `hp.ud_grade(apod, nside_out=2048)`.


change code to for each frequency pick 100 fits
 cd  /sptgrid/analysis/spt3g_d1_midell_tqu_healpix/real_data_maps/signflip_noise/

 sth like that

 /home/creichardt/spt3g_software/sources/mask_lists/

 ' high_ell_TT_2021_1500d_source_mask_6mJy_plusedges.txt'