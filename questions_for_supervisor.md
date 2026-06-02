# Questions for Supervisor Meeting
_Last updated: 2026-06-02_

---

> **Before writing more code — need answers to §10 and §11 first.**
> The whole pipeline structure depends on these.

---

## 1. Data structure — what maps do I actually have?

- The data seems to be organised by **subfield** and **observation ID**:
  - Subfields: `ra0hdec-44.75`, `ra0hdec-52.25`, `ra0hdec-59.75`, `ra0hdec-67.25`
  - Each subfield has many observation IDs → many `.g3` files
- **Q: Are these per-observation (bundle) maps, or already coadded maps?**
- **Q: What does one observation ID correspond to? (one night? one scan pass? one "bundle"?)**
- **Q: Is there already a coadded map product I should be using, or do I need to coadd myself?**

---

## 2. What maps do I actually need for the BB power spectrum?

- I have maps per subfield and per observation ID — that's a lot of files.
- **Q: Do I work with all subfields, or one at a time, or combined?**
- **Q: For the power spectrum, should I use the coadded map, or individual bundles (for noise estimation)?**
- **Q: What is the standard noise-bias subtraction strategy here? (e.g. half-mission splits, bundle cross-spectra?)**

---

## 3. Left / Right scan splits

- The frame keys in the existing code are named `"Left90GHz"`, `"Right90GHz"`, etc.
- **Q: Do "Left" and "Right" refer to scan direction (left-going vs right-going telescope scans)?**
- **Q: Are these useful for the BB analysis, or only for systematics checks?**
- **Q: Should I be working with Left+Right coadded, or keeping them separate?**

---

## 4. Frequency bands

- There are 3 frequency bands: 90 GHz, 150 GHz, 220 GHz.
- **Q: For the mid-ell BB spectrum, should I analyse each band independently, or combine/multifrequency from the start?**
- **Q: Is foreground separation relevant at this stage?**

---

## 5. What is "mid-ell" exactly?

- The project goal says "mid-ell BB power spectrum".
- **Q: What ell range counts as mid-ell in your analysis? (e.g. 300 < ℓ < 3000?)**
- **Q: What `lmax` should I be using for the harmonic transforms?**

---

## 6. Masks

- `mask_utils.py` builds a mask from weight maps (works for any map).
- **Q: Is there a pre-computed apodisation mask I should use, or should I build it from the weight maps?**
- **Q: Is there a point-source mask I need to apply?**
- **Q: Should the mask be the same for all subfields, or per-subfield?**

---

## 7. Pipeline scope — what am I responsible for building?

- **Q: Is there an existing SPT-3G BB pipeline I should follow or interface with?**
- **Q: Am I computing the power spectrum from scratch with NaMaster, or does the collaboration have standard tools?**
- **Q: What is the expected deliverable at the end — a spectrum plot? a data file? a paper-ready result?**

---

## 8. Plotting — do I need to visualise all maps?

- There are many maps (4 subfields × many obs IDs × 3 frequencies × 2 scan directions).
- **Q: Is visual inspection of individual maps expected, or just the coadd?**
- **Q: Are there known bad observations I should flag/exclude?**
- **Q: Is there a data quality cut procedure I should follow?**

---

## 9. Input files to locate on the cluster ⚠️ PRIORITY

Before writing any more pipeline code, I need to find these 3 things on the cluster:

### 9a. Input sky maps (the observed data)
- The actual T/Q/U signal maps (+ weight maps TT, QQ, UU)
- These are what get fed into the power spectrum estimator
- **Q: Where are the coadded maps stored? What is the exact path/naming convention?**
- **Q: Are they in `.g3` format, or converted to FITS/HEALPix already?**
- **Q: Which map product should I use — are there different versions/reductions?**

### 9b. Filtered maps (signal simulations passed through the pipeline)
- In SPT-3G, the time-domain processing applies filters (polynomial subtraction,
  high-pass scan filter, etc.) that suppress large-scale modes.
- "Filtered maps" = simulated CMB skies that have been run through the **same
  filtering pipeline** as the real data → used to measure the filter transfer function.
- The transfer function corrects for modes lost by the filter in the power spectrum.
- **Q: Where are the filtered signal simulations stored on the cluster?**
- **Q: How many simulations are there? (need enough for a stable transfer function estimate)**
- **Q: Is the transfer function already computed, or do I need to compute it from the sims?**
- **Q: Is there documentation on what filters were applied to the data?**

### 9c. Apodisation mask
- A smooth window map (values 0→1) applied to the sky maps before computing
  pseudo-Cls, to taper the sharp field edges and reduce spectral leakage.
- Different from the binary obs mask (which just marks observed pixels).
- **Q: Is there a pre-made apodisation mask for each subfield?**
- **Q: Where is it stored? (`.fits`, `.npy`, or `.g3`?)**
- **Q: If not, should I build it from the weight maps using NaMaster's C² apodisation?**
- **Q: Should there be a separate point-source hole mask multiplied in?**

---

## 10. Do I need to combine (stack) the 4 subfields? ⚠️ PIPELINE-DEFINING

The 4 subfields are separate declination strips of sky:

```
ra0hdec-44.75   ━━━━━━━━━━━━━━━━━  Dec ≈ -44.75°
ra0hdec-52.25   ━━━━━━━━━━━━━━━━━  Dec ≈ -52.25°
ra0hdec-59.75   ━━━━━━━━━━━━━━━━━  Dec ≈ -59.75°
ra0hdec-67.25   ━━━━━━━━━━━━━━━━━  Dec ≈ -67.25°
```

There are two different approaches — I don't know which one applies here:

**Option A — Analyse subfields separately, then combine spectra**
- Compute BB power spectrum independently per subfield
- Combine the 4 spectra at the end (inverse-variance weighted)
- Each subfield has its own mask

**Option B — Stitch maps together into one large footprint**
- Mosaic the 4 strips into a single map
- Compute one power spectrum from the full combined map
- Requires a single combined mask

- **Q: Should I treat each subfield independently or combine them?**
- **Q: Is there a standard SPT-3G way of handling the multi-subfield footprint?**
- **Q: If combining, is there a map-level mosaic tool I should use?**

---

## 11. Do I need to look at every observation ID? ⚠️ PIPELINE-DEFINING

Each subfield has many observation IDs. Almost certainly the answer is **no** — but the question is what to do with them:

**Option A — Use a pre-existing coadded map**
- Someone has already averaged all the observations → one map per subfield
- I just load and use that → no need to touch individual obs IDs
- **Most likely for a first analysis**

**Option B — Coadd the bundles myself**
- Load all obs IDs, average them (weighted by weight maps)
- Needed if no coadd product exists yet
- Then plot and analyse the coadd only

**Option C — Keep bundles separate for noise estimation**
- Use pairs of bundles (e.g. even/odd obs IDs) to form "splits"
- Compute cross-spectrum between splits → noise bias cancels
- Individual bundle maps are never plotted, just averaged in pairs

Regarding **visual inspection**:
- Almost certainly do **not** plot every obs ID
- Instead: flag bad observations using **data quality metrics** (e.g. map noise level, 
  number of detector hits, weather conditions) → automated cut
- Then spot-check a handful visually

- **Q: Is there already a coadded map product I should start from?**
- **Q: Is there a data quality cut / flagging procedure already defined?**
- **Q: What is the noise-bias subtraction strategy? (bundle cross-spectra? half-mission splits?)**
- **Q: How many observations are there per subfield roughly?**

---

## 12. Full depth vs half depth coadds — clarifications needed

- I understand that full depth = all observations coadded, half depth A/B = two independent halves
- The cross-spectrum C_ℓ^{A×B} cancels noise bias → this is the BB estimator strategy
- **Q: Are there already full depth AND half depth coadds on the cluster, or do I build them?**
- **Q: How are the halves defined — odd/even obs IDs, or first/second half of the season?**
- **Q: Should I use half-depth cross-spectra as the primary BB estimator?**

---

## 13. Where are files stored on `scott`?

Currently syncing code to: `/home/vwelke/Initial_test/`

Need to locate on the cluster:
- [ ] Full depth coadded maps (per subfield)
- [ ] Half depth A / B maps (per subfield)
- [ ] Apodisation mask(s) — `.fits` or `.npy`
- [ ] Point source mask (if separate)
- [ ] Filtered signal simulations (for transfer function)

**Q: Where on `scott` (or sptgrid) are the map products and masks stored?**
**Q: Is there a shared group analysis directory with standard products?**

---

## Notes / answers (fill in after meeting)

| Question | Answer | Date |
|----------|--------|------|
| | | |

