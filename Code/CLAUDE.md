# CLAUDE.md

## Project context

This is a local code-development folder for my PhD research on SPT-3G CMB polarization analysis.

The scientific goal is to build an efficient, modular Python analysis pipeline for producing a mid-ell BB power spectrum from SPT-3G map products.

The longer-term goal may involve combining or comparing mid-ell SPT maps/spectra with low-ell B-mode maps/spectra, so the code should be organized in a way that can later support:

- multiple map products
- masks
- frequency bands
- split maps
- simulations
- power spectrum stages
- plotting and diagnostic checks

This local folder does **not** contain real SPT data.

The real SPT data live only on the remote shared cluster. Local code should be written and refactored here, then copied to the cluster and tested there manually.

---

## Scope and ownership rule

Do not read files outside `~/Initial_test` (this repository). Do not access, browse, or infer contents of other directories on this machine or the cluster.

Do not edit other people's code. This includes shared SPT-3G collaboration pipeline tools kept in this repo only for reference, such as `Code/Source_Code/master_field_mapmaker.py` and `Code/Source_Code/make_3g_sims.py` — treat these as read-only unless I explicitly ask for a change. If a task seems to require editing them, stop and ask first rather than assuming it's in scope.

---

## Two-machine workflow

This project is developed across two machines:

- **Local laptop**: AI-assisted planning, code drafting, refactoring, notes, and documentation.
- **HPC/scott**: real data access, testing, running SPT code, inspecting map products, and executing notebooks.

Do not treat notebooks as the main source of truth. Jupyter notebooks are for exploration and execution only.

When generating reusable code, write it into `.py` files under `src/`.

When planning, write short dated notes under `notes/`.

When creating notebook-style analysis steps, write them as clear Python scripts or markdown plans first, so they can be copied/tested on the HPC.

Avoid making large edits to `.ipynb` files unless explicitly asked.

---

## Notebook editing rule

Avoid making large edits to `.ipynb` files.

For tiny debugging edits in notebooks:

- Do not edit the notebook directly unless I explicitly ask.
- Instead, show me the exact small code cell or replacement snippet.
- I will manually copy it into the HPC notebook and test it there.
- After the code is stable, move reusable parts into `.py` files under `src/`.

Notebooks should be treated as temporary testing surfaces. Stable logic should gradually be moved into Python modules.

---

## Data-access rule

Do not assume access to real SPT data locally.

Do not request, create, copy, download, inspect, or simulate private SPT data files.

Do not hardcode real data paths such as:

- `/sptgrid/...`
- `/sptlocal/...`
- cluster scratch/data paths
- private collaboration repository paths

Use placeholders or user-supplied arguments instead, for example:

```python
map_file = "/path/to/map_file.g3"

---

## Research-coding interaction rule

This is research code, not app or website development.

Do not behave like an autonomous app-building agent that silently makes large changes until something works.

I need to understand the scientific and coding logic at each step. I am often better at answering focused questions than writing one large, fully specified prompt, so do not guess missing details.

For any non-trivial task:

1. First read the relevant file, notebook section, or code context.
2. Summarize what you think the current code is doing.
3. Identify any missing or ambiguous details.
4. Ask me focused follow-up questions before writing code.
5. Do not make assumptions unless clearly stated and confirmed.
6. Propose a short step-by-step plan only after the key details are clear.
7. For each step, explain:
   - what code you plan to use
   - why that code is needed
   - what assumptions it makes
   - what output/check I should expect
8. Ask me before moving to the next step.
9. Keep edits small and reviewable.

Do not complete a large task in one pass unless I explicitly ask.

Prefer an interactive workflow:

- read the relevant context
- summarize what you found
- ask me focused questions
- propose one small step
- explain the code logic
- wait for my answer or confirmation
- then give the exact code snippet
- wait for my test result
- then continue

For notebook debugging, do not directly rewrite the whole notebook. Give me the exact small cell or replacement snippet, and explain where it should go.

When plotting or analyzing maps, first ask clarifying questions about details such as:

- which file/path is being used
- which map object or key is being plotted
- whether the map is T/Q/U, E/B, or another product
- whether the map is full bundle, split map, coadd, mask, or diagnostic product
- expected pixel scale
- expected projection/coordinate system
- desired figure layout
- whether the output should be saved or only displayed

The goal is not only to produce working code. The goal is for me to understand and be able to maintain the analysis pipeline.

---

## Code comment style

Do not use decorative separator lines in comments such as:

```python
# ── Section label ─────────────────────────────────────────────────────────────
```

Use plain comments only:

```python
# Section label
```

This applies to all Python files and notebook cells.

---

## Repository layout

- `Code/Map_Operation/` — the active analysis pipeline. This is the code developed and maintained here.
  - `mask_utils.py` — sky-mask helpers: `obs_mask_from_weights` (pure numpy, works without healpy), `apodise_mask`, `load_mask` (both healpy, imported lazily).
  - `eb_pipeline.py` — Q/U → E/B diagnostic conversion via `healpy.map2alm_lsq` on the partial sky. Does **not** correct E→B leakage; it's a sanity-check tool, not the final BB estimator (that will need a proper pseudo-Cl method, e.g. NaMaster).
  - `Plot.py` — SPT-3G plotting helpers: `apply_spt_style()` (shared rcParams), `show_map_full_field`/`show_map_thumbnail` (single-panel healpy views), `show_1d_functions_of_ell`, `show_alm_triangle`, and the `MapPlotter` class (multi-panel gnomview/imshow grids).
  - `noise_psd_*.py` — a family of standalone scripts that compute the BB noise pseudo-Cl from sign-flip null maps, split by data format and frequency:
    - `noise_psd_fits_{90,220}ghz.py` — mid-ℓ FITS maps (HEALPix nside=8192).
    - `noise_psd_g3{,_90ghz,_220ghz}.py` — low-ℓ `g3.gz` maps (HEALPix nside=2048, requires the `spt3g` package).
    All follow the same pattern: load an apodisation mask → mask/apodise Q,U per realization → `hp.anafast` → average `cl_BB` over N sign-flip files → save `ell`, `N_l` to an `.npz` in `outputs/`. Config (paths, mask choice, `LMAX`, file count) is set via constants at the top of the file, not CLI args.
  - `test_eb_pipeline.py` — pytest suite for `mask_utils`/`eb_pipeline`, split into numpy-only tests (run anywhere) and `@needs_healpy` tests (skipped locally, run on the cluster).
  - `plot_lowl_bb_tiles.py`, `bb_*_inspect*.ipynb` — cluster-only inspection notebooks/scripts for tiling and viewing coadded low-ℓ maps; not runnable locally (no data, and `plot_lowl_bb_tiles.py` imports `spt3g`).
- `Code/Source_Code/` — reference copies of shared SPT-3G collaboration pipeline tools (`master_field_mapmaker.py`, `make_3g_sims.py`), pulled in for reference (see commit messages "found mapmaker" / "simulation code"). These are collaboration-wide code, not authored in this repo — treat as read-only reference, do not refactor.
- `Code/data_paths.md` — canonical reference for real cluster data/mask paths (mid-ℓ FITS, low-ℓ g3.gz, masks). Real paths belong only in this doc and in cluster-side config constants, never hardcoded elsewhere per the data-access rule above.
- `outputs/`, `Code/Map_Operation/outputs/` — generated `.npz` power-spectrum results; gitignored, regenerated by the `noise_psd_*.py` scripts.

### Data regimes

- **Mid-ℓ**: FITS, HEALPix nside=8192, unweighted T/Q/U (`WEIGHTED = F`).
- **Low-ℓ**: `g3.gz`, HEALPix nside=2048, weighted — call `maps.RemoveWeights(frame)` before extracting arrays; 3 frequency frames per file (`90GHz`/`150GHz`/`220GHz`).
- **Masks**: nside=8192 `.npz` files, single `'mask'` key, values in [0, 1]; `hp.ud_grade` to match the target map's nside. See `Code/data_paths.md` for the mask-name → filename table.

### Import convention

Local modules are imported as `from Initial_test.Map_Operation.mask_utils import ...`, i.e. the repo root is expected on `PYTHONPATH` as the `Initial_test` package (this matters on the cluster, where these imports actually resolve).

## Commands

- Run a noise-PSD script (edit the config constants at the top of the file first): `cd Code/Map_Operation && python noise_psd_fits_90ghz.py`
- Long cluster runs (pattern documented in each script's header comment): `nohup python -u <script>.py > <script>.log 2>&1 &` then `tail -f <script>.log`
- Tests: `python -m pytest Code/Map_Operation/test_eb_pipeline.py -v` (numpy-only tests run locally; healpy tests are skipped unless healpy is installed) — or `python test_eb_pipeline.py` for a standalone runner with no pytest dependency.