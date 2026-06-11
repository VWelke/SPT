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