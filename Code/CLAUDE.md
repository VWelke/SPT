# CLAUDE.md

## Project context

This is a local code-development folder for my PhD research on SPT-3G CMB polarization analysis.

The scientific goal is to build an efficient, modular Python analysis pipeline for producing a mid-ell BB power spectrum from SPT-3G map products. The longer-term goal may involve combining or comparing mid-ell SPT maps/spectra with low-ell B-mode maps/spectra, so the code should be organized in a way that can later support multiple map products, masks, frequency bands, and analysis stages.

This local folder does **not** contain real SPT data.

The real SPT data live only on the remote shared cluster. Local code should be written and refactored here, then copied to the cluster and tested there manually.

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