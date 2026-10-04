# Data notice

This repository contains code, stored cross-validation partitions, per-participant
and per-fold results, and model predictions derived from WESAD
(Schmidt et al., ICMI 2018). It does **not** contain the raw WESAD files.

WESAD's distribution page states: "You may use this data for scientific,
non-commercial purposes, provided that you give credit to the owners when
publishing any work based on this data" (https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html,
checked 2026-10-04). Redistribution of derived data is not addressed there, so the
window-level feature tables are **withheld**; their SHA-256 hashes are recorded in
`results/feature_table_manifest.json` and `results/extended_table_manifest.json`, and
`extract_extended.py` regenerates the extended table from the official WESAD files.
Anyone reusing anything here must cite Schmidt et al. (2018) and respect WESAD's
non-commercial terms. No model, feature table or result derived from WESAD is used
in any E14 Health product.

The `wesad_readme.pdf` shipped inside the official archive was read on 2026-10-04: it
contains no licence terms beyond "If you publish material based on this dataset, please
reference the publication [1]" (Schmidt et al., ICMI 2018). The terms quoted above from the
distribution page therefore govern.
