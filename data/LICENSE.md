# Terms for the data in this directory

The root `LICENSE` is MIT and covers the software, not these files.

| Files | Source | Licence |
|---|---|---|
| `test.csv`, `test_ms.csv`, `test_ic50_500.csv`, `test_ic50_1000.csv`, `dataset_demo.csv` | IEDB, `mhc_ligand_full` export of 2025-04-21 | **CC BY 4.0**. Filtered, relabelled and re-split, i.e. modified. Redistributed under the same licence. |
| `mhc_mapping.csv`, HLA rows | IPD-IMGT/HLA, release 3.59.0, via [`ANHIG/IMGTHLA`](https://github.com/ANHIG/IMGTHLA) | **CC BY-NoDerivs**, redistributed by permission of Anthony Nolan (Dr James Robinson, 24 September 2026) on one condition: should the work become commercial, or otherwise make profit, Anthony Nolan must be contacted to review the agreement. Processed (gap-free full-length chains, with the start and end of the peptide-binding domain), not an official IPD-IMGT/HLA release. |
| `mhc_mapping.csv`, H2 (murine) rows | UniProt | **CC BY 4.0**. Free to redistribute in modified form with attribution. |

Attribution, citations and the exact snapshot identity are in the
[third-party notices](https://github.com/daylight-00/PREpiBind/blob/main/THIRD_PARTY_NOTICES.md)
of the PREpiBind repository. Cite the sources and state that the data was modified when you reuse
anything here.
