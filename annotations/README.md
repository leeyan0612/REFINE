# CaSMO reference annotations

This directory contains the publication exports listed in `manifest.json`. Only explicitly listed source fields and annotation payloads are released. Reference targets preserve known and unknown coverage; there are no source photographs, feature caches or model prediction arrays in this export.

IDRiD references inherit official lesion masks, qualified foveal coordinates and optic-disc annotations. DDR inherits official lesion masks and the recorded anatomical-reference provenance. Spatial and morphology query targets are deterministic derived annotations, not newly collected clinical diagnoses. Source roles retain the existing CORE/META/CAL assignment by content group.

Annotation assets are released under CC BY-NC-SA 4.0, with original attribution retained:
https://creativecommons.org/licenses/by-nc-sa/4.0/

- IDRiD: Porwal et al., *Indian Diabetic Retinopathy Image Dataset (IDRiD)*, Data 2018, 3(3), 25. https://doi.org/10.3390/data3030025
- DDR: Li et al., *Diagnostic Assessment of Deep Learning Algorithms for Diabetic Retinopathy Screening*, Information Sciences 2019. https://github.com/nkicsl/DDR-dataset

Original IDRiD annotations retain their original CC BY 4.0 attribution. DDR dataset material and adaptations follow its CC BY-NC-SA 4.0 terms. The repository's MIT source-code license does not relicense dataset assets.

## Files and matching

Directories use the official image filename without its extension. `image_file_name` gives the complete official filename; `public_id` is its stem. SHA-256 is provided for content verification. `source_order_index` defines record ordering. `content_group` is a publication identifier preserving content-group isolation, without copying internal identity fields.

Each record supplies four files:

- `targets.npz`: exact positive/known native pixel masses on the 256 and 28 grids; C4/S32/S-child8/M20 truth, completeness and confidence fields. Unknown image truth is −1. Fitting selects only the allowed C4/S28/M16 objectives from the full annotation inventory.
- `members.npz`: component-to-cell sparse masses and native areas.
- `components.npz`: native lesion membership, geometry, foveal point, OD centroid and disc scale, together with their defined/eligible flags. Field sources are recorded separately; model-derived anatomy is not an official manual reference.
- `groups.json`: qualifying group membership and geometric query metadata.

The original native-coordinate convention and square-padding transform are retained in `geometry`. Every payload has a byte count and SHA-256 in the manifest. `catalog_CS.json`, `catalog_M.json` and `M_core.json` contain query programs, text variants and the selected morphology core.
