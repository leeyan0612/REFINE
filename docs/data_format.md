# Frozen-feature and supervision interface

`refine.training.training_step` accepts prepared CORE and optional META batches. Reference tensors are consumed by losses and never passed to the image or text encoders.

## Prepared batch

Each prepared batch is a Python dictionary containing tensor values and ordinary lists/dictionaries:

- `local`: `[B,784,local_dim]`; `global_features`: `[B,global_dim]`.
- `support`: image-only Boolean `[B,256,256]`.
- `vectors`: C/S/M mappings to `[Q_axis,512]` sentence vectors.
- `axes`: C/S/M mappings containing `P`, `K`, `cells`, `truth`, `complete`, `weight`.
- `lesion_ids`: C/S/M mappings to integer category indices 0–3.
- `query_indices`: C `[0,1,2,3]`, S the 28 allowed indices, M the 16 allowed indices in the release configuration.
- `query_wording_indices`: the corresponding original training wording indices, each in 0–2.
- `members`: sparse member and group structures below.
- `positive_weights`: C/S/M and `primitive` vectors computed once from CORE reference masses; these are reused for META and never fitted on CAL or target evaluation data.

`P` and `K` have shape `[B,Q_axis,256,256]`. They contain positive and known reference masses. `cells` is `[B,256,256]` and contains total native area per output cell. The constraints are `0 <= P <= K <= cells`. `truth`, `complete`, and `weight` have shape `[B,Q_axis]`; truth is 1/0/−1 for positive/known-negative/unknown. Unknown support must not be recoded as negative. Role and query permissions are fixed before training.

## Member and group structures

The `members` dictionary uses the arrays `member_image`, `member_query`, `member_ptr`, `cell_index`, `member_area`, `member_cell_mass`. The CSR pointer partitions cell indices and native pixel masses by original lesion component. Members are deduplicated by original component identity within each image/query, and their masses sum exactly to their areas.

For META, also supply `group_image`, `group_query`, `group_ptr`, `group_member_indices`, preserving every qualifying group's original member associations. Required semantic flags are `members_deduplicated_by_original_component_id=True`, `original_group_member_association_preserved=True`, `group_members_uncropped=True`, `role='META'`, `loss_only=True`. The group-level metadata fields required by the numerical helper are documented by its validation checks in `satisfaction.py`.

Direct fitting averages omission over unique members. META feedback uses a smooth minimum over members of a group and a smooth maximum over qualifying groups; temperatures are both 0.05. Conditional spatial META feedback is computed only for fully qualified paired distance predicates and matched wording indices.

## Batch bundle for the training CLI

Store prepared tensor-only dictionaries using `torch.save`. They are read with `weights_only=True`. The bundle JSON has this structure:

```json
{
  "local_dim": 768,
  "global_dim": 512,
  "core_content_groups": ["core-example"],
  "meta_content_groups": ["meta-example"],
  "core_batches": ["batches/core000.pt"],
  "meta_batches": ["batches/meta000.pt"]
}
```

Use the dimensions in the model manifest. Batch paths are relative to the JSON, and the CLI visits the supplied streams cyclically.

Checkpoints contain the query-head parameters and registered buffers. Load them with `refine.load_model`.
