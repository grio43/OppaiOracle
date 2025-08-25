# Orientation Map – How flips are decided

This JSON is consumed by the augmentation/inference pipeline to translate directional tags when an image is horizontally flipped.

## Sections

- **`explicit_mappings`**: Hard‑coded left↔right pairs (e.g., `left_eye_closed` ↔ `right_eye_closed`).
- **`regex_patterns`**: Generic patterns to swap left/right substrings in tags.
- **`symmetric_tags`**: Tags that are visually symmetric; flipping should not change them.
- **`skip_flip_tags`**: Tags where flipping is unsafe (e.g., any kind of text/watermark).
- **`complex_asymmetric_tags`**: Tags that need special handling or are only conditionally symmetric (e.g., `heterochromia`).

## Safety levels

The `data.orientation_safety_mode` sets how strictly we require a known mapping before flipping:

- `conservative` – Only flip when an explicit mapping applies.
- `balanced` – Flip when an explicit mapping or regex can swap tags; skip risky tags.
- `permissive` – Flip unless a tag appears in `skip_flip_tags`.

## Tips

- When adding new tags, prefer explicit pairs to avoid ambiguity.
- Keep `skip_flip_tags` conservative; UI/UX labels and watermarks should never be mirrored.
- If you need inline documentation in JSON, add `_comment` keys beside sections—these are ignored by our loaders but keep the file self‑describing.
