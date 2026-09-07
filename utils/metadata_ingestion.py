# utils/metadata_ingestion.py

"""
Common utilities for ingesting image metadata files.

Functions here convert a raw “tags” field into a list of tag tokens.
These helpers perform *no* normalisation on punctuation or case;
upstream code must preserve tags exactly.
"""

from typing import List, Optional
import hashlib
from pathlib import Path


RATING_TAGS = (
    "rating:general", "rating:sensitive", "rating:questionable", "rating:explicit",
)
RATING_ALIASES = {
    "g": RATING_TAGS[0], "general": RATING_TAGS[0], "safe": RATING_TAGS[0],
    "s": RATING_TAGS[1], "sensitive": RATING_TAGS[1],
    "q": RATING_TAGS[2], "questionable": RATING_TAGS[2],
    "e": RATING_TAGS[3], "explicit": RATING_TAGS[3],
    **{tag: tag for tag in RATING_TAGS},
}


def rating_to_tag(rating) -> Optional[str]:
    """Danbooru s means sensitive; legacy spelled-out safe means general."""
    if isinstance(rating, int) and not isinstance(rating, bool):
        return RATING_TAGS[rating] if 0 <= rating < len(RATING_TAGS) else None
    return RATING_ALIASES.get(str(rating).strip().lower())

def encode_rating_targets(tag_vec, tag_to_index, rating):
    """Overwrite rating targets: one-hot when known, -1 (unobserved) otherwise.

    The separate rating field wins over rating tokens in the tag list.
    ASL and metrics ignore -1; it is neither a positive nor a negative.
    """
    known = rating_to_tag(rating)
    for tag in RATING_TAGS:
        idx = tag_to_index.get(tag)
        if idx is not None and 0 <= idx < len(tag_vec):
            tag_vec[idx] = float(tag == known) if known else -1.0
    return tag_vec


def parse_tags_field(tags_field) -> List[str]:
    """Turn a raw tags field into a list of tokens.

    V2 namespaced strings are whitespace-delimited. Legacy comma-delimited
    strings and lists retain their exact tag spelling, punctuation and case.
    """
    if tags_field is None:
        return []
    if isinstance(tags_field, str):
        prefixes = ("gen:", "char:", "copyright:", "artist:", "meta:", "rating:")
        if tags_field.lstrip().startswith(prefixes) or ',' not in tags_field:
            return tags_field.split()
        return [p.strip() for p in tags_field.split(',') if p.strip()]
    if isinstance(tags_field, (list, tuple)):
        return [str(t).strip() for t in tags_field if str(t).strip()]
    return []


def annotation_tags(annotation) -> List[str]:
    """Unique positives per image, with the rating field as the source of truth."""
    tags = [t for t in parse_tags_field(annotation.get("tags"))
            if not t.startswith("rating:")]
    rating = rating_to_tag(annotation.get("rating"))
    if rating:
        tags.append(rating)
    return list(dict.fromkeys(tags))


def sidecar_image_id(path, filename, tags):
    """Namespace V2 IDs by directory so different subsets may reuse numeric IDs."""
    from utils.path_utils import sanitize_identifier
    stem = sanitize_identifier(Path(filename).stem)
    if any(t.startswith(("gen:", "char:", "copyright:", "artist:", "meta:")) for t in tags):
        namespace = hashlib.sha256(str(Path(path).parent).encode()).hexdigest()[:16]
        return f'{namespace}_{stem}'
    return stem
